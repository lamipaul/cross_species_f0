import numpy as np, torch
from metadata import list_files
from preprocess import cache_key, species_files
from cqt import hz2bin, hop_length


def get_files(meta, split=None, test=False, n_folds=5, folder=None, specie=None):
    """ 5-fold split over the sorted file list, as in paper_experiments/train_crepe.py (split=None: all files are train).
    With folder (cache folder) and specie, the list falls back to the cache manifest when the audio data is absent (preprocess.species_files) """
    files = species_files(meta, folder, specie) if folder else list_files(meta)
    if split is None:
        return [] if test else files
    a, b = int(len(files) / n_folds * split), int(len(files) / n_folds * (split + 1))
    return files[a:b] if test else files[:a] + files[b:]


def loader_kwargs(num_workers):
    """ DataLoader worker options. Workers are forked explicitly: since Python 3.14 the default start method on Linux is forkserver,
    which re-imports the main script in each worker, and our training scripts run at module level (no __main__ guard).
    Fork also avoids pickling the dataset for every worker. Workers only use the CPU, so forking after CUDA initialisation is safe. """
    if num_workers == 0:
        return dict(num_workers=0)
    return dict(num_workers=num_workers, multiprocessing_context='fork', persistent_workers=True)


def load_cached(folder, specie, fn):
    """ Cached log-magnitude CQT (dB, float16) and frame-level F0 (Hz, 0 = unvoiced, NaN = unknown).
    The CQT is not normalised: the encoder starts with a per-frame LayerNorm, which removes any per-clip affine transform of the dB values """
    return np.load(f'{folder}/{specie}/{cache_key(fn)}.npy'), np.load(f'{folder}/{specie}/{cache_key(fn)}_f0.npy')


def label_uncertainty(bins, meta, p, max_cents=100.):
    """ Uncertainty (std, in bins) of frame labels traced on spectrograms (bins: label bins of one file, NaN where there is no pitch label).
    Two independent sources, added in quadrature:
    - timing: an annotation point is only located within one annotation step (metadata 'annot_step', s), i.e. a uniform time error of
      std step / sqrt(12); on the pitch axis it becomes |contour slope| x step / sqrt(12). The slope is the largest label change between
      the frame and its neighbours, in bins per second.
    - frequency grid: when the annotated values are quantised on a spectrogram grid of 'annot_grid_hz' Hz, the rounding error has
      std (grid width in bins at the label frequency) / sqrt(12).
    The total is capped at max_cents: the timing model is pessimistic for long annotation steps (it ignores that points are placed on the
    contour, so the error between points comes from interpolation rather than from a full step of timing error).
    Missing metadata fields give no contribution. Frames without a pitch label get 0. """
    bpo = p['bins_per_octave']
    unc2 = np.zeros(len(bins))
    if meta.get('annot_step'):
        hop_s = hop_length(meta['FS'], meta['fmin'], p) / meta['FS']
        slope = np.fmax(np.abs(np.diff(bins, prepend=np.nan)), np.abs(np.diff(bins, append=np.nan))) / hop_s  # bins / s, NaN-ignoring
        unc2 += np.nan_to_num(slope * meta['annot_step'] / np.sqrt(12)) ** 2
    if meta.get('annot_grid_hz'):
        d, f = meta['annot_grid_hz'], meta['fmin'] * 2 ** (bins / bpo)
        f = np.where(f > d, f, np.nan)
        unc2 += np.nan_to_num(bpo * np.log2((f + d / 2) / (f - d / 2)) / np.sqrt(12)) ** 2
    return np.where(np.isnan(bins), 0, np.minimum(np.sqrt(unc2), max_cents * bpo / 1200)).astype(np.float32)


class WindowDataset(torch.utils.data.Dataset):
    """ Windows of `context` consecutive CQT frames with their frame-level labels.

    x is the log-magnitude CQT in dB (see load_cached), so that noise can be mixed in the power domain.

    Returns x (context, n_bins + 2 * max_shift), pitch bin targets (context,) (NaN if unvoiced, unknown or out of the grid),
    their uncertainty in bins (context,) (see label_uncertainty),
    voicing (context,), a validity mask (context,) (0 for padding and frames without annotation),
    a frame mask (context,) (bool, False for padding: clips shorter than the context window),
    and a noise segment like x (dB, taken from unvoiced annotated frames of the dataset's files, tiled if needed),
    or an empty tensor if noise=False.
    random_offset: shifts each window start by a random amount in [0, stride) (training).
    time_aug, p_time: with probability p_time, the window is rescaled in time by a factor 2 ** u, u uniform in [-time_aug, time_aug] (octaves):
    its context frames are read every 2 ** u cached frames (linear interpolation of the CQT and of the pitch labels, nearest frame for the
    other labels). This is equivalent to computing the CQT with a hop 2 ** u times longer, since the analysis windows do not depend on the hop.
    in_memory: load all CQTs in RAM at construction (float16, about 1.7 kB per frame) instead of memory-mapping them on access.
    Use it when RAM allows (e.g. cluster nodes): random reads from disk can be the bottleneck. With forked
    DataLoader workers (loader_kwargs), the arrays are shared copy-on-write, not duplicated.
    """
    def __init__(self, species, folder, p, files, context=10, stride=None, labelled_only=True, random_offset=False, noise=False, min_noise_run=10, in_memory=False,
                 time_aug=0., p_time=0.):
        super(WindowDataset, self).__init__()
        self.context, self.p, self.stride, self.random_offset = context, p, stride or context, random_offset
        self.time_aug, self.p_time = time_aug, p_time
        self._open, self.max_open = {}, 256  # memory maps kept open by read()
        self.x = [] if in_memory else None  # in-memory CQTs
        # CQTs stay on disk and are memory-mapped when a window is read (the OS page cache keeps the frequently used parts in RAM);
        # files are opened on access only, to keep the number of open files low
        self.paths, self.lengths, self.bins, self.unc, self.voiced, self.valid = [], [], [], [], [], []
        win_file, win_start, specie_of = [], [], []
        for s, (specie, meta) in enumerate(species.items()):
            for fn in files[specie]:
                f0 = np.load(f'{folder}/{specie}/{cache_key(fn)}_f0.npy')
                if labelled_only and np.isnan(f0).all():
                    continue
                bins = hz2bin(f0, meta['fmin'], p['bins_per_octave'])
                bins[(bins < 0) | (bins > p['n_bins'] - 1)] = np.nan  # voiced but out of the pitch grid: no pitch target
                i, n = len(self.paths), len(f0)
                self.paths.append(f'{folder}/{specie}/{cache_key(fn)}.npy')
                if in_memory:
                    self.x.append(np.load(self.paths[-1]))
                self.lengths.append(n)
                self.bins.append(bins.astype(np.float32))
                self.unc.append(label_uncertainty(bins, meta, p))
                self.voiced.append((f0 > 0).astype(np.float32))
                self.valid.append((~np.isnan(f0)).astype(np.float32))
                starts = np.arange(0, max(1, n - context + 1), self.stride)
                if n > context and starts[-1] != n - context:
                    starts = np.append(starts, n - context)
                win_file.append(np.full(len(starts), i, dtype=np.int32)), win_start.append(starts.astype(np.int32)), specie_of.append(np.full(len(starts), s, dtype=np.int16))
        if not win_file:
            raise ValueError(f'No cached file found in {folder} for species {list(species)} ({sum(len(f) for f in files.values())} sound files listed): '
                             'check that the data paths of metadata.py exist from the current folder and that preprocess.py has run')
        self.win_file, self.win_start, self.specie_of = np.concatenate(win_file), np.concatenate(win_start), np.concatenate(specie_of)
        # noise bank: runs of annotated unvoiced frames (file index, start, length)
        self.noise_runs = []
        if noise:
            min_run = min(min_noise_run, context)
            for i, (v, ok) in enumerate(zip(self.voiced, self.valid)):
                unv = np.concatenate([[0], (ok > 0) & (v == 0), [0]]).astype(np.int8)
                d = np.diff(unv)
                for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                    if b - a >= min_run:
                        self.noise_runs.append((i, a, b - a))
            if not self.noise_runs:
                print('Warning: no unvoiced annotated segment found, noise augmentation disabled')
        self.noise = noise and len(self.noise_runs) > 0
        if in_memory:
            print(f'CQTs loaded in RAM: {sum(x.nbytes for x in self.x) / 1e9:.1f} GB')
        if self.noise:
            print(f'Noise bank: {len(self.noise_runs)} unvoiced segments, {sum(r[2] for r in self.noise_runs)} frames')

    def __len__(self):
        return len(self.win_file)

    def read(self, i, a, b):
        """ CQT frames a:b of file i (dB), as float32. The most recently used memory maps are kept open (per process, at most
        max_open files): opening a .npy file costs about 1 ms, much more than reading a window from it """
        if self.x is not None:
            return self.x[i][a:b].astype(np.float32)
        mm = self._open.pop(i, None)
        if mm is None:
            mm = np.load(self.paths[i], mmap_mode='r')
            if len(self._open) >= self.max_open:
                self._open.pop(next(iter(self._open)))  # least recently used
        self._open[i] = mm
        return np.asarray(mm[a:b], dtype=np.float32)

    def sample_noise(self):
        i, a, L = self.noise_runs[torch.randint(len(self.noise_runs), (1,)).item()]
        if L >= self.context:  # random segment of the run
            s0 = a + torch.randint(L - self.context + 1, (1,)).item()
            return torch.tensor(self.read(i, s0, s0 + self.context))
        seg = self.read(i, a, a + L)  # short run: tiled from a random start
        return torch.tensor(seg[(torch.randint(L, (1,)).item() + np.arange(self.context)) % L])

    def rescaled(self, i, st, r):
        """ context frames of file i read every r cached frames from frame st: CQT and pitch labels linearly interpolated, other labels
        from the nearest frame. Positions past the end of the file are dropped (the window is then padded by the caller) """
        n = self.lengths[i]
        pos = st + np.arange(self.context) * r
        pos = pos[pos <= n - 1]
        a, b = int(np.floor(pos[0])), min(n - 1, int(np.ceil(pos[-1])))
        seg = self.read(i, a, b + 1)
        lo = np.floor(pos).astype(int) - a; hi = np.minimum(lo + 1, b - a); w = (pos - a - lo)[:, None]
        x = (1 - w) * seg[lo] + w * seg[hi]
        bl, bh = self.bins[i][a + lo], self.bins[i][a + hi]
        near = a + np.round(pos - a).astype(int).clip(0, b - a)
        bins = np.where(np.isnan(bl) | np.isnan(bh), self.bins[i][near], (1 - w[:, 0]) * bl + w[:, 0] * bh).astype(np.float32)
        return x.astype(np.float32), bins, self.unc[i][near], self.voiced[i][near], self.valid[i][near]

    def __getitem__(self, idx):
        i, st = int(self.win_file[idx]), int(self.win_start[idx])
        n = self.lengths[i]
        if self.random_offset and n > self.context:
            st = min(st + torch.randint(self.stride, (1,)).item(), n - self.context)
        if self.time_aug > 0 and torch.rand(1).item() < self.p_time:
            r = 2 ** ((2 * torch.rand(1).item() - 1) * self.time_aug)
            x, bins, unc, voiced, valid = self.rescaled(i, st, r)
        else:
            sl = slice(st, st + self.context)
            x, bins, unc, voiced, valid = self.read(i, sl.start, sl.stop), self.bins[i][sl], self.unc[i][sl], self.voiced[i][sl], self.valid[i][sl]
        pad = self.context - len(x)
        present = np.arange(self.context) < len(x)
        if pad > 0:  # clips shorter than the context window (padded frames are masked out, their value does not matter)
            x = np.pad(x, ((0, pad), (0, 0)), mode='edge')
            bins, unc, voiced, valid = np.pad(bins, (0, pad), constant_values=np.nan), np.pad(unc, (0, pad)), np.pad(voiced, (0, pad)), np.pad(valid, (0, pad))
        noise = self.sample_noise() if self.noise else torch.zeros(0)
        return torch.tensor(x), torch.tensor(bins), torch.tensor(unc), torch.tensor(voiced), torch.tensor(valid), torch.tensor(present), noise

    def sample_weights(self):
        """ Inverse species frequency, as in paper_experiments/train_crepe.py """
        counts = np.bincount(self.specie_of)
        return torch.tensor(len(self) / counts[self.specie_of], dtype=torch.double)
