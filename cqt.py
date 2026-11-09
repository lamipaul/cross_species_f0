"""Constant-Q transform used to build the relative-pitch input of the model.

The CQT of a species spans [fmin * 2**(-max_shift/bpo), fmin * 2**((n_bins + max_shift)/bpo)[ with bpo bins per octave:
the n_bins central bins form the model input (bin 0 = fmin), and max_shift extra bins on each side allow pitch shifts by cropping
(supervised pitch-shift augmentation and PESTO's losses). Only the grid itself must be below Nyquist: margin bins above Nyquist are
filled with the frame minimum (as if the file had been upsampled: no content there).
The time axis is relative too: the hop is hop_periods periods of fmin (hop = hop_periods / fmin seconds). Since the constant-Q
windows also scale as 1 / frequency, a vocalisation slowed down and pitched down by the same factor (as with the downsampling
trick of bioacoustic_f0_estimation) gives the same input, whatever the species.
Kernels are computed in the time domain (like nnAudio's CQT1992v2 used by PESTO), grouped per octave to avoid
convolving high-frequency bins with the long kernels of the lowest octave.
"""
import hashlib, json, functools
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


def add_cqt_args(parser):
    parser.add_argument('--bins_per_octave', type=int, default=120, help="CQT resolution (120 = 10 cents)")
    parser.add_argument('--n_octaves', type=int, default=6, help="Number of octaves above fmin covered by the model")
    parser.add_argument('--hop_periods', type=float, default=0.9, help="CQT hop in periods of the species' fmin (hop = hop_periods / fmin seconds; 0.9: 10 ms for canids, 0.4 ms for bats, 47 ms for lions)")
    parser.add_argument('--filter_scale', type=float, default=0.3, help="Scales the CQT kernel lengths. 1: windows of 1 / (2 ** (1 / bins_per_octave) - 1) = 173 periods at 120 bins/octave; 0.3 (default): 52 periods, the time-frequency trade-off of PESTO (Q of a 3 bins/semitone CQT)")
    parser.add_argument('--max_shift', type=int, default=180, help="Extra cached bins on each side of the grid, i.e. the largest pitch shift by cropping (180 = 1.5 octaves)")
    parser.add_argument('--cache_dir', type=str, default='cache', help="Folder for the cached CQTs")


def get_params(args):
    return dict(bins_per_octave=args.bins_per_octave, n_bins=args.bins_per_octave * args.n_octaves,
                hop_periods=args.hop_periods, filter_scale=args.filter_scale, max_shift=args.max_shift)


def hop_length(FS, fmin, p):
    """ CQT hop in samples for a species: hop_periods periods of fmin """
    return max(1, round(p['hop_periods'] * FS / fmin))


def cache_folder(cache_dir, p):
    """ One cache folder per set of CQT parameters (as PESTO does with its hash-named cache files) """
    h = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:10]
    return f'{cache_dir}/cqt_{h}'


def max_fmin(FS, p):
    """ Highest fmin such that the top bin of the grid stays under Nyquist (margin bins above Nyquist are filled, see compute_cqt) """
    return FS / 2 / 2 ** ((p['n_bins'] - 1) / p['bins_per_octave'])


def bin2hz(bins, fmin, bpo):
    return fmin * 2 ** (np.asarray(bins) / bpo)


def hz2bin(f, fmin, bpo):
    f = np.asarray(f, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(f > 0, bpo * np.log2(f / fmin), np.nan)


class CQT(nn.Module):
    def __init__(self, sr, hop_length, fmin, n_bins, bins_per_octave=120, filter_scale=1.):
        super(CQT, self).__init__()
        self.hop_length = hop_length
        Q = filter_scale / (2 ** (1 / bins_per_octave) - 1)
        freqs = fmin * 2 ** (np.arange(n_bins) / bins_per_octave)
        assert freqs[-1] < sr / 2, f'Highest CQT bin ({freqs[-1]:.0f}Hz) is above Nyquist ({sr/2:.0f}Hz)'
        lengths = np.ceil(Q * sr / freqs).astype(int)
        self.groups = []
        for i, start in enumerate(range(0, n_bins, bins_per_octave)):
            idx = np.arange(start, min(start + bins_per_octave, n_bins))
            L = lengths[idx].max() | 1  # odd length so that frames are centred
            kernels = np.zeros((len(idx), L), dtype=np.complex64)
            for j, k in enumerate(idx):
                l = lengths[k]
                win = np.hanning(l)
                t = (np.arange(l) - (l - 1) / 2) / sr
                off = (L - l) // 2
                kernels[j, off:off + l] = win * np.exp(2j * np.pi * freqs[k] * t) / win.sum()
            self.register_buffer(f'real_{i}', torch.tensor(kernels.real).unsqueeze(1), persistent=False)
            self.register_buffer(f'imag_{i}', torch.tensor(kernels.imag).unsqueeze(1), persistent=False)
            self.groups.append(L)

    def forward(self, x):
        """ x: (batch, samples) -> magnitudes (batch, n_bins, frames), frame j centred on sample j * hop_length """
        x = x.unsqueeze(1)
        n_frames = (x.shape[-1] - 1) // self.hop_length + 1
        out = []
        for i, L in enumerate(self.groups):
            xp = F.pad(x, (L // 2, L // 2))
            re = F.conv1d(xp, getattr(self, f'real_{i}'), stride=self.hop_length)
            im = F.conv1d(xp, getattr(self, f'imag_{i}'), stride=self.hop_length)
            out.append(torch.sqrt(re ** 2 + im ** 2)[..., :n_frames])
        return torch.cat(out, dim=1)


@functools.lru_cache(maxsize=8)
def get_cqt(FS, fmin, p_json, device):
    p = json.loads(p_json)
    hop = hop_length(FS, fmin, p)
    f0 = fmin * 2 ** (-p['max_shift'] / p['bins_per_octave'])
    n_total = p['n_bins'] + 2 * p['max_shift']
    n_valid = int(np.sum(f0 * 2 ** (np.arange(n_total) / p['bins_per_octave']) < FS / 2))  # bins below Nyquist
    return CQT(FS, hop, f0, n_valid, p['bins_per_octave'], p['filter_scale']).to(device), hop, n_total


def compute_cqt(sig, FS, fmin, p, device='cpu'):
    """ Returns the log-magnitude CQT (in dB) of shape (frames, n_bins + 2 * max_shift) as float16, and the frame times
    (frame j is centred on sample j * hop_length(FS, fmin, p)). Bins above Nyquist get the frame minimum. """
    if fmin > max_fmin(FS, p):
        raise ValueError(f'fmin={fmin}Hz is too high for FS={FS}Hz: the grid would go over Nyquist (max fmin is {max_fmin(FS, p):.1f}Hz)')
    cqt, hop, n_total = get_cqt(FS, fmin, json.dumps(p, sort_keys=True), device)
    with torch.no_grad():
        mag = cqt(torch.as_tensor(sig, dtype=torch.float32, device=device).unsqueeze(0))[0]
    x = 20 * torch.log10(mag.clamp(min=1e-8)).T
    if x.shape[1] < n_total:
        x = torch.cat([x, x.amin(1, keepdim=True).expand(len(x), n_total - x.shape[1])], 1)
    x = x.cpu().numpy().astype(np.float16)
    return x, np.arange(len(x)) * hop / FS
