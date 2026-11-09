"""Computes and caches the CQT and frame-level F0 labels of every sound file.

For each file, stores in {cache_folder}/{specie}/{key}.npy the log-magnitude CQT (frames, n_bins + 2 * max_shift),
and in {key}_f0.npy the F0 annotation resampled on the CQT frames (Hz, 0 = unvoiced, NaN = unknown).
"""
import os, argparse, json
from glob import glob
import numpy as np, pandas as pd, librosa, mir_eval, torch, soundfile
from tqdm import tqdm
from metadata import get_species, list_files
from cqt import add_cqt_args, get_params, cache_folder, compute_cqt, hop_length


def cache_key(fn):
    """ Unique cache name for a sound file (several species have files with identical basenames in different folders) """
    return os.path.splitext(os.path.normpath(fn))[0].replace('/', '__').replace(' ', '_')


def manifest_path(folder, specie):
    """ List of the species' sound files written by preprocess.py next to their cached CQTs, so that training and evaluation can run
    from the cache alone (e.g. on a cluster where only the cache was copied) with the same files and the same fold assignment """
    return f'{folder}/{specie}/files.txt'


def species_files(meta, folder, specie):
    """ Sorted sound files of a species: listed from the audio data when it is present, from the cache manifest otherwise """
    files = list_files(meta)
    if not files and os.path.isfile(manifest_path(folder, specie)):
        files = open(manifest_path(folder, specie)).read().splitlines()
    return files


def annot_offset(fn, meta):
    """ Time correction of a file's annotations (s), from metadata 'annot_offset_samples' (in samples at the file's native sampling rate).
    La Palma chaffinches: the annotation tool time-stamped spectrogram frames at their start (frame index x hop) instead of their centre,
    i.e. NFFT / 2 = 256 samples too early (annotation step = hop = 256 samples, 5.33 ms at 48 kHz / 5.80 ms at 44.1 kHz). Measured on
    2026-11-07: the CQT energy at the label is maximal with this correction (mean z 2.04 -> 2.29), and the cut times of the clips are not involved. """
    n = meta.get('annot_offset_samples', 0)
    return n / soundfile.info(fn).samplerate if n else 0.


def load_annot(fn, time, unknown_gap=1.5, offset=0.):
    """ Resamples annotations on the given frame times, following paper_experiments/run_all.py.
    Returns the frame F0 (Hz, 0 = unvoiced, NaN = unknown).
    mir_eval holds voicing between annotation points, so a gap without points is interpolated as voiced. As in run_all.py, zeros
    are inserted in gaps longer than 4 times the median annotation step (unvoiced). Additionally, frames inside gaps between
    unknown_gap and 4 times the median step are set to NaN (unknown, excluded from losses and evaluation): measured on canids
    (irregular annotation steps), their spectral level at the label lies between voiced and unvoiced frames, i.e. many are silences.
    offset: added to the annotation times (s), to correct a systematic time offset of a species' annotations (see annot_offset).
    """
    if not os.path.isfile(fn[:-4] + '.csv'):
        return np.full(len(time), np.nan, dtype=np.float32)
    annot = pd.read_csv(fn[:-4] + '.csv').drop_duplicates(subset='Time').sort_values('Time')
    annot['Time'] += offset
    t = annot.Time.values
    # add a 0 at starts and ends of large gaps to avoid interpolating between vocalisations
    med_diff = annot.Time.diff().median()
    rgaps, lgaps = annot.Time[annot.Time.diff() > med_diff * 4], annot.Time[annot.Time.diff(-1) < - med_diff * 4]
    annot = pd.concat([annot, pd.DataFrame({'Time': np.concatenate([lgaps + med_diff, rgaps - med_diff]), 'Freq': [0] * (len(lgaps) + len(rgaps))})]).sort_values('Time')
    f0 = np.zeros(len(time), dtype=np.float32)  # frames outside of the annotated section are unvoiced
    mask = (time > annot.Time.min()) & (time < annot.Time.max())
    if mask.sum() > 0:
        f0[mask] = mir_eval.melody.resample_melody_series(annot.Time.values, annot.Freq.values, annot.Freq.values > 0, time[mask])[0]
    if len(t) > 1 and unknown_gap:
        i = np.clip(np.searchsorted(t, time), 1, len(t) - 1)
        gap = t[i] - t[i - 1]
        inside = (time > t[i - 1]) & (time < t[i]) & (gap > unknown_gap * med_diff) & (gap <= 4 * med_diff)
        f0[inside & (f0 > 0)] = np.nan
    return f0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('specie', type=str, nargs='?', default=None, help="Species to process (default: all species with an fmin in metadata.py)")
    parser.add_argument('--fake', action='store_true', help="Use the synthetic species from make_fake_data.py")
    parser.add_argument('--overwrite', action='store_true', help="Recompute already cached files")
    parser.add_argument('--labels_only', action='store_true', help="Only recompute the frame labels ({key}_f0.npy) of already cached CQTs")
    add_cqt_args(parser)
    args = parser.parse_args()
    import warnings
    warnings.filterwarnings('ignore', message='Non-uniform timescale')  # expected after zero insertion, see load_annot

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p = get_params(args)
    folder = cache_folder(args.cache_dir, p)
    os.makedirs(folder, exist_ok=True)
    json.dump(p, open(f'{folder}/params.json', 'w'), indent=1)

    for specie, meta in get_species(only=args.specie, fake=args.fake).items():
        os.makedirs(f'{folder}/{specie}', exist_ok=True)
        # the cache folder only depends on CQT parameters: refuse to mix files computed with another FS / fmin
        sp = {'FS': meta['FS'], 'fmin': meta['fmin']}
        sp_file = f'{folder}/{specie}/species.json'
        if os.path.isfile(sp_file) and json.load(open(sp_file)) != sp and not args.overwrite:
            raise ValueError(f'{specie} was cached with {json.load(open(sp_file))}, metadata.py now says {sp}: rerun with --overwrite')
        json.dump(sp, open(sp_file, 'w'))
        fns = list_files(meta)
        if not fns and os.path.isfile(manifest_path(folder, specie)):
            print(f'{specie}: no audio data, using the {len(species_files(meta, folder, specie))} files listed in the cache ({manifest_path(folder, specie)})')
            continue
        if fns:
            with open(manifest_path(folder, specie), 'w') as f:
                f.write('\n'.join(fns) + '\n')
        else:
            parts = meta['wavpath'].split('/')
            broken = [d for d in ('/'.join(parts[:k]) for k in range(1, len(parts))) if os.path.islink(d) and not os.path.exists(d)]
            print(f"WARNING: no sound file found for {specie} at {meta['wavpath']} (from {os.getcwd()})"
                  + (f': {broken[0]} is a broken symbolic link to {os.readlink(broken[0])} (copy the data with rsync -L)' if broken else ''))
        for fn in tqdm(fns, desc=specie):
            out = f'{folder}/{specie}/{cache_key(fn)}'
            if args.labels_only:
                n_frames = len(np.load(out + '.npy', mmap_mode='r'))
                hop = hop_length(meta['FS'], meta['fmin'], p)
                np.save(out + '_f0.npy', load_annot(fn, np.arange(n_frames) * hop / meta['FS'], offset=annot_offset(fn, meta)))
                continue
            if os.path.isfile(out + '_f0.npy') and not args.overwrite:
                continue
            sig, fs = librosa.load(fn, sr=meta['FS'])
            try:
                x, time = compute_cqt(sig, meta['FS'], meta['fmin'], p, device)
            except torch.cuda.OutOfMemoryError:  # very long files with long low-frequency kernels: fall back to the CPU
                torch.cuda.empty_cache()
                tqdm.write(f'GPU out of memory for {fn} ({len(sig) / meta["FS"]:.0f}s), computing on CPU')
                x, time = compute_cqt(sig, meta['FS'], meta['fmin'], p, 'cpu')
            np.save(out + '.npy', x)
            np.save(out + '_f0.npy', load_annot(fn, time, offset=annot_offset(fn, meta)))
    print('CQTs cached in', folder)
