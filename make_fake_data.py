"""Generates synthetic vocalisations to test the pipeline, in the dataset format of bioacoustic_f0_estimation:
one .wav per vocalisation and a .csv with columns Time (s) and Freq (Hz) covering the voiced section.
Contours combine a glide, vibrato and occasional frequency jumps; harmonics decay randomly; background noise has random SNR.
"""
import os, argparse
import numpy as np, pandas as pd, soundfile as sf
from metadata import fake_species

parser = argparse.ArgumentParser()
parser.add_argument('--n_files', type=int, default=20, help="Number of files per species")
parser.add_argument('--max_duration', type=float, default=1.5, help="Maximum file duration (s)")
parser.add_argument('--n_unlabelled', type=int, default=5, help="Number of extra files without annotations per species (for SSL)")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)

for specie, meta in fake_species.items():
    folder = os.path.dirname(meta['wavpath'])
    os.makedirs(folder, exist_ok=True)
    FS, (lo, hi) = meta['FS'], meta['f0_range']
    for i in range(args.n_files + args.n_unlabelled):
        dur = rng.uniform(0.3, args.max_duration)
        t = np.arange(int(dur * FS)) / FS
        on, off = rng.uniform(0.05, 0.3) * dur, rng.uniform(0.7, 0.95) * dur
        # F0 contour: log-linear glide + vibrato + optional jump
        f_start, f_end = np.exp(rng.uniform(np.log(lo), np.log(hi), 2))
        contour = f_start * (f_end / f_start) ** np.clip((t - on) / (off - on), 0, 1)
        contour *= 2 ** (rng.uniform(0, 1) * np.sin(2 * np.pi * rng.uniform(2, 15) * t) / 12)
        if rng.random() < 0.3:
            contour[t > rng.uniform(on, off)] *= 2 ** (rng.choice([-1, 1]) * rng.uniform(2, 7) / 12)
        contour = np.clip(contour, lo, hi)
        phase = 2 * np.pi * np.cumsum(contour) / FS
        decay = rng.uniform(0.3, 0.9)
        sig = sum(decay ** (h - 1) * np.sin(h * phase) for h in range(1, 8) if np.all(h * contour < FS / 2))
        env = ((t > on) & (t < off)).astype(float)
        ramp = int(0.01 * FS)
        env = np.convolve(env, np.hanning(2 * ramp) / ramp, mode='same')
        sig = sig * env
        snr_db = rng.uniform(0, 30)
        noise = rng.standard_normal(len(t))
        noise *= np.sqrt(np.mean(sig[env > 0.5] ** 2) / 10 ** (snr_db / 10)) / noise.std()
        sig = sig + noise
        fn = f'{folder}/{specie}_{i:03d}.wav'
        sf.write(fn, (0.5 * sig / np.abs(sig).max()).astype(np.float32), FS)
        if i < args.n_files:  # annotations every 5 ms over the voiced section
            ta = np.arange(on + 0.01, off - 0.01, 0.005)
            pd.DataFrame({'Time': ta, 'Freq': np.interp(ta, t, contour)}).to_csv(fn[:-4] + '.csv', index=False)
    print(f'{specie}: {args.n_files} labelled and {args.n_unlabelled} unlabelled files in {folder}')
