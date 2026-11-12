"""Estimates F0 for new recordings with a trained model, following predict.py of bioacoustic_f0_estimation.
Saves {basename}_f0.csv with time, f0, confidence, salience_max (peak of the model's pitch distribution) and salience (spectral salience
of the estimated F0, see contour_salience) columns, and optionally a spectrogram with overlaid predictions.
Sampling rate and fmin (bottom of the 6-octave pitch grid) come from --specie, --FS / --fmin, or by default from each file: native
sampling rate and the highest fmin it allows (the grid then covers the 6 octaves below Nyquist). The resulting range is printed.
"""
import math, argparse, os, warnings
import numpy as np, pandas as pd, librosa, soundfile, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metadata import species, fake_species
from cqt import compute_cqt, bin2hz, max_fmin, hop_length
from model import load_model, infer
from decode import weighted_argmax
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help="Directory with sound files to process, or a single file to process")
parser.add_argument('--model_path', type=str, required=True, help="Path of model weights (checkpoint saved by train.py)")
parser.add_argument('--specie', type=str, default=None, help="Species from metadata.py, to get the sampling rate and fmin")
parser.add_argument('--FS', type=int, default=None, help="Sampling rate (overrides --specie; default: native rate of each file)")
parser.add_argument('--fmin', type=float, default=None, help="Bottom of the 6-octave pitch grid in Hz (overrides --specie; default: highest value allowed by the sampling rate)")
parser.add_argument('--step', type=float, default=None, help="Time step between predictions in seconds, i.e. the CQT hop given to the model (default: the model's own hop, 0.9 / fmin). The hop is part of the input representation: models trained with time rescaling (train.py --time_aug) have seen hops within 2 ** +-time_aug of their own; outside that range a warning is printed")
parser.add_argument('--num_chunks', type=int, default=None, help="Number of chunks the CQT of each file is computed in, to bound memory (default: from the file length, FS, fmin and the free memory, see cqt.compute_cqt); results do not change")
parser.add_argument('--use_channel', type=int, help='ID of the channel to use (starting from 0, if not specified, will use a mix of all)', default=None)
parser.add_argument('--no_print', action='store_true', help="Skip printing spectrograms with overlaid F0 predictions")
parser.add_argument('--threshold', type=float, default=0.5, help="Confidence threshold used when printing F0 predictions on spectrograms")
parser.add_argument('--NFFT', type=int, default=1024, help="Window size used for printing spectrograms")
args = parser.parse_args()

meta = {**species, **fake_species}.get(args.specie, {}) if args.specie else {}
assert not args.specie or meta, f'{args.specie} is not a species of metadata.py'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, p = load_model(args.model_path, device)
bps = p['bins_per_octave'] // 12
time_aug = torch.load(args.model_path, map_location='cpu', weights_only=False).get('args', {}).get('time_aug') or 0.  # hop range seen in training (octaves)
reported, step_reported = set(), set()


def settings(filepath):
    """ Sampling rate and fmin for a file: --FS / --fmin, else --specie, else native rate and the highest allowed fmin.
    If only fmin is given and it is too high for the native rate, the file is upsampled to the lowest rate that allows it
    (bins above the native Nyquist frequency are just empty), as done for little owls in metadata.py """
    native = soundfile.info(filepath).samplerate
    FS, fmin, note = args.FS or meta.get('FS'), args.fmin or meta.get('fmin'), ''
    if FS is None:
        FS = native
        if fmin and fmin > max_fmin(FS, p):
            FS = int(math.ceil(fmin / max_fmin(1, p)))  # max_fmin is proportional to the sampling rate
            note = f' (upsampled from {native}Hz to allow fmin={fmin}Hz)'
    if fmin is None:
        fmin, note = math.floor(max_fmin(FS, p)), ' (fmin set automatically, use --fmin or --specie to change it)'
    if fmin > max_fmin(FS, p):
        raise ValueError(f'fmin={fmin}Hz is too high for FS={FS}Hz (at most {max_fmin(FS, p):.1f}Hz): lower --fmin or raise --FS')
    if (FS, fmin) not in reported:
        reported.add((FS, fmin))
        print(f'FS={FS}Hz, fmin={fmin}Hz: the model\'s F0 estimations range from {fmin:.1f}Hz to {bin2hz(p["n_bins"], fmin, p["bins_per_octave"]):.1f}Hz' + note)
    return FS, fmin

def contour_salience(x, bins, start, bpo):
    """ Spectral salience of the estimated F0 in each frame, as in bioacoustic_f0_estimation (predict.py, voc_characterisation.py):
    energy within +-1 semitone of the F0 over energy within +-6 semitones, on magnitudes whitened per frequency bin with the file's
    median and standard deviation over frames (clipped to be positive). Computed on the model's CQT instead of a fixed-size FFT:
    the +-1 / +-6 semitone bands always span the same number of bins, so values do not depend on the F0 or the sampling rate.
    x: CQT in dB (frames, cached bins), bins: estimated F0 (grid bins), start: index of grid bin 0 in x. Values in ]0, 1]. """
    mag = 10 ** (x.astype(np.float32) / 20)
    z = np.clip((mag - np.median(mag, 0)) / (mag.std(0) + 1e-12), 1e-12, 1e3)
    k1, k6 = bpo // 12, bpo // 2
    zp = np.pad(z, ((0, 0), (k6, k6)))  # bands reaching outside the cached bins count as empty
    win = zp[np.arange(len(z))[:, None], np.round(bins).astype(int)[:, None] + start + np.arange(2 * k6 + 1)]  # +-6 semitones around the F0
    return win[:, k6 - k1:k6 + k1 + 1].sum(1) / win.sum(1)


if os.path.isdir(args.input):
    files = [os.path.join(args.input, fn) for fn in sorted(os.listdir(args.input)) if fn.split('.')[-1].upper() in list(soundfile._formats) + ['AIF']]
elif os.path.isfile(args.input) and args.input.split('.')[-1].upper() in list(soundfile._formats) + ['AIF']:
    files = [args.input]
else:
    files = []
if len(files) == 0:
    raise Exception(f"The given input argument {args.input} is either not a valid directory/file path, or it doesn't contain any sound file of the supported formats")

for ifile, filepath in enumerate(files):
    desc = f'{ifile + 1}/{len(files)}: {os.path.basename(filepath)}'  # progress bars per file, as in bioacoustic_f0_estimation
    FS, fmin = settings(filepath)
    sig, fs = librosa.load(filepath, sr=FS, mono=args.use_channel is None)
    if sig.ndim > 1:
        sig = sig[args.use_channel]
    pf = {**p, 'hop_periods': args.step * fmin} if args.step else p  # CQT parameters for this file: same bins, requested hop
    if args.step and (FS, fmin) not in step_reported:
        step_reported.add((FS, fmin))
        native, used = hop_length(FS, fmin, p) / FS, hop_length(FS, fmin, pf) / FS
        print(f'Step {used * 1000:.2f}ms (model\'s own hop at fmin={fmin}Hz: {native * 1000:.2f}ms, ratio {used / native:.2f})'
              + (f'. WARNING: outside the hops seen in training ({native * 2 ** -time_aug * 1000:.2f}-{native * 2 ** time_aug * 1000:.2f}ms), predictions may degrade'
                 if abs(np.log2(used / native)) > time_aug + 1e-3 else ''))
    x, time = compute_cqt(sig, FS, fmin, pf, device, args.num_chunks, desc=desc + ' (CQT)')
    salience, confidence = infer(model, x, p, device, desc=desc)
    bins = weighted_argmax(salience, bps)
    f0 = bin2hz(bins, fmin, p['bins_per_octave'])
    pd.DataFrame({'time': time, 'f0': f0, 'confidence': confidence, 'salience_max': salience.max(1),
                  'salience': contour_salience(x, bins, p['max_shift'], p['bins_per_octave'])}).to_csv(filepath.rsplit('.', 1)[0] + '_f0.csv', index=False)

    if not args.no_print and len(sig) / FS <= 60:
        mask = confidence > args.threshold
        plt.figure(figsize=(max(6.4, 6.4 * time[-1] / 3), 4.8))
        plt.specgram(sig, Fs=FS, NFFT=args.NFFT, noverlap=args.NFFT - args.NFFT // 8, cmap='inferno')
        if mask.any():
            plt.scatter(time[mask], f0[mask], c=confidence[mask], s=5, cmap='cool', vmin=0, vmax=1)
            plt.colorbar(label='Confidence')
            plt.ylim(0, f0[mask].max() * 1.5)
        plt.xlim(0, len(sig) / FS)
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.savefig(filepath.rsplit('.', 1)[0] + '_f0.png')
        plt.close()
