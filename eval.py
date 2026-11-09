"""Evaluates a trained model on the cached CQTs, following paper_experiments/eval_all.py:
voicing threshold chosen at the ROC optimum per species, metrics computed per vocalisation (50 cents tolerance) then averaged.
Scores are stored in scores/{name}_{specie}_scores.csv

Two secondary pitch accuracies compensate for the known resolution of the annotations (metadata 'annot_grid_hz' and 'annot_step'), see
label_interval: a frame is correct if the estimate is within the tolerance of any value the label could have taken. 'Pitch acc' stays the
standard metric (comparable with bioacoustic_f0_estimation); 'grid-comp.' is objective (measured quantisation, 50 cents). 'grid+timing-comp.'
uses a stricter tolerance (--comp_cent_thr, 10 cents as CREPE's strictest metric): once the label interval absorbs the annotation resolution
(frequency grid and timing), the tolerance only has to cover the estimate's own error. It measures precision, mostly meaningful where labels
are fine (on fast sweeps the timing interval is wide, so the metric is lenient there).
"""
import os, argparse
import numpy as np, pandas as pd, torch, mir_eval
from sklearn import metrics as skmetrics
from tqdm import tqdm
from metadata import get_species
from cqt import cache_folder, bin2hz, hop_length
from dataset import get_files, load_cached
from model import load_model, infer
from decode import weighted_argmax

def label_interval(f0, meta, hop_s, timing=True):
    """ Interval of F0 values (cents, mir_eval reference of 10 Hz) the label of each frame could have taken given the annotation resolution.
    - frequency grid ('annot_grid_hz' = D): the annotated value f stands for [f - D/2, f + D/2]
    - timing ('annot_step' = s, only if timing): the label is only located within +-s/2 in time, so the interval spans the label contour over
      +-s/2 around the frame (frame labels inside the window, plus linear interpolation at its edges); the contour being continuous, every value
      between its minimum and maximum is reachable. Each value is widened by its grid cell.
    f0: frame labels (Hz, 0 = unvoiced, NaN = unknown). Returns (low, high) arrays, NaN where the frame has no pitch label. """
    d = meta.get('annot_grid_hz') or 0.
    lab = np.where(f0 > d / 2, f0, np.nan)
    lo, hi = mir_eval.melody.hz2cents(np.nan_to_num(lab - d / 2)), mir_eval.melody.hz2cents(np.nan_to_num(lab + d / 2))
    lo, hi = np.where(np.isnan(lab), np.nan, lo), np.where(np.isnan(lab), np.nan, hi)
    if timing and meta.get('annot_step'):
        half = meta['annot_step'] / 2 / hop_s  # half window in frames
        n, k = len(f0), int(np.floor(half))
        lo_w, hi_w = lo.copy(), hi.copy()
        for j in range(1, k + 1):  # whole frames inside the window
            for sh in (j, -j):
                a, b = np.roll(lo, sh), np.roll(hi, sh)
                if sh > 0: a[:sh], b[:sh] = np.nan, np.nan
                else: a[sh:], b[sh:] = np.nan, np.nan
                lo_w, hi_w = np.fmin(lo_w, a), np.fmax(hi_w, b)
        frac = half - k
        if frac > 0:  # window edges, interpolated between frame k and k + 1
            for sgn in (1, -1):
                idx = np.arange(n)
                i0, i1 = idx + sgn * k, idx + sgn * (k + 1)
                ok = (i1 >= 0) & (i1 < n) & (i0 >= 0) & (i0 < n)
                for arr, red in ((lo, np.fmin), (hi, np.fmax)):
                    e = np.full(n, np.nan)
                    e[ok] = (1 - frac) * arr[i0[ok]] + frac * arr[i1[ok]]
                    if red is np.fmin: lo_w = red(lo_w, e)
                    else: hi_w = red(hi_w, e)
        lo, hi = np.where(np.isnan(lab), np.nan, lo_w), np.where(np.isnan(lab), np.nan, hi_w)
    return lo, hi


def compensated_accuracy(ref_v, est_c, lo, hi, tol):
    """ Share of reference-voiced frames whose estimate is within tol cents of the label interval [lo, hi] """
    if not ref_v.any():
        return np.nan
    ok = (est_c >= lo - tol) & (est_c <= hi + tol)
    return np.mean(ok[ref_v])


parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str, help="Checkpoint saved by train.py")
parser.add_argument('specie', type=str, nargs='?', default=None, help="Species to evaluate (default: all)")
parser.add_argument('--split', type=int, default=None, help="Test fold (between 0 and 4) to evaluate on (default: all files)")
parser.add_argument('--fake', action='store_true', help="Use the synthetic species from make_fake_data.py")
parser.add_argument('--cent_thr', type=float, default=50)
parser.add_argument('--comp_cent_thr', type=float, default=10, help="Tolerance of the grid+timing-compensated accuracy (cents): stricter, since the label interval already absorbs the annotation resolution (10 cents as CREPE's strictest metric on exact labels)")
parser.add_argument('--cache_dir', type=str, default='cache')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, p = load_model(args.model_path, device)
folder, bpo = cache_folder(args.cache_dir, p), p['bins_per_octave']
name = os.path.splitext(os.path.basename(args.model_path))[0]
os.makedirs('scores', exist_ok=True)
summary = []

for specie, meta in get_species(only=args.specie, fake=args.fake).items():
    fns = get_files(meta, args.split, test=args.split is not None, folder=folder, specie=specie)
    preds = []
    for fn in tqdm(fns, desc=specie, leave=False):
        x, f0 = load_cached(folder, specie, fn)
        if np.isnan(f0).all():
            continue
        salience, voicing = infer(model, x, p, device)
        bins, conf = weighted_argmax(salience, bpo // 12), voicing
        keep = ~np.isnan(f0)  # frames with unknown labels (no annotation, or annotation gaps, see preprocess.load_annot) are not scored
        hop_s = hop_length(meta['FS'], meta['fmin'], p) / meta['FS']
        intervals = [i[keep] for i in label_interval(f0, meta, hop_s, timing=False) + label_interval(f0, meta, hop_s, timing=True)]
        preds.append((fn, f0[keep], bin2hz(bins, meta['fmin'], bpo)[keep], conf[keep], intervals))
    if not preds:
        continue
    # ROC optimal voicing threshold over all frames of the species
    labels, confs = np.concatenate([f0 > 0 for _, f0, _, _, _ in preds]), np.concatenate([c for _, _, _, c, _ in preds])
    fpr, tpr, thr = skmetrics.roc_curve(labels, confs)
    threshold = thr[np.argmin(abs(tpr + fpr - 1))]

    rows = []
    for fn, f0, est, conf, (lo_g, hi_g, lo_t, hi_t) in preds:
        ref_v, ref_c = f0 > 0, mir_eval.melody.hz2cents(np.where(f0 > 0, f0, 0))
        est_c = mir_eval.melody.hz2cents(est)
        recall, fa = mir_eval.melody.voicing_measures(ref_v, conf > threshold)
        correct = ref_v & (np.abs(est_c - ref_c) < args.cent_thr)
        jumps = np.abs(np.diff(est_c))[correct[1:] & correct[:-1]]
        rows.append({'fn': fn, 'Recall': recall, 'False alarm': fa,
                     'Pitch acc': mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, conf, est_c, cent_tolerance=args.cent_thr),
                     'Chroma acc': mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, conf, est_c, cent_tolerance=args.cent_thr),
                     'Pitch acc (grid-comp.)': compensated_accuracy(ref_v, est_c, lo_g, hi_g, args.cent_thr),
                     f'Pitch acc (grid+timing-comp., {args.comp_cent_thr:g} cents)': compensated_accuracy(ref_v, est_c, lo_t, hi_t, args.comp_cent_thr),
                     'Overall acc': mir_eval.melody.overall_accuracy(ref_v, ref_c, conf > threshold, est_c, cent_tolerance=args.cent_thr),
                     'Smoothness (cents)': jumps.mean() if len(jumps) else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(f'scores/{name}_{specie}_per_file.csv', index=False)
    scores = df.drop(columns='fn').mean().to_frame(specie).T
    scores['count'], scores['threshold'] = len(df), threshold
    scores.to_csv(f'scores/{name}_{specie}_scores.csv')
    summary.append(scores)

summary = pd.concat(summary)
print(summary.round(3).to_string())
summary.to_csv(f'scores/{name}_summary.csv')
