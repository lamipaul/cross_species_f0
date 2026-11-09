"""Robustness to the position of the vocalisations in the pitch grid: accuracy as a function of an fmin offset.
For each test file, the CQT is recomputed from the audio with fmin' = fmin * 2 ** offset (offset in octaves; > 0: calls lower in the grid,
< 0: higher), which also changes the hop (hop = hop_periods / fmin'), exactly as choosing another fmin in predict.py would.
Reported per model, species and offset (scores/robustness_{names}.csv, one row each, resumable; plot with --plot):
  in_grid     share of voiced frames whose label lies inside the 6-octave grid (the others cannot be estimated)
  pitch_acc   raw pitch accuracy (50 cents) on in-grid voiced frames, per file then averaged (as eval.py)
  pitch_acc_all  same, counting out-of-grid voiced frames as errors
  voicing_auc ROC AUC of the voicing confidence (threshold-free), over all labelled frames of the species
The audio must be available (it is not in the cache). The sampling rate is the species' FS, raised when fmin' requires it (as predict.py).
"""
import os, math, json, argparse, warnings
import numpy as np, pandas as pd, librosa, torch, mir_eval
from sklearn import metrics as skmetrics
from tqdm import tqdm
from metadata import get_species
from cqt import compute_cqt, bin2hz, hz2bin, max_fmin
from dataset import get_files
from model import load_model, infer
from decode import weighted_argmax
from preprocess import load_annot, annot_offset
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('model_paths', type=str, nargs='+', help="Checkpoints saved by train.py (compared on the same files)")
parser.add_argument('--specie', type=str, default=None, help="Single species (default: all)")
parser.add_argument('--split', type=int, default=0, help="Test fold to use (unseen during training)")
parser.add_argument('--offsets', type=float, nargs='+', default=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2], help="fmin offsets in octaves")
parser.add_argument('--max_files', type=int, default=20, help="Test files per species, evenly spread over the fold (CPU cost)")
parser.add_argument('--cent_thr', type=float, default=50)
parser.add_argument('--plot', action='store_true', help="Only plot an existing results file")
args = parser.parse_args()

names = [os.path.splitext(os.path.basename(m))[0] for m in args.model_paths]
out = f"scores/robustness_{'_vs_'.join(names)}.csv"
os.makedirs('scores', exist_ok=True)


def plot(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2563eb', '#ea580c', '#16a34a', '#9333ea']  # one fixed colour per model, in argument order
    species = list(dict.fromkeys(df.specie))
    mean = df.groupby(['model', 'offset'], sort=False)[['pitch_acc', 'pitch_acc_all', 'in_grid', 'voicing_auc']].mean().reset_index()
    mean['specie'] = 'mean over species'
    df = pd.concat([mean, df])
    panels = ['mean over species'] + species
    nc = 4
    fig, axes = plt.subplots(math.ceil(len(panels) / nc), nc, figsize=(4 * nc, 3 * math.ceil(len(panels) / nc)), sharex=True, sharey=True, squeeze=False)
    for ax, sp in zip(axes.flat, panels):
        d = df[df.specie == sp]
        g = d[d.model == names[0]].sort_values('offset')
        ax.fill_between(g.offset, 0, g.in_grid, color='0.9', lw=0, label='voiced frames in grid')
        for name, c in zip(names, colors):
            g = d[d.model == name].sort_values('offset')
            ax.plot(g.offset, g.pitch_acc, color=c, lw=2, marker='o', ms=4, label=f'{name}: pitch acc (in grid)')
            ax.plot(g.offset, g.voicing_auc, color=c, lw=1, ls='--', label=f'{name}: voicing AUC')
        ax.axvline(0, color='0.6', lw=0.8)
        ax.set_title(sp, fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, lw=0.5)
    for ax in axes.flat[len(panels):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('fmin offset (octaves, > 0: calls lower in the grid)')
    axes[0, 0].legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(out.replace('.csv', '.png'), dpi=110)
    print('Saved', out.replace('.csv', '.png'))


if args.plot:
    plot(pd.read_csv(out))
    raise SystemExit

device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = [load_model(m, device) for m in args.model_paths]
done = pd.read_csv(out) if os.path.isfile(out) else pd.DataFrame(columns=['model', 'specie', 'offset'])
done_keys = set(zip(done.model, done.specie, done.offset.astype(float)))

for specie, meta in get_species(only=args.specie).items():
    fns = get_files(meta, args.split, test=True)
    if not fns:
        print(f'{specie}: no audio found, skipped')
        continue
    fns = [fns[i] for i in np.unique(np.linspace(0, len(fns) - 1, args.max_files).round().astype(int))]
    for off in args.offsets:
        todo = [(n, m) for n, m in zip(names, models) if (n, specie, float(off)) not in done_keys]
        if not todo:
            continue
        fmin = meta['fmin'] * 2 ** off
        FS = max(meta['FS'], int(math.ceil(fmin / max_fmin(1, todo[0][1][1]))))
        res = {n: [] for n, _ in todo}  # per file: (labels Hz, estimates Hz, voicing confidence)
        for fn in tqdm(fns, desc=f'{specie} offset {off:+g} (fmin {fmin:.4g}Hz, FS {FS}Hz)', leave=False):
            sig, _ = librosa.load(fn, sr=FS)
            cqts = {}  # one CQT per set of CQT parameters (e.g. models trained with different cache margins)
            for n, (model, p) in todo:
                key = json.dumps(p, sort_keys=True)
                if key not in cqts:
                    cqts[key] = compute_cqt(sig, FS, fmin, p, device)
                x, time = cqts[key]
                f0 = load_annot(fn, time, offset=annot_offset(fn, meta))
                if np.isnan(f0).all():
                    continue
                sal, conf = infer(model, x, p, device)
                est = bin2hz(weighted_argmax(sal, p['bins_per_octave'] // 12), fmin, p['bins_per_octave'])
                keep = ~np.isnan(f0)
                res[n].append((f0[keep], est[keep], conf[keep], p))
        rows = []
        for n, files in res.items():
            if not files:
                continue
            accs, accs_all, n_v, n_in = [], [], 0, 0
            for f0, est, conf, p in files:
                v = f0 > 0
                b = hz2bin(np.where(v, f0, np.nan), fmin, p['bins_per_octave'])
                inside = v & (b >= -0.5) & (b <= p['n_bins'] - 0.5)
                n_v, n_in = n_v + v.sum(), n_in + inside.sum()
                ok = np.abs(mir_eval.melody.hz2cents(np.where(v, f0, 0)) - mir_eval.melody.hz2cents(est)) < args.cent_thr
                if inside.any():
                    accs.append(np.mean(ok[inside]))
                if v.any():
                    accs_all.append(np.mean((ok & inside)[v]))
            labels, confs = np.concatenate([f[0] > 0 for f in files]), np.concatenate([f[2] for f in files])
            auc = skmetrics.roc_auc_score(labels, confs) if 0 < labels.mean() < 1 else np.nan
            rows.append({'model': n, 'specie': specie, 'offset': float(off), 'fmin': fmin, 'FS': FS, 'files': len(files),
                         'in_grid': n_in / max(n_v, 1), 'pitch_acc': np.mean(accs) if accs else np.nan,
                         'pitch_acc_all': np.mean(accs_all) if accs_all else np.nan, 'voicing_auc': auc})
            print(f"{n} {specie} offset {off:+g}: in grid {rows[-1]['in_grid']:.3f}, pitch acc {rows[-1]['pitch_acc']:.3f}, voicing AUC {auc:.3f}")
        if rows:  # saved after each (species, offset): an interrupted run resumes where it stopped
            done = pd.concat([done, pd.DataFrame(rows)]) if len(done) else pd.DataFrame(rows)
            done.to_csv(out, index=False)

plot(done)
