"""Training of the encoder and the transformer temporal head in a single phase, from scratch.

L_total = L_pitch(refined) + local_weight * L_pitch(local) + voicing_weight * L_voicing + ssl_weight * L_PESTO
- the pitch targets are Gaussians whose width combines --sigma with each label's uncertainty (annotation timing on fast contours,
  annotation frequency grid, see dataset.label_uncertainty)
- the supervised terms anchor the absolute pitch of the relative grid; the PESTO losses (equivariance, shift cross-entropy, invariance,
  combined by PESTO's gradient-based weighting) regularise the encoder throughout training. They are computed on voiced annotated
  frames only: unvoiced frames have no pitch to be equivariant to
- species are balanced by inverse frequency sampling, as in paper_experiments/train_crepe.py
- augmentations of the training windows: random window offsets, real background noise mixed in the power domain (--p_noise, --noise_snr),
  supervised pitch shift (the CQT crop and the pitch targets are shifted by the same random number of bins, --shift_aug, --p_shift),
  time rescaling (as with a different hop, --time_aug, --p_time; independent of the pitch shift, so that the model does not tie
  contour speed to pitch position),
  random truncation (--p_truncate). The PESTO views get independent real noise (see losses.PESTOLosses)
"""
import os, argparse, math
import numpy as np, torch, torch.nn.functional as F, tqdm
from torch.utils.tensorboard import SummaryWriter
from metadata import get_species
from cqt import add_cqt_args, get_params, cache_folder
from dataset import get_files, WindowDataset, loader_kwargs
from model import add_model_args, encoder_hparams, temporal_hparams, F0Model, harmonic_stack
from losses import PESTOLosses, pitch_loss, add_aug_args, mix_noise

parser = argparse.ArgumentParser()
parser.add_argument('--omit', type=str, help="Species to rule out of the training set", default=None)
parser.add_argument('--only', type=str, help="Train only on the given species", default=None)
parser.add_argument('--split', type=int, help="Portion out of between 0 and 4 to use as test set (default: train on all files)", default=None)
parser.add_argument('--fake', action='store_true', help="Use the synthetic species from make_fake_data.py")
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=16, help="Number of windows of --context frames per batch")
parser.add_argument('--p_truncate', type=float, default=0.25, help="Probability of truncating a training window to a random shorter length, so that the temporal head learns to handle short clips")
parser.add_argument('--stride', type=int, default=None, help="Hop between training windows, in frames (default: context // 2)")
parser.add_argument('--no_random_offset', action='store_true', help="Disable random offsets of the training windows (in [0, stride))")
parser.add_argument('--shift_aug', type=int, default=None, help="Maximum supervised pitch shift in bins (default: max_shift, i.e. the cache margin; 0 to disable)")
parser.add_argument('--p_shift', type=float, default=0.8, help="Probability of pitch-shifting a training window")
parser.add_argument('--time_aug', type=float, default=1.5, help="Maximum time rescaling of a training window, in octaves (factor 2 ** u, u uniform in [-time_aug, time_aug]; 0 to disable)")
parser.add_argument('--p_time', type=float, default=0.8, help="Probability of rescaling a training window in time (independent of the pitch shift)")
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--ssl_weight', type=float, default=1., help="Weight of the PESTO losses (constant)")
parser.add_argument('--ssl_frames', type=int, default=256, help="Number of voiced frames of the batch used for the PESTO losses")
parser.add_argument('--sigma', type=float, default=2.5, help="Base std of the Gaussian smooth labels in bins (2.5 bins = 25 cents, as CREPE), combined in quadrature with each label's uncertainty (dataset.label_uncertainty)")
parser.add_argument('--local_weight', type=float, default=0.5, help="Weight of the pitch loss on the local (encoder) salience")
parser.add_argument('--voicing_weight', type=float, default=1.)
parser.add_argument('--ema_rate', type=float, default=0.999, help="EMA rate of PESTO's gradient-based loss weighting")
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--name', type=str, default=None, help="Run name (default: built from --omit/--only/--split)")
parser.add_argument('--in_memory', action='store_true', help="Load all cached CQTs in RAM instead of reading them from disk")
parser.add_argument('--resume', action='store_true', help="Continue from weights/{name}.pth if it exists (optimizer, scheduler, epoch included)")
add_cqt_args(parser)
add_model_args(parser)
add_aug_args(parser)
args = parser.parse_args()

suffix = args.name or ("omit_" + args.omit if args.omit else f"only-{args.split}_" + args.only if args.only else "all")
os.makedirs('weights', exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True  # fixed input sizes: let cuDNN pick the fastest convolution algorithms
ckpt_path = f'weights/{suffix}.pth'

p = get_params(args)
folder = cache_folder(args.cache_dir, p)
species = get_species(args.only, args.omit, args.fake)
ds_args = dict(species=species, folder=folder, p=p, context=args.context, stride=args.stride or max(1, args.context // 2), in_memory=args.in_memory)
train_set = WindowDataset(files={s: get_files(m, args.split, folder=folder, specie=s) for s, m in species.items()},
                          random_offset=not args.no_random_offset, noise=True, time_aug=args.time_aug, p_time=args.p_time, **ds_args)
sampler = torch.utils.data.WeightedRandomSampler(train_set.sample_weights(), len(train_set), replacement=True)
loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, drop_last=True, **loader_kwargs(args.num_workers))
val_loader = None
if args.split is not None:
    val_set = WindowDataset(files={s: get_files(m, args.split, test=True, folder=folder, specie=s) for s, m in species.items()}, **ds_args)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, **loader_kwargs(args.num_workers))
print(f'{len(train_set)} training windows from {len(train_set.paths)} files' + (f', {len(val_set)} validation windows' if val_loader else ''))

enc_hp, tmp_hp = encoder_hparams(args, p['n_bins']), temporal_hparams(args, p['n_bins'])
model = F0Model(enc_hp, tmp_hp, args.harmonics).to(device)
pesto = PESTOLosses(model.encoder, p['n_bins'], p['bins_per_octave'], p['max_shift'], args.ema_rate, args.noise_snr, args.p_noise, args.harmonics).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
total_steps = args.epochs * len(loader)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * s / total_steps)))
ms, nb = p['max_shift'], p['n_bins']
shift_aug = ms if args.shift_aug is None else args.shift_aug
assert 0 <= shift_aug <= ms, f'--shift_aug must be between 0 and max_shift ({ms})'


def truncate(present, valid):
    """ Masks out a random suffix of a fraction p_truncate of the windows (as if the clip ended there) """
    (B, T), dev = present.shape, present.device
    lengths = torch.where(torch.rand(B, device=dev) < args.p_truncate, torch.randint(1, T + 1, (B,), device=dev), torch.full((B,), T, device=dev))
    keep = torch.arange(T, device=dev).unsqueeze(0) < lengths.unsqueeze(1)
    return present & keep, valid * keep


def augment(x, bins, noise):
    """ Real noise mixing, then supervised pitch shift and harmonic stacking: the input crop moves by k bins (content at bin j goes to bin j + k)
    and the targets follow. Targets shifted out of the grid become NaN (no pitch target, voicing target kept) """
    if noise is not None and args.p_noise > 0:
        x = mix_noise(x, noise, args.noise_snr, args.p_noise)
    k = torch.randint(-shift_aug, shift_aug + 1, (len(x),), device=x.device) * (torch.rand(len(x), device=x.device) < args.p_shift)
    xin = harmonic_stack(x, (ms - k).view(-1, 1), nb, args.harmonics, p['bins_per_octave'])  # all harmonics shifted together
    bins = bins + k.view(-1, 1)
    bins[(bins < 0) | (bins > nb - 1)] = float('nan')
    return xin, bins


def compute_losses(x, bins, unc, voiced, valid, present, noise=None, train=False):
    """ x: CQT windows in dB (batch, context, n_bins + 2 * max_shift), unc: label uncertainty in bins (dataset.label_uncertainty).
    Returns the supervised loss, its terms and monitoring stats """
    xin, bins = augment(x, bins, noise) if train else (harmonic_stack(x, ms, nb, args.harmonics, p['bins_per_octave']), bins)
    local, logits, vlogits = model(xin, present)
    pmask = (valid > 0) & ~torch.isnan(bins)
    bins = torch.nan_to_num(bins)
    sigma = torch.sqrt(args.sigma ** 2 + unc ** 2)  # target width per frame, in bins
    losses = {'pitch': pitch_loss(logits, bins, pmask, sigma),
              'pitch_local': pitch_loss(local, bins, pmask, sigma),
              'voicing': (F.binary_cross_entropy_with_logits(vlogits, voiced, reduction='none') * valid).sum() / valid.sum().clip(min=1)}
    total = losses['pitch'] + args.local_weight * losses['pitch_local'] + args.voicing_weight * losses['voicing']
    # raw pitch accuracy (50 cents) of the argmax on voiced frames, for monitoring
    err = (logits.argmax(-1).float() - bins).abs()[pmask] * 1200 / p['bins_per_octave']
    stats = {'acc': (err < 50).float().mean() if pmask.any() else torch.tensor(float('nan')),
             'voicing_acc': (((vlogits > 0).float() == voiced).float() * valid).sum() / valid.sum().clip(min=1)}
    return total, losses, stats


n_iter, best_val, start_epoch = 0, float('inf'), 0
if args.resume and os.path.isfile(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model']), optimizer.load_state_dict(ckpt['optimizer']), scheduler.load_state_dict(ckpt['scheduler'])
    pesto.weighting.weights, pesto.weighting.grads = ckpt['weighting']
    n_iter, best_val, start_epoch = ckpt['n_iter'], ckpt['best_val'], ckpt['epoch'] + 1
    print(f'Resuming {ckpt_path} at epoch {start_epoch}')
writer = SummaryWriter('runs/' + suffix, purge_step=n_iter if start_epoch else None)  # purge: drop logs written after the checkpoint
for epoch in range(start_epoch, args.epochs):
    model.train()
    for x, bins, unc, voiced, valid, present, noise in tqdm.tqdm(loader, desc=str(epoch), leave=False):
        x, bins, unc, voiced, valid, present, noise = (t.to(device) for t in (x, bins, unc, voiced, valid, present, noise))
        noise = noise if noise.numel() > 0 else None  # empty noise bank (no unvoiced annotated segment): no noise augmentation
        present, valid = truncate(present, valid)
        optimizer.zero_grad()
        total, losses, stats = compute_losses(x, bins, unc, voiced, valid, present, noise, train=True)
        ssl_mask = present & (valid > 0) & (voiced > 0)  # PESTO losses on voiced annotated frames only (clean CQT, noise added in PESTOLosses)
        if args.ssl_weight > 0 and ssl_mask.sum() > 1:
            sel = torch.randperm(int(ssl_mask.sum()), device=device)[:args.ssl_frames]
            ssl_loss, ssl_losses = pesto(x[ssl_mask].unsqueeze(1)[sel], None if noise is None else noise[ssl_mask].unsqueeze(1)[sel])
            total = total + args.ssl_weight * ssl_loss
            losses.update({f'ssl_{k}': v for k, v in ssl_losses.items()})
            losses.update({f'ssl_weight_{k}': torch.as_tensor(float(w)) for k, w in pesto.weighting.weights.items()})
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        scheduler.step()
        writer.add_scalar('loss', total.item(), n_iter)
        for k, v in {**losses, **stats}.items():
            writer.add_scalar(f'train/{k}', v.item(), n_iter)
        n_iter += 1

    ckpt = {'model': model.state_dict(), 'encoder_hparams': enc_hp, 'temporal_hparams': tmp_hp, 'harmonics': args.harmonics, 'cqt_params': p,
            'species': list(species), 'args': vars(args)}
    if val_loader:
        model.eval()
        vals = []
        with torch.no_grad():
            for x, bins, unc, voiced, valid, present, _ in val_loader:
                total, losses, stats = compute_losses(*(t.to(device) for t in (x, bins, unc, voiced, valid, present)))
                vals.append([total.item(), stats['acc'].item(), stats['voicing_acc'].item()])
        vloss, vacc, vvacc = (float(v) for v in np.nanmean(vals, axis=0))  # python floats: checkpoints must load with weights_only
        writer.add_scalar('val/loss', vloss, n_iter), writer.add_scalar('val/acc', vacc, n_iter), writer.add_scalar('val/voicing_acc', vvacc, n_iter)
        print(f'epoch {epoch}: val loss {vloss:.3f}, pitch acc {vacc:.3f}, voicing acc {vvacc:.3f}')
        if vloss < best_val:
            best_val = vloss
            torch.save(ckpt, f'weights/{suffix}_best.pth')
    # last checkpoint, with what is needed to resume (written atomically: a job killed while saving does not corrupt it)
    torch.save({**ckpt, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'n_iter': n_iter,
                'best_val': best_val, 'weighting': (pesto.weighting.weights, pesto.weighting.grads)}, ckpt_path + '.tmp')
    os.replace(ckpt_path + '.tmp', ckpt_path)
print(f'Saved {ckpt_path}')
