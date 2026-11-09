"""Self-supervised losses and gradient-based loss weighting copied from PESTO (pesto-full/src/losses, src/data,
src/callbacks/loss_weighting.py), without the lightning dependencies, real-noise augmentation, and the supervised losses.

As in PESTO, the SSL losses receive softmax-normalised encoder outputs (the CrossEntropy backend is applied to them), so their
cross-entropies range from about 5.6 (sharp, consistent outputs) to log(n_bins) = 6.6 (flat or inconsistent outputs).
"""
import torch, torch.nn as nn, torch.nn.functional as F
from model import harmonic_stack


# ---------------------------------------------------------------- PESTO: equivariance
class HuberLoss(nn.Module):
    def __init__(self, tau: float):
        super(HuberLoss, self).__init__()
        self.register_buffer('tau', torch.tensor(tau), persistent=False)

    def forward(self, x):
        x = x.abs()
        return torch.where(x.le(self.tau), x ** 2 / 2, self.tau ** 2 / 2 + self.tau * (x - self.tau))


class PowerSeries(nn.Module):
    def __init__(self, value: float, power_min, power_max, tau: float = 1.):
        super(PowerSeries, self).__init__()
        self.value = value
        self.register_buffer('weights', self.value ** torch.arange(power_min, power_max).float(), persistent=False)
        self.loss_fn = HuberLoss(tau)

    def forward(self, x1, x2, target):
        """ x2[i] is the pitch-shifted version of x1[i] by target[i] bins """
        z1, z2 = x1.mv(self.weights), x2.mv(self.weights)
        freq_ratios = self.value ** target.float()
        loss_12 = self.loss_fn(z2 / z1 - freq_ratios).mean()
        loss_21 = self.loss_fn(z1 / z2 - 1 / freq_ratios).mean()
        return (loss_12 + loss_21) / 2


# ---------------------------------------------------------------- PESTO: invariance and shift cross-entropy
class CrossEntropyLoss(nn.Module):
    def __init__(self, symmetric=False, detach_targets=False):
        super(CrossEntropyLoss, self).__init__()
        self.symmetric, self.detach_targets = symmetric, detach_targets
        self.backend = nn.CrossEntropyLoss()

    def forward(self, input, target):
        if self.symmetric:
            return (self.compute_loss(input, target) + self.compute_loss(target, input)) / 2
        return self.compute_loss(input, target)

    def compute_loss(self, input, target):
        return self.backend(input, target.detach() if self.detach_targets else target)


class ShiftCrossEntropy(nn.Module):
    def __init__(self, pad_length=5, criterion=None):
        super(ShiftCrossEntropy, self).__init__()
        self.criterion = criterion or CrossEntropyLoss()
        self.pad_length = pad_length

    def forward(self, x1, x2, target):
        x1 = F.pad(x1, (self.pad_length, self.pad_length))
        x2 = F.pad(x2, (2 * self.pad_length, 2 * self.pad_length))
        idx = target.unsqueeze(1) + torch.arange(x1.size(-1), device=target.device) + self.pad_length
        return self.criterion(x1, torch.gather(x2, dim=1, index=idx))


# ---------------------------------------------------------------- PESTO: gradient-based loss weighting
class GradientsLossWeighting:
    def __init__(self, weights, last_layer, ema_rate=0.):
        self.weights = dict(weights)
        self.last_layer, self.ema_rate = last_layer, ema_rate
        self.grads = {k: 1 - v for k, v in weights.items()}

    def combine_losses(self, **losses):
        self.update_weights(losses)
        return sum([self.weights[key] * losses[key] for key in self.weights.keys()])

    def update_weights(self, losses):
        grads = []
        for k, loss in losses.items():
            if not loss.requires_grad:
                return
            g = torch.autograd.grad(loss, self.last_layer, retain_graph=True)[0].norm().detach()
            g = self.ema_rate * self.grads[k] + (1 - self.ema_rate) * g
            self.grads[k] = g
            grads.append(g)
        grads = torch.stack(grads)
        grads = 1 - grads / grads.sum().clip(min=1e-7)
        for i, k in enumerate(losses.keys()):
            self.weights[k] = grads[i]


# ---------------------------------------------------------------- Real noise augmentation (replaces PESTO's Gaussian noise on dB values)
def add_aug_args(parser):
    g = parser.add_argument_group('augmentations')
    g.add_argument('--p_noise', type=float, default=0.5, help="Probability of mixing real noise into a sample (PESTO views and training windows)")
    g.add_argument('--noise_snr', type=float, nargs=2, default=[-5., 20.], help="SNR range (dB) of the real noise mixing")


def mix_noise(x, noise, snr_db=(-5., 20.), p=0.5):
    """ Mixes noise into x in the power domain, x and noise being log-magnitude CQTs in dB of identical shape (batch, ...).
    For each item, with probability p, noise is scaled to a random SNR (uniform in snr_db, in dB, computed over all the item's
    frames and bins) and the powers are added: 10 log10(10 ** (x / 10) + g * 10 ** (noise / 10)). Other items are returned unchanged.
    """
    B = x.shape[0]
    dims = tuple(range(1, x.ndim))
    snr = torch.empty(B, device=x.device).uniform_(*snr_db)
    # powers relative to the item's maximum to avoid overflow
    ref = torch.maximum(x.amax(dims, keepdim=True), noise.amax(dims, keepdim=True))
    ps, pn = 10 ** ((x - ref) / 10), 10 ** ((noise - ref) / 10)
    g = ps.mean(dims) / pn.mean(dims).clamp(min=1e-30) / 10 ** (snr / 10)
    mixed = 10 * torch.log10(ps + g.view(-1, *[1] * (x.ndim - 1)) * pn + 1e-30) + ref
    apply = (torch.rand(B, device=x.device) < p).view(-1, *[1] * (x.ndim - 1))
    return torch.where(apply, mixed, x)


class PESTOLosses(nn.Module):
    """ PESTO training step (pesto-full/src/models/pesto.py, training_step) with its default config (configs/model/default.yaml),
    except for the augmentation of the augmented and pitch-shifted views: real background noise (unvoiced frames from the dataset)
    mixed in the power domain, independently for both views (see mix_noise), instead of PESTO's Gaussian noise on dB values and gain
    (the gain is a no-op anyway: the encoder's first LayerNorm cancels any scaling of the input).
    Noise is mixed into the full-width CQT, before cropping the pitch-shifted view.
    """
    def __init__(self, encoder, n_bins, bins_per_octave, margin, ema_rate=0.999, noise_snr=(-5., 20.), p_noise=0.5, harmonics=(1,), max_steps=55):
        super(PESTOLosses, self).__init__()
        self.encoder = encoder
        self.n_bins, self.margin, self.max_steps, self.bpo, self.harmonics = n_bins, margin, max_steps, bins_per_octave, list(harmonics)
        self.noise_snr, self.p_noise = tuple(noise_snr), p_noise
        self.equiv_loss_fn = PowerSeries(2 ** (1 / bins_per_octave), 1 - n_bins, 1, tau=2 ** (1 / 6) - 1)
        self.inv_loss_fn = CrossEntropyLoss(symmetric=True, detach_targets=True)
        self.sce_loss_fn = ShiftCrossEntropy(pad_length=max_steps, criterion=self.inv_loss_fn)
        self.weighting = GradientsLossWeighting({'invariance': 0., 'shift_entropy': 1., 'equivariance': 0.}, encoder.fc.weight, ema_rate)

    def forward(self, x, noise=None):
        """ x: cached CQT frames in dB (batch, 1, n_bins + 2 * margin), noise: noise frames like x (None: views without noise).
        The pitch-shifted view is shifted by up to max_steps bins (PESTO default: 5.5 semitones = 55 bins at 120 bins / octave), within the cache margin.
        The encoder receives harmonic stacks of the views (model.harmonic_stack)
        -> total loss, dict of losses """
        if noise is not None:  # independent noise for both views: the second one is taken from another item of the batch
            xa_full = mix_noise(x, noise, self.noise_snr, self.p_noise)
            xt_full = mix_noise(x, noise.roll(1, 0), self.noise_snr, self.p_noise)
        else:
            xa_full = xt_full = x
        ms = self.margin  # index of grid bin 0 in the cached CQT
        n_steps = torch.randint(-self.max_steps, self.max_steps + 1, (x.shape[0],), device=x.device)  # as PESTO's PitchShiftCQT: cropping the cached CQT
        stack = lambda v, start: harmonic_stack(v[:, 0], start, self.n_bins, self.harmonics, self.bpo)  # (batch, n_harmonics, n_bins)
        x, xa, xt = stack(x, ms), stack(xa_full, ms), stack(xt_full, ms - n_steps)  # all harmonics are shifted together
        y, ya, yt = [F.softmax(self.encoder(v), dim=-1) for v in (x, xa, xt)]
        losses = dict(invariance=self.inv_loss_fn(y, ya),
                      shift_entropy=self.sce_loss_fn(ya, yt, n_steps),
                      equivariance=self.equiv_loss_fn(ya, yt, n_steps))
        return self.weighting.combine_losses(**losses), losses


# ---------------------------------------------------------------- Supervised losses
def gaussian_targets(bins, n_bins, sigma=1.):
    """ Smooth label: Gaussian centred on the (fractional) target bin, normalised to sum to 1. sigma: scalar or one value per target (bins) """
    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=bins.device)
    sigma = sigma.unsqueeze(-1) if sigma.ndim else sigma
    g = torch.exp(-(torch.arange(n_bins, device=bins.device) - bins.unsqueeze(-1)) ** 2 / (2 * sigma ** 2))
    return g / g.sum(-1, keepdim=True)


def pitch_loss(logits, bins, mask, sigma=1.):
    """ Cross-entropy (= KL up to a constant) between predicted salience and the smooth target, over masked frames.
    sigma: target width in bins, scalar or per frame (same shape as bins) """
    if mask.sum() == 0:
        return logits.sum() * 0
    target = gaussian_targets(bins[mask], logits.shape[-1], sigma[mask] if torch.is_tensor(sigma) else sigma)
    return -(target * F.log_softmax(logits[mask], dim=-1)).sum(-1).mean()

