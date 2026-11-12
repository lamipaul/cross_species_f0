from functools import partial
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm


def add_model_args(parser):
    g = parser.add_argument_group('model')
    g.add_argument('--harmonics', type=float, nargs='+', default=[0.5, 1, 2, 3], help="Harmonic CQT input: one channel per harmonic, shifted so that harmonic h lines up with the F0 bin ([1]: plain CQT)")
    g.add_argument('--n_chan_layers', type=int, nargs='+', default=[64, 48, 48, 16, 4], help="Encoder channels (PESTO default: 40 30 30 10 3)")
    g.add_argument('--n_prefilt_layers', type=int, default=3, help="(PESTO default: 2)")
    g.add_argument('--prefilt_kernel_size', type=int, default=15)
    g.add_argument('--p_dropout', type=float, default=0.1, help="(PESTO default: 0.2)")
    g.add_argument('--d_model', type=int, default=128, help="Transformer head dimension")
    g.add_argument('--n_layers', type=int, default=2, help="Transformer head layers")
    g.add_argument('--n_heads', type=int, default=4)
    g.add_argument('--conv_channels', type=int, default=32, help="Channels of the full-resolution 2D-conv refinement after the transformer (0: none)")
    g.add_argument('--conv_layers', type=int, default=5, help="Layers of the refinement (time dilation doubles per layer: receptive field 2 ** (conv_layers + 1) + 1 frames)")
    g.add_argument('--context', type=int, default=64, help="Maximum number of frames seen by the temporal head (shorter clips are handled with padding masks)")


def encoder_hparams(args, n_bins):
    return dict(n_chan_input=len(args.harmonics), n_chan_layers=list(args.n_chan_layers), n_prefilt_layers=args.n_prefilt_layers,
                prefilt_kernel_size=args.prefilt_kernel_size, residual=True, n_bins_in=n_bins, output_dim=n_bins, p_dropout=args.p_dropout)


def temporal_hparams(args, n_bins):
    return dict(n_bins=n_bins, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, context=args.context,
                conv_channels=args.conv_channels, conv_layers=args.conv_layers)


def harmonic_stack(x, start, n_bins, harmonics, bins_per_octave):
    """ Harmonic CQT from cached CQT frames: channel h at grid bin b is the cached bin start + b + round(bins_per_octave * log2(h)),
    i.e. the energy at h times the frequency of b, so that all harmonics of an F0 line up on its bin (as in Bittner et al. 2017).
    In a constant-Q transform the h-th harmonic is analysed with a window h times shorter: higher harmonics bring time resolution.
    x: (..., n_cached_bins) dB; start: index of grid bin 0 in the cache (max_shift without pitch shift, max_shift - k for a shift of k bins),
    int or tensor broadcastable to x.shape[:-1]. Bins falling outside the cache are filled with the frame minimum.
    Returns (..., len(harmonics), n_bins). """
    offs = [int(round(bins_per_octave * np.log2(h))) for h in harmonics]
    lo, hi = max(0, -min(offs)), max(0, max(offs))
    fill = x.amin(-1, keepdim=True)
    xp = torch.cat([fill.expand(*x.shape[:-1], lo), x, fill.expand(*x.shape[:-1], hi)], -1)
    start = torch.as_tensor(start, device=x.device)
    idx = (start + lo)[..., None, None] + torch.tensor(offs, device=x.device)[:, None] + torch.arange(n_bins, device=x.device)
    shape = (*x.shape[:-1], len(offs))
    return xp.unsqueeze(-2).expand(*shape, xp.shape[-1]).gather(-1, idx.expand(*shape, n_bins))


class ToeplitzLinear(nn.Conv1d):
    def __init__(self, in_features, out_features):
        super(ToeplitzLinear, self).__init__(in_channels=1, out_channels=1, kernel_size=in_features + out_features - 1,
                                             padding=out_features - 1, bias=False)

    def forward(self, input):
        return super(ToeplitzLinear, self).forward(input.unsqueeze(-2)).squeeze(-2)


class Resnet1d(nn.Module):
    """ Frame-level encoder, copied from PESTO (pesto-full/src/models/networks/resnet1d.py).
    Only difference: returns logits, the softmax is applied by the caller.
    """
    def __init__(self, n_chan_input=1, n_chan_layers=(20, 20, 10, 1), n_prefilt_layers=1, prefilt_kernel_size=15, residual=False,
                 n_bins_in=216, output_dim=128, activation_fn='leaky', a_lrelu=0.3, p_dropout=0.2):
        super(Resnet1d, self).__init__()
        self.hparams = dict(n_chan_input=n_chan_input, n_chan_layers=list(n_chan_layers), n_prefilt_layers=n_prefilt_layers,
                            prefilt_kernel_size=prefilt_kernel_size, residual=residual, n_bins_in=n_bins_in, output_dim=output_dim,
                            activation_fn=activation_fn, a_lrelu=a_lrelu, p_dropout=p_dropout)
        activation_layer = {'relu': nn.ReLU, 'silu': nn.SiLU, 'leaky': partial(nn.LeakyReLU, negative_slope=a_lrelu)}[activation_fn]
        n_in, n_ch = n_chan_input, list(n_chan_layers)
        if len(n_ch) < 5:
            n_ch.append(1)

        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])
        prefilt_padding = prefilt_kernel_size // 2
        self.conv1 = nn.Sequential(nn.Conv1d(n_in, n_ch[0], kernel_size=prefilt_kernel_size, padding=prefilt_padding), activation_layer(), nn.Dropout(p=p_dropout))
        self.n_prefilt_layers = n_prefilt_layers
        self.prefilt_layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(n_ch[0], n_ch[0], kernel_size=prefilt_kernel_size, padding=prefilt_padding), activation_layer(), nn.Dropout(p=p_dropout))
            for _ in range(n_prefilt_layers - 1)])
        self.residual = residual
        conv_layers = []
        for i in range(len(n_ch) - 1):
            conv_layers.extend([nn.Conv1d(n_ch[i], n_ch[i + 1], kernel_size=1), activation_layer(), nn.Dropout(p=p_dropout)])
        self.conv_layers = nn.Sequential(*conv_layers)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = ToeplitzLinear(n_bins_in * n_ch[-1], output_dim)

    def forward(self, x):
        """ x: (batch, channels, freq_bins) -> logits (batch, output_dim) """
        x = self.conv1(self.layernorm(x))
        for layer in self.prefilt_layers:
            x = layer(x) + x if self.residual else layer(x)
        return self.fc(self.flatten(self.conv_layers(x)))


class ConvRefine(nn.Module):
    """ Residual 2D convolutions over the (time, pitch) log-salience map at full pitch resolution, applied after the transformer (whose
    corrections go through a d_model bottleneck). Time dilation doubles at each layer, pitch dilations cycle over freq_dilations.
    Pitch-shift equivariant. The output layer is initialised at zero (identity at start). Padded frames are zeroed after every layer,
    so that a padded + masked clip gives the same output as the clip alone. """
    def __init__(self, n_bins, channels=32, n_layers=5, freq_kernel=9, freq_dilations=(1, 3, 9), p_dropout=0.1, a_lrelu=0.3):
        super(ConvRefine, self).__init__()
        self.log_n_bins, fk = float(np.log(n_bins)), freq_kernel // 2
        self.inp = nn.Conv2d(1, channels, (3, freq_kernel), padding=(1, fk))
        self.layers = nn.ModuleList([nn.Conv2d(channels, channels, (3, freq_kernel), dilation=(2 ** i, freq_dilations[i % len(freq_dilations)]),
                                               padding=(2 ** i, freq_dilations[i % len(freq_dilations)] * fk)) for i in range(n_layers)])
        self.act, self.drop = nn.LeakyReLU(a_lrelu), nn.Dropout(p_dropout)
        self.out = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.out.weight), nn.init.zeros_(self.out.bias)

    def forward(self, logits, mask=None):
        """ logits: (batch, T, n_bins), mask: (batch, T) bool -> refined logits (batch, T, n_bins) """
        m = (torch.ones(logits.shape[:2], device=logits.device) if mask is None else mask.float())[:, None, :, None]
        z = (F.log_softmax(logits, dim=-1) + self.log_n_bins).clamp(min=-20).unsqueeze(1) * m  # 0 = flat distribution
        z = self.act(self.inp(z)) * m
        for conv in self.layers:
            z = z + self.drop(self.act(conv(z))) * m
        return logits + self.out(z).squeeze(1)


class TemporalHead(nn.Module):
    """ Small transformer refining the frame-level salience with up to `context` frames of temporal context, followed by an optional
    full-resolution 2D-conv refinement (ConvRefine, conv_channels > 0).
    Each token is the full log-salience distribution of a frame (all n_bins), projected to d_model.
    Sequences can be shorter than `context` (short clips), padded frames are excluded from attention with `mask`.
    The refinements are residual (output layers initialised at zero), so the head starts as the identity of the local salience.
    """
    def __init__(self, n_bins, d_model=128, n_layers=2, n_heads=4, context=64, p_dropout=0.1, conv_channels=0, conv_layers=5):
        super(TemporalHead, self).__init__()
        self.in_proj = nn.Linear(n_bins, d_model)
        self.pos = nn.Parameter(torch.zeros(1, context, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model, dropout=p_dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, n_bins)
        nn.init.zeros_(self.out_proj.weight), nn.init.zeros_(self.out_proj.bias)
        self.voicing = nn.Linear(d_model, 1)
        self.refine = ConvRefine(n_bins, conv_channels, conv_layers) if conv_channels else None

    def forward(self, local_logits, mask=None):
        """ local_logits: (batch, T <= context, n_bins), mask: (batch, T) bool, True for real frames (None: all real)
        -> refined logits (batch, T, n_bins), voicing logits (batch, T) """
        z = self.in_proj(F.log_softmax(local_logits, dim=-1)) + self.pos[:, :local_logits.shape[1]]
        z = self.norm(self.transformer(z, src_key_padding_mask=None if mask is None else ~mask))
        logits = local_logits + self.out_proj(z)
        if self.refine is not None:
            logits = self.refine(logits, mask)
        return logits, self.voicing(z).squeeze(-1)


class F0Model(nn.Module):
    def __init__(self, encoder_hparams, temporal_hparams, harmonics=(1,)):
        super(F0Model, self).__init__()
        self.encoder = Resnet1d(**encoder_hparams)
        self.temporal = TemporalHead(**temporal_hparams)
        self.context, self.harmonics = temporal_hparams['context'], list(harmonics)

    def forward(self, x, mask=None):
        """ x: (batch, T, n_harmonics, n_bins) harmonic CQT frames (harmonic_stack), mask: (batch, T) bool (True for real frames)
        -> local logits, refined logits (batch, T, n_bins), voicing logits (batch, T) """
        B, T, H, Fb = x.shape
        local = self.encoder(x.reshape(B * T, H, Fb)).reshape(B, T, -1)
        logits, voicing = self.temporal(local, mask)
        return local, logits, voicing


def load_model(path, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model = F0Model(ckpt['encoder_hparams'], ckpt['temporal_hparams'], ckpt.get('harmonics', [1]))
    model.load_state_dict(ckpt['model'])
    return model.eval().to(device), ckpt['cqt_params']


@torch.no_grad()
def infer(model, x, p, device='cpu', hop=None, batch_size=64, enc_batch_size=2048, desc=None):
    """ Runs the model over a whole clip.
    x: cached CQT in dB (frames, n_bins + 2 * max_shift). The encoder runs once per frame; the temporal head then sees
    the whole clip in one pass if it is not longer than model.context frames, otherwise overlapping windows of model.context
    frames (hop: default context // 4) whose predictions are averaged with a Hann weighting (frames near a window's centre,
    which have context on both sides, count more).
    desc: if given, shows progress bars over the batches with this description.
    Returns the salience (frames, n_bins) and voicing probabilities (frames,) as numpy arrays.
    """
    ctx, ms, nb = model.context, p['max_shift'], p['n_bins']
    T = len(x)
    # harmonic stack and encoder per batch of frames, local logits kept on the CPU: memory does not grow with the file length on the GPU
    local = torch.cat([model.encoder(harmonic_stack(torch.as_tensor(np.asarray(x[i:i + enc_batch_size], dtype=np.float32), device=device),
                                                    ms, nb, model.harmonics, p['bins_per_octave'])).cpu()
                       for i in tqdm(range(0, T, enc_batch_size), desc=desc and desc + ' (encoder)', leave=False, disable=desc is None)])
    if T <= ctx:
        logits, voicing = model.temporal(local.unsqueeze(0).to(device))
        return F.softmax(logits[0], dim=-1).cpu().numpy(), torch.sigmoid(voicing[0]).cpu().numpy()
    hop = hop or max(1, ctx // 4)
    starts = list(range(0, T - ctx + 1, hop))
    if starts[-1] != T - ctx:
        starts.append(T - ctx)
    w = torch.hann_window(ctx + 2, periodic=False)[1:-1]  # strictly positive weights
    sal, voi, cnt = torch.zeros(T, nb), torch.zeros(T), torch.zeros(T)
    for b in tqdm(range(0, len(starts), batch_size), desc=desc and desc + ' (temporal head)', leave=False, disable=desc is None):
        idx = torch.tensor(starts[b:b + batch_size]).unsqueeze(1) + torch.arange(ctx)
        logits, voicing = model.temporal(local[idx].to(device))
        sal.index_add_(0, idx.flatten(), (F.softmax(logits, dim=-1).cpu() * w[None, :, None]).flatten(0, 1))
        voi.index_add_(0, idx.flatten(), (torch.sigmoid(voicing).cpu() * w).flatten())
        cnt.index_add_(0, idx.flatten(), w.repeat(len(idx)))
    return (sal / cnt.unsqueeze(1)).numpy(), (voi / cnt).numpy()
