# Per-species characteristics, following bioacoustic_f0_estimation/paper_experiments/metadata.py
#   wavpath : glob pattern of sound files (one vocalisation per file, annotations in {basename}.csv with columns Time (s), Freq (Hz))
#   FS      : sampling rate at which files are loaded (per-species). There is no downsampling trick: the CQT hop is relative to fmin
#             instead (cqt.hop_length), which has the same effect of putting all species on a common time scale
#   fmin    : bottom of the species' 6-octave relative-pitch grid (bin 0). Used only for preprocessing (CQT range) and decoding
#             (f = fmin * 2 ** (bin / bins_per_octave)). The top of the cached CQT must stay under Nyquist: see cqt.max_fmin
#
# fmin rule: half the 0.5th percentile of annotated F0 (one octave of margin below the species' range, room for subharmonics),
# capped by cqt.max_fmin(FS) (about FS / 175) and rounded down. FS may be above the native sampling rate when the cap would
# otherwise push the F0 range to the top of the grid (99.5th percentile more than 5 octaves above fmin, e.g. little owls, native 4 kHz,
# hummingbirds, hermits): upsampled bins above the native Nyquist are just empty.
# Annotated F0 percentiles (0.5 / 50 / 99.5, Hz) measured on 2026-11-06 are given in comments.
# fmin = None: species skipped by all scripts (data not available yet)
# Optional:
#   exclude           : fnmatch patterns of file basenames to leave out
#   max_per_recording : keep at most this many clips per source recording (clips named {recording}_{n}.wav), evenly spread
#   annot_offset_samples : added to the annotation times, in samples at the file's native rate (see preprocess.annot_offset)
#   annot_step        : median time between annotation points (s), measured on 2026-11-07; used for the label uncertainty (dataset.label_uncertainty)
#   annot_grid_hz     : frequency grid the annotations are quantised on (spectrogram bins of the annotation tool), measured on 2026-11-07;
#                       used for the label uncertainty (dataset.label_uncertainty)
import os
from glob import glob
from fnmatch import fnmatch
import numpy as np

species = {
    'canids':                   {'wavpath': 'data/wolves/canids/*/*.wav',                                   'FS': 16000,  'annot_step': 0.0468, 'fmin': 90},   # 197 / 496 / 1432, native 8-22 kHz
    'dolphins':                 {'wavpath': 'data/dclmmpa2011/cut_no_overlap/*.wav',                         'FS': 192000, 'fmin': None},
    'spotted_hyenas':           {'wavpath': 'data/hyenas/lehmann hyena whoop traces/*.wav',                  'FS': 16000,  'annot_grid_hz': 3.91, 'annot_step': 0.005, 'fmin': 80},   # 164 / 301 / 746, native 32 kHz
    'vocadito':                 {'wavpath': 'data/vocadito/*.wav',                                           'FS': 44100,  'annot_step': 0.0058, 'fmin': 44},   # 88 / 219 / 411 (human singing)
    'vocalnotes':               {'wavpath': 'data/vocalnotes/*.wav',                                         'FS': 44100,  'annot_step': 0.0058, 'fmin': 50},   # 102 / 221 / 586 (human singing)
    'orangutans':               {'wavpath': 'data/orangutans/*.wav',                                        'FS': 44100,  'annot_step': 0.0143, 'fmin': 25},   # 39 / 277 / 1087: 4.8-octave range, fmin centres it in the grid (rule would give 19)
    'rodents':                  {'wavpath': 'data/LiuLabData/cut_no_overlap/*.wav',                          'FS': 250000, 'fmin': None},
    'lions':                    {'wavpath': 'data/lions/Biologger Roars Expanded (500ms)/*.WAV',            'FS': 16000,  'annot_grid_hz': 1.95, 'annot_step': 0.0405, 'fmin': 19},   # 39 / 148 / 256
    'monk_parakeets':           {'wavpath': 'data/monk_parakeets/pre-processed_calls/*.WAV',                'FS': 44100,  'annot_step': 0.0017, 'fmin': 250},  # 687 / 1760 / 3186
    'La_Palma_chaffinches':     {'wavpath': 'data/La_Palma_chaffinches/cut/*.wav',                          'FS': 44100,  'annot_grid_hz': 90., 'annot_offset_samples': 256, 'annot_step': 0.0053, 'fmin': 250},  # 1206 / 3094 / 5857, native 44.1-48 kHz
    'little_owls':              {'wavpath': 'data/little_owls/cut/*.wav',                                    'FS': 32000,  'annot_grid_hz': 7.81, 'annot_step': 0.0082, 'fmin': 180},  # 609 / 922 / 1289, native 4 kHz (upsampled)
    'Reunion_grey_white_eyes':  {'wavpath': 'data/Reunion_grey_white_eyes/cut/*.wav',                       'FS': 44100,  'annot_grid_hz': 90., 'annot_step': 0.0054, 'fmin': 250},  # 2670 / 3962 / 6029, native 44.1-48 kHz
    'long-billed_hermits':      {'wavpath': 'data/long-billed_hermits/*.wav',                               'FS': 88200,  'annot_step': 0.0014, 'fmin': 500},  # 2093 / 4514 / 8201, native 44.1 kHz (upsampled)
    'hummingbirds':             {'wavpath': 'data/hummingbirds/*.wav',                                      'FS': 88200,  'annot_grid_hz': 86.13, 'annot_step': 0.0061, 'fmin': 500,   # 1206 / 6546 / 10767, native 44.1 kHz (upsampled)
                                 # 13680 clips from 881 recordings of 278 Apodiformes species (hummingbirds and ~1130 swift clips). Capped to limit
                                 # redundancy and cache size; Phaethornis longirostris = long-billed hermit, excluded to keep --omit long-billed_hermits clean
                                 'max_per_recording': 5, 'exclude': ['Phaethornis-longirostris-*']},
    'disk-winged_bats':         {'wavpath': 'data/disk-winged_bats/spixs_disc_winged_bat_*/*.wav',          'FS': 400000, 'annot_grid_hz': 732.4, 'annot_step': 0.0007, 'fmin': 2270}, # 11719 / 22705 / 48340, native 375-400 kHz
    'bottlenose_dolphins':      {'wavpath': 'data/sayigh/*/*.wav',                                           'FS': 96000,  'fmin': None},
}

# Synthetic species generated by make_fake_data.py, used to test the pipeline end-to-end
fake_species = {
    'fake_lowfreq':  {'wavpath': 'data/fake/fake_lowfreq/*.wav',  'FS': 16000,  'annot_step': 0.005, 'fmin': 50,   'f0_range': (100, 1000)},
    'fake_bird':     {'wavpath': 'data/fake/fake_bird/*.wav',     'FS': 44100,  'annot_step': 0.005, 'fmin': 200,  'f0_range': (1000, 6000)},
    'fake_ultra':    {'wavpath': 'data/fake/fake_ultra/*.wav',    'FS': 250000, 'annot_step': 0.005, 'fmin': 1000, 'f0_range': (20000, 50000)},
}


def get_species(only=None, omit=None, fake=False):
    """ Returns the {name: characteristics} dict of species to use (skips species without fmin) """
    pool = fake_species if fake else species
    names = [only] if only else [s for s in pool if s != omit]
    out = {}
    for s in names:
        if pool[s]['fmin'] is None:
            print(f'Skipping {s}: fmin not set in metadata.py')
            continue
        out[s] = pool[s]
    return out


def list_files(meta):
    """ Sorted sound files of a species (metadata dict, or a wavpath glob pattern), skipping broken symbolic links,
    excluded files and clips beyond max_per_recording """
    meta = {'wavpath': meta} if isinstance(meta, str) else meta
    files = [f for f in sorted(glob(meta['wavpath'])) if os.path.exists(f)]
    files = [f for f in files if not any(fnmatch(os.path.basename(f), pat) for pat in meta.get('exclude', []))]
    if meta.get('max_per_recording'):
        recs = {}
        for f in files:
            recs.setdefault(os.path.basename(f).rsplit('_', 1)[0], []).append(f)
        k = meta['max_per_recording']
        files = sorted(f for clips in recs.values() for f in (clips if len(clips) <= k else [clips[i] for i in np.linspace(0, len(clips) - 1, k).round().astype(int)]))
    return files
