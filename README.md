# Cross-species F0 estimation

Training scripts for *Cross-Species F0 Tracking: Hybrid Training Framework*: a PESTO-style frame encoder on a relative-pitch CQT
and a small transformer temporal head, trained in a single phase with supervised losses and PESTO's self-supervised losses as a regulariser.
Code structure and conventions follow [bioacoustic_f0_estimation](https://github.com/lamipaul/bioacoustic_f0_estimation)
(dataset format, `metadata.py` species dictionary, `--omit/--only/--split` options, per-vocalisation evaluation).


## Model and training details
- **Input**: log-magnitude CQT in dB over `[fmin, fmin * 2**6]` at 120 bins/octave (720 bins), windows of 52 periods (`--filter_scale 0.3`, PESTO's
  time-frequency trade-off) and a hop relative to the species (`--hop_periods 0.9`: hop = 0.9 / fmin, e.g. 0.4 ms for bats, 10 ms for canids, 47 ms for lions),
  so that time and pitch are both relative to fmin (equivalent to the downsampling trick of bioacoustic_f0_estimation). Cached with `max_shift` extra bins
  on each side for pitch shifts by cropping. Not normalised: the encoder's first per-frame LayerNorm makes per-clip normalisation unnecessary.
  `fmin` is never given to the network, only used for preprocessing and decoding (`f = fmin * 2 ** (bin / 120)`).
  The encoder receives a harmonic CQT (`--harmonics`, default 0.5 1 2 3, as in Bittner et al. 2017's deep salience): channel h is the cached CQT
  shifted so that harmonic h lines up with the F0 bin (`model.harmonic_stack`, built on the fly, no extra cache). Higher harmonics are analysed
  with shorter constant-Q windows, which helps on fast sweeps. Pitch shifts move all channels together.
- **Encoder**: PESTO's `Resnet1d` per frame (LayerNorm, prefiltering convs, 1x1 convs, Toeplitz linear layer), 720 output bins, larger than PESTO's
  default (`--n_chan_layers 64 48 48 16 4 --n_prefilt_layers 3 --p_dropout 0.1`, 0.74 M parameters with the head).
- **Temporal head**: transformer over up to `--context` frames (default 64). Each token is the full local log-salience distribution of a frame
  (all 720 bins, linearly projected). It outputs a residual refinement of the salience (initialised at identity) and a voicing logit per frame.
  Clips shorter than the context are padded and padded frames are masked out of attention. During training, `--p_truncate` of the windows are cut to a
  random length so that the head also learns short inputs. At inference the encoder runs once per frame. Clips up to `context` frames go through the
  head in one pass; longer clips use windows with a hop of `context // 4`, averaged with a Hann weighting.
  The transformer is followed by a full-resolution refinement (`ConvRefine`, `--conv_channels 32 --conv_layers 5`): residual 2D convolutions
  over the (time, pitch) log-salience map, dilated in time (65-frame receptive field), so that corrections are not limited by the d_model bottleneck.
- **Training** (single phase, from scratch): cross-entropy with Gaussian smooth labels on the refined and local salience, voicing BCE,
  and PESTO's losses (equivariance, shift cross-entropy, invariance, combined by its gradient-based weighting) with a constant weight `--ssl_weight`,
  computed on voiced annotated frames only. The supervised terms anchor the absolute pitch, so no calibration of the PESTO output is needed.
  Species are sampled with inverse frequency weights.
- **Augmentations**: real background noise (unvoiced annotated frames of the training files, all species) mixed in the power domain at a random SNR
  (`--p_noise`, `--noise_snr`, default 0.5 and -5 to 20 dB), for the training windows and, independently, for PESTO's augmented and pitch-shifted views.
  Supervised pitch shift: with probability `--p_shift` (0.8), the CQT crop and the targets move by the same random number of bins, up to
  `--shift_aug` (default `max_shift` = 180 bins, 1.5 octaves), so that vocalisations are seen anywhere in the grid. Time rescaling: with probability
  `--p_time` (0.8), independently of the pitch shift, frames are resampled by a factor 2^u with u uniform in +-`--time_aug` (1.5 octaves),
  as with another hop (`dataset.WindowDataset.rescaled`, labels interpolated), so that the model does not tie a contour speed to a pitch range.
  Also random window offsets and random truncation. PESTO's pitch-shifted views keep shifts of up to 55 bins.
- **Decoding**: weighted argmax around the salience peak; voicing from the temporal head, thresholded at the ROC optimum per species in `eval.py`.
- **Label width**: each pitch target is a Gaussian of std sqrt(`--sigma`^2 + u^2), with `--sigma` 2.5 bins (25 cents, as CREPE) and u the label's
  uncertainty (`dataset.label_uncertainty`): annotation timing (|contour slope| x `annot_step` / sqrt(12)) and, for quantised annotations,
  the spectrogram grid (`annot_grid_hz`, rounding std = grid width / sqrt(12)). Both are measured per species in `metadata.py`.
  Evaluation is unchanged (50 cents tolerance).
- **Evaluation**: as `eval_all.py` (50 cents, per vocalisation, ROC-optimal voicing threshold per species). Two secondary pitch accuracies
  compensate for the annotation resolution (`eval.label_interval`): a frame is correct if the estimate is within 50 cents of any value the
  label could have taken, given the frequency grid (`grid-comp.`, objective, 50 cents) or the grid and +- half an annotation step in time
  (`grid+timing-comp.`, with a stricter tolerance of 10 cents, `--comp_cent_thr`, as CREPE's strictest metric: the interval absorbs the
  annotation resolution, so the tolerance only covers the estimate's own error). `Pitch acc` remains the standard, comparable metric.
- **Labels**: frames inside annotation gaps of 1.5 to 4 times the median annotation step are set to unknown (NaN, excluded from losses and evaluation),
  see `preprocess.load_annot`. La Palma chaffinch annotations are shifted by NFFT/2 = 256 samples (`annot_offset_samples`, see
  `preprocess.annot_offset`). `python preprocess.py --labels_only` recomputes labels without recomputing CQTs.

## Results (fold 0) vs bioacoustic_f0_estimation
Pitch accuracy (50 cents) of `split0_v3` (`scores/split0_v3_best_summary.csv`) against the methods of Best et al. 2025
(`bioacoustic_f0_estimation/paper_experiments/scores/{specie}_scores.csv`), on the 11 species both evaluate. Same metric code (`eval_all.py`).

| Species | pYIN | CREPE | PESTO ft. species | CREPE ft. other species | CREPE ft. species | **ours (v3)** |
| --- | --- | --- | --- | --- | --- | --- |
| canids | 0.562 | 0.688 | 0.758 | 0.686 | 0.734 | **0.822** |
| spotted hyenas | 0.359 | 0.635 | 0.911 | 0.721 | **0.942** | 0.922 |
| orangutans | 0.162 | 0.414 | 0.038 | 0.432 | **0.506** | 0.478 |
| lions | 0.354 | **0.531** | 0.283 | 0.519 | 0.475 | 0.518 |
| monk parakeets | 0.084 | 0.186 | 0.229 | 0.246 | 0.297 | **0.662** |
| La Palma chaffinches | 0.093 | 0.194 | 0.094 | 0.294 | 0.279 | **0.498** |
| little owls | 0.530 | 0.822 | 0.905 | 0.766 | 0.888 | **0.960** |
| Reunion grey white-eyes | 0.233 | 0.623 | 0.649 | 0.684 | 0.709 | **0.733** |
| long-billed hermits | 0.140 | 0.149 | 0.040 | 0.152 | 0.415 | **0.608** |
| hummingbirds | 0.428 | 0.533 | 0.021 | 0.571 | 0.630 | **0.714** |
| disk-winged bats | 0.438 | 0.478 | 0.063 | 0.704 | 0.732 | **0.815** |
| **mean** | 0.308 | 0.477 | 0.363 | 0.525 | 0.601 | **0.703** |
| mean voicing recall / false alarm | 0.773 / 0.207 | 0.744 / 0.231 | 0.837 / 0.128 | 0.753 / 0.226 | 0.826 / 0.169 | **0.909 / 0.105** |

CREPE ft. species: one CREPE per species, fine-tuned with 5-fold cross-validation (`tcrepe_ftsp`); CREPE ft. other species: fine-tuned on all
other species (`tcrepe_ftoth`, the target species is unseen); PESTO ft. species: PESTO trained self-supervised on the target species (`pesto_ft`).
Ours: one model for all species, trained on folds 1-4 of every species. Not an exact comparison yet:
- ours is scored on test fold 0 only (1/5 of the files), the baselines on all files (5-fold cross-validation for `tcrepe_ftsp`);
- labels differ slightly: La Palma chaffinch annotations are shifted by 256 samples here, canid annotation gaps of 1.5-4 median steps are unknown
  (not scored) here, hummingbirds are capped at 5 clips per recording here (13680 clips in the baseline);
- `tcrepe_ftoth` never saw the target species: the matching experiment for our model is `--omit`.


## Data format
As in bioacoustic_f0_estimation: one sound file per vocalisation, with annotations in `{basename}.csv` (columns `Time` in seconds and `Freq` in Hz).
Species are declared in `metadata.py` with their glob pattern `wavpath`, sampling rate `FS` and `fmin`, the bottom of their 6-octave pitch grid
(the rule used to choose them is written there). `fmin` must be low enough for the 6-octave grid to stay under Nyquist:
`fmin <= FS / 2 / 2 ** ((n_bins - 1) / bins_per_octave)` (about `FS / 127` with default parameters, `cqt.max_fmin`); `preprocess.py` raises an
error otherwise. The cached margin bins above Nyquist (used by pitch shifts and harmonic channels) are filled with the frame minimum.

## Pipeline
```
python preprocess.py                                    # cache CQTs, frame labels and file lists (cache/cqt_{hash}/{specie}/)
python train.py --split 0 --name split0                 # -> weights/split0.pth and weights/split0_best.pth
python eval.py weights/split0_best.pth --split 0        # scores/ per species
python predict.py my_recordings/ --model_path weights/split0_best.pth   # FS and fmin from each file, or --specie / --FS / --fmin
python predict.py my_recordings/ --model_path weights/split0_best.pth --step 0.002   # another time step (CQT hop given to the model)
```
`predict.py` writes `{basename}_f0.csv` with `time`, `f0`, `confidence` (voicing), `salience_max` (peak of the model's pitch distribution)
and `salience`, the spectral salience of the estimated F0 as in bioacoustic_f0_estimation (energy within +-1 semitone over energy within
+-6 semitones, on magnitudes whitened per frequency bin over the file), computed on the CQT so that it does not depend on F0 or sampling rate.
`--omit specie`, `--only specie` and `--split k` (5-fold, test fold k) work as in `paper_experiments/train_crepe.py`.
With `--split`, `train.py` validates on the held-out fold each epoch and also saves `weights/{name}_best.pth`.
Training and evaluation only need the cache: when the audio is absent, the file lists stored in the cache are used (same folds).
Logs are written to `runs/` for TensorBoard. `./smoke_test.sh` runs everything on synthetic data (`make_fake_data.py`, `--fake` option).

## Running on a Slurm cluster
`run_slurm.sh` runs preprocessing, training and evaluation, trained on folds 1-4 and evaluated on fold 0 (`S` in the script).
From the project folder, with the conda environment activated (the job calls `$CONDA_PREFIX/bin/python` by full path):
```
sbatch run_slurm.sh
```
GPU type, epochs etc. are set directly in the script. The job loads the cache in RAM (`--in_memory`) and resumes from its last epoch (`--resume`),
so it can be resubmitted as is.

## Files
| File | Content |
| --- | --- |
| `metadata.py` | species dictionary (+ synthetic test species), file listing |
| `cqt.py` | time-domain CQT, relative hop, cache naming, Hz <-> bin conversions |
| `preprocess.py` | CQT + label caching (annotation resampling as in `run_all.py`), file lists |
| `dataset.py` | 5-fold splits, windows of `context` frames, noise bank, species-balanced sampling weights |
| `model.py` | PESTO `Resnet1d` encoder (Toeplitz output layer), transformer `TemporalHead`, sliding-window inference |
| `losses.py` | PESTO losses (equivariance, shift cross-entropy, invariance), pitch shift, gradient-based loss weighting, real-noise mixing; supervised losses |
| `decode.py` | weighted argmax |
| `train.py` | training |
| `eval.py`, `predict.py` | evaluation (as `eval_all.py`) and inference (as `predict.py`) |
| `robustness.py` | accuracy vs fmin offset (grid position), CQT recomputed from the audio of test files, several models compared |
| `run_slurm.sh`, `smoke_test.sh` | Slurm job (fold 0), synthetic end-to-end test |

