# Project context: cross_species_f0

Training code for the paper *Cross-Species F0 Tracking: Hybrid Training Framework*. It estimates F0 for many species
(canids, dolphins, hyenas, rodents, bats, birds, ...) with one model working on a relative-pitch grid (6 octaves above a per-species `fmin`).
See `README.md` for the pipeline, file table and model/training details. Read it before making changes.

## Conventions
- Mirrors [bioacoustic_f0_estimation](https://github.com/lamipaul/bioacoustic_f0_estimation) (`paper_experiments/`): dataset format
  (one wav per vocalisation + `{basename}.csv` with `Time`, `Freq`), `metadata.py` species dict, `--omit/--only/--split` (5-fold on the sorted file list),
  per-vocalisation evaluation with ROC-optimal voicing threshold (as `eval_all.py`). Keep this compatibility.
- PESTO code (`Resnet1d`, losses, pitch shift, gradient loss weighting) is copied from `pesto-full` without the Lightning dependencies.
  Keep it close to the original (the augmentation is replaced by real-noise mixing).
- Style: compact scripts, argparse at module level, no frameworks beyond torch. Comments in English.
- `fmin` is never an input to the network. It is only used for the CQT range (preprocessing) and for decoding bins to Hz.
- The CQT cache lives in `cache/cqt_{hash of CQT params}/{specie}/`. Changing CQT args creates a new cache folder.
- Not a git repo. `data/ cache/ weights/ runs/ scores/ logs/` are gitignored outputs.
- Keep the code lean: the user asked to remove options that are not used (2026-11-07).

## Status (as of 2026-11-07)
- **Single-phase training** (user's decision, 2026-11-07): `train.py` trains encoder + transformer head from scratch with the supervised losses and
  PESTO's losses at a constant `--ssl_weight` (default 1), on voiced annotated frames only. Replaces the former two-phase design (PESTO pretraining, then
  joint training with a frozen encoder and decaying SSL weight). Why: the Phase 1 run (cluster, fold 0) learned relative pitch well (95% of voiced frames
  shift exactly with the input; 65-80% accuracy for most species at a constant offset of -136 bins, harmonic confusions for parakeets, hyenas and singing),
  but its logits exploded (top-to-median gap 500-2000) and its absolute offset was arbitrary, so the frozen part of Phase 2 learned nothing
  (train accuracy < 4%). After unfreezing, Phase 2 reached 0.68 validation pitch accuracy (epoch 11 onward). Those results are the baseline for the new design.
- Removed at the user's request (not used): two-phase code, `--temporal_head conv`, Viterbi decoder, `--ssl_aug pesto/both` (PESTO's Gaussian noise + gain),
  `--precision bf16`, smoothness loss, per-clip normalisation, `run_real.sh`, compatibility code for old checkpoints and caches. Old checkpoints
  (`ssl_*`, `joint_*`) no longer load. Decoding is weighted argmax only.
- Data: 13 species with `fmin` in `metadata.py` (canids, hyenas, little owls and the two singing sets from `~/Documents/bioacoustic_f0_estimation/paper_experiments/data/`,
  8 more from the external drive `/mnt/wwn-0x50014ee26b98a243-part2/F0/`, which must be mounted on the laptop). Dolphins, rodents and bottlenose dolphins
  have no data yet. Hummingbirds are capped at 5 clips per source recording and exclude *Phaethornis longirostris* (= long-billed hermit); the folder also
  holds about 1130 swift clips. Canid annotation gaps of 1.5-4x the median step are unknown labels.
- CQT: `filter_scale` 0.3 (windows of 52 periods, as PESTO) and relative hop `hop_periods` 0.9 (hop = 0.9 / fmin), both chosen by the user.
  Cache `cache/cqt_9da21d82db` (6.9 GB, with `files.txt` per species so that training runs without audio). Unused old caches:
  `cqt_adce121ac8`, `cqt_318bd16737`.
- Cluster (Slurm, partition `g`: B200 GPUs and MIG slices `nvidia_b200_1g.23gb`, 1.5 TB RAM per node, no node-local disk, cbe-g5-6 drained;
  `--ntasks=1` required with GPU requests; jobs hit a time limit despite `MaxTime=UNLIMITED`, resubmitting resumes). Python 3.15 + PyTorch 2.15 (cu132).
  Compute nodes put another python first on the PATH: `run_slurm.sh` calls `$CONDA_PREFIX/bin/python`, so the env must be active at `sbatch` time.
  The cluster has the cache but no audio. GPU runs are launched by the user: the Claude Code sandbox hides `/dev/nvidia*` and forbids the sockets
  that multi-worker DataLoaders use, so only CPU and `--num_workers 0` can be tested here.
- Python >= 3.14 starts DataLoader workers with forkserver, which re-imports the unguarded scripts: `dataset.loader_kwargs` forks explicitly.
- `--sigma` default changed from 1 to 2.5 bins (25 cents, CREPE's label smoothing; user's choice, 2026-11-07).
- 2026-11-07: single-phase run (cluster, fold 0) at 3.2 steps/s (~34 min/epoch). Epoch 0 val pitch acc 0.687 (= best of the two-phase run).
  Eval of epoch 2 vs the two-phase Phase 2 epoch 14 (`scores/split0_summary.csv`, `scores/twophase_split0_summary.csv`, plot `scores/split0_vs_twophase.png`):
  mean over species pitch acc 0.651 vs 0.629, overall acc 0.680 vs 0.635, voicing recall 0.862 vs 0.811, false alarm 0.157 vs 0.204.
  Hardest: hermits 0.32, chaffinches 0.33, parakeets 0.42, orangutans 0.45, lions 0.50 (chroma acc ~ pitch acc: not octave errors).
  PESTO invariance/shift-entropy barely move with smooth outputs (their CE on probabilities ~ log(720) - <p,q>); equivariance decreases.
  Planned later (user): ablation `--ssl_weight 0`. The user declined per-species validation logging in TensorBoard for now.
  To evaluate old two-phase checkpoints, drop `temporal_hparams['type']` from a copy (done in $TMPDIR, not in the repo).
- 2026-11-07: epoch 7 (best so far) on fold 0: mean pitch acc 0.666, overall 0.714, recall 0.888, false alarm 0.129 (`scores/split0_ep7_summary.csv`,
  progression plot `scores/split0_progress.png`; epoch-2 scores kept as `scores/split0_ep2_*`, since eval.py names outputs after the checkpoint file).
  Chaffinches stuck at 0.34; hummingbirds have ~6% octave errors (chroma 0.68 vs pitch 0.62). Plot script lives in the session scratchpad only.
- 2026-11-07 diagnosis (epoch 7, test fold 0; scripts in the session scratchpad): the single model already matches per-species fine-tuned CREPE of
  Best et al. 2025 (`~/Documents/bioacoustic_f0_estimation/paper_experiments/scores`: mean pitch acc 0.617 vs 0.601 for tcrepe_ftsp, 0.525 for tcrepe_ftoth).
  Hard species are hard for all methods. Gross label errors <= 3% of frames. Errors on hard species are mostly near misses (50-200 cents) and grow
  with contour slope (acc 0.88 slow -> 0.19 fast): the 52-period CQT window spans several semitones of sweep for hermits, parakeets, chaffinches,
  white-eyes, lions, orangutans. Chaffinch labels are ~7 ms late (acc 0.32 -> 0.46 with a 2-frame shift; all other species peak at 0 shift).
  Annotations are quantised on spectrogram FFT grids (chaffinches/white-eyes 86-94 Hz, bats 732 Hz, hummingbirds 86 Hz, hyenas 3.9 Hz, owls 7.8 Hz).
  Proposed (awaiting user): chaffinch `annot_offset`, test shorter / two-resolution CQT windows, 100-cent tolerance in eval. Loss weighting by
  label/model agreement judged low priority (confirmation bias, few gross errors).
- 2026-11-07 (user's decisions after the diagnosis): no agreement-based loss weighting (keep in mind as alternatives: noise-robust loss such as
  generalised cross-entropy, soft bootstrapping targets); no wider evaluation tolerance (stays 50 cents).
  - Chaffinch labels corrected by NFFT/2 = 256 samples at each file's native rate (user's hypothesis: the annotation tool stamped frame starts;
    annotation step = hop = 256 samples). CQT z at label 2.04 -> 2.29. The clip cut times (rounded to 10 ms in file names) are not involved:
    checked by locating all 347 clips in the original recordings `/mnt/.../F0/FCPalmae_2023/FCPalmae` (no annotations there).
    Cache labels of the chaffinches were recomputed locally; the cluster needs `cache/cqt_9da21d82db/La_Palma_chaffinches/*_f0.npy`.
  - Per-frame target width sqrt(sigma^2 + u^2), u computed in memory by `dataset.label_uncertainty` from `annot_step` and `annot_grid_hz` in
    `metadata.py` (user preferred this over cache files). Median widths: 25 cents (singing, owls, canids) to 58 cents (chaffinches); p90 up to
    ~210 cents for lions, ~175 orangutans, ~170 chaffinches (the timing assumption may be pessimistic for long annotation steps). The user chose a cap: u <= 100 cents (widest target ~103 cents;
    capped on 21-26% of frames for lions, orangutans and chaffinches, 10% for hermits).
  - A shorter CQT window does not help even on fast sweeps (filter_scale 0.15 / 0.075 worse than 0.3), so no multi-resolution input for now.
  - Don't run steps the user has not explicitly agreed to (they stopped a label recomputation to check what was being written).
- `run_slurm.sh` takes the run name from `NAME` (default `split0`): the user runs the uncertainty-aware version as
  `sbatch --export=ALL,NAME=split0_unc run_slurm.sh` in parallel with the running `split0` job (which should not be resubmitted with the new code).
- 2026-11-07: `eval.py` reports two extra columns (user's agreement, as secondary results next to the standard `Pitch acc`):
  `Pitch acc (grid-comp.)` and `Pitch acc (grid+timing-comp.)`, see `eval.label_interval` (unit-tested on synthetic cases).
- 2026-11-08 plateau analysis (`split0_unc` epoch 6, val 0.717 vs 0.697 for `split0`): **underfitting**, not overfitting: clean training-file
  accuracy = test accuracy (0.667 vs 0.668), train acc on augmented windows (0.63) < val (0.71). The temporal head adds only +2.9 points over the
  per-frame encoder (nothing for chaffinches, lions, orangutans, white-eyes). User's decision: implement 4 changes together in one run `split0_v2`
  (attribution needs ablations later): harmonic CQT input (`--harmonics 0.5 1 2 3`), bigger encoder (`64 48 48 16 4`, 3 prefilter layers,
  dropout 0.1), `--lr 3e-4`, full-resolution 2D-conv refinement after the transformer (`ConvRefine`, `--conv_channels 32`). These are now the
  defaults; old checkpoints load as single-channel models without refinement. CPU step time x2.7 (0.74 M parameters vs 0.49 M).
  `runs/split0` contains a resubmission of the old run (job 3000503) after the code update: do not count it as the old setup.
- 2026-11-08 **split0_v2 results** (20 epochs, 11.5 steps/s, ~3.5 h; best = last checkpoint; `scores/split0_v2_best_summary.csv`):
  mean pitch acc 0.746 (previous model 0.676 with the same corrected labels), overall 0.775, recall 0.908, false alarm 0.102,
  grid-comp. 0.773, grid+timing-comp. 0.868. Against Best et al. 2025 on the 11 shared species: 0.710 vs 0.601 (tcrepe_ftsp) and 0.525
  (tcrepe_ftoth), better on 9/11 (orangutans 0.494 vs 0.506, hyenas 0.927 vs 0.942). Biggest gains: hermits 0.35 -> 0.61, parakeets 0.44 -> 0.64,
  hummingbirds 0.62 -> 0.74. Still weak: orangutans 0.49 (grid+timing only 0.61, not octave errors), chaffinches 0.51, lions 0.53; canid voicing
  (recall 0.79, FA 0.22). Val acc still rising slowly at the end (0.779), train acc on augmented windows 0.735 < val.
  Proposed next (awaiting user): ablations of the 4 changes + `--ssl_weight 0`, `--omit` experiments (vs tcrepe_ftoth), 5-fold CV, orangutan diagnosis.
- 2026-11-08 grid-position robustness (user's decisions): the model failed on gibbons with the automatic fmin (calls high in the grid, coarse hop);
  `--fmin 250` fixed most of it; single-channel WPE dereverberation (scratch implementation) removed < 0.5 dB, no effect. `predict.py` now infers FS
  (native rate) and fmin (highest allowed) per file, upsamples when a requested fmin needs it, and prints the range. Planned: robustness curve
  (accuracy vs fmin offset +-2 octaves, from audio), two-pass automatic fmin in predict.py, CQT-domain reverberation augmentation.
  Implemented for `split0_v3`: cache margin 180 bins (`--max_shift 180`, cache `cqt_a0d647935b`, ~9 GB; bins above Nyquist filled with the frame
  minimum, only the grid must be below Nyquist), supervised pitch shift up to +-180 bins (`--p_shift 0.8`) and independent time rescaling
  (`--time_aug 1.5` octaves, `--p_time 0.8`, `dataset.rescaled`, equivalent to another hop), PESTO shift kept at +-55 bins. The user computes
  the cache on the laptop GPU (CPU took too long; 2575 files were done on CPU) and copies it to the cluster. Gibbon detection plots:
  `plots/gibbons_CAR07_20220530_070000/`.
- 2026-11-08: `robustness.py` (fmin' = fmin * 2**offset, offsets -2..2 octaves, 20 test files per species, CPU from audio, resumable CSV +
  small-multiples plot). Tested on 3 owl files with v2 (acc 0.99 / 0.99 / 0.94, voicing AUC 0.99 / 1.00 / 0.73 at -2 / 0 / +2). v3 (`split0_v3`)
  at epoch 4: val acc ~1 point below v2, val voicing 2-7 points below and noisier (watch; `--p_time 0.5` if it stays behind).
  User's decision: run the robustness analysis only at the end, and only if really needed.
- **Orangutans: planned to be dropped** (user's decision, 2026-11-08; still in `metadata.py` and in v2/v3 training until then). Spectrograms
  (`scores/orangutan_examples.png`, `scores/orangutan_worst.png`) show octave errors, spikes and missed calls in the annotations. Before dropping,
  justify it by quantifying annotation jumps (share of frame-to-frame jumps > e.g. 1 semitone / near an octave, per species, orangutans vs others).
- 2026-11-09 **split0_v3 results** (20 epochs, best = last, val acc 0.769 vs 0.778 for v2; `scores/split0_v3_best_summary.csv`): slightly worse
  than v2 on the standard test fold: mean pitch acc 0.740 vs 0.746, recall 0.892 vs 0.908, false alarm 0.119 vs 0.102, overall 0.761 vs 0.775.
  Worse on 10/13 species (hummingbirds -3.0, orangutans -1.6, bats -1.5), better on parakeets (+2.6). Voicing pays most (FA +3-4 points for
  white-eyes, hermits, hummingbirds, orangutans). Train acc on augmented windows 0.715 (v2 0.735). Whether it buys robustness off the usual
  grid position is not measured yet (gibbon file / robustness.py).
- 2026-11-09 gibbon test (`plots/gibbons_CAR07_20220530_070000/v2_vs_v3_{autofmin,fmin125,zoom}/`, script in the session scratchpad):
  at fmin 125 (original failure setting) v2 again tracks the ~1.2 kHz reverberation band between sweeps; v3 at fmin 125 looks like v3 at
  the automatic fmin (173): grid-position failure fixed. Both still track steep upsweeps only partially (sweep speed / reverberation).
  v3 has fewer false alarms on stationary noise bands (voiced 24% vs 44% of frames at auto fmin).
- 2026-11-09 (user's decision): `eval.py` grid+timing-compensated accuracy now uses 10 cents (`--comp_cent_thr`, column
  `Pitch acc (grid+timing-comp., 10 cents)`); `Pitch acc` and grid-comp. stay at 50 cents. Earlier grid+timing numbers (e.g. 0.868) were at 50 cents.
  Little owls v3: 0.979 at 50 cents -> 0.885 at 10 cents.
- Open items (2026-11-09):
  - Planned: orangutan jump quantification then drop (above); two-pass automatic fmin in `predict.py`; CQT-domain reverberation
    augmentation (gibbons follow reverberation tails instead of sweeps; WPE did not help).
  - Proposed, awaiting user: ablations of the 4 v2 changes and `--ssl_weight 0`; `--omit` experiments (vs tcrepe_ftoth); 5-fold CV.
  - At the end, only if needed: robustness curve (`robustness.py`).
  - Ideas kept in mind, not planned: noise-robust loss (generalised cross-entropy), soft bootstrapping targets, LoRA adapters, curriculum,
    differentiable ridge tracking, baselines; canid voicing (recall 0.79, FA 0.22); data for dolphins, rodents, bottlenose dolphins.
