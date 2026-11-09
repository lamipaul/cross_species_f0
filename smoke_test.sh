#!/bin/bash
# End-to-end test of the pipeline on synthetic data with a tiny model (CPU-friendly, a few minutes), including resuming a run
set -e
cd "$(dirname "$0")"
TINY="--n_chan_layers 16 16 8 2 --n_prefilt_layers 2 --d_model 32 --n_layers 1 --n_heads 2 --conv_channels 8"
python make_fake_data.py --n_files 10 --n_unlabelled 3 --max_duration 1.0
python preprocess.py --fake
python train.py --fake --split 0 --epochs 2 --batch_size 16 --lr 1e-3 --num_workers 0 --name smoke $TINY
python train.py --fake --split 0 --epochs 3 --batch_size 16 --lr 1e-3 --num_workers 0 --name smoke $TINY --resume
for ckpt in smoke smoke_best; do python eval.py weights/$ckpt.pth --fake --split 0; done
python predict.py data/fake/fake_bird --specie fake_bird --model_path weights/smoke.pth
echo "Smoke test passed"
