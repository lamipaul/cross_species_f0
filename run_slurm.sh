#!/bin/bash
#SBATCH -J f0_train
#SBATCH -p g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_b200_1g.23gb:1
#SBATCH --exclude=cbe-g5-6
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Full pipeline on the cluster, trained on folds 1-4 and evaluated on fold 0. From the project folder, with the conda environment activated:
#   sbatch run_slurm.sh
#   sbatch --export=ALL,NAME=split0_unc run_slurm.sh     another run name (weights/{NAME}*.pth, runs/{NAME}), e.g. to run two trainings side by side
# Preprocessing skips files already cached (or uses the file lists of the cache when the audio is absent);
# resubmitting continues training from the last saved epoch (--resume).
set -e
PY=${PY:-${CONDA_PREFIX:?activate the conda environment before sbatch}/bin/python}  # full path: compute nodes put another python first on the PATH
S=0  # test fold
NAME=${NAME:-split$S}

$PY preprocess.py
$PY train.py --split $S --name $NAME --epochs 20 --in_memory --resume --num_workers 6
for CKPT in $NAME ${NAME}_best; do
    $PY eval.py weights/$CKPT.pth --split $S
done
