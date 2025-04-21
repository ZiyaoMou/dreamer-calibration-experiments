#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=dreamer
#SBATCH --output=./slurm_output/%j.out
#SBATCH --mail-user=zmou1@jh.edu
#SBATCH --mail-type=END,FAIL


nvidia-smi

python3 dreamerv2/train.py --logdir log/logdir/atari_pong/dreamerv2/1 \
  --configs atari --task atari_pong

python3 dreamerv2/collect_calibration.py --configs atari --task atari_pong