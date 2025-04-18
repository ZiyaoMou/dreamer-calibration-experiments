#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=dreamer
#SBATCH --output=./slurm_output/%j.out

nvidia-smi

python3 dreamerv2/train.py --logdir ~/logdir/atari_pong/dreamerv2/1 \
  --configs atari --task atari_pong