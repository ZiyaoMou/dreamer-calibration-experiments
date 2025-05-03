#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=36:00:00
#SBATCH --job-name=dreamer-platt
#SBATCH --output=./slurm_cal_output/%j.out
#SBATCH --mail-user=zmou1@jh.edu
#SBATCH --mail-type=END,FAIL


nvidia-smi

python3 dreamerv2/calibrate.py --logdir log/cal_logdir_platt/atari_pong/dreamerv2/1 \
  --configs atari --task atari_pong --rssm_calibrate_mode platt
