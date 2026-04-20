#!/bin/bash
#SBATCH --job-name=train_diff_altconf5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs_train_diff_altconf5.txt
#SBATCH --export=ALL

/programs/pytorch/envs/pt/bin/python train.py \
    --data data_n10_N10altconf5_refmac_n1000 \
    --outdir checkpoints_diff_altconf5 \
    --epochs 200
