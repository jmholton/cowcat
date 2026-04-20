#!/bin/bash
#SBATCH --job-name=train_diff_n1del
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs_train_diff_n1del.txt
#SBATCH --export=ALL

/programs/pytorch/envs/pt/bin/python train.py \
    --data data_n10_N1del_hydr_n1000 \
    --outdir checkpoints_diff_n1del \
    --epochs 200
