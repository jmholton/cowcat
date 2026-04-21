#!/bin/bash
#SBATCH --job-name=train_diff_n1000ac
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs_train_diff_n1000allclust.txt
#SBATCH --export=ALL

/programs/pytorch/envs/pt/bin/python train.py \
    --data data_n1000_allclust_altconf5_n1000 \
    --outdir checkpoints_diff_n1000allclust \
    --epochs 200
