#!/bin/bash
#SBATCH --job-name=train_diff_r2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs_train_diff_r2.txt
#SBATCH --export=ALL

/programs/pytorch/envs/pt/bin/python train.py \
    --data data_protein_200res_3000flood_occ25_5alt_n1000 \
    --outdir checkpoints_protein_occ25_diff_r2 \
    --epochs 200
