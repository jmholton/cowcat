#!/bin/bash
#SBATCH --job-name=train_nf_warmstart
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs_train_diff_noflood_warmstart.txt
#SBATCH --export=ALL

/programs/pytorch/envs/pt/bin/python train.py \
    --data data_protein_200res_noflood_5alt_n1000 \
    --pretrain checkpoints_diff_protein_noflood_scratch/best.pt \
    --outdir checkpoints_diff_protein_noflood_warmstart \
    --epochs 200
