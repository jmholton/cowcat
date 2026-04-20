#!/bin/bash
#SBATCH --job-name=train_pretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs_train_pretrain.txt
#SBATCH --export=ALL

/programs/pytorch/envs/pt/bin/python train.py --data data_protein_200res_3000flood_occ25_5alt_n1000 --pretrain checkpoints_n10_N1del_hydr/best.pt --outdir checkpoints_protein_occ25_N1del_pretrain --epochs 200
