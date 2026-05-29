#!/bin/bash
#SBATCH --job-name=cowcat_train
#SBATCH --partition=es1
#SBATCH --account=pc_als831
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A40:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-train-%j.out

cd "$SLURM_SUBMIT_DIR"
source cluster.sh
setup_pytorch
PORT=$(( 29500 + SLURM_JOB_ID % 1000 ))
python3 train.py "$@"
