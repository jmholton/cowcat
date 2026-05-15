#!/bin/bash
#SBATCH --job-name=untangler_1aho
#SBATCH --partition=xds
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --output=untangler_1aho_%j.out
#SBATCH --error=untangler_1aho_%j.err
#SBATCH --export=ALL

cd "$SLURM_SUBMIT_DIR"

/programs/pytorch/envs/pt/bin/python run_untangler_1aho.py "$@"
