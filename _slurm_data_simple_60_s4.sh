#!/bin/bash
#SBATCH --job-name=gen_simple60_s4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=debug
#SBATCH --time=08:00:00
#SBATCH --export=ALL

cd /home/jamesh/projects/squish_solvent/claude_CNN

ccp4-python generate_simple.py \
    --nsamples 1000 \
    --outdir data_n10_N1del_hydr_n1000_s4 \
    --natoms 10 --modified 1 \
    --cell 40.0 40.0 40.0 --dmin 2.0 --spacegroup "P 1" \
    --submit --partition debug \
    2>&1 | tee logs_gen_simple_60_s4.txt
