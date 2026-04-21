#!/bin/bash
#SBATCH --job-name=prot_data
#SBATCH --partition=debug
#SBATCH --array=0-99999%50
#SBATCH --output=/home/jamesh/projects/squish_solvent/claude_CNN/data_protein_200res_noflood_5alt_n1000/logs/%A_%a.log
#SBATCH --error=/home/jamesh/projects/squish_solvent/claude_CNN/data_protein_200res_noflood_5alt_n1000/logs/%A_%a.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=5

mkdir -p /home/jamesh/projects/squish_solvent/claude_CNN/data_protein_200res_noflood_5alt_n1000/logs
/programs/ccp4-8.0/libexec/python3.7 /home/jamesh/projects/squish_solvent/claude_CNN/generate_protein.py \
    --sample-id $SLURM_ARRAY_TASK_ID \
    --outdir /home/jamesh/projects/squish_solvent/claude_CNN/data_protein_200res_noflood_5alt_n1000 \
    --nresidues 200 \
    --nwaters 200 \
    --n-flood 0 \
    --flood-avoid-fullocc \
    --shift-scale 0.5 \
    --n-altlocs 5 \
    --missing-fraction 0.05 \
    --never-collected-fraction 0.05 \
    --cell 40.0 40.0 40.0 \
    --dmin 2.0 \
