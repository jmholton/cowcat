#!/bin/tcsh -f
# generate.csh -- submit a v4 protein-altloc training-data SLURM array.
#
# Usage:    generate.csh <outdir> [nsamples]
#
# Defaults: 1000 samples
#           P 21 21 21,  cell 45.9 40.7 30.1,  dmin 0.965
#           64 residues, 30 waters
#           20 altlocs, 5000 flood waters @ occ 0.08, 5 altloc swaps/res
#           per-conf phenix.geommin (parallel)
#           refmac partition, exclude GPU nodes voltron + graphics2
#
# Edit defaults below if you want different cell / SG / sample params.

if ($#argv < 1) then
    echo "Usage: $0 <outdir> [nsamples]"
    exit 1
endif

set outdir   = $1
set nsamples = 1000
if ($#argv >= 2) set nsamples = $2

ccp4-python generate_protein.py --submit --nsamples $nsamples \
    --outdir $outdir \
    --cell 45.9 40.7 30.1 --dmin 0.965 --spacegroup "P 21 21 21" \
    --nresidues 64 --nwaters 30 \
    --n-altlocs 20 --n-flood 5000 --flood-occ 0.08 \
    --altloc-swaps-per-res 5 \
    --per-conf-geommin --exclude-nodes voltron,graphics2 \
    --partition refmac --max-array 20 --time 00:15:00
