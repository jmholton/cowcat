#!/bin/tcsh -f
# pack.csh -- pack sample_NNNNN dirs into X/Y/S npy arrays for training,
#             with raw cross-Patterson at channel 3 (matches *_rawcrossp).
#
# Usage:    pack.csh <data_dir>
#
# Writes to <data_dir>_rawcrossp/  (X.npy, Y.npy, S.npy + log).

if ($#argv < 1) then
    echo "Usage: $0 <data_dir>"
    exit 1
endif

set data   = $1
set outdir = ${data}_rawcrossp
set name   = `basename $data`

mkdir -p $outdir
sbatch --partition=refmac --ntasks=1 --cpus-per-task=8 \
    --exclude=voltron,graphics2 \
    --job-name=pack_$name \
    --output=$outdir/pack_%j.log \
    --wrap="/programs/pytorch/envs/pt/bin/python pack.py --data $data --outdir $outdir --crossp-raw --workers 8 --force"
