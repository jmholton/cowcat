#!/bin/tcsh -f
# pack.csh -- pack sample_NNNNN dirs into X/Y/S npy arrays for training,
#             with signed-sqrt of cross-Patterson at channel 3 (matches *_ssqrt).
#
# Usage:    pack.csh <data_dir>
#
# Writes to <data_dir>_ssqrt/  (X.npy, Y.npy, S.npy + log).

if ($#argv < 1) then
    echo "Usage: $0 <data_dir>"
    exit 1
endif

set data   = $1
set outdir = ${data}_ssqrt
set name   = `basename $data`

mkdir -p $outdir
sbatch --partition=debug --ntasks=1 --cpus-per-task=8 \
    --exclude=voltron,graphics2 \
    --job-name=pack_$name \
    --output=$outdir/pack_%j.log \
    --wrap="/programs/pytorch/envs/pt/bin/python pack.py --data $data --outdir $outdir --workers 8 --force"

