#!/bin/tcsh -f
# pack.csh -- pack sample_NNNNN dirs into X/Y/S npy arrays for training.
#
# Usage:    pack.csh <data_dir> [encoding]
#
# encoding (default: ssqrt):
#   ssqrt     -- signed-sqrt cross-Patterson  -> <data_dir>_ssqrt/
#   unitratio -- unit-ratio deconvolution     -> <data_dir>_unitratio/
#   raw       -- raw cross-Patterson          -> <data_dir>_rawcrossp/

if ($#argv < 1) then
    echo "Usage: $0 <data_dir> [ssqrt|unitratio|raw]"
    exit 1
endif

set data     = $1
set encoding = ssqrt
if ($#argv >= 2) set encoding = $2

set name = `basename $data`

set extra_flags = ""
if ( "$encoding" == "unitratio" ) then
    set outdir = ${data}_unitratio
    set extra_flags = "--crossp-unitratio"
else if ( "$encoding" == "raw" ) then
    set outdir = ${data}_rawcrossp
    set extra_flags = "--crossp-raw"
else
    set outdir = ${data}_ssqrt
endif

mkdir -p $outdir
sbatch --partition=debug --ntasks=1 --cpus-per-task=8 \
    --exclude=voltron,graphics2 \
    --job-name=pack_$name \
    --output=$outdir/pack_%j.log \
    --wrap="/programs/pytorch/envs/pt/bin/python pack.py --data $data --outdir $outdir $extra_flags --workers 8 --force"

