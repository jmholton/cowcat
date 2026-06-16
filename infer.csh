#!/bin/tcsh -f
# infer.csh -- run the inference gamut on one checkpoint.
#
# Three tests, each run via `srun --partition=gpu --gres=gpu:1`:
#   1. v4 protein sample  -- infer.py + rfactor.py vs refmacout.mtz
#   2. b10 simple sample  -- infer.py only (no MTZ in simple pipeline)
#   3. real 1aho data     -- infer.py + rfactor.py vs refmacout_minRfree.mtz
#
# Output maps land in each sample dir as predicted_diff_<tag>.map
# (default tag = basename of the checkpoint's parent dir).
#
# Usage:
#   infer.csh checkpoint=<path/best.pt>
#   infer.csh checkpoint=<path> tag=<tag>
#   infer.csh checkpoint=<path> protein=<dir> simple=<dir> real=<dir>
#   infer.csh checkpoint=<path> base_features=64
#   infer.csh checkpoint=<path> skip=simple,real
#
# Defaults:
#   protein = data/data_protein_v4_s0/sample_00000
#   simple  = data/data_simple_b10/sample_00000
#   real    = 1aho_test
#   base_features = 32
#
# rfactor.py is given the *total* map predicted.map (pred + fc), which
# infer.py writes alongside the tagged diff. MTZ column labels:
#   protein  refmacout.mtz           -> F, FreeR_flag    (defaults)
#   real     refmacout_minRfree.mtz  -> FP, FreeR_flag

set checkpoint    = ""
set tag           = ""
set base_features = 32
set protein       = data/data_protein_v4_s0/sample_00000
set simple        = data/data_simple_b10/sample_00000
set real          = 1aho_test
set real_fo_label = FP
set real_mtz      = refmacout_minRfree.mtz
set skip          = ""
set crossp_raw      = 0    # 1 → pass --crossp-raw to infer.py (for *_rawcrossp-trained models)
set crossp_unitratio = 0   # 1 → pass --crossp-unitratio to infer.py (for *_unitratio-trained models)
set mobius           = 0   # 1 → pass --mobius to infer.py (for *_mobius-trained models)

# read command line: key=value pairs
foreach Arg ( $* )
    set assign = `echo $Arg | awk '{print ( /=/ )}'`
    set Key = `echo $Arg | awk -F "=" '{print $1}'`
    set Val = `echo $Arg | awk '{print substr($0,index($0,"=")+1)}'`
    if( $assign ) then
      set test = `set | awk -F "\t" '{print $1}' | egrep "^${Key}"'$' | wc -l`
      if ( $test ) then
          set $Key = "$Val"
          echo "$Key = $Val"
          continue
      endif
    endif
end

if ( "$checkpoint" == "" ) then
    echo "Usage: $0 checkpoint=<path/best.pt> [tag=<tag>] [protein=<dir>] [simple=<dir>] [real=<dir>] [base_features=N] [skip=simple,real]"
    exit 1
endif

if ( ! -e "$checkpoint" ) then
    echo "checkpoint not found: $checkpoint"
    exit 1
endif

if ( "$tag" == "" ) then
    set tag = `dirname $checkpoint | xargs basename`
endif

set PYTHON = /programs/pytorch/envs/pt/bin/python
set SRUN   = "srun --partition=gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=4"

set infer_extra = ""
if ( "$crossp_raw" == "1" )       set infer_extra = "$infer_extra --crossp-raw"
if ( "$crossp_unitratio" == "1" ) set infer_extra = "$infer_extra --crossp-unitratio"
if ( "$mobius" == "1" )           set infer_extra = "$infer_extra --mobius"

echo "================================================================"
echo "inference gamut"
echo "  checkpoint     = $checkpoint"
echo "  tag            = $tag"
echo "  base_features  = $base_features"
echo "  protein dir    = $protein"
echo "  simple  dir    = $simple"
echo "  real    dir    = $real"
echo "  skip           = $skip"
echo "================================================================"

# ── 1. v4 protein sample ─────────────────────────────────────────────
set do_protein = 1
if ( "$skip" =~ *protein* ) set do_protein = 0
if ( ! -d "$protein" )      set do_protein = 0

if ( $do_protein ) then
    echo ""
    echo "── [1/3] v4 protein sample: $protein ──────────────────────"
    set out_map = $protein/predicted_diff_${tag}.map
    $SRUN $PYTHON infer.py \
        --checkpoint $checkpoint \
        --base-features $base_features \
        --2fofc $protein/2fofc.map \
        --fofc  $protein/fofc.map \
        --fc    $protein/fc.map \
        --output $out_map $infer_extra
    if ( -e "$protein/refmacout.mtz" ) then
        echo ""
        echo "-- rfactor.py vs refmacout.mtz --"
        $SRUN $PYTHON rfactor.py \
            --mtz   $protein/refmacout.mtz \
            --fc    $protein/fc.map \
            --pred  $protein/predicted.map \
            --truth $protein/truth.map
    else
        echo "  (no refmacout.mtz; skipping rfactor)"
    endif
else
    echo "skipped: v4 protein"
endif

# ── 2. b10 simple sample ─────────────────────────────────────────────
set do_simple = 1
if ( "$skip" =~ *simple* ) set do_simple = 0
if ( ! -d "$simple" )      set do_simple = 0

if ( $do_simple ) then
    echo ""
    echo "── [2/3] b10 simple sample: $simple ───────────────────────"
    set out_map = $simple/predicted_diff_${tag}.map
    $SRUN $PYTHON infer.py \
        --checkpoint $checkpoint \
        --base-features $base_features \
        --2fofc $simple/2fofc.map \
        --fofc  $simple/fofc.map \
        --fc    $simple/fc.map \
        --output $out_map $infer_extra
    # simple pipeline has no MTZ; infer.py reports CC vs truth.map if present
else
    echo "skipped: b10 simple"
endif

# ── 3. real 1aho data ────────────────────────────────────────────────
set do_real = 1
if ( "$skip" =~ *real* ) set do_real = 0
if ( ! -d "$real" )      set do_real = 0

if ( $do_real ) then
    echo ""
    echo "── [3/3] real 1aho data: $real ────────────────────────────"
    set out_map = $real/predicted_diff_${tag}.map
    $SRUN $PYTHON infer.py \
        --checkpoint $checkpoint \
        --base-features $base_features \
        --2fofc $real/2fofc.map \
        --fofc  $real/fofc.map \
        --fc    $real/fc.map \
        --output $out_map $infer_extra
    if ( -e "$real/$real_mtz" ) then
        echo ""
        echo "-- rfactor.py vs $real_mtz (fo-label=$real_fo_label) --"
        $SRUN $PYTHON rfactor.py \
            --mtz       $real/$real_mtz \
            --fc        $real/fc.map \
            --pred      $real/predicted.map \
            --fo-label  $real_fo_label
    else
        echo "  (no $real_mtz; skipping rfactor)"
    endif
else
    echo "skipped: real 1aho"
endif

echo ""
echo "── done. predicted maps tagged '_${tag}' in each sample dir ────"
