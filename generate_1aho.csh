#!/bin/tcsh -f
# generate_1aho.csh -- submit a 1AHO training-data SLURM array.
#
# Usage:    generate_1aho.csh outdir=<dir> [key=value ...]
#
# Defaults:
#   nsamples       = 1000
#   shift_scale    = 0.50    (B-scaled jiggle; 0 = no jiggle)
#   swaps_per_res  = 0.0     (pairwise conformer label swaps per residue)
#   n_flood        = 1764
#   flood_occ_lo   = -0.144  (occ_max = FLOOD_LINE_K*sqrt(3)/sqrt(n_flood) → ~11% Rfree)
#   flood_occ_hi   =  0.144
#   flood_b_lo     = 1.0     (min flood water B factor, Å²; log-uniform [lo,hi])
#   flood_b_hi     = 15.0    (max flood water B factor, Å²; covers gt48 atom B range)
#   flood_nf_min   = 700     (vary_flood/random_flood: min n_flood, log-uniform)
#   flood_nf_max   = 4000    (vary_flood/random_flood: max n_flood)
#   flood_peak_sigma = 3.0   (random_flood: peak target in units of gt48 map RMS)
#
# NOTE: use key=value format only (no --flags). Wrong: --flood-nf-range 10 2000
#       Correct: flood_nf_min=10 flood_nf_max=2000
#   flood_min_dist = 0.0     (0 = everywhere; old default was 2.0)
#   vary_flood     = 0       (1 → log-uniform N in [nf_min,nf_max], ±occ targeting ~11%)
#   ncyc           = 10
#   k_conformers   = 32
#   seed           = 0
#   partition      = debug
#   account        = ""      (e.g. pc_als831)
#   qos            = ""      (e.g. lr_normal)
#   max_array      = 300
#
# Examples:
#   # default: jiggle=0.5, ±flood, everywhere, 1000 samples
#   generate_1aho.csh outdir=data/data_1aho_s0
#
#   # zero jiggle, zero swaps, measure flood-water contribution only
#   generate_1aho.csh outdir=data/data_1aho_flood10 nsamples=10 shift_scale=0
#
#   # Lawrencium
#   generate_1aho.csh outdir=data/data_1aho_s0 partition=lr6 account=pc_als831 qos=lr_normal

set outdir        = ""
set nsamples      = 1000
set shift_scale   = 0.0
set swaps_per_res = 0.0
set n_flood       = 1764
set flood_occ_lo  = -0.144
set flood_occ_hi  =  0.144
set flood_b_lo    = 1.0
set flood_b_hi    = 15.0
set flood_nf_min     = 700
set flood_nf_max     = 4000
set flood_peak_sigma = 3.0
set flood_occ_max    = ""   # "" = no clip; set to e.g. 0.5 to limit high-B occ
set flood_min_dist = 0.0
set vary_flood         = 0
set random_flood       = 0
set flood_rfree_target = ""   # "" = use hardcoded FLOOD_LINE_K; set e.g. 0.11 to compute K analytically
set ncyc          = 10
set k_conformers  = 32
set seed          = 0
set partition     = refmac
set account       = ""
set qos           = ""
set max_array     = 300

# ── parse key=value args ──────────────────────────────────────────────────────
foreach Arg ( $* )
    set assign = `echo $Arg | awk '{print ( /=/ )}'`
    set Key = `echo $Arg | awk -F "=" '{print $1}'`
    set Val = `echo $Arg | awk '{print substr($0,index($0,"=")+1)}'`
    if ( $assign ) then
        set test = `set | awk -F "\t" '{print $1}' | egrep "^${Key}"'$' | wc -l`
        if ( $test ) then
            set $Key = "$Val"
            echo "$Key = $Val"
            continue
        endif
    endif
end

if ( "$outdir" == "" ) then
    echo "Usage: $0 outdir=<dir> [key=value ...]"
    echo "       See header for all options and defaults."
    exit 1
endif

# ── build flood args ──────────────────────────────────────────────────────────
set flood_args = "--n-flood $n_flood --flood-occ-range $flood_occ_lo $flood_occ_hi"
set flood_args = "$flood_args --flood-b-range $flood_b_lo $flood_b_hi"
set flood_args = "$flood_args --flood-min-dist $flood_min_dist"
if ( "$random_flood" == "1" ) then
    # N, occ amplitude, B all independent per sample — no Rfree targeting
    set flood_args = "--random-flood"
    set flood_args = "$flood_args --flood-nf-range $flood_nf_min $flood_nf_max"
    set flood_args = "$flood_args --flood-peak-sigma $flood_peak_sigma"
    if ( "$flood_occ_max" != "" ) set flood_args = "$flood_args --flood-occ-max $flood_occ_max"
    set flood_args = "$flood_args --flood-b-range $flood_b_lo $flood_b_hi"
else if ( "$vary_flood" == "1" ) then
    # N random, occ scaled to target Rfree (calibrated ft7-11, shift_scale=0, B[5,80])
    set flood_args = "--vary-flood --flood-nf-range $flood_nf_min $flood_nf_max"
    set flood_args = "$flood_args --flood-b-range $flood_b_lo $flood_b_hi"
    set flood_args = "$flood_args --flood-min-dist $flood_min_dist"
    if ( "$flood_rfree_target" != "" ) set flood_args = "$flood_args --flood-rfree-target $flood_rfree_target"
endif

# ── build account/qos args ────────────────────────────────────────────────────
set cluster_args = ""
if ( "$account" != "" ) set cluster_args = "$cluster_args --account $account"
if ( "$qos"     != "" ) set cluster_args = "$cluster_args --qos $qos"

echo "================================================================"
echo "generate_1aho.csh"
echo "  outdir        = $outdir"
echo "  nsamples      = $nsamples"
echo "  shift_scale   = $shift_scale"
echo "  swaps_per_res = $swaps_per_res"
echo "  flood         = $flood_args"
echo "  flood_b       = [$flood_b_lo, $flood_b_hi] Å² (log-uniform)"
echo "  ncyc          = $ncyc"
echo "  k_conformers  = $k_conformers"
echo "  seed          = $seed"
echo "  partition     = $partition"
echo "  cluster_args  = $cluster_args"
echo "================================================================"

ccp4-python generate_1aho.py --submit \
    --outdir $outdir \
    --nsamples $nsamples \
    --shift-scale $shift_scale \
    --swaps-per-residue $swaps_per_res \
    $flood_args \
    --ncyc $ncyc \
    --k-conformers $k_conformers \
    --seed $seed \
    --partition $partition \
    --max-array $max_array \
    $cluster_args

exit

 generate_1aho.csh \
      outdir=data/data_1aho_floodtest17 \
      nsamples=50 \
      shift_scale=0.0 \
      swaps_per_res=50 \
      vary_flood=1 \
      flood_b_lo=5 \
      flood_b_hi=80 \
      flood_nf_min=700 \
      flood_nf_max=4000 \
      flood_min_dist=0.5



 ./generate_1aho.csh \
      outdir=data/data_1aho_f0 \
      nsamples=1000 \
      shift_scale=0.0 \
      swaps_per_res=50 \
      k_conformers=8 \
      vary_flood=1 flood_rfree_target=0.08 \
      flood_b_lo=5 \
      flood_b_hi=80 \
      flood_nf_min=700 \
      flood_nf_max=4000 \
      flood_min_dist=1.0



