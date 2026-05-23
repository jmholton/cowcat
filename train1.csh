#!/bin/tcsh -f
# train.csh -- submit a 1-GPU DDP training run.
#
# Usage:    train.csh <outdir> <data_dir1> [<data_dir2> ...]
#
# Defaults: 200 epochs, lr 1e-3, alpha 0.5, base-features 32
#           accum-steps 1 (eff_batch = 1 x 1 GPUs x 1 = 1)
#           Rfree-vs-real-1aho diagnostic enabled (1aho_test/)
#
# Env-var overrides:
#   TRAIN_EPOCHS         (default 200)
#   TRAIN_LR             (default 1e-3)
#   TRAIN_BASE_FEATURES  (default 32)
#   TRAIN_ACCUM_STEPS    (default 1)
#   TRAIN_ALPHA          (default 0.5)
#   TRAIN_PRETRAIN       path to a checkpoint to warm-start from
#   TRAIN_RESUME         path to a checkpoint to resume training from
#   TRAIN_EVAL_1AHO      (default 1aho_test;   set to "" to disable)


set outdir = ""
set data   = ""

set nGPUs         = 1
set epochs        = 200
set lr            = 1e-3
set base_features = 32
set accum_steps   = 1
set alpha         = 0.5
set eval_1aho     = 1aho_test
set train_pretrain = ""
set train_resume = ""

# read the command line to update variables and other settings
foreach Arg ( $* )
    set arg = `echo $Arg | awk '{print tolower($0)}'`
    set assign = `echo $arg | awk '{print ( /=/ )}'`
    set Key = `echo $Arg | awk -F "=" '{print $1}'`
    set Val = `echo $Arg | awk '{print substr($0,index($0,"=")+1)}'`
    set Csv = `echo $Val | awk 'BEGIN{RS=","} {print}'`
    set key = `echo $Key | awk '{print tolower($1)}'`
    set num = `echo $Val | awk '{print $1+0}'`
    set int = `echo $Val | awk '{print int($1+0)}'`

    if( $assign ) then
      # re-set any existing variables
      set test = `set | awk -F "\t" '{print $1}' | egrep "^${Key}"'$' | wc -l`
      if ( $test ) then
          set $Key = $Val
          echo "$Key = $Val"
          continue
      endif
      # synonyms
    else
      # no equal sign
      if( -d "$Arg" && -e "${Arg}/X.npy" ) then
        set data = ( $data $Arg )
        continue
      endif
    endif
    if("$key" == "debug") set debug = "1"
end

if ( "$outdir" == "" ) then
    echo "Usage: $0 outdir=<outdir> <data_dir1> [<data_dir2> ...]"
    exit 1
endif

set name   = `basename $outdir`

set extra = ""
if ("$train_pretrain")  set extra = "$extra --pretrain $train_pretrain"
if ("$train_resume" != "")    set extra = "$extra --resume   $train_resume"
if ("$eval_1aho" != "") set extra = "$extra --eval-1aho-dir $eval_1aho"

sbatch --partition=gpu --gres=gpu:$nGPUs --ntasks=1 --cpus-per-task=8 \
    --job-name=train_$name \
    --output=slurm-%j.out \
    --wrap="/programs/pytorch/envs/pt/bin/torchrun --nproc_per_node=$nGPUs train.py --data $data --outdir $outdir --epochs $epochs --batch-size 1 --lr $lr --alpha $alpha --base-features $base_features --accum-steps $accum_steps $extra"


