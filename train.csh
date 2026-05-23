#!/bin/tcsh -f
# train.csh -- submit a 4-GPU DDP training run.
#
# Usage:    train.csh <outdir> <data_dir1> [<data_dir2> ...]
#
# Defaults: 200 epochs, lr 1e-3, alpha 0.5, base-features 32
#           accum-steps 4 (eff_batch = 1 x 4 GPUs x 4 = 16)
#           Rfree-vs-real-1aho diagnostic enabled (1aho_test/)
#
# Env-var overrides:
#   TRAIN_EPOCHS         (default 200)
#   TRAIN_LR             (default 1e-3)
#   TRAIN_BASE_FEATURES  (default 32)
#   TRAIN_ACCUM_STEPS    (default 4)
#   TRAIN_ALPHA          (default 0.5)
#   TRAIN_PRETRAIN       path to a checkpoint to warm-start from
#   TRAIN_RESUME         path to a checkpoint to resume training from
#   TRAIN_EVAL_1AHO      (default 1aho_test;   set to "" to disable)

if ($#argv < 2) then
    echo "Usage: $0 <outdir> <data_dir1> [<data_dir2> ...]"
    exit 1
endif

set outdir = $1
shift
set data   = "$argv"
set name   = `basename $outdir`

set epochs        = 200
set lr            = 1e-3
set base_features = 32
set accum_steps   = 4
set alpha         = 0.5
set eval_1aho     = 1aho_test
if ($?TRAIN_EPOCHS)         set epochs        = $TRAIN_EPOCHS
if ($?TRAIN_LR)             set lr            = $TRAIN_LR
if ($?TRAIN_BASE_FEATURES)  set base_features = $TRAIN_BASE_FEATURES
if ($?TRAIN_ACCUM_STEPS)    set accum_steps   = $TRAIN_ACCUM_STEPS
if ($?TRAIN_ALPHA)          set alpha         = $TRAIN_ALPHA
if ($?TRAIN_EVAL_1AHO)      set eval_1aho     = $TRAIN_EVAL_1AHO

set extra = ""
if ($?TRAIN_PRETRAIN)  set extra = "$extra --pretrain $TRAIN_PRETRAIN"
if ($?TRAIN_RESUME)    set extra = "$extra --resume   $TRAIN_RESUME"
if ("$eval_1aho" != "") set extra = "$extra --eval-1aho-dir $eval_1aho"

sbatch --partition=gpu --gres=gpu:4 --ntasks=1 --cpus-per-task=8 \
    --job-name=train_$name \
    --output=slurm-%j.out \
    --wrap="/programs/pytorch/envs/pt/bin/torchrun --nproc_per_node=4 train.py --data $data --outdir $outdir --epochs $epochs --batch-size 1 --lr $lr --alpha $alpha --base-features $base_features --accum-steps $accum_steps $extra"
