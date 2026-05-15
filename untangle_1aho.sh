#!/bin/tcsh -f
# Wrapper to run Untangler on 1AHO varconf structure.
# Requires CCP4 env (for converge_refmac.com on PATH) and pytorch python.
#
# Usage: ./untangle_1aho.sh [--pdb 1aho/varconf_opt6.pdb] [--mtz 1aho/refme_minRfree.mtz]
#
# On original cluster:
#   source /global/home/groups-sw/ac_als831/ccp4-X/bin/ccp4.setup-sh
#   ./untangle_1aho.sh

set SCRIPT_DIR = `dirname $0`
set PYTORCH_PYTHON = /programs/pytorch/envs/pt/bin/python

$PYTORCH_PYTHON $SCRIPT_DIR/run_untangler_1aho.py $argv
