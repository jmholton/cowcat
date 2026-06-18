#!/usr/bin/env ccp4-python
"""Build the pre-fix varconf model: O atoms NOT limited by next-residue count.

This recreates the model where ARG18 O gets stretched C=O bonds after refmac.
At k=16 (or whatever max_k you choose), chains where res_k[r+1] < res_k[r] have
no omega-plane restraint on O(r) → 1/occ gradient overshoot blows the C=O bond.

Output: 1aho/test_broken_o/k{max_k}/starthere.pdb
Run refmac on it and inspect ARG18 O in the output PDB.
"""
import argparse, os, sys
from pathlib import Path

os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from explore_1aho_fusion import parse_conformers, build_varconf_pdb

PDB   = SCRIPT_DIR / '1aho/gt48.pdb'
WATER = SCRIPT_DIR / '1aho/gt48_water.pdb'

ap = argparse.ArgumentParser()
ap.add_argument('--max-k', type=int, default=16)
args = ap.parse_args()

outdir = SCRIPT_DIR / f'1aho/test_broken_o/k{args.max_k}'
outdir.mkdir(parents=True, exist_ok=True)
out_pdb = outdir / 'starthere.pdb'

print(f'Parsing conformers from {PDB}...')
st_orig, chain_names, conf_data = parse_conformers(str(PDB))

print(f'Building broken model (limit_o=False, max_k={args.max_k})...')
slot_names, res_k, per_res_sel = build_varconf_pdb(
    chain_names, conf_data, st_orig,
    out_pdb=out_pdb, workdir=outdir,
    max_k=args.max_k,
    water_pdb=str(WATER) if WATER.exists() else None,
    limit_o=False,
)

# Show how many chains ARG18 O appears in vs how many it *should*
ref_chain_data = conf_data[chain_names[0]]
residue_keys = list(ref_chain_data.keys())
for i, rk in enumerate(residue_keys):
    resname = ref_chain_data[rk]['resname']
    k_r = res_k[rk]
    k_next = res_k[residue_keys[i+1]] if i+1 < len(residue_keys) else k_r
    if k_r > k_next:
        print(f'  {resname}{rk[1] if isinstance(rk,tuple) else rk}: '
              f'O in {k_r} chains (BROKEN: should be {k_next} — no omega restraint for chains {k_next+1}–{k_r})')

print(f'\nOutput: {out_pdb}')
print(f'Run refmac with 1aho/refme.mtz and check C-O bond length in ARG18.')
