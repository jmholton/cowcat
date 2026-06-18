#!/usr/bin/env ccp4-python
"""Build a variable-conformer-count reference model.

Uses chain_id=altloc format (refmac-compatible for >20 confs).
Per-residue k from dev_to_nconf; chains ranked by global maximin.
Output: 1aho/varconf/starthere.pdb + refmacout.pdb
"""
import os, shutil, tempfile
from pathlib import Path

os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)

SCRIPT_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(SCRIPT_DIR))

from explore_1aho_fusion import (
    parse_conformers, build_varconf_pdb, build_fobs_mtz,
)
from explore_condensation import run_weightsnap

PDB   = SCRIPT_DIR / '1aho/gt48.pdb'
REFME = SCRIPT_DIR / '1aho/gt48.mtz'
OUTDIR = SCRIPT_DIR / '1aho/varconf'
OUTDIR.mkdir(parents=True, exist_ok=True)

print('Parsing conformers...')
st_orig, chain_names, conf_data = parse_conformers(str(PDB))
print(f'  {len(chain_names)} chains, {len(conf_data[chain_names[0]])} residues')

starthere = OUTDIR / 'starthere.pdb'
print('Building variable-conformer model...')
selected, res_k, per_res_sel = build_varconf_pdb(
    chain_names, conf_data, st_orig,
    out_pdb=starthere, workdir=OUTDIR,
)

print('Building fobs.mtz...')
with tempfile.TemporaryDirectory(prefix='varconf_fobs_') as td:
    fobs = build_fobs_mtz(str(PDB), str(REFME), Path(td))
    fobs_out = OUTDIR / 'fobs.mtz'
    shutil.copy2(fobs, fobs_out)
print(f'  → {fobs_out}')

print('Running weight-snap refinement...')
with tempfile.TemporaryDirectory(prefix='varconf_ref_') as td:
    r_i, rf_i, r_f, rf_f, elapsed, final_mtz, final_pdb, refmac_log = run_weightsnap(
        starthere, fobs_out, Path(td))
    if final_pdb and final_pdb.exists():
        shutil.copy2(final_pdb, OUTDIR / 'refmacout.pdb')
    if final_mtz and final_mtz.exists():
        shutil.copy2(final_mtz, OUTDIR / 'refmacout.mtz')
    (OUTDIR / 'refmac.log').write_text(refmac_log)

print(f'R_init={r_i:.4f} Rf_init={rf_i:.4f}  R_final={r_f:.4f} Rf_final={rf_f:.4f}  t={elapsed:.0f}s')
print(f'Output: {OUTDIR}')
