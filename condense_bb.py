#!/usr/bin/env ccp4-python
"""
condense_bb.py — Backbone-only maximin condensation sweep on gt48.

Strips gt48.pdb to main-chain atoms (N/CA/C/O/OXT/H/HA*), computes Fc from
this backbone-only model as the data, then for each k in K_LEVELS:
  - maximin-selects k chains from the 48-conformer backbone model
  - runs the standard weight-snap refinement
Goal: find smallest k that achieves R ≤ 3% against backbone-only Fc data.

Usage:
  ccp4-python condense_bb.py --submit [--partition lr6 --account pc_als831 --qos lr_normal]
  ccp4-python condense_bb.py --collect
  ccp4-python condense_bb.py --max-k 8 --fobs-mtz outdir/fobs.mtz   # worker
"""

import json
import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import shutil
import subprocess
import tempfile
from pathlib import Path

import gemmi
import numpy as np

from explore_1aho_fusion import (
    DMIN, MAINCHAIN_ATOMS, run, parse_conformers, build_starthere_pdb,
)
import explore_condensation
from explore_condensation import run_weightsnap, K_LEVELS, collect

# Use refmac5-newhess for this sweep
explore_condensation.REFMAC5 = Path('/programs/ccp4-8.0/bin/refmac5-newhess')

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PDB    = Path('1aho/gt48.pdb')
DEFAULT_REFME  = Path('1aho/refme_minRfree.mtz')   # only used for FreeR_flag
DEFAULT_OUTDIR = Path('1aho/condense_bb')


# ── Backbone strip ────────────────────────────────────────────────────────────

def strip_to_backbone(in_pdb, out_pdb, keep_disulfides=True):
    """Write a PDB with only main-chain atoms (text-based filter on ATOM/HETATM lines).
    Drops waters (HOH/WAT/H2O) entirely. Preserves altlocs, chains, CRYST1, etc.
    If keep_disulfides=True, also keeps CB and SG atoms (and HB*) for CYS residues
    so disulfide bridges remain part of the rigid covalent network.
    """
    cys_extra = frozenset({'CB', 'SG', 'HB', 'HB1', 'HB2', 'HB3', 'HG'}) if keep_disulfides else frozenset()
    out_lines = []
    with open(in_pdb) as f:
        for line in f:
            if line.startswith('ATOM  ') or line.startswith('HETATM'):
                resname = line[17:20].strip()
                if resname in ('HOH', 'WAT', 'H2O'):
                    continue
                atom_name = line[12:16].strip()
                keep_set = MAINCHAIN_ATOMS
                if resname == 'CYS':
                    keep_set = MAINCHAIN_ATOMS | cys_extra
                if atom_name not in keep_set:
                    continue
            out_lines.append(line)
    with open(out_pdb, 'w') as f:
        f.writelines(out_lines)


# ── Fobs MTZ from backbone Fc only (no Fpart, no bulk solvent) ───────────────

def build_fobs_calc_only(pdb_path, refme_path, tmpdir):
    """Build fobs MTZ where FP=|Fc(pdb)|, Fpart=0; FreeR_flag from refme."""
    pdb_path   = Path(pdb_path).resolve()
    refme_path = Path(refme_path).resolve()
    tmpdir     = Path(tmpdir).resolve()
    tmpdir.mkdir(parents=True, exist_ok=True)
    fc_mtz = tmpdir / '_fc.mtz'
    run(['gemmi', 'sfcalc', f'--dmin={DMIN}', f'--to-mtz={fc_mtz}', str(pdb_path)],
        cwd=tmpdir)
    fc    = gemmi.read_mtz_file(str(fc_mtz))
    refme = gemmi.read_mtz_file(str(refme_path))

    h  = np.array(fc.column_with_label('H'),   dtype=np.int32)
    k  = np.array(fc.column_with_label('K'),   dtype=np.int32)
    l  = np.array(fc.column_with_label('L'),   dtype=np.int32)
    fp = np.array(fc.column_with_label('FC'),  dtype=np.float32)

    h_r  = np.array(refme.column_with_label('H'),          dtype=np.int32)
    k_r  = np.array(refme.column_with_label('K'),          dtype=np.int32)
    l_r  = np.array(refme.column_with_label('L'),          dtype=np.int32)
    fr_r = np.array(refme.column_with_label('FreeR_flag'), dtype=np.float32)
    fr_dict = {(int(h_r[i]), int(k_r[i]), int(l_r[i])): fr_r[i]
               for i in range(len(h_r))}

    sigfp   = np.maximum(0.01, 0.02 * fp).astype(np.float32)
    fr      = np.array([fr_dict.get((int(h[i]), int(k[i]), int(l[i])), 0)
                        for i in range(len(h))], dtype=np.float32)
    fpart   = np.zeros(len(h), dtype=np.float32)
    phipart = np.zeros(len(h), dtype=np.float32)

    out = gemmi.Mtz()
    out.cell       = fc.cell
    out.spacegroup = fc.spacegroup
    out.add_dataset('HKL_base')
    for lbl in ('H', 'K', 'L'):
        out.add_column(lbl, 'H')
    out.add_dataset('data')
    out.add_column('FP',         'F')
    out.add_column('SIGFP',      'Q')
    out.add_column('FreeR_flag', 'I')
    out.add_column('Fpart',      'F')
    out.add_column('PHIpart',    'P')
    data = np.column_stack([h, k, l, fp, sigfp, fr, fpart, phipart])
    out.set_data(data.astype(np.float32))

    out_mtz = tmpdir / 'fobs.mtz'
    out.write_to_file(str(out_mtz))
    n_free = int((fr == 0).sum())
    print(f'  Fobs MTZ: {len(h)} reflections (work={len(h)-n_free}, free={n_free}), '
          f'dmin={DMIN} Å, mean FP={fp.mean():.1f}')
    return out_mtz


# ── Worker ────────────────────────────────────────────────────────────────────

def run_one_k(max_k, bb_pdb, fobs_mtz, outdir):
    import time
    os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)

    print(f'[k={max_k}] Parsing conformers...')
    st_orig, chain_names, conf_data = parse_conformers(bb_pdb)

    k_dir = outdir / f'k{max_k}'
    k_dir.mkdir(parents=True, exist_ok=True)
    starthere_pdb = k_dir / 'starthere.pdb'

    print(f'[k={max_k}] Building reduced PDB ({len(chain_names)} chains → {max_k})...')
    n_alt = build_starthere_pdb(chain_names, conf_data, st_orig, max_k,
                                ref_pdb=bb_pdb, out_pdb=starthere_pdb,
                                workdir=k_dir)
    print(f'[k={max_k}] Running weight-snap refinement...')
    r_i, rf_i, r_f, rf_f, elapsed, final_mtz, final_pdb, refmac_log = run_weightsnap(
        starthere_pdb, fobs_mtz, k_dir)
    if final_mtz and final_mtz.exists():
        final_mtz.rename(k_dir / 'refmacout.mtz')
    if final_pdb and final_pdb.exists():
        final_pdb.rename(k_dir / 'refmacout.pdb')
    (k_dir / 'refmac.log').write_text(refmac_log)

    result = dict(max_k=max_k, n_alt=n_alt, elapsed=elapsed,
                  r_init=r_i, rf_init=rf_i, r_final=r_f, rf_final=rf_f)
    (k_dir / 'result.json').write_text(json.dumps(result))
    print(f'[k={max_k}] Done. R_init={r_i:.4f} Rf_init={rf_i:.4f} '
          f'R_final={r_f:.4f} Rf_final={rf_f:.4f} t={elapsed:.1f}s')


# ── Submit ────────────────────────────────────────────────────────────────────

def submit(gt48_pdb, refme_path, outdir, partition, account=None, qos=None):
    os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)
    bb_pdb = outdir / 'gt48_bb.pdb'
    if not bb_pdb.exists():
        print(f'Stripping {gt48_pdb} to backbone → {bb_pdb}')
        strip_to_backbone(gt48_pdb, bb_pdb)
    else:
        print(f'Backbone PDB already exists: {bb_pdb}')

    fobs_mtz = outdir / 'fobs.mtz'
    if not fobs_mtz.exists():
        print('Building Fobs MTZ from backbone Fc...')
        fobs_tmp = outdir / '_fobs_tmp'
        fobs_tmp.mkdir(exist_ok=True)
        built = build_fobs_calc_only(bb_pdb, refme_path, fobs_tmp)
        shutil.copy2(built, fobs_mtz)
        print(f'  saved → {fobs_mtz}')
    else:
        print(f'Fobs MTZ already exists: {fobs_mtz}')

    me = Path(__file__).resolve()
    for max_k in K_LEVELS:
        script = SCRIPT_DIR / f'_cond_bb_k{max_k}.sh'
        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name=cond_bb_k{max_k}',
            '#SBATCH --ntasks=1',
            '#SBATCH --cpus-per-task=1',
            '#SBATCH --mem=8G',
            f'#SBATCH --partition={partition}',
        ]
        if account:
            lines.append(f'#SBATCH --account={account}')
        if qos:
            lines.append(f'#SBATCH --qos={qos}')
        lines += [
            '#SBATCH --export=ALL',
            '',
            'mkdir -p "${CCP4_SCR:-/tmp}"',
            f'cd {SCRIPT_DIR}',
            f'{sys.executable} {me} \\',
            f'  --max-k {max_k} \\',
            f'  --bb-pdb {bb_pdb} \\',
            f'  --fobs-mtz {fobs_mtz} \\',
            f'  --outdir {outdir}',
            '',
        ]
        script.write_text('\n'.join(lines))
        script.chmod(0o755)
        r = subprocess.run(['sbatch', str(script)], capture_output=True, text=True)
        print(f'  k={max_k}: {r.stdout.strip() or r.stderr.strip()}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt48-pdb', default=str(DEFAULT_PDB),
                    help='Source 48-conformer PDB (default: 1aho/gt48.pdb)')
    ap.add_argument('--refme',    default=str(DEFAULT_REFME),
                    help='MTZ providing FreeR_flag (default: 1aho/refme_minRfree.mtz)')
    ap.add_argument('--outdir',   default=str(DEFAULT_OUTDIR))
    ap.add_argument('--partition', default='lr6')
    ap.add_argument('--account')
    ap.add_argument('--qos')
    # Worker args
    ap.add_argument('--max-k',    type=int, help='Run single k level (worker mode)')
    ap.add_argument('--bb-pdb',   help='Pre-built backbone PDB (worker mode)')
    ap.add_argument('--fobs-mtz', help='Pre-built fobs MTZ (worker mode)')
    # Modes
    ap.add_argument('--submit',  action='store_true', help='Submit one job per k')
    ap.add_argument('--collect', action='store_true', help='Print results table')
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.submit:
        submit(Path(args.gt48_pdb).resolve(), Path(args.refme).resolve(), outdir,
               partition=args.partition, account=args.account, qos=args.qos)

    elif args.max_k is not None:
        bb_pdb   = Path(args.bb_pdb).resolve()
        fobs_mtz = Path(args.fobs_mtz).resolve()
        run_one_k(args.max_k, bb_pdb, fobs_mtz, outdir)

    elif args.collect:
        collect(outdir)

    else:
        # Sequential fallback (testing)
        bb_pdb = outdir / 'gt48_bb.pdb'
        if not bb_pdb.exists():
            strip_to_backbone(Path(args.gt48_pdb).resolve(), bb_pdb)
        fobs_mtz = outdir / 'fobs.mtz'
        if not fobs_mtz.exists():
            with tempfile.TemporaryDirectory(prefix='cond_bb_fobs_') as _td:
                built = build_fobs_calc_only(bb_pdb, Path(args.refme).resolve(), Path(_td))
                shutil.copy2(built, fobs_mtz)
        for max_k in K_LEVELS:
            run_one_k(max_k, bb_pdb, fobs_mtz, outdir)
        collect(outdir)


if __name__ == '__main__':
    main()
