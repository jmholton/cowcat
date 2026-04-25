#!/usr/bin/env ccp4-python
"""
explore_condensation.py — Sweep max_k conformer count vs refmac R/time.

Runs a weight-snap refinement schedule for each k in K_LEVELS:
  NCYC 10 @ wm=10  →  NCYC 10 @ wm=0.01  →  NCYC 50 @ wm=0.5

Usage:
  # Build fobs.mtz once, then submit one SLURM job per k:
  ccp4-python explore_condensation.py --submit [--partition debug]

  # Collect results after all jobs finish:
  ccp4-python explore_condensation.py --collect

  # Single k level (called by SLURM workers):
  ccp4-python explore_condensation.py --max-k 4 --fobs-mtz outdir/fobs.mtz
"""

import json
import os
import sys
sys.stdout.reconfigure(line_buffering=True)
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from explore_1aho_fusion import (
    REFMAC5, run,
    parse_conformers, build_fobs_mtz,
    generate_occ_groups, parse_rfactors, load_density_map,
    _maximin_select,
)

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_PDB = Path('1aho/refmacout_minRfree.pdb')
DEFAULT_MTZ = Path('1aho/refme_minRfree.mtz')

K_LEVELS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]


COMBINE_PDBS = SCRIPT_DIR / 'combine_pdbs_runme.com'


# ── Chain-level maximin selection ─────────────────────────────────────────────

def select_chains_maximin(chain_names, conf_data, k):
    """Select k representative chains by maximin over CA centroid positions.

    Returns (selected_chain_names, voronoi_occs).
    Each chain has equal prior weight (1/n_chains); representative occ = sum
    of assigned chains' weights.
    """
    n = len(chain_names)
    k = min(k, n)
    centroids = np.zeros((n, 3))
    for i, cn in enumerate(chain_names):
        pts = []
        for reskey, rd in conf_data[cn].items():
            ca = rd['atoms'].get('CA')
            if ca:
                pts.append([ca.pos.x, ca.pos.y, ca.pos.z])
        if pts:
            centroids[i] = np.mean(pts, axis=0)

    selected_idx = _maximin_select(centroids, k)
    sel_pts = centroids[selected_idx]
    dists = np.linalg.norm(centroids[:, None, :] - sel_pts[None, :, :], axis=2)
    assign = np.argmin(dists, axis=1)
    w = 1.0 / n
    voronoi_occs = [float((assign == gi).sum()) * w for gi in range(k)]
    selected = [chain_names[i] for i in selected_idx]
    return selected, voronoi_occs


def _write_chain_pdb(chain_name, occ, conf_data, ref_chain_data, cell, spacegroup_hm, outpath):
    """Write one conformer chain as a PDB with chain_id = altloc = chain_name."""
    lines = [f"CRYST1{cell.a:9.3f}{cell.b:9.3f}{cell.c:9.3f}"
             f"{cell.alpha:7.2f}{cell.beta:7.2f}{cell.gamma:7.2f}"
             f" {spacegroup_hm:<11s}1\n"]
    serial = 1
    rd_this = conf_data[chain_name]
    for reskey, ref_rd in ref_chain_data.items():
        if reskey not in rd_this:
            continue
        rd = rd_this[reskey]
        resname = rd['resname']
        is_het = resname in ('HOH', 'WAT', 'H2O')
        rec = 'HETATM' if is_het else 'ATOM  '
        seqnum = ref_rd['seqid'].num
        icode  = ref_rd['seqid'].icode if ref_rd['seqid'].icode != ' ' else ' '
        for aname, ref_atom in ref_rd['atoms'].items():
            atom = rd['atoms'].get(aname)
            if atom is None:
                atom = ref_atom
            elem = atom.element.name.upper()
            name4 = f' {aname:<3s}' if len(elem) == 1 and len(aname) < 4 else f'{aname:<4s}'
            lines.append(
                f'{rec}{serial:5d} {name4}{chain_name}'
                f'{resname:<3s} {chain_name:1s}'
                f'{seqnum:4d}{icode:1s}   '
                f'{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}'
                f'{occ:6.2f}{atom.b_iso:6.2f}'
                f'          {elem:>2s}\n'
            )
            serial += 1
    lines.append('END\n')
    Path(outpath).write_text(''.join(lines))


def build_starthere_pdb(chain_names, conf_data, st_orig, k, ref_pdb, out_pdb, workdir):
    """Select k chains by maximin, write individual PDBs, combine via combine_pdbs_runme.com."""
    selected, occs = select_chains_maximin(chain_names, conf_data, k)
    print(f'    selected chains: {selected}')
    ref_chain_data = conf_data[chain_names[0]]

    chain_pdbs = []
    for cn, occ in zip(selected, occs):
        p = workdir / f'_chain_{cn}.pdb'
        _write_chain_pdb(cn, occ, conf_data, ref_chain_data,
                         st_orig.cell, st_orig.spacegroup_hm, p)
        chain_pdbs.append(str(p))

    result = subprocess.run(
        ['tcsh', str(COMBINE_PDBS)] + chain_pdbs +
        [f'refpdb={ref_pdb}', f'outfile={out_pdb}'],
        capture_output=True, text=True, cwd=workdir,
    )
    if result.returncode != 0 or not Path(out_pdb).exists():
        raise RuntimeError(f'combine_pdbs_runme.com failed:\n{result.stdout}\n{result.stderr}')
    print(f'    k={k}  n_chains={len(selected)}  occs={[f"{o:.3f}" for o in occs]}')
    return len(selected)


# ── Refmac helpers ────────────────────────────────────────────────────────────

def _run_one(xyzin, xyzout, hklout, fobs_mtz, ncyc, weight_matrix, tmpdir):
    """Single refmac run. Returns (R, Rfree, log_text)."""
    occ_kw = generate_occ_groups(xyzin)
    kw  = b'LABIN FP=FP SIGFP=SIGFP FPART1=Fpart PHIP1=PHIpart FREE=FreeR_flag\n'
    kw += b'LABOUT FC=FC PHIC=PHIC FWT=FWT PHWT=PHWT DELFWT=DELFWT PHDELWT=PHDELWT\n'
    kw += b'solvent no\nscpart 1\ndamp 0.5 0.5\nmake hout Y\nmake hydr Y\n'
    kw += f'weight matrix {weight_matrix}\nNCYC {ncyc}\n'.encode()
    kw += occ_kw
    kw += b'END\n'
    log = run(
        [REFMAC5,
         'XYZIN',  xyzin,  'XYZOUT', xyzout,
         'HKLIN',  fobs_mtz, 'HKLOUT', hklout,
         'LIBOUT', tmpdir / '_refmac.lib'],
        input_bytes=kw, cwd=tmpdir, check=False,
    )
    r, rf = parse_rfactors(log)
    return r, rf, log


def run_weightsnap(starthere_pdb, fobs_mtz, tmpdir):
    """Three-stage weight-snap: NCYC10@wm10 → NCYC10@wm0.01 → NCYC50@wm0.5.

    Returns (r_init, rf_init, r_final, rf_final, elapsed_s, final_mtz_path).
    """
    t0 = time.time()
    pdb_a, pdb_b, pdb_c = tmpdir/'_a.pdb', tmpdir/'_b.pdb', tmpdir/'_c.pdb'
    mtz_a, mtz_b, mtz_c = tmpdir/'_a.mtz', tmpdir/'_b.mtz', tmpdir/'_c.mtz'

    r1, rf1, log1 = _run_one(starthere_pdb, pdb_a, mtz_a, fobs_mtz, 10,  10,   tmpdir)
    if not pdb_a.exists():
        (tmpdir / '_refmac_stage1.log').write_text(log1)
        raise RuntimeError(f'refmac stage1 failed; log saved to {tmpdir}/_refmac_stage1.log\n'
                           + log1[-3000:])
    r2, rf2, log2 = _run_one(pdb_a,         pdb_b, mtz_b, fobs_mtz, 10,  0.01, tmpdir)
    if not pdb_b.exists():
        (tmpdir / '_refmac_stage2.log').write_text(log2)
        raise RuntimeError(f'refmac stage2 failed; log saved to {tmpdir}/_refmac_stage2.log\n'
                           + log2[-3000:])
    r3, rf3, log3 = _run_one(pdb_b,         pdb_c, mtz_c, fobs_mtz, 50,  0.5,  tmpdir)

    elapsed = time.time() - t0
    return r1, rf1, r3, rf3, elapsed, (mtz_c if mtz_c.exists() else None), (pdb_c if pdb_c.exists() else None)


# ── Worker: run one k level ───────────────────────────────────────────────────

def run_one_k(max_k, pdb_path, fobs_mtz, outdir):
    os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)

    print(f'[k={max_k}] Parsing conformers...')
    st_orig, chain_names, conf_data = parse_conformers(pdb_path)

    k_dir = outdir / f'k{max_k}'
    k_dir.mkdir(parents=True, exist_ok=True)
    starthere_pdb = k_dir / 'starthere.pdb'

    print(f'[k={max_k}] Building reduced PDB...')
    n_alt = build_starthere_pdb(chain_names, conf_data, st_orig, max_k,
                                ref_pdb=pdb_path, out_pdb=starthere_pdb,
                                workdir=k_dir)
    print(f'[k={max_k}] Running weight-snap refinement...')
    r_i, rf_i, r_f, rf_f, elapsed, final_mtz, final_pdb = run_weightsnap(
        starthere_pdb, fobs_mtz, k_dir)
    if final_mtz and final_mtz.exists():
        final_mtz.rename(k_dir / 'refmacout.mtz')
    if final_pdb and final_pdb.exists():
        final_pdb.rename(k_dir / 'refmacout.pdb')

    result = dict(max_k=max_k, n_alt=n_alt, elapsed=elapsed,
                  r_init=r_i, rf_init=rf_i, r_final=r_f, rf_final=rf_f)
    (k_dir / 'result.json').write_text(json.dumps(result))
    print(f'[k={max_k}] Done. R_init={r_i:.4f} Rf_init={rf_i:.4f} '
          f'R_final={r_f:.4f} Rf_final={rf_f:.4f} t={elapsed:.1f}s')


# ── Submit: build fobs once, then one job per k ───────────────────────────────

def submit(pdb_path, refme_path, outdir, partition, account=None, qos=None):
    os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)
    fobs_mtz = outdir / 'fobs.mtz'
    if not fobs_mtz.exists():
        print('Building Fobs MTZ...')
        fobs_tmp = outdir / '_fobs_tmp'
        fobs_tmp.mkdir(exist_ok=True)
        built = build_fobs_mtz(pdb_path, refme_path, fobs_tmp)
        shutil.copy2(built, fobs_mtz)
        print(f'  saved → {fobs_mtz}')
    else:
        print(f'Fobs MTZ already exists: {fobs_mtz}')

    me = Path(__file__).resolve()
    for max_k in K_LEVELS:
        script = SCRIPT_DIR / f'_cond_k{max_k}.sh'
        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name=cond_k{max_k}',
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
            f'  --fobs-mtz {fobs_mtz} \\',
            f'  --pdb {pdb_path} \\',
            f'  --outdir {outdir}',
            '',
        ]
        script.write_text('\n'.join(lines))
        script.chmod(0o755)
        r = subprocess.run(['sbatch', str(script)], capture_output=True, text=True)
        print(f'  k={max_k}: {r.stdout.strip() or r.stderr.strip()}')


# ── Collect: print table from saved result JSONs ──────────────────────────────

def collect(outdir):
    rows = []
    for max_k in K_LEVELS:
        p = outdir / f'k{max_k}/result.json'
        if p.exists():
            rows.append(json.loads(p.read_text()))
        else:
            print(f'  k={max_k}: no result.json yet')

    if not rows:
        print('No results found.')
        return

    hdr = (f"{'max_k':>5}  {'n_alt':>5}  {'time_s':>7}  "
           f"{'R_init':>7}  {'Rf_init':>7}  {'R_final':>7}  {'Rf_final':>8}")
    print()
    print(hdr)
    print('-' * len(hdr))
    for d in sorted(rows, key=lambda x: x['max_k']):
        def fmt(v):
            return f'{v:.4f}' if v is not None else '  N/A '
        t_str = f"{d['elapsed']:.1f}" if d.get('elapsed') is not None else 'N/A'
        print(f"{d['max_k']:>5}  {d['n_alt']:>5}  {t_str:>7}  "
              f"{fmt(d.get('r_init')):>7}  {fmt(d.get('rf_init')):>7}  "
              f"{fmt(d.get('r_final')):>7}  {fmt(d.get('rf_final')):>8}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb',       default=str(DEFAULT_PDB))
    ap.add_argument('--mtz',       default=str(DEFAULT_MTZ))
    ap.add_argument('--outdir',    default='1aho/explore_condensation')
    ap.add_argument('--partition', default='debug')
    ap.add_argument('--account')
    ap.add_argument('--qos')
    # Worker args
    ap.add_argument('--max-k',    type=int, help='Run single k level (worker mode)')
    ap.add_argument('--fobs-mtz', help='Pre-built fobs MTZ (worker mode)')
    # Modes
    ap.add_argument('--submit',  action='store_true', help='Submit one job per k to SLURM')
    ap.add_argument('--collect', action='store_true', help='Print table from saved results')
    args = ap.parse_args()

    pdb_path   = Path(args.pdb).resolve()
    refme_path = Path(args.mtz).resolve()
    outdir     = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.submit:
        submit(pdb_path, refme_path, outdir,
               partition=args.partition, account=args.account, qos=args.qos)

    elif args.max_k is not None:
        fobs_mtz = Path(args.fobs_mtz).resolve()
        run_one_k(args.max_k, pdb_path, fobs_mtz, outdir)

    elif args.collect:
        collect(outdir)

    else:
        # Sequential fallback
        os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)
        fobs_mtz = outdir / 'fobs.mtz'
        if not fobs_mtz.exists():
            with tempfile.TemporaryDirectory(prefix='cond_fobs_') as _td:
                built = build_fobs_mtz(pdb_path, refme_path, Path(_td))
                shutil.copy2(built, fobs_mtz)
        for max_k in K_LEVELS:
            run_one_k(max_k, pdb_path, fobs_mtz, outdir)
        collect(outdir)


if __name__ == '__main__':
    main()
