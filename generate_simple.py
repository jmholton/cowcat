#!/usr/bin/env ccp4-python
"""
generate_simple.py — Simple N-O-atom training samples in the 1AHO P1 cell.

Truth: N_ATOMS random O atoms in a P1 cell with the same dimensions as the
1AHO training data (45.9×40.7×30.1 Å, d_min=0.965 Å → 96×128×144 grid).
Partial: N_MISS atoms randomly deleted → starthere.pdb.
Refmac refines the partial model; maps written as sample_NNNNN/.

The Fo-Fc map will have a clean positive peak at each missing atom — a
super-obvious signal for the network to learn on, in the same grid as the
1AHO and swapscan datasets so all data can be mixed in training.

Usage:
    ccp4-python generate_simple.py --nsamples 500 --outdir data/data_simple_n500
    ccp4-python generate_simple.py --submit --nsamples 1000 --seed 0 \\
        --outdir data/data_simple_s0 \\
        --partition lr6 --account pc_als831 --qos lr_normal
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gemmi
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

CELL       = (45.9, 40.7, 30.1, 90.0, 90.0, 90.0)   # matches 1AHO training data → 96×128×144
DMIN       = 0.965
N_ATOMS    = 20
N_MISS     = 1
SPACEGROUP = 'P 21 21 21'
NCYC       = 20
SAMPLE_RATE = 3.0
MAX_ARRAY  = 1000

REFMAC5 = Path(shutil.which('refmac5') or '/programs/ccp4-8.0/bin/refmac5')


# ── Atom placement ────────────────────────────────────────────────────────────

def place_atoms(n, rng, min_dist=2.5):
    """Return list of (x, y, z, b_iso) for n O atoms in the unit cell."""
    a, b, c = CELL[:3]
    positions = []
    for _ in range(n * 300):
        if len(positions) >= n:
            break
        x, y, z = rng.random() * a, rng.random() * b, rng.random() * c
        if positions:
            dists = [((x-px)**2 + (y-py)**2 + (z-pz)**2)**0.5
                     for px, py, pz, _ in positions]
            if min(dists) < min_dist:
                continue
        positions.append((x, y, z, float(rng.uniform(10.0, 30.0))))
    return positions[:n]


def write_pdb(positions, out_path):
    a, b, c, al, be, ga = CELL
    import gemmi as _gemmi
    _sg = _gemmi.find_spacegroup_by_name(SPACEGROUP)
    _z  = len(list(_sg.operations())) if _sg else 1
    _hm = _sg.hm if _sg else SPACEGROUP
    lines = [
        f'CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} {_hm:<11s}{_z:3d}\n'
    ]
    for i, (x, y, z, b_iso) in enumerate(positions):
        lines.append(
            f'HETATM{i+1:5d}  O   HOH A{i+1:4d}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}'
            f'{1.00:6.2f}{b_iso:6.2f}          '
            f'  O\n'
        )
    lines.append('END\n')
    Path(out_path).write_text(''.join(lines))


# ── MTZ construction ──────────────────────────────────────────────────────────

def build_fobs_mtz(truth_sf_mtz, out_path, rng, freer_fraction=0.05):
    """FP = |FC_truth|, SIGFP = 0.02·FP, random FreeR_flag."""
    mtz  = gemmi.read_mtz_file(str(truth_sf_mtz))
    fc   = np.array(mtz.column_with_label('FC'),   dtype=np.float32)
    h    = np.array(mtz.column_with_label('H'),    dtype=np.int32)
    k    = np.array(mtz.column_with_label('K'),    dtype=np.int32)
    l    = np.array(mtz.column_with_label('L'),    dtype=np.int32)
    sigf = np.maximum(0.01, 0.02 * fc)
    free = (rng.random(len(h)) < freer_fraction).astype(np.float32)

    out = gemmi.Mtz()
    out.cell       = mtz.cell
    out.spacegroup = mtz.spacegroup
    out.add_dataset('HKL_base')
    for lbl in ('H', 'K', 'L'):
        out.add_column(lbl, 'H')
    out.add_dataset('data')
    out.add_column('FP',         'F')
    out.add_column('SIGFP',      'Q')
    out.add_column('FreeR_flag', 'I')
    out.set_data(np.column_stack([h, k, l, fc, sigf, free]))
    out.write_to_file(str(out_path))


def mtz_to_ccp4(mtz_path, f_col, phi_col, out_path):
    mtz  = gemmi.read_mtz_file(str(mtz_path))
    grid = mtz.transform_f_phi_to_map(f_col, phi_col, sample_rate=SAMPLE_RATE)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(out_path))


# ── Refmac ────────────────────────────────────────────────────────────────────

def run_refmac(starthere_pdb, fobs_mtz, tmpdir):
    os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)
    kw  = b'LABIN FP=FP SIGFP=SIGFP FREE=FreeR_flag\n'
    kw += b'LABOUT FC=FC PHIC=PHIC FWT=FWT PHWT=PHWT'
    kw += b' DELFWT=DELFWT PHDELWT=PHDELWT\n'
    kw += f'NCYC {NCYC}\n'.encode()
    kw += b'REFI TYPE REST RESI MLKF\n'
    kw += b'SCALE LSSC ANISO BULK\n'
    kw += b'SOLVENT YES\n'
    kw += b'MAKE HYDR NO\n'

    out_mtz = tmpdir / 'refmacout.mtz'
    out_pdb = tmpdir / 'refmacout.pdb'
    r = subprocess.run(
        [str(REFMAC5),
         'XYZIN',  str(starthere_pdb.resolve()),
         'XYZOUT', str(out_pdb),
         'HKLIN',  str(fobs_mtz.resolve()),
         'HKLOUT', str(out_mtz),
         'LIBOUT', str(tmpdir / '_refmac.lib')],
        input=kw, capture_output=True, cwd=str(tmpdir),
    )
    log = r.stdout.decode(errors='replace')
    (tmpdir / 'refmac.log').write_text(log)
    rwork = rfree = None
    for line in reversed(log.splitlines()):
        if 'R factor' in line and 'Rfree' in line:
            try:
                parts = line.split()
                rwork, rfree = float(parts[-2]), float(parts[-1])
            except Exception:
                pass
            break
    return rwork, rfree, out_mtz if out_mtz.exists() else None


# ── Per-sample pipeline ───────────────────────────────────────────────────────

def generate_sample(sample_idx, outdir, seed=None):
    outdir     = Path(outdir).resolve()
    sample_dir = outdir / f'sample_{sample_idx:05d}'

    if sample_dir.exists() and (sample_dir / 'metadata.json').exists():
        return sample_idx, True, 'already done'

    rng_seed = sample_idx if seed is None else seed + sample_idx
    rng      = np.random.default_rng(rng_seed)

    ccp4_scr = Path(os.environ.get('CCP4_SCR', '/tmp')) / outdir.name
    ccp4_scr.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix=f'simple_{sample_idx:05d}_', dir=ccp4_scr))

    try:
        positions = place_atoms(N_ATOMS, rng)
        if len(positions) < N_ATOMS:
            return sample_idx, False, f'only placed {len(positions)}/{N_ATOMS} atoms'

        truth_pdb = tmpdir / 'truth.pdb'
        write_pdb(positions, truth_pdb)

        missing_idx = sorted(rng.choice(N_ATOMS, size=N_MISS, replace=False).tolist())
        partial = [p for i, p in enumerate(positions) if i not in missing_idx]
        starthere_pdb = tmpdir / 'starthere.pdb'
        write_pdb(partial, starthere_pdb)

        truth_sf_mtz = tmpdir / 'truth_sf.mtz'
        r = subprocess.run(
            ['gemmi', 'sfcalc', f'--dmin={DMIN}', f'--to-mtz={truth_sf_mtz}', str(truth_pdb)],
            capture_output=True,
        )
        if r.returncode != 0:
            return sample_idx, False, f'sfcalc failed: {r.stderr.decode()[-300:]}'

        fobs_mtz = tmpdir / 'fobs.mtz'
        build_fobs_mtz(truth_sf_mtz, fobs_mtz, rng)

        rwork, rfree, out_mtz = run_refmac(starthere_pdb, fobs_mtz, tmpdir)
        if out_mtz is None:
            return sample_idx, False, 'refmac produced no output MTZ'

        sample_dir.mkdir(parents=True, exist_ok=True)
        mtz_to_ccp4(truth_sf_mtz, 'FC',     'PHIC',    sample_dir / 'truth.map')
        mtz_to_ccp4(out_mtz,      'FWT',    'PHWT',    sample_dir / '2fofc.map')
        mtz_to_ccp4(out_mtz,      'DELFWT', 'PHDELWT', sample_dir / 'fofc.map')
        mtz_to_ccp4(out_mtz,      'FC',     'PHIC',    sample_dir / 'fc.map')

        meta = {
            'sample_idx':  sample_idx,
            'missing_idx': missing_idx,
            'n_atoms':     N_ATOMS,
            'n_miss':      N_MISS,
            'r':           rwork,
            'rf':          rfree,
        }
        (sample_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))
        r_str = f'R={rwork:.4f} Rf={rfree:.4f}' if rwork is not None else 'R=n/a'
        return sample_idx, True, r_str

    except Exception:
        import traceback
        return sample_idx, False, traceback.format_exc()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── SLURM submission ──────────────────────────────────────────────────────────

def submit(outdir, nsamples, seed, partition, account, qos):
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    n_batches = (nsamples + MAX_ARRAY - 1) // MAX_ARRAY
    for b in range(n_batches):
        start = b * MAX_ARRAY
        end   = min(start + MAX_ARRAY - 1, nsamples - 1)
        sh    = outdir / f'_batch{b}.sh'
        log   = outdir / f'slurm_b{b}_%a.out'
        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name=simple_{outdir.name}',
            f'#SBATCH --partition={partition}',
            '#SBATCH --ntasks=1',
            f'#SBATCH --array=0-{end - start}',
            f'#SBATCH --output={log}',
            '#SBATCH --export=ALL',
        ]
        if account:
            lines.append(f'#SBATCH --account={account}')
        if qos:
            lines.append(f'#SBATCH --qos={qos}')
        lines += [
            'mkdir -p "${CCP4_SCR:-/tmp}"',
            f'cd {SCRIPT_DIR}',
            f'ccp4-python {script} --task $(( {start} + $SLURM_ARRAY_TASK_ID ))'
            f' --outdir {outdir} --seed {seed}'
            f' --n-atoms {N_ATOMS} --spacegroup "{SPACEGROUP}"',
        ]
        sh.write_text('\n'.join(lines) + '\n')
        r = subprocess.run(['sbatch', str(sh)], capture_output=True, text=True)
        print(f'  Batch {b} (samples {start}–{end}): {r.stdout.strip() or r.stderr.strip()}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global N_ATOMS, SPACEGROUP
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nsamples',   type=int, default=100)
    ap.add_argument('--seed',       type=int, default=0)
    ap.add_argument('--outdir',     required=True)
    ap.add_argument('--submit',     action='store_true')
    ap.add_argument('--task',       type=int, default=None,
                    help='SLURM array task: generate sample_idx=TASK')
    ap.add_argument('--partition',  default='lr6')
    ap.add_argument('--account',    default='pc_als831')
    ap.add_argument('--qos',        default='lr_normal')
    ap.add_argument('--n-atoms',    type=int, default=None,
                    help=f'Number of O atoms (default: {N_ATOMS})')
    ap.add_argument('--spacegroup', default=None,
                    help=f'Space group HM symbol (default: {SPACEGROUP})')
    args = ap.parse_args()

    if args.n_atoms is not None:
        N_ATOMS = args.n_atoms
    if args.spacegroup is not None:
        SPACEGROUP = args.spacegroup

    if args.submit:
        submit(args.outdir, args.nsamples, args.seed,
               args.partition, args.account, args.qos)
        return

    if args.task is not None:
        idx, ok, msg = generate_sample(args.task, args.outdir, seed=args.seed)
        print(f'sample {idx}: {"ok" if ok else "FAILED"}  {msg}')
        return

    # Local sequential run
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for i in range(args.nsamples):
        idx, success, msg = generate_sample(i, outdir, seed=args.seed)
        if success:
            ok += 1
        else:
            fail += 1
            print(f'  FAILED sample {idx}: {msg}')
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{args.nsamples}  ok={ok}  fail={fail}')
    print(f'Done. ok={ok}  fail={fail}')


if __name__ == '__main__':
    main()
