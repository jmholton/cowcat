#!/usr/bin/env ccp4-python
"""
swapscan_to_samples.py — Convert swapscan refmacout.mtz files to training sample dirs.

Each trial with a refmacout.mtz becomes a sample_NNNNN/ directory compatible
with pack.py / dataset.py / train.py:

    sample_NNNNN/
        2fofc.map     – FWT/PHWT        from trial refmacout.mtz
        fofc.map      – DELFWT/PHDELWT  from trial refmacout.mtz
        fc.map        – FC/PHIC         from trial refmacout.mtz
        truth.map     – Fgt/PHIgt       from 1aho/gt48.mtz (shared symlink)
        metadata.json – trial_id, swaps, R, Rf, rmsd_e, wE

Truth map is built once from gt48.mtz (Fgt/PHIgt columns) and symlinked into
every sample directory.

Run preprocess.py afterwards to build crossp.npy, then pack.py to pack to npy.

Usage:
    ccp4-python swapscan_to_samples.py \\
        --swapscan /path/to/swapscan_under20_exhaust \\
        --gt48mtz  1aho/gt48.mtz \\
        --outdir   data/swapscan_under20 \\
        [--workers 8]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gemmi

SAMPLE_RATE = 3.0
SCRIPT_DIR  = Path(__file__).resolve().parent


# ── Map generation ────────────────────────────────────────────────────────────

def mtz_to_ccp4(mtz_path, f_col, phi_col, out_path):
    mtz  = gemmi.read_mtz_file(str(mtz_path))
    grid = mtz.transform_f_phi_to_map(f_col, phi_col, sample_rate=SAMPLE_RATE)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(out_path))


def build_truth_map(gt48_mtz, outdir):
    """Build truth.map from Fgt/PHIgt in gt48.mtz. Returns absolute path."""
    truth_map = outdir / '_truth.map'
    if not truth_map.exists():
        print(f'Building truth map from {gt48_mtz} (Fgt/PHIgt) ...')
        mtz_to_ccp4(gt48_mtz, 'Fgt', 'PHIgt', truth_map)
        print(f'  truth map → {truth_map}')
    return truth_map.resolve()


# ── Per-trial conversion ──────────────────────────────────────────────────────

def process_trial(args):
    """Convert one trial directory → sample_NNNNN/. Returns (name, status)."""
    trial_dir, outdir, truth_map_abs = args
    trial_dir     = Path(trial_dir)
    outdir        = Path(outdir)
    truth_map_abs = Path(truth_map_abs)

    rjson = trial_dir / 'result.json'
    rmtz  = trial_dir / 'refmacout.mtz'

    if not rjson.exists() or not rmtz.exists():
        return trial_dir.name, 'skip (missing result.json or refmacout.mtz)'

    result = json.loads(rjson.read_text())
    if result.get('status') != 'ok':
        return trial_dir.name, f'skip (status={result.get("status")})'

    tid        = result['trial_id']
    sample_dir = outdir / f'sample_{tid:05d}'

    if (sample_dir / 'metadata.json').exists():
        return trial_dir.name, 'already done'

    sample_dir.mkdir(parents=True, exist_ok=True)
    try:
        mtz_to_ccp4(rmtz, 'FWT',    'PHWT',    sample_dir / '2fofc.map')
        mtz_to_ccp4(rmtz, 'DELFWT', 'PHDELWT', sample_dir / 'fofc.map')
        mtz_to_ccp4(rmtz, 'FC',     'PHIC',    sample_dir / 'fc.map')

        truth_link = sample_dir / 'truth.map'
        if not truth_link.exists():
            truth_link.symlink_to(truth_map_abs)

        meta = {
            'trial_id': tid,
            'swaps':    result.get('swaps', []),
            'r':        result.get('r'),
            'rf':       result.get('rf'),
            'rmsd_e':   result.get('rmsd_e'),
            'wE':       result.get('wE'),
            'source':   str(trial_dir),
        }
        (sample_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))
        return trial_dir.name, 'ok'

    except Exception as e:
        import traceback
        return trial_dir.name, f'ERROR: {traceback.format_exc()}'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--swapscan', required=True,
                    help='Swapscan output dir (contains trial_NNNNN/ subdirs)')
    ap.add_argument('--gt48mtz', default='1aho/gt48.mtz',
                    help='Ground-truth MTZ with Fgt/PHIgt columns (default: 1aho/gt48.mtz)')
    ap.add_argument('--outdir',  required=True,
                    help='Output data dir (will contain sample_NNNNN/ subdirs)')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    swapscan = Path(args.swapscan).resolve()
    gt48_mtz = Path(args.gt48mtz).resolve()
    outdir   = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    truth_map = build_truth_map(gt48_mtz, outdir)

    trial_dirs = sorted(swapscan.glob('trial_*'))
    print(f'Found {len(trial_dirs)} trial directories')

    work = [(str(td), str(outdir), str(truth_map)) for td in trial_dirs]
    ok = skipped = errors = 0
    n  = len(work)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_trial, item): item for item in work}
        done = 0
        for fut in as_completed(futures):
            name, status = fut.result()
            done += 1
            if status == 'ok':
                ok += 1
            elif 'ERROR' in status:
                errors += 1
                print(f'  {name}: {status}')
            else:
                skipped += 1
            if done % 500 == 0 or done == n:
                print(f'  {done}/{n}  ok={ok}  skip={skipped}  err={errors}',
                      flush=True)

    print(f'\nDone. ok={ok}  skipped={skipped}  errors={errors}')
    print(f'Output: {outdir}')
    print(f'\nNext steps:')
    print(f'  python3 preprocess.py --data {outdir} --workers {args.workers}')
    print(f'  python3 pack.py --data {outdir} --workers {args.workers}')


if __name__ == '__main__':
    main()
