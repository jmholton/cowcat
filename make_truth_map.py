#!/usr/bin/env ccp4-python
"""make_truth_map.py — Compute truth.map from truth_full.pdb.

Two modes depending on whether --obs-mtz is supplied:

  Default (generate_protein style):
    1. gemmi sfcalc on truth_full.pdb → protein SFs
    2. cavenv bulk solvent mask (chain A only)
    3. Scale + box-filter smoothing + B envelope
    4. F_total = F_protein + F_solvent → truth.mtz → truth.map

  --obs-mtz gt48.mtz (generate_1aho style):
    1. gemmi sfcalc on truth_full.pdb → protein SFs
    2. Wilson B correction: scale FC to match Wilson B of Fgt/FP in obs-mtz
    3. truth.map from corrected sfcalc — no bulk solvent
    4. refme.mtz beside truth_full.pdb regenerated if present:
       FP = |F_truth + F_bulk| using Fpart/PHIpart already in refme.mtz
    (matches generate_1aho.py build_sample_mtz exactly)

Cell and space group are read from the CRYST1 record in the PDB.
Resolution (--dmin) defaults to 0.965 Å (the 1AHO boiled-data setting).

Usage:
  ccp4-python make_truth_map.py truth_full.pdb
  ccp4-python make_truth_map.py truth_full.pdb --dmin 2.0 --out truth.map
  ccp4-python make_truth_map.py sample_*/truth_full.pdb --workers 4
  ccp4-python make_truth_map.py truth_full.pdb --obs-mtz 1aho/gt48.mtz
  ccp4-python make_truth_map.py sample_*/truth_full.pdb --obs-mtz 1aho/gt48.mtz --workers 8
"""

import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gemmi
import numpy as np
from scipy.ndimage import uniform_filter

SAMPLE_RATE   = 3.0      # map oversampling
SOLVENT_RAD   = 1.41     # cavenv RADMAX (Å)
SOLVENT_SCALE = 0.334    # bulk water e⁻/Å³
SOLVENT_B     = 50.0     # B_sol (Å²)


def _read_cryst1(pdb_path):
    """Return (cell_tuple, sg_hm) from CRYST1 line."""
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith('CRYST1'):
            a, b, c = float(line[6:15]), float(line[15:24]), float(line[24:33])
            al, be, ga = float(line[33:40]), float(line[40:47]), float(line[47:54])
            sg = line[55:66].strip()
            return (a, b, c, al, be, ga), sg
    raise RuntimeError(f'No CRYST1 in {pdb_path}')


def _mask_pdb(pdb_path, out_path):
    """Write single-conformer protein-only PDB for cavenv masking.

    Keeps only chain A atoms to stay well below cavenv's 50,000-atom-with-
    symmetry-mates limit (multi-conformer truth_full.pdb can have 48 chains).
    One representative conformer is sufficient for bulk solvent envelope.
    """
    seen = set()
    with open(pdb_path) as fin, open(out_path, 'w') as fout:
        for line in fin:
            if line.startswith('CRYST1'):
                fout.write(line)
                continue
            if line[:6] not in ('ATOM  ', 'HETATM'):
                continue
            chain = line[21]
            if chain != 'A':
                continue
            key = (line[22:26], line[12:16])
            if key in seen:
                continue
            seen.add(key)
            fout.write(line[:16] + ' ' + line[17:])   # blank altloc col
        fout.write('END\n')


def _wilson_b(F, s2, n_bins=20, min_per_bin=10):
    """Wilson B from arrays of |F| and s²=1/d².  Returns 0.0 if insufficient data."""
    valid = (F > 0) & np.isfinite(F) & np.isfinite(s2)
    F, s2 = F[valid].astype(np.float64), s2[valid].astype(np.float64)
    if s2.size < min_per_bin * 3:
        return 0.0
    edges = np.linspace(s2.min(), s2.max(), n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m = (s2 >= edges[i]) & (s2 < edges[i + 1])
        if m.sum() < min_per_bin:
            continue
        xs.append(float(s2[m].mean()))
        ys.append(float(np.log((F[m] ** 2).mean())))
    if len(xs) < 3:
        return 0.0
    slope, _ = np.polyfit(xs, ys, 1)
    return float(-2.0 * slope)


def _ref_wilson_b_from_mtz(mtz_path):
    """Compute Wilson B from an amplitude column of an MTZ file.

    Tries FP, F, Fgt in order — works with 1aho.mtz (FP) and gt48.mtz (Fgt).
    Returns None if no suitable column is found.
    """
    mtz = gemmi.read_mtz_file(str(mtz_path))
    labels = mtz.column_labels()
    f_col = next((c for c in ('FP', 'F', 'Fgt') if c in labels), None)
    if f_col is None:
        print(f'    Wilson B: no amplitude column in {mtz_path} (have: {labels})')
        return None
    h = mtz.column_with_label('H').array.astype(np.int32)
    k = mtz.column_with_label('K').array.astype(np.int32)
    l = mtz.column_with_label('L').array.astype(np.int32)
    F = mtz.column_with_label(f_col).array.astype(np.float32)
    cell = mtz.cell
    s2 = np.array([cell.calculate_1_d2([int(h_), int(k_), int(l_)])
                   for h_, k_, l_ in zip(h, k, l)], dtype=np.float64)
    print(f'    Wilson B reference: using column {f_col!r} from {Path(mtz_path).name}')
    return _wilson_b(F, s2)


def _regen_refme_mtz(refme_path, F_truth_dict):
    """Rewrite refme.mtz with FP = |F_truth + F_bulk| for all observed HKLs.

    Uses existing refme.mtz as template: keeps FreeR_flag, Fpart, PHIpart, and
    the valid/missing HKL mask (NaN FP = never observed).  Only FP and SIGFP
    are recomputed from the new F_truth_dict.

    F_truth_dict: {(H, K, L): complex amplitude} from corrected sfcalc.
    """
    mtz = gemmi.read_mtz_file(str(refme_path))
    arr = np.array(mtz)
    labels = mtz.column_labels()
    ih = labels.index('H')
    ik = labels.index('K')
    il = labels.index('L')
    ifp     = labels.index('FP')
    isigfp  = labels.index('SIGFP')
    ifpart  = labels.index('Fpart')
    iphipart = labels.index('PHIpart')

    for i in range(len(arr)):
        if np.isnan(arr[i, ifp]):
            continue   # missing reflection — leave as NaN
        hkl = (int(arr[i, ih]), int(arr[i, ik]), int(arr[i, il]))
        F_t    = F_truth_dict.get(hkl, 0.0 + 0.0j)
        F_bulk = arr[i, ifpart] * np.exp(1j * np.radians(arr[i, iphipart]))
        amp    = float(np.abs(F_t + F_bulk))
        arr[i, ifp]    = amp
        arr[i, isigfp] = max(0.01, 0.02 * amp)

    mtz.set_data(arr)
    mtz.write_to_file(str(refme_path))


def make_truth_map(pdb_path, out_map=None, dmin=0.965, obs_mtz=None):
    """Compute truth.map for one truth_full.pdb.  Returns output map path.

    obs_mtz: if given, use generate_1aho mode (Wilson B correction, no cavenv).
             Should be gt48.mtz or similar with Fgt/FP column.
             Also regenerates refme.mtz beside truth_full.pdb if it exists.
    """
    pdb_path = Path(pdb_path).resolve()
    if out_map is None:
        out_map = pdb_path.parent / 'truth.map'
    out_map = Path(out_map)

    cell, sg_hm = _read_cryst1(pdb_path)
    a, b, c = cell[0], cell[1], cell[2]
    na = round(a * SAMPLE_RATE / dmin)
    nb = round(b * SAMPLE_RATE / dmin)
    nc = round(c * SAMPLE_RATE / dmin)
    sg = gemmi.find_spacegroup_by_name(sg_hm)
    sg_num = sg.number if sg else 1

    with tempfile.TemporaryDirectory(prefix='truth_map_') as tmpd:
        tmp = Path(tmpd)

        # ── 1. Protein SFs via gemmi sfcalc ──────────────────────────────────
        mtz_prot = tmp / 'protein.mtz'
        r = subprocess.run(
            ['gemmi', 'sfcalc', f'--dmin={dmin:.4f}', f'--rate={SAMPLE_RATE}',
             f'--to-mtz={mtz_prot}', str(pdb_path)],
            cwd=str(tmp), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if r.returncode != 0 or not mtz_prot.exists():
            raise RuntimeError(
                f'gemmi sfcalc failed:\n{r.stderr.decode(errors="replace")[-800:]}')

        mtz = gemmi.read_mtz_file(str(mtz_prot))
        h_p   = mtz.column_with_label('H').array.astype(np.float64)
        k_p   = mtz.column_with_label('K').array.astype(np.float64)
        l_p   = mtz.column_with_label('L').array.astype(np.float64)
        fc_p  = mtz.column_with_label('FC').array.astype(np.float64)
        phi_p = mtz.column_with_label('PHIC').array.astype(np.float64)

        if obs_mtz is not None:
            # ── generate_1aho mode: Wilson B correction, no bulk solvent ──────
            cell_p = mtz.cell
            s2_p = np.array([
                cell_p.calculate_1_d2([int(h_p[i]), int(k_p[i]), int(l_p[i])])
                for i in range(len(h_p))
            ], dtype=np.float64)
            ref_B = _ref_wilson_b_from_mtz(obs_mtz)
            gen_B = _wilson_b(fc_p, s2_p)
            if ref_B and gen_B:
                delta_B = ref_B - gen_B
                fc_p = fc_p * np.exp(-delta_B * s2_p / 4.0)
                print(f'    Wilson B: ref={ref_B:.2f} gen={gen_B:.2f} '
                      f'ΔB={delta_B:+.2f} Å² applied')
            else:
                print(f'    Wilson B: ref={ref_B} gen={gen_B} — skipped')

            fc_out  = fc_p.astype(np.float32)
            phi_out = phi_p.astype(np.float32)

            # build F_truth dict for refme.mtz update
            F_truth_dict = {
                (int(h_p[i]), int(k_p[i]), int(l_p[i])): fc_p[i] * np.exp(1j * np.radians(phi_p[i]))
                for i in range(len(h_p))
            }

        else:
            # ── default mode: cavenv bulk solvent ─────────────────────────────
            mask_pdb = str(tmp / 'mask.pdb')
            _mask_pdb(pdb_path, mask_pdb)

            cell_kw = f'{a} {b} {c} {cell[3]} {cell[4]} {cell[5]}'
            cavenv_in = (
                f'CELL {cell_kw}\nSYMM {sg_num}\nENVSOLVENT\n'
                f'GRID {na} {nb} {nc}\nRADMAX {SOLVENT_RAD}\n'
            ).encode()
            cv = subprocess.run(
                ['cavenv', 'xyzin', mask_pdb, 'mapout', 'solvent.map'],
                input=cavenv_in, cwd=str(tmp),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            if cv.returncode != 0 or not (tmp / 'solvent.map').exists():
                raise RuntimeError(
                    f'cavenv failed:\n{cv.stdout.decode(errors="replace")[-800:]}')

            ccp4 = gemmi.read_ccp4_map(str(tmp / 'solvent.map'))
            ccp4.setup(float('nan'))
            arr = np.array(ccp4.grid, copy=False)
            max_val = float(arr.max()) or 1.0
            arr *= SOLVENT_SCALE / max_val

            grid_spacing = a / arr.shape[2]
            per_iter = 2.0 * (grid_spacing * np.pi * 1.468) ** 2
            n_smooth = int(SOLVENT_B / per_iter) if per_iter > 0 else 0
            if n_smooth < 3:
                n_smooth = 0
            smooth_B  = n_smooth * per_iter
            rs_solv_B = SOLVENT_B - smooth_B
            if n_smooth > 0:
                pad    = n_smooth + 1
                padded = np.pad(arr, pad, mode='wrap')
                for _ in range(n_smooth):
                    padded = uniform_filter(padded, size=3, mode='nearest')
                arr[:] = padded[pad:-pad, pad:-pad, pad:-pad]

            hkl      = gemmi.transform_map_to_f_phi(ccp4.grid, half_l=True)
            s_sq     = (h_p / a)**2 + (k_p / b)**2 + (l_p / c)**2
            bfac     = np.exp(-rs_solv_B * s_sq / 4.0)
            hkl_arr  = np.column_stack([h_p, k_p, l_p]).astype(np.int32)
            F_solv   = hkl.get_value_by_hkl(hkl_arr).astype(complex) * bfac
            F_prot   = fc_p * np.exp(1j * np.radians(phi_p))
            F_tot    = F_prot + F_solv
            fc_out   = np.abs(F_tot).astype(np.float32)
            phi_out  = np.degrees(np.angle(F_tot)).astype(np.float32)

        # ── Write truth.mtz and FFT → truth.map ──────────────────────────────
        out_mtz_path = tmp / 'truth.mtz'
        out_mtz = gemmi.Mtz()
        out_mtz.cell       = mtz.cell
        out_mtz.spacegroup = mtz.spacegroup
        out_mtz.add_dataset('HKL_base')
        for lbl in ('H', 'K', 'L'):
            out_mtz.add_column(lbl, 'H')
        out_mtz.add_dataset('data')
        out_mtz.add_column('FC',   'F')
        out_mtz.add_column('PHIC', 'P')
        out_mtz.set_data(
            np.column_stack([h_p, k_p, l_p, fc_out, phi_out]).astype(np.float32))
        out_mtz.write_to_file(str(out_mtz_path))

        grid = out_mtz.transform_f_phi_to_map('FC', 'PHIC', sample_rate=SAMPLE_RATE)
        ccp4_out = gemmi.Ccp4Map()
        ccp4_out.grid = grid
        ccp4_out.update_ccp4_header()
        ccp4_out.write_ccp4_map(str(out_map))

    # ── Regenerate refme.mtz if present (generate_1aho mode only) ────────────
    if obs_mtz is not None:
        refme_path = pdb_path.parent / 'refme.mtz'
        if refme_path.exists():
            _regen_refme_mtz(refme_path, F_truth_dict)
            print(f'    refme.mtz updated: {refme_path}')
        else:
            print(f'    refme.mtz not found beside {pdb_path.name} — skipped')

    return out_map


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pdbs', nargs='+', metavar='truth_full.pdb')
    ap.add_argument('--dmin',    type=float, default=0.965,
                    help='Resolution cutoff in Å (default 0.965)')
    ap.add_argument('--obs-mtz', default=None, metavar='gt48.mtz',
                    help='generate_1aho mode: apply Wilson B correction from '
                         'FP column of this MTZ; skip cavenv bulk solvent')
    ap.add_argument('--out',     default=None,
                    help='Output map path (only valid for single-PDB runs; '
                         'default: truth.map beside each PDB)')
    ap.add_argument('--workers', type=int, default=1,
                    help='Parallel workers (default 1)')
    args = ap.parse_args()

    if args.out and len(args.pdbs) > 1:
        ap.error('--out can only be used with a single input PDB')

    obs_mtz = Path(args.obs_mtz).resolve() if args.obs_mtz else None
    if obs_mtz and not obs_mtz.exists():
        ap.error(f'--obs-mtz not found: {obs_mtz}')

    targets = [(p, args.out, args.dmin, obs_mtz) for p in args.pdbs]

    def _run(item):
        pdb, out, dmin, obs_mtz = item
        try:
            result = make_truth_map(pdb, out_map=out, dmin=dmin, obs_mtz=obs_mtz)
            return pdb, True, str(result)
        except Exception as e:
            return pdb, False, str(e)

    ok = fail = 0
    if args.workers == 1:
        for item in targets:
            pdb, success, msg = _run(item)
            print(f'{"ok" if success else "FAILED"}: {pdb}  →  {msg}')
            if success: ok += 1
            else:       fail += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run, item): item for item in targets}
            for fut in as_completed(futs):
                pdb, success, msg = fut.result()
                print(f'{"ok" if success else "FAILED"}: {pdb}  →  {msg}')
                if success: ok += 1
                else:       fail += 1

    print(f'\n{ok + fail} processed: {ok} ok, {fail} failed')
    if fail:
        sys.exit(1)


if __name__ == '__main__':
    main()
