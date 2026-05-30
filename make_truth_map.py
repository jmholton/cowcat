#!/usr/bin/env ccp4-python
"""make_truth_map.py — Compute truth.map from truth_full.pdb.

Pipeline:
  1. gemmi sfcalc on truth_full.pdb (H already present) → protein SFs
  2. cavenv bulk solvent mask (single-conformer protein only, no waters/flood)
  3. Scale mask + box-filter smoothing + B envelope (matches ano_sfall.com)
  4. F_total = F_protein + F_solvent → truth.mtz
  5. FFT truth.mtz FC/PHIC → truth.map

Cell and space group are read from the CRYST1 record in the PDB.
Resolution (--dmin) defaults to 0.965 Å (the 1AHO boiled-data setting).

Usage:
  ccp4-python make_truth_map.py truth_full.pdb
  ccp4-python make_truth_map.py truth_full.pdb --dmin 2.0 --out truth.map
  ccp4-python make_truth_map.py sample_*/truth_full.pdb --workers 4
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

    Strips:
      - altlocs (keeps first occurrence of each chain/resnum/atom)
      - water chains (S, W) and flood chain (F)
      - lower-case chain names (symmetry copies in multi-chain models)
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
            if chain in ('S', 'W', 'F') or chain.islower():
                continue
            key = (chain, line[22:26], line[12:16])
            if key in seen:
                continue
            seen.add(key)
            fout.write(line[:16] + ' ' + line[17:])   # blank altloc col
        fout.write('END\n')


def make_truth_map(pdb_path, out_map=None, dmin=0.965):
    """Compute truth.map for one truth_full.pdb.  Returns output map path."""
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
        F_prot = fc_p * np.exp(1j * np.radians(phi_p))

        # ── 2. Cavenv bulk solvent mask ───────────────────────────────────────
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

        # ── 3. Scale mask to bulk water density + box-filter smoothing ────────
        ccp4 = gemmi.read_ccp4_map(str(tmp / 'solvent.map'))
        ccp4.setup(float('nan'))
        arr = np.array(ccp4.grid, copy=False)
        max_val = float(arr.max()) or 1.0
        arr *= SOLVENT_SCALE / max_val

        grid_spacing = a / arr.shape[2]   # Å/voxel along a-axis
        per_iter = 2.0 * (grid_spacing * np.pi * 1.468) ** 2
        n_smooth = int(SOLVENT_B / per_iter) if per_iter > 0 else 0
        if n_smooth < 3:
            n_smooth = 0
        smooth_B   = n_smooth * per_iter
        rs_solv_B  = SOLVENT_B - smooth_B
        if n_smooth > 0:
            pad    = n_smooth + 1
            padded = np.pad(arr, pad, mode='wrap')
            for _ in range(n_smooth):
                padded = uniform_filter(padded, size=3, mode='nearest')
            arr[:] = padded[pad:-pad, pad:-pad, pad:-pad]

        # ── 4. Mask → SFs + B envelope ────────────────────────────────────────
        hkl   = gemmi.transform_map_to_f_phi(ccp4.grid, half_l=True)
        s_sq  = (h_p / a)**2 + (k_p / b)**2 + (l_p / c)**2
        bfac  = np.exp(-rs_solv_B * s_sq / 4.0)
        hkl_array = np.column_stack([h_p, k_p, l_p]).astype(np.int32)
        F_solv = hkl.get_value_by_hkl(hkl_array).astype(complex) * bfac

        # ── 5. Combine and write truth.mtz ────────────────────────────────────
        F_tot   = F_prot + F_solv
        fc_out  = np.abs(F_tot).astype(np.float32)
        phi_out = np.degrees(np.angle(F_tot)).astype(np.float32)

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

        # ── 6. FFT → truth.map ───────────────────────────────────────────────
        grid = out_mtz.transform_f_phi_to_map('FC', 'PHIC', sample_rate=SAMPLE_RATE)
        ccp4_out = gemmi.Ccp4Map()
        ccp4_out.grid = grid
        ccp4_out.update_ccp4_header()
        ccp4_out.write_ccp4_map(str(out_map))

    return out_map


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pdbs', nargs='+', metavar='truth_full.pdb')
    ap.add_argument('--dmin',    type=float, default=0.965,
                    help='Resolution cutoff in Å (default 0.965)')
    ap.add_argument('--out',     default=None,
                    help='Output map path (only valid for single-PDB runs; '
                         'default: truth.map beside each PDB)')
    ap.add_argument('--workers', type=int, default=1,
                    help='Parallel workers (default 1)')
    args = ap.parse_args()

    if args.out and len(args.pdbs) > 1:
        ap.error('--out can only be used with a single input PDB')

    targets = [(p, args.out, args.dmin) for p in args.pdbs]

    def _run(item):
        pdb, out, dmin = item
        try:
            result = make_truth_map(pdb, out_map=out, dmin=dmin)
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
