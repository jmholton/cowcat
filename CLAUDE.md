# Claude CNN — Electron Density Reconstruction

## Project Goal

Train a 3D U-Net to reconstruct ground-truth electron density (Fo) from phased maps that contain systematic errors. The current focus is on **alternate conformations**: the partial model has a single centroid atom where the truth has a disordered cluster; refmac refines the centroid's occupancy/position against the true multi-conf density. The CNN learns to recover the true multi-Gaussian density from the blurred/biased 2Fo-Fc, Fo-Fc, and Fc maps.

---

## Key Files

| File | Purpose |
|------|---------|
| `generate_protein.py` | Main data generation pipeline — protein backbone + side chains + altloc + refmac |
| `generate_data.py` | Older pipeline — random O atoms, atom deletion, no refmac (kept for reference) |
| `model.py` | UNet3D: 4 input channels, 1 output, base_features=32, circular padding, 5.84M params |
| `train.py` | Training loop: peak_weighted_mse loss, AdamW, CosineAnnealingLR |
| `dataset.py` | `make_splits` / `make_splits_multi` — dataset loading utilities |
| `preprocess.py` | Packs `.map` files into memory-mapped `.bin` for fast loading |
| `jigglepdb.awk` | Displaces atom positions to generate alternate conformers |
| `converge_refmac.com` | Wrapper: runs refmac5 N cycles on `starthere.pdb` + `refme.mtz` → `refmacout.mtz` |
| `randompdb.com` | Generates random O-atom structure in a P1 cell |

---

## Architecture: Fixed Parameters

- **Unit cell**: P1, 40×40×40 Å, 90/90/90
- **Resolution**: d_min = 2.0 Å
- **Grid**: 60×60×60 voxels (sample_rate=3.0)
- **Map channels (input)**: 2Fo-Fc (FWT/PHWT), Fo-Fc (DELFWT/PHDELWT), Fc (FC_ALL/PHIC_ALL), cross-Patterson
- **Output**: truth density (FC/PHIC of full model)

---

## Python Interpreters

**Critical — do not mix these up:**

- `ccp4-python` — use for `generate_protein.py`, `generate_data.py`, and any script importing `gemmi`. Gemmi is installed here; the pytorch python cannot use it (GLIBC too old for prebuilt wheels).
- `/programs/pytorch/envs/pt/bin/python` — use for `train.py`, `dataset.py`, anything using torch.
- System `python` is Python 2. Never use it.
- The symlink `./python3` in this directory points to the base pytorch python (no torch) — use the full `envs/pt` path for training.

---

## SLURM Cluster

- **Data generation**: `--partition debug`, `--workers 300` (cluster ~400 cores total)
- **Training**: `--partition gpu --gres=gpu:1`
- Always pass `--export=ALL` to SLURM array jobs so CCP4 environment variables propagate.
- `CCP4_SCR` directory must exist on compute nodes before refmac5 runs. Fix in code: `os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)`.

Common generation invocation:
```
ccp4-python generate_protein.py --submit --nsamples 1000 \
    --outdir data_n10_N1altconf3_refmac_n1000 --partition debug --max-array 300
```

Common training invocation:
```
srun --partition gpu --gres=gpu:1 \
    /programs/pytorch/envs/pt/bin/python train.py \
    --data data_n10_N1altconf3_refmac_n1000 --epochs 200
```

---

## generate_protein.py Pipeline (Current)

Each sample runs in an isolated `tempfile.mkdtemp()` directory.

1. Random 20-residue AA sequence (natural UniProt frequencies)
2. Build backbone via `build_n2c.awk` (Ramachandran-sampled phi/psi)
3. Build side chains via `build_side.awk` (Ponder-Richards chi rotamers)
4. Add random water molecules; set CRYST1 P1 40×40×40; centre in box; randomise B factors
5. `phenix.geometry_minimization` → `minimized.pdb`
6. `jigglepdb.awk` × 2 seeds → two conformers merged as altloc A/B
7. `phenix.geometry_minimization` → `truth_full.pdb`
8. `gemmi sfcalc truth_full.pdb` → `truth.mtz`
9. Build `refme.mtz` (F=|FC|, SIGF=0.02·|FC|), run `uniqueify`, simulate missing/never-collected reflections
10. Extract single conformer (altloc A) as `starthere.pdb`, with alt-conf cluster replaced by centroid atom
11. `refmac5` 20 cycles on `starthere.pdb` → `refmacout.mtz` (occupancy refinement)
12. Merge: apply refmac-refined x/y/z/occ/B **only to centroid atom**; all other atoms keep ground-truth values → `partial.pdb`
13. Convert MTZ columns → CCP4 `.map` files

**Merge strategy is critical:** saving `refmacout.pdb` wholesale corrupts non-disordered atoms' B factors. Only the centroid atom gets refmac values.

---

## Key Tool Paths

```python
BUILD_N2C  = Path('/home/jamesh/projects/git/build_pdb/build_n2c.awk')
BUILD_SIDE = Path('/home/jamesh/projects/git/build_pdb/build_side.awk')
PHENIX_GM  = Path('/programs/phenix-2.0-5936/phenix_bin/phenix.geometry_minimization')
REFMAC5    = Path('/programs/ccp4-8.0/bin/refmac5')
JIGGLEPDB  = SCRIPT_DIR / 'jigglepdb.awk'
UNIQUEIFY  = 'uniqueify'   # on PATH via CCP4 environment
```

---

## Current Training Status (as of 2026-04-16)

Best checkpoint: `checkpoints_n10_N1altconf2_5/` — epoch 81, val loss **0.00182** (warm-start source for current runs).

Active runs comparing altloc cluster sizes:

| Run | Dataset | Best val |
|-----|---------|----------|
| `checkpoints_n10_N1del_altconf2refmac` | N1del + altconf2_refmac | 0.00305 (~ep 21) |
| `checkpoints_n10_N1del_altconf3refmac` | N1del + altconf3_refmac | **0.00261** (~ep 10) |
| `checkpoints_n10_N1del_altconf5refmac` | N1del + altconf5_refmac | 0.00335 (~ep 9) |

3-conf converges fastest at early epochs.

---

## Known Gotchas

- **Blank chain from randompdb.com**: omit `chain` keyword from refmac occupancy group entirely.
- **uniqueify path**: `uniqueify` (not the full CCP4 path) — it must be on `$PATH` via CCP4 env.
- **B factor randomisation**: happens after phenix.reduce adds H (step 4), not before, so H atoms inherit the parent heavy-atom B factor.
- **simulate_missing_data**: two non-overlapping categories — `missing` (F/SIGF→NaN, row kept with FreeR_flag) and `never_collected` (rows deleted entirely). `uniqueify` must run before this.
