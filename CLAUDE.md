# Claude CNN — Electron Density Reconstruction

## Project Goal

Train a 3D U-Net to reconstruct ground-truth electron density (Fo) from phased maps that contain systematic errors. The current focus is on **alternate conformations**: the partial model has a single centroid atom where the truth has a disordered cluster; refmac refines the centroid's occupancy/position against the true multi-conf density. The CNN learns to recover the true multi-Gaussian density from the blurred/biased 2Fo-Fc, Fo-Fc, and Fc maps.

---

## Key Files

| File | Purpose |
|------|---------|
| `generate_1aho.py` | Current data pipeline — jiggle 1AHO PDB, flood waters, sfcalc, refmac, CCP4 maps |
| `generate_protein.py` | Older pipeline — random 20-residue backbone + side chains + altloc + refmac |
| `generate_data.py` | Oldest pipeline — random O atoms, atom deletion, no refmac (kept for reference) |
| `model.py` | UNet3D: 4 input channels, 1 output, base_features=32, circular padding, 5.84M params |
| `train.py` | Training loop: heteroscedastic NLL loss, AdamW, CosineAnnealingLR, DataParallel |
| `dataset.py` | `ElectronDensityDataset`, `PackedDataset`, `make_splits` / `make_splits_multi` |
| `pack.py` | Packs `sample_NNNNN/` dirs into `X.npy`/`Y.npy`/`S.npy` for fast mmap loading |
| `preprocess.py` | Pre-computes cross-Patterson channel → `crossp.npy` per sample (run before pack.py) |
| `jigglepdb.awk` | Displaces atom positions to generate alternate conformers |
| `converge_refmac.com` | Wrapper: runs refmac5 N cycles on `starthere.pdb` + `refme.mtz` → `refmacout.mtz` |
| `randompdb.com` | Generates random O-atom structure in a P1 cell |
| `explore_1aho_fusion.py` | Conformer scoring, rebuild, refmac utilities for the 1AHO iterative pipeline |
| `rebuild_iterate.py` | Standalone iterative rebuild loop: score Fo-Fc outliers → rebuild → refmac → repeat |

---

## Architecture: Fixed Parameters

- **Unit cell**: P1, 40×40×40 Å, 90/90/90
- **Resolution**: d_min = 2.0 Å
- **Grid**: 60×60×60 voxels (sample_rate=3.0)
- **Map channels (input)**: 2Fo-Fc (FWT/PHWT), Fo-Fc (DELFWT/PHDELWT), Fc (FC_ALL/PHIC_ALL), cross-Patterson
- **Output**: truth density (FC/PHIC of full model)

---

## Python Interpreters

**Critical — do not mix these up. Rules apply on both clusters:**

- `ccp4-python` — use for `generate_1aho.py`, `generate_protein.py`, and any script importing `gemmi`. Gemmi is only installed here.
- PyTorch python — use for `train.py`, `pack.py`, `dataset.py`, anything using torch.
- Never use the system `python` for project code.

**Original cluster** (`/programs/` paths visible):
- CCP4 python: `ccp4-python` (on PATH after `source /global/home/groups-sw/ac_als831/ccp4-X/bin/ccp4.setup-sh`)
- PyTorch python: `/programs/pytorch/envs/pt/bin/python`
- System `python` is Python 2. The symlink `./python3` points to the base pytorch python (no torch) — use the full `envs/pt` path for training.

**Einsteinium / Lawrencium** (`cluster.sh` handles both):
- CCP4 python: `ccp4-python` via `setup_ccp4` (sources `/global/home/groups-sw/ac_als831/ccp4-9/bin/ccp4.setup-sh`)
- PyTorch python: `python3` after `setup_pytorch`:
  ```bash
  source /etc/profile.d/modules.sh
  export MODULEPATH=$MODULEPATH:/global/software/rocky-8.x86_64/modfiles/Core
  module load --force ml/pytorch/2.3.1-py3.11.7-mf
  ```

---

## SLURM Cluster

Always pass `--export=ALL` to SLURM array jobs so CCP4 environment variables propagate.
`CCP4_SCR` directory must exist on compute nodes: `os.makedirs(os.environ.get('CCP4_SCR', '/tmp'), exist_ok=True)`.

**Original cluster:**
- **Data generation**: `--partition debug`, `--workers 300` (cluster ~400 cores total)
- **Training**: `--partition gpu --gres=gpu:1`

```bash
ccp4-python generate_protein.py --submit --nsamples 1000 \
    --outdir data_n10_N1altconf3_refmac_n1000 --partition debug --max-array 300

srun --partition gpu --gres=gpu:1 \
    /programs/pytorch/envs/pt/bin/python train.py \
    --data data_n10_N1altconf3_refmac_n1000 --epochs 200
```

**Einsteinium / Lawrencium:**
- **Account**: `pc_als831`; **GPU QOS**: `es_normal` (min 16 CPUs); **CPU QOS**: `lr_normal`
- **GPU partition**: `es1` (4× A40 per node); **CPU partition**: `lr6`
- **MaxArraySize**: 1001 — submit ≤1000 samples per array job, one directory per batch
- Use `cd "$SLURM_SUBMIT_DIR"` before sourcing `cluster.sh` — SLURM runs from a spool directory.

```bash
# Generate: one 1000-sample batch per directory (avoids large-dir filesystem slowdown)
ccp4-python generate_1aho.py --submit --nsamples 1000 --seed 100 \
    --outdir data/data_1aho_s100 --partition lr6 --account pc_als831 --qos lr_normal

# Pack (once per dir, before training)
python3 pack.py --data data/data_1aho_s100 --workers 8

# Train
sbatch train.sh --data data/data_1aho_s100 data/data_1aho_s200 \
    --outdir checkpoints_1aho --epochs 200
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

## Current Training Status (as of 2026-04-30)

Loss is now heteroscedastic NLL — values are negative and not comparable to earlier MSE-based runs.

| Checkpoint | Dataset | Best val | Notes |
|-----------|---------|----------|-------|
| `checkpoints_n10_N1altconf2_5/` | protein n1000 | 0.00182 (ep 81) | MSE loss; warm-start source |
| `checkpoints_n10_N1del_altconf3refmac/` | N1del + altconf3_refmac | 0.00261 (ep 10) | MSE loss; 3-conf fastest |
| `checkpoints_1aho_n1000v3/` | 1AHO n=1000 | **-0.7376** (ep 18) | NLL loss; 2-GPU, 100 ep done |

Next: generate 9× 1000-sample batches (seeds 100–900) → pack each → retrain on 9k samples, 4 GPUs.

## 1AHO Iterative Rebuild (varconf_sweep)

`explore_1aho_fusion.py` + `rebuild_iterate.py` implement a post-weight-snap rebuild loop for the 1AHO model: score Fo-Fc density outliers, apply top-N prune/add actions on gt48 conformers, run refmac NCYC=5, repeat.

**k16_old results** (`1aho/varconf_sweep/k16_old/`):
- Starting R=0.046/Rf=0.050 (16-slot weight-snap)
- After 10 rounds: R=0.039/Rf=0.042
- Peak candidate criterion (atom-position sampling of Fo-Fc at absent conformers) added but was always redundant with the regular scorer — peak was always already in the top-N list
- CYS26/CYS48 disulfide never appeared as a candidate; disulfide residues have low per-atom Fo-Fc signal

**Key implementation notes:**
- `run_refmac_quick` must resolve input paths to absolute before calling refmac (refmac runs in a tmpdir via `cwd=`, so relative paths fail)
- `find_map_peak_candidate` samples the DELFWT map at absent conformer atom positions (not the global map maximum, which lands in symmetry copies in P2₁2₁2₁)
- SS-coupled residues (CYS16/36, CYS22/46, CYS26/48, CYS12/63) are added as pairs

---

## Known Gotchas

- **Blank chain from randompdb.com**: omit `chain` keyword from refmac occupancy group entirely.
- **uniqueify path**: `uniqueify` (not the full CCP4 path) — it must be on `$PATH` via CCP4 env.
- **B factor randomisation**: happens after phenix.reduce adds H (step 4), not before, so H atoms inherit the parent heavy-atom B factor.
- **simulate_missing_data**: two non-overlapping categories — `missing` (F/SIGF→NaN, row kept with FreeR_flag) and `never_collected` (rows deleted entirely). `uniqueify` must run before this.
- **Data generation directories**: keep ≤1000 samples per directory to avoid Lustre metadata slowdown. Use separate dirs per batch (e.g. `data_1aho_s100`, `data_1aho_s200`); `make_splits_multi` concatenates them transparently.
- **DataParallel + pin_memory deadlock**: `pin_memory=True` with DataParallel and forked DataLoader workers causes a hang. `train.py` disables pin_memory when `n_gpus > 1`.
- **pack.py target**: Y.npy stores `znorm(truth - fc)` (the difference map), not `znorm(truth)`. S.npy stores `log(std(truth - fc))`. Matches `ElectronDensityDataset` exactly.
- **Einsteinium module loading**: `module load --force ml/pytorch/2.3.1-py3.11.7-mf` requires `Core` in MODULEPATH and the `--force` flag; CUDA/cuDNN warnings on login nodes are harmless.
- **refmac cwd vs relative paths**: `run_refmac_quick` runs refmac with `cwd=tmpdir`; always resolve XYZIN and HKLIN to absolute paths before passing them in, or refmac will fail with "Cannot find input file".
- **P2₁2₁2₁ map global peak in symmetry copies**: `transform_f_phi_to_map` returns the full unit cell (4 ASU copies). The global maximum is often in a non-ASU region. When searching for density near protein atoms, sample the map at known atom positions rather than finding the global peak and searching nearby.
