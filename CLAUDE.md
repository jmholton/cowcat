# Claude CNN — Electron Density Reconstruction

## Project Goal

Train a 3D U-Net to reconstruct ground-truth electron density (Fo) from phased maps that contain systematic errors. The current focus is on **alternate conformations**: the partial model has a single centroid atom where the truth has a disordered cluster; refmac refines the centroid's occupancy/position against the true multi-conf density. The CNN learns to recover the true multi-Gaussian density from the blurred/biased 2Fo-Fc, Fo-Fc, and Fc maps.

---

## Key Files

| File | Purpose |
|------|---------|
| `generate_protein.py` | **Primary data pipeline** — random protein backbone + altloc conformers + flood waters + refmac |
| `generate_simple.py` | Simple O-atom pipeline — N random O atoms in configurable cell/SG, 1 missing, refmac |
| `generate_1aho.py` | 1AHO-specific pipeline — jiggle real protein, flood waters, sfcalc, refmac, CCP4 maps |
| `generate_data.py` | Oldest pipeline — random O atoms, atom deletion, no refmac (kept for reference) |
| `model.py` | UNet3D: 4 input channels, 1 output, base_features=32, circular padding, 5.84M params |
| `train.py` | Training loop: heteroscedastic NLL loss, AdamW, CosineAnnealingLR, DataParallel, --accum-steps |
| `dataset.py` | `ElectronDensityDataset`, `PackedDataset`, `make_splits` / `make_splits_multi` |
| `pack.py` | Packs `sample_NNNNN/` dirs into `X.npy`/`Y.npy`/`S.npy` for fast mmap loading |
| `jigglepdb.awk` | Displaces atom positions to generate alternate conformers |
| `explore_1aho_fusion.py` | Conformer scoring, rebuild, refmac utilities for the 1AHO iterative pipeline |
| `swapscan_varconf.py` | Chain-letter swap optimisation: random/geo-targeted trials, refmac NCYC=50, wE scoring |
| `swapscan_to_samples.py` | Convert swapscan `refmacout.mtz` files → `sample_NNNNN/` dirs for UNet3D training |

---

## Architecture

- **Map channels (input)**: 2Fo-Fc (FWT/PHWT), Fo-Fc (DELFWT/PHDELWT), Fc (FC_ALL_LS/PHIC_ALL_LS), cross-Patterson
- **Target (Y)**: `znorm(truth.map − fc.map)` — true phased difference map (what Fo-Fc would be with perfect phases)
- **Scale (S)**: `log(std(truth − fc))` — predicted by a scalar head; used in heteroscedastic NLL loss
- **Model**: UNet3D, base_features=32, circular padding, 5.84M params; fully convolutional (handles any grid size)
- **Grid**: 60×60×60 for 40×40×40 Å / dmin=2.0; 144×128×96 for 45.9×40.7×30.1 Å / dmin=0.965 (sample_rate=3.0)
- **Note**: CCP4 maps cover the full P1 unit cell regardless of space group (4 ASU copies for P 21 21 21)

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

## generate_protein.py Pipeline (Current — v2)

Each sample runs in an isolated `tempfile.mkdtemp()` directory.

1. Random 64-residue AA sequence (natural UniProt frequencies)
2. Build backbone via `build_n2c.awk` (Ramachandran-sampled phi/psi)
3. Build side chains via `build_side.awk` (Ponder-Richards chi rotamers)
4. Add random water molecules; set CRYST1 P 21 21 21 45.9×40.7×30.1; centre in box; randomise B factors
5. `phenix.geometry_minimization` → `minimized.pdb`
6. Self-refine B factors: refmac NCYC=20 against own SFs → realistic correlated B factors
7. `jigglepdb.awk` × N seeds (byB mode, shift_scale=0.5, through-bond correlated) → N conformers, each independently minimized
8. Combine into single-chain altloc format: **chain A** (protein, altlocs A–E, occ=1/N each), **chain S** (waters, altlocs A–E, occ=1/N each) → `multiconf.pdb` / `truth_full.pdb`
9. Inject flood waters (chain F, ±occ) into `truth_full.pdb` to simulate ghost solvent peaks
10. `_sfcalc_with_bulksolv(truth_full.pdb)` → `truth.mtz` (includes H via phenix.reduce, cavenv bulk solvent)
11. Build `refme.mtz` (F=|FC|, SIGF=0.02·|FC|), `uniqueify`, simulate missing/never-collected reflections
12. `step8_build_mixed_model`: reads N altloc chains → collapses to starthere.pdb:
    - All residues kept as altlocs A–E at occ=1/N (ALTLOC_DIST_THRESHOLD=0)
    - max_confs=min(N,3) — so with N=5, only 3 altlocs in starthere (2 conformers "missing")
    - Waters → chain S with 5-way altlocs at occ=1/N
    - H atoms stripped (refmac MAKE HYDR A adds them back)
    - `refmac_occupancy_setup.com` generates per-residue occ groups
13. `step9_refmac`: 2 rounds of NCYC=20, occupancy refinement via `refmac_occupancy_setup.com`
14. `step10_convert_maps`: truth.map, 2fofc.map, fofc.map, fc.map, **truediff.map** (= truth−fc, the training target)

**Key design decisions:**
- `truth_full.pdb` uses single-chain altloc format matching `refmacout.pdb` (both chain A + chain S) for direct visual comparison in Coot/PyMOL
- Flood waters (chain F) are injected AFTER all geommin steps to avoid false clash detection
- Waters in starthere.pdb start at occ=1/N so refmac can refine toward true occupancy
- `--flood-avoid-fullocc` flag is accepted but currently has no effect (legacy; `_generate_flood_waters` always avoids all existing atoms)

**v2 generation parameters (data_protein_v2_s*):**
```bash
ccp4-python generate_protein.py --submit --nsamples 1000 \
    --outdir data/data_protein_v2_s0 \
    --cell 45.9 40.7 30.1 --dmin 0.965 --spacegroup "P 21 21 21" \
    --nresidues 64 --nwaters 30 \
    --n-altlocs 5 --n-flood 5000 --flood-occ 0.08 \
    --altloc-swaps-per-res 5 --seed 0 \
    --partition lr6 --account pc_als831 --qos lr_normal
# flood-occ 0.08 → Rfree ≈ 10.7% from flood contribution alone
# altloc-swaps-per-res 5 = Poisson(5) random pairwise altloc label swaps per residue
# no --weight-matrix (refmac auto weighting)
```

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

## Current Training Status (as of 2026-05-14)

Loss is heteroscedastic NLL — negative values are normal and not comparable to earlier MSE-based runs.

| Checkpoint | Dataset | Best val | Notes |
|-----------|---------|----------|-------|
| `checkpoints_n10_N1altconf2_5/` | protein n1000 | 0.00182 (ep 81) | MSE loss; warm-start source |
| `checkpoints_1aho_n1000v3/` | 1AHO n=1000 | **-0.7376** (ep 18) | NLL loss; best to date; 2-GPU, 100 ep |

**Datasets available (packed):**

| Path | N | Grid | Notes |
|------|---|------|-------|
| `data/data_1aho_n1000` | 991 | 144×128×96 | Best proven dataset; X.npy=6G |
| `data/data_simple_v2_s0` | 1000 | 144×128×96 | 20 O-atoms, P 21 21 21 |
| `data/data_simple_v2_p1_s0` | 1000 | 144×128×96 | 20 O-atoms, P1; X.npy=17G |
| `data/data_simple_60_s0` | 1000 | 60×60×60 | 20 O-atoms, P1, 40Å, dmin=2.0 |
| `data/data_protein_v2_s0` | 962 | 144×128×96 | protein v2; **training unstable — see below** |

**Training instability finding (2026-05-14):**
- All runs from Apr 11–Apr 19 used MSE loss (positive, train≈val, healthy)
- NLL loss introduced Apr 20: first runs fine, but some val spikes appeared early
- 1AHO NLL runs (Apr 23–24) trained well: train and val tracked together
- **protein v2 runs (May 14) broken**: ep0 train=3815, val=169,262 — extreme values in packed arrays
- Root cause under investigation: likely extreme cross-Patterson values from P 21 21 21 symmetry with flood waters (spikey Patterson) and/or data pipeline artifacts
- Simple P1 runs also show train/val divergence (train→-3, val→40+) due to batch_size=1 + heteroscedastic NLL overconfidence collapse

**Training instability fixes to try:**
- Use `--accum-steps 8` to simulate larger batch size (added to train.py)
- Diagnose protein_v2 X.npy for extreme values before retraining
- Consider dropping cross-Patterson channel for P 21 21 21 data (spikey Patterson)
- 60×60×60 simple dataset available as clean baseline

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

## 1AHO Varconf Optimisation (swapscan_varconf.py)

`swapscan_varconf.py` randomly trials chain-letter swap moves on a varconf PDB, runs refmac NCYC=50, and measures Rfree + rmsd_e (RMSD between |FC_ALL_LS| and LS-scaled |Fgt|) + wE (molprobify geometry score, lower=better).

**Optimisation trajectory** (each built on the previous, starting from gt48.pdb / varconf_opt3.pdb):

| File | Description | rmsd_e | Rf | wE |
|------|-------------|--------|-----|-----|
| `1aho/varconf_opt1.pdb` | Baseline (gt48 condensed to k=32) | 2.3522 | — | — |
| `1aho/varconf_opt2.pdb` | sc H→L ARG62 | 2.3522 | — | — |
| `1aho/varconf_opt3.pdb` | +2 more swaps | 2.3334 | 0.0398 | — |
| `1aho/varconf_opt4.pdb` | spr_0p12 t189 (8 swaps, from swapscan_spr_fine) | 2.3187 | 0.0398 | 161.933 |
| `1aho/varconf_opt5.pdb` | targeted t13: GLU32+LYS50+PHE15+VAL10+TYR42+ARG18 | 2.3395 | 0.0398 | 155.495 |
| `1aho/varconf_opt6.pdb` | swapscan_opt5 spr_0p06 t28: LYS2+ARG18+THR27+GLU24 | 2.3431 | 0.0395 | 154.349 |
| `1aho/varconf_opt7.pdb` | targeted: HIS64 sc A→L + ARG62 sc A→O | — | 0.0373 | 148.928 |
| `1aho/varconf_opt8.pdb` | geo-submit opt7_geo: LYS30 sc I→J | — | — | ~144.6 |
| `1aho/varconf_opt9.pdb` | geo-submit opt8_geo: ASP8 sc C→O | — | — | ~144.0 |
| `1aho/varconf_opt10.pdb` | geo-submit opt9_geo: ASP8 pep N→P | — | — | 142.921 |
| `1aho/varconf_opt11.pdb` | geo-submit opt10_geo: GLU32 sc E→F | — | — | 142.375 |
| `1aho/varconf_opt12.pdb` | geo-submit opt11_geo: GLU24 pep D→L | — | — | 142.062 |
| `1aho/varconf_opt13.pdb` | geo-submit opt12_geo: CYS63 sc B→E | — | — | 141.521 |
| `1aho/varconf_opt14.pdb` | targeted opt12: CYS63sc+ASP9pep simultaneous | — | — | 141.303 |
| `1aho/varconf_opt15.pdb` | geo-submit opt13_geo: ASP9 pep Q→S (sequential better than simultaneous) | — | — | 141.153 |

**Scoring metrics:**
- **rmsd_e**: lower = better agreement with ground-truth structure factors
- **wE (weighted energy)**: `molprobify_runme.com` output — sum of weighted phenix geometry deviations (ramalyze, rotalyze, omegalyze, cbetadev, etc.). Lower = better geometry. Started at ~161.5, now ~143.5.
- `molprobify_runme.com` is at `~/Develop/molprobify_runme.com`; takes ~5–10 min per run on the 10101-atom 23-chain varconf structure. Runs inline in every trial (no separate rescore needed).

**Standard workflow per round (Lawrencium):**
```bash
# 1. Submit sweep from current opt PDB (4 spr values × 250 = 1000 jobs)
ccp4-python swapscan_varconf.py --submit \
    --pdb 1aho/varconf_optN.pdb --fobs 1aho/refme.mtz --truth 1aho/gt48.mtz \
    --outdir 1aho/swapscan_optN --sweep-spr 0.063,0.094,0.125,0.188 \
    --n-trials 250 --ncyc 50 \
    --partition lr6 --account pc_als831 --qos lr_normal

# 2. Collate when done — prints ΔRf table, wE table, AND compatible combos
ccp4-python swapscan_varconf.py --collate --outdir 1aho/swapscan_optN

# 3. Save winner as next opt PDB
cp 1aho/swapscan_optN/spr_0pXX/trial_NNNNN/swap.pdb 1aho/varconf_opt{N+1}.pdb

# 4. Update targeted_submit() in swapscan_varconf.py:
#    - set base_pdb to varconf_opt{N+1}.pdb
#    - replace building blocks + combos with groups from combo-finder output
# Then submit:
ccp4-python swapscan_varconf.py --targeted-submit --outdir 1aho/swapscan_targeted_opt{N+1} \
    --partition lr6 --account pc_als831 --qos lr_normal

# 5. Collate targeted results; save best as next opt PDB; repeat
```

**Geo-targeted workflow (--geo-submit, used from opt7 onward):**

Runs `phenix.ramalyze` + `phenix.rotalyze` on the base PDB, restricts the swap catalog to residues with OUTLIER geometry, and submits only those (~2,700–3,900 trials vs 11,218 exhaustive). Yields one best-ΔwE swap per round. Uses `refmacout.pdb` from the winning trial as the next round's base PDB.

```bash
# Submit geo-targeted scan from current opt PDB
ccp4-python swapscan_varconf.py --geo-submit \
    --pdb 1aho/varconf_optN.pdb --fobs 1aho/refme.mtz --truth 1aho/gt48.mtz \
    --outdir 1aho/swapscan_optN_geo --ncyc 50 \
    --partition lr6 --account pc_als831 --qos lr_normal

# Collate when done — winner is the trial with best ΔwE
ccp4-python swapscan_varconf.py --collate --outdir 1aho/swapscan_optN_geo

# Save winner as next opt PDB (use refmacout.pdb, not swap.pdb)
cp 1aho/swapscan_optN_geo/trial_NNNNN/refmacout.pdb 1aho/varconf_opt{N+1}.pdb

# Repeat
```

- `--include-allowed` flag also adds Ramachandran Allowed residues to the scan target
- Outlier count decreases each round as geometry improves (13 → 10 → 10 → 9 over opt7–opt10)

**Compatible-combo detection** (automatic in `--collate`):
- After each per-subdir collation, `find_compatible_combos()` scans the top-20 wE trials
- Two trials are compatible if no swap residue in one is within 1 sequence position of any in the other
- Skipped automatically if trials have >8 swaps (too complex to interpret)
- Prints top-5 compatible pairs and top-3 compatible triples ranked by combined ΔwE
- Use the printed groups to populate `targeted_submit()` for the next targeted round

**Other CLI modes:**
```bash
# Rescore old runs that predate molprobify integration
ccp4-python swapscan_varconf.py --rescore-submit --outdir 1aho/swapscan_spr_fine \
    --partition lr6 --account pc_als831 --qos lr_normal
```

**Lawrencium notes:**
- Partition `lr6`, account `pc_als831`, QOS `lr_normal`; no special time limit needed (lr_normal default is generous)
- molprobify calls phenix tools — verify phenix is on PATH after `setup_ccp4` on compute nodes before first run
- MaxArraySize 1001: our 251-trial batches fit in a single array job with no splitting needed
- Home directory (`~/Develop/molprobify_runme.com`) is NFS-mounted and accessible from compute nodes

**Gotchas:**
- wE and rmsd_e are in tension: high-spr trials improve wE but hurt rmsd_e/Rf. Low-spr (0.063–0.094) gives best balance.
- `targeted_submit()` has hardcoded base PDB and building blocks — update both when advancing to a new opt PDB.
- spr=2× previous optimal swaps is a good rule of thumb for the sweep range upper bound.
- `--geo-submit` winner: copy `refmacout.pdb` (not `swap.pdb`) as the next opt PDB — refmacout.pdb is the fully refined structure after the swap.
- CCP4 is already set up in the shell environment on Lawrencium — no need to re-source setup scripts before running swapscan commands.

---

## Known Gotchas

- **pack.py target**: Y.npy stores `znorm(truth - fc)` (the difference map), not `znorm(truth)`. S.npy stores `log(std(truth - fc))`. Matches `ElectronDensityDataset` exactly. pack.py never reads `metadata.json`.
- **CCP4 map header size**: gemmi writes 1344-byte headers for P 21 21 21 (1024 standard + 320 symmetry records). `_load_map` uses `offset = filesize - 4*n` so it's header-size agnostic and always correct.
- **Map grid vs metadata grid_shape**: gemmi's `transform_f_phi_to_map` rounds up to FFT-friendly numbers (e.g. 144×128×96), not the `round(cell/dmin*3)` estimate stored in metadata. Read grid from map header, not metadata.
- **Full unit cell in maps**: `transform_f_phi_to_map` returns the full P1 unit cell regardless of space group. For P 21 21 21, the network sees 4 ASU copies of the protein.
- **Cross-Patterson spikiness in P 21 21 21**: the full-cell map has 4 ASU copies → cross-Patterson accumulates Harker cross-terms → much spikier than P1. May cause extreme X.npy values and training instability.
- **Heteroscedastic NLL overconfidence collapse**: with batch_size=1, the model quickly learns to set log_var→-3 (clamp floor) on training samples. Val loss explodes when it's wrong with high confidence. Fix: use `--accum-steps 8` to simulate larger batch.
- **--vary-flood overrides --n-flood**: when `--vary-flood` is set, n_flood is drawn randomly from log-uniform [FLOOD_NF_MIN=700, FLOOD_NF_MAX=4000] ignoring --n-flood. Use `--n-flood N --flood-occ O` without `--vary-flood` for deterministic flood parameters.
- **--flood-avoid-fullocc is a no-op**: accepted by argparse and passed through but never used in `_generate_flood_waters` (which always avoids all existing atoms).
- **SLURM bash arrays**: bash array indexing `${arr[$ID]}` fails silently on compute nodes. Use Python to decode task ID: `N=$(python3 -c "print([200,500,1000][$SLURM_ARRAY_TASK_ID])")`.
- **Data generation directories**: keep ≤1000 samples per directory to avoid Lustre metadata slowdown.
- **DataParallel + pin_memory deadlock**: `pin_memory=True` with DataParallel and forked DataLoader workers causes a hang. `train.py` disables pin_memory when `n_gpus > 1`.
- **Einsteinium module loading**: `module load --force ml/pytorch/2.3.1-py3.11.7-mf` requires `Core` in MODULEPATH and the `--force` flag; CUDA/cuDNN warnings on login nodes are harmless.
- **refmac cwd vs relative paths**: always resolve XYZIN and HKLIN to absolute paths; refmac runs with `cwd=tmpdir` so relative paths fail.
- **cavenv SYMM**: must be integer space group number (e.g. 19 for P 21 21 21), not HM string. Use `gemmi.find_spacegroup_by_name(sg).number`.
