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
| `run_untangler_1aho.py` | Wrapper to run Untangler ILP on 1AHO varconf structure via SLURM |
| `slurm_untangler_1aho.sh` | SLURM batch script for Untangler (xds partition, 100 CPUs) |
| `condense_bb.py` | Backbone(+SS)-only maximin condensation sweep (gt48 → various k); refines with refmac5-newhess |
| `condense_bb_varconf.py` | Per-residue varconf condensation on a multi-chain conformer model with backbone-tuned thresholds |
| `condense_singlechain.py` | Per-residue altloc maximin on a single-chain altloc PDB (e.g. deconform output); refines with refmac5-newhess + occupancy refinement |

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

**Scoring metrics:**
- **rmsd_e**: lower = better agreement with ground-truth structure factors
- **wE (weighted energy)**: `molprobify_runme.com` output — sum of weighted phenix geometry deviations (ramalyze, rotalyze, omegalyze, cbetadev, etc.). Lower = better geometry. Started at ~161.5, now ~154.
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

---

## Backbone Conformer Condensation

Three scripts implement increasingly sophisticated conformer-count reduction on 1AHO models, all refining a backbone+disulfide-only model against Fc data computed from the same backbone+SS atoms.

| script | input format | conformer selection | starting atoms |
|--------|--------------|--------------------|----------------|
| `condense_bb.py` | multi-chain (e.g. gt48.pdb, 48 chains) | global maximin → flat k chains | ~18,700 (gt48 backbone) |
| `condense_bb_varconf.py` | multi-chain | per-residue maximin via `build_varconf_pdb` (uses chain A as residue/atom template) | as input |
| `condense_singlechain.py` | single-chain altloc (e.g. deconform output) | per-residue altloc maximin (no template, no duplicates) | as input |

**Pipeline (`condense_singlechain.py`, the current best):**

1. Strip input PDB to backbone + cysteine CB/SG (preserves disulfide network) → `bbss.pdb`
2. Compute Fobs MTZ as `FP = |Fc(bbss)|`, `Fpart = 0`, `FreeR_flag` from `1aho/refme_minRfree.mtz` (the actual diffraction free-R split)
3. For each residue: heavy-atom max-deviation across altlocs → look up target k from a threshold table → maximin-pick that many altlocs
4. `reoccupy.awk` (`~/Develop/`) renormalizes per-residue occupancies to sum=1
5. 4 rounds of refmac5-newhess weight-snap (NCYC 10 each at wm 0.01→0.1→1→10→0.5)
6. Rounds ≥3 generate occupancy-group refmac keywords via `~/Develop/refmac_occupancy_setup.com` (instead of the built-in `generate_occ_groups`)

**Threshold sets** are defined in `condense_singlechain.py:THRESHOLD_SETS` (top of the file). To list available names from the CLI: `ccp4-python condense_singlechain.py --help` shows them under `--threshold-set`. Each set maps heavy-atom max-deviation (Å) to per-residue k:

| set | (dev, k) brackets | bottom k |
|---|---|---|
| `floor1lean` | (0.5,1) (0.8,2) (1.2,3) (1.8,5) (2.5,7) (99,10) | 1 |
| `floor1` | (0.4,1) (0.6,2) (0.8,3) (1.2,5) (2.0,8) (99,12) | 1 |
| `floor2` | (0.6,2) (0.8,3) (1.2,5) (2.0,8) (99,12) | 2 |
| `default` | (0.6,2) (0.8,4) (1.0,6) (1.5,8) (2.5,12) (99,16) | 2 |
| `lean` | (0.6,1) (0.8,2) (1.0,4) (1.5,6) (2.5,8) (99,12) | 1 |
| `midrich` | (0.6,3) (0.8,5) (1.0,7) (1.5,10) (2.5,14) (99,20) | 3 |
| `rich` | (0.6,4) (0.8,6) (1.0,8) (1.5,12) (2.5,16) (99,24) | 4 |

**Best results from `1aho/deconform_under20_best_0025.pdb` (22-altloc deconform output)** — backbone+SS, 4 weight-snap rounds, `--cys-floor 3`:

| set | atoms | R rd 4 | Rfree rd 4 |
|---|---|---|---|
| floor1lean | 945 | 6.69% | 6.82% |
| floor1 | 1281 | 5.57% | 5.80% |
| **floor2** | **1381** | **4.40%** | **4.71%** |
| midrich | 1966 | 3.16% | 3.37% |

`floor2` is the sweet spot: under 5% Rfree budget, ~30% smaller atom count than `midrich`.

### floor2 recipe (Lawrencium)

```bash
# On Lawrencium (lr6 partition, pc_als831 account, lr_normal QOS).
# Source code already includes refmac5-newhess override and reoccupy/occ-setup wiring.

cd /path/to/claude_CNN
source cluster.sh && setup_ccp4

# Verify required external tools are reachable from the compute node
ls ~/Develop/reoccupy.awk ~/Develop/refmac_occupancy_setup.com
which refmac5-newhess  # path is hardcoded in condense_singlechain.py:
                       # /programs/ccp4-8.0/bin/refmac5-newhess — adjust if cluster differs

sbatch --partition=lr6 --account=pc_als831 --qos=lr_normal \
       --ntasks=1 --cpus-per-task=1 --mem=8G --export=ALL \
       --job-name=floor2 --output=floor2_%j.log \
       --wrap="cd \$SLURM_SUBMIT_DIR && \
               ccp4-python condense_singlechain.py \
                 --threshold-set floor2 --n-rounds 4 --cys-floor 3 \
                 --singlechain-pdb 1aho/deconform_under20_best_0025.pdb \
                 --outdir 1aho/condense_singlechain"

# Single ~10-minute job. Result lands in 1aho/condense_singlechain/floor2/result.json
ccp4-python -c "import json; d=json.load(open('1aho/condense_singlechain/floor2/result.json')); print(d['rounds'][-1])"
```

**Notes for the Lawrencium counterpart:**
- The hardcoded refmac5-newhess path (`/programs/ccp4-8.0/bin/refmac5-newhess`) in `condense_singlechain.py` works on the original cluster only. On Lawrencium, point it at the equivalent newhess binary or revert to plain refmac5 in `ccp4-9` if newhess is unavailable.
- `~/Develop/reoccupy.awk` and `~/Develop/refmac_occupancy_setup.com` are NFS-mounted from the user's home, so they should be available on Lawrencium compute nodes.
- The `--cys-floor 3` keeps the disulfide-bonded cysteines at ≥3 conformers, which buys ~0.2–0.7 % Rfree at +60–100 atom cost — worth it for SS-rich structures like 1AHO (4 disulfides).
- Threshold sets are defined inline in `condense_singlechain.py:THRESHOLD_SETS`; add new ones there.

---

## Untangler (ILP Conformer Label Optimisation)

`untangler/` is a git submodule (branch `2_conformer_challenge_solution` of github.com/Phoelionix/Untangler). It resolves tangled altloc assignments by ILP, scoring via wE (phenix geometry) + Rfree from phenix.refine.

**Run:**
```bash
sbatch slurm_untangler_1aho.sh --pdb 1aho/conf3norm_fitGT48.pdb \
    --altloc-subset-size 3 --max-runs 5
```

**Input requirements (critical):**
- Single-chain altloc format: all protein atoms in chain A with altloc letters (A, B, C, ...)
- **Uniform altloc coverage**: every disordered residue must have ALL altloc letters present — no partial sets (e.g., B+C without A). Fill gaps by copying the available altloc line with the altloc column changed.
- **Aromatic side chains required**: `relabel_ring` crashes on backbone-only structures. Use full-atom PDB.
- `altloc_subset_size` must equal the number of conformers, or the highest-letter conformer is left unpaired.

**`1aho/conf3norm_fitGT48.pdb`** (working 3-conformer input):
- 2865 atoms = 955 atoms × 3 altlocs (A, B, C), fully normalized coverage
- Derived from `under20_fitGT48.pdb` altlocs A/B/C; missing altlocs filled by copying nearest available

**Submodule patches** (applied directly in `untangler/`, not committable from parent):
- `LinearOptimizer/ConstraintsHandler.py` lines 641–644: changed duplicate clash-distance `assert` to a warning (f-string bug accessed wrong dict key during assert message formatting)
- `LinearOptimizer/Solver.py` lines ~1144 and ~1316: changed `if lp_problem.sol_status==LpStatusInfeasible` → `if lp_problem.sol_status != 1` — PuLP returns `sol_status=0` (no solution found) for infeasible next-best problems, not `-1` (infeasible), so the original guard was never triggered
- `untangle.py` line ~1503: replaced `shutil.rmtree(tmp_refine_subdir)` with `subprocess.run(['rm', '-rf', tmp_refine_subdir], check=False)` — `shutil.rmtree` raises `OSError: ENOTEMPTY` on NFS after unlinking files (NFS directory-cache race); `rm -rf` is more robust

**SLURM routing:** `Untangler.refine_shell_file` is set to `untangler/Refinement/Refine_slurm.sh`, which submits each phenix.refine call to the `refmac` partition via `sbatch --parsable`, polls until the job exits, and cancels the SLURM job via `trap "scancel $SLURM_JOB_ID" EXIT` if the wrapper is killed (e.g. by Python's subprocess timeout). This prevents zombie jobs from piling up in the refmac queue.

**Skip cross-conformer subprocess calls:** set BOTH `NonbondConstraint: 0` AND `ClashConstraint: 0` in `weight_factors`. Setting only one still runs both cross-conformer scripts (see `skip_nonbonds` logic in ConstraintsHandler.py).

**Performance note:** `GenerateHoltonData.sh` → `untangle_score_weighted.csh` → `phenix.molprobity` runs single-threaded on the full model every ILP loop. With 22 conformers (~10,000 atoms) this is hours per call; the 3-conformer model (~955 atoms) is fast.

**For Untangler to make swaps**, the input must be deliberately tangled (conformer labels not yet optimized). A well-refined structure scores 0 high-tension connections each ILP loop and makes no moves (`"moves": {}` in every `xLO-toFlip_*.json`). Score improvements in that case come entirely from the phenix.refine cycles Untangler runs between ILP iterations, not from any relabelling.

**Completed run on `conf3norm_fitGT48.pdb`** (5 loops, `--altloc-subset-size 3`):
- No altloc swaps in any loop — input was already optimally labelled
- Rfree improved 18.62 → 18.51% purely from refinement; wE 87.7 → 84.0
- Final output: `untangler/output/conf3norm_fitGT48_loopEnd4.pdb` (same conformer assignments, slightly better geometry)

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
- **shutil.rmtree ENOTEMPTY on NFS**: Python's `shutil.rmtree` calls `os.rmdir` after unlinking all files; on NFS the server may still report the directory non-empty due to caching, raising `OSError: [Errno 39]`. Use `subprocess.run(['rm', '-rf', path])` instead for cleanup of refinement tmp dirs.
- **Flood water occupancy**: `flood_occ = FLOOD_LINE_K / sqrt(n_flood)` is always positive. The `--flood-occ` CLI flag is unchecked — passing a negative value would propagate to gemmi and write negative occupancies. Normal code paths (including `--vary-flood`) can never produce negative occ.
- **Space group domain gap for CNN inference on real 1AHO data**: Training uses synthetic P1 40×40×40 Å cells (one molecule, no symmetry). 1AHO is P2₁2₁2₁ with 4 ASU copies per unit cell. Applying the trained CNN to real 1AHO maps means each 60×60×60 patch sees portions of 2–3 symmetry-related molecules simultaneously — a pattern the network was never trained on. The cross-Patterson channel implicitly encodes inter-ASU vectors but the CNN has no way to exploit that. Expect degraded performance if ever applied to non-P1 experimental data.
