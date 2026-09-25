# 🔥🧠 Physics-Informed Neural Operator (PINO) for Forest-Fire Spatial Mapping — India (2000–2022)

<!-- AUDIT-UPDATE-2026-09-25 -->
> ### Audit update (2026-09-25)
> This repository's step was recalculated independently from the raw data in a full end-to-end audit.
> **Corrected results, reproduction checks and audit code: [`AUDIT_2026-09-25.md`](AUDIT_2026-09-25.md)** and `audit_2026-09-25/`.
> Earlier text below is kept for the record (it also remains in the git history). Statements superseded by the audit:
>
> - **Term ablation (0.602 / 0.924 / 0.940)**: historical single-seed, fixed-budget runs, with no no-physics arm. Under one validated protocol with 3 seeds: **no physics 0.945, diffusion 0.924, diff + adv 0.939, full 0.939** (Track A). No physics configuration beats no physics on any track.
> - **Physics vs no physics**: confirmed and extended (3 seeds, paired tests). Full − none: −0.006 (A, p < 0.02 for every seed), CI including 0 (B1, B2), −0.011 (B3).
> - **CDR-PINO historical numbers** (0.9398, 0.7510 ± 0.0182, 0.6187 ± 0.0680, 0.8960) reproduce bit-exactly. Under the unified protocol (3 seeds), full CDR scores 0.939 / 0.719 / 0.570 / 0.893 (A / B1 / B2 / B3), and Track A and B1–B3 were separate models trained under different protocols.
> - **Elevation dominance**: specific to CDR-PINO's 7-covariate model. With the full predictor set, removing terrain changes AUC by −0.0001; Biswas also ranks elevation low (2.4%).
<!-- AUDIT-UPDATE-2026-09-25 -->

## Current results — unified protocol (audit 2026-09-24/25; code updated 2026-09-25)

**These are the numbers to cite.** Paper numbers come only from the project-level
`results/FINAL_MANUSCRIPT_NUMBERS.md`.

**How they were produced.** Every CDR-PINO configuration was retrained under **one**
protocol:
- AdamW, `ReduceLROnPlateau`, early stopping on validation AUC;
- validation carved from whole spatial blocks;
- 3 seeds (42/43/44);
- 4 physics configurations: none, diffusion, diffusion + advection, full CDR.

The code is [`cdr_pinn/run_unified_protocol.py`](cdr_pinn/run_unified_protocol.py). The
result files are in [`CDR_PINN_Data/unified/`](CDR_PINN_Data/unified/): per-run
predictions (`pred_*.npz`), `track_*.json`, `partitions.npz` and `term_magnitudes_*.json`.
Checkpoints are git-ignored because they total about 820 MB.
[`cdr_pinn/analyze_unified.py`](cdr_pinn/analyze_unified.py) rebuilds every table below
from those files; its output goes to `CDR_PINN_Data/unified/analysis/`. Figures come from
`generate_figures.py --map` and go to `CDR_PINN_Data/unified/figures/`.

**ROC-AUC (mean ± SD over seeds × folds; 12 km cells):**

| Track | No physics | Diffusion | Diff + adv | **Full CDR** | Forest cells, full (no physics) |
|---|---:|---:|---:|---:|---:|
| A — random cells | **0.945** ± 0.002 | 0.924 ± 0.005 | 0.939 ± 0.001 | 0.939 ± 0.002 | 0.897 (0.910) |
| B1 — spatial blocks (3 folds) | 0.724 ± 0.010 | 0.642 ± 0.013 | **0.730** ± 0.004 | 0.718 ± 0.007 | 0.701 (0.702) |
| B2 — held-out regions (6) | **0.614** ± 0.021 | — | — | 0.570 ± 0.007 | 0.569 (0.635) |
| B3 — held-out years, leak-free label | **0.904** ± 0.001 | 0.886 ± 0.006 | 0.894 ± 0.001 | 0.893 ± 0.003 | 0.833 (0.848) |

**What the results show:**
- **No physics configuration improves on the same network without physics.** Paired
  tests (`PAIRED_physics_vs_nophys.csv`):
  - Track A, full − none = −0.006, DeLong p < 0.02 for every seed.
  - B1 and B2: block-bootstrap CIs include 0.
  - B3: −0.011.
- **Classical models on the same cells, splits and 7 covariates beat CDR-PINO on every
  track** (`PAIRED_cdr_vs_classical_same_cells.csv`). RF scores 0.980 (A), 0.974 (B1) and
  0.959 (B2).
- **B3 falls below covariate-free persistence baselines** on the same cell-months:
  - climatological fire frequency 0.908;
  - seasonal frequency 0.930;
  - RF with monthly covariates and month 0.972.

  The earlier claim that temporal generalisation is CDR-PINO's advantage is withdrawn.
- **The trained field does not behave as the PDE describes.** The PDE residual is
  71–616× the mean |∂u/∂t| (`TERM_MAGNITUDES_summary.csv`). Advection carries 83–98% of
  the right-hand side and diffusion < 0.4%. In practice the field acts as a static
  susceptibility map.
- **Historical numbers reproduce bit-exactly** (0.9398 / 0.7510 ± 0.0182 /
  0.6187 ± 0.0680 / 0.8960). However, Track A and B1–B3 were separately trained models
  under different protocols, and the historical B3 label leaked the test period. They are
  superseded, and the older text below is kept for the record.

**Reproduce:**
```bash
cd cdr_pinn
python analyze_unified.py          # tables from the saved result files (CPU, minutes)
python generate_figures.py --map   # figures + CDR-PINO susceptibility map (CPU)
python run_unified_protocol.py     # full retrain of all 102 runs (GPU, ~4.5 GPU-hours)
```
Use the `cdr_pinn_env` interpreter (torch + CUDA). The published results used the monthly
stack with SHA-256 `ea7828a5…`. `build_monthly_stacks.py` now contains two corrections:
- VI_Quality masking for 2007-03/04;
- exact days-in-month for precipitation.

Rebuilding the stack therefore changes it slightly, and every CDR-PINO result would then
need to be re-run.

---

**Historical description (pre-audit; superseded where the audit block above says so).**

**This repository's headline contribution is the CDR-PINN**: a convection-diffusion-
reaction (CDR) partial differential equation over a latent fire-susceptibility
field, solved by a physics-informed Fourier neural operator (PINO, Li et al. 2023)
and trained on 22 years (2000–2022) of real, monthly-resolved fire observations
across India. Each governing-equation term maps to a distinct fire-behavior
mechanism (vegetation/moisture-driven diffusion, terrain-driven advection,
human-ignition-driven reaction), and global-in-time well-posedness of the equation
is proven, not assumed.

- **Implementation**: [`cdr_pinn/`](cdr_pinn/) — spectral differential operators,
  FNO/PINO backbone with 3 physics heads, adaptive loss balancing, monthly data
  pipeline, term-ablation/generalization-track/Jackknife/causal-weighting/
  curriculum-learning/validation-split experiment scripts, all GPU-verified with
  real results (see result JSONs in `CDR_PINN_Data/`, checkpoints excluded via
  `.gitignore` as regeneratable).
- **Design and full manuscript**: [`Design_and_Paper/`](Design_and_Paper/) — the PDE
  construction and proofs for each term, the consolidated architecture/training
  design, the full paper draft, methodology section, novelty/comparison argument
  against Biswas, Mahato & Joshi (2025), and reviewer-facing Q&A clarifications.
- **Headline results**: term-ablation AUC 0.602 (diffusion-only) → 0.9398 (full CDR,
  final standard train/val/test protocol, superseding the earlier 0.941/0.9406
  pre-standard-protocol figures still visible in this study's raw experiment log);
  temporal generalization (leave-years-out) AUC 0.8960; all three of the reference
  paper's variable-understanding analyses (permutation importance, response curves,
  Jackknife) reproduced, six independent methods converging on near-total elevation
  dominance. A direct physics-vs-no-physics comparison on all four generalization
  tracks (not just the random split) found no accuracy benefit from the physics
  constraint anywhere — noise-level on the spatial-block track, a real cost on
  leave-one-region-out and leave-years-out — a disclosed negative result, not
  hidden. Full numbers: `Design_and_Paper/CDR_PINN_Full_Paper_Draft.md`.

---

## Also in this repository: Step 8 — Physics-Informed Fire-Risk Model (PINN vs. baseline ladder)

The plain-monotonicity PINN ladder below **predates and is superseded by the
CDR-PINN above** for this study's actual novel contribution — kept here as an
honest, disclosed-negative-result baseline (the physics-informed monotonicity
penalty did not measurably beat a same-capacity plain MLP).

**Notebook:** [`Step8_PINN_FireRisk_Model.ipynb`](Step8_PINN_FireRisk_Model.ipynb)
**Kernel:** `firerisk-anaconda3` (Python 3.12.7, base `C:\Users\Admin\anaconda3\python.exe`)

## What this is

> **Renumbered twice.** On 2026-08-17: the Integrated_Analysis steps this README
> refers to were renumbered to Step 5 (integration) and Step 6 (model) for a cleaner
> paper narrative — training was genuinely the last step in both execution order and
> documentation labels at the time. On 2026-08-19: a new Step 5 (Terrain &
> Accessibility Analysis) was inserted between FLDAS and Integration, bumping
> Integration to Step 6, the Model to Step 7, and this PINN step to Step 8 (was
> Step 7). This file's prose below has been updated to match both times; the results
> and conclusions are unaffected (label-only change) — this step was already built
> with real results (run 2026-08-08/09) before the project started tracking it as a
> numbered step at all.

Step 7 trains a Random Forest on Step 6's integrated feature table and gets
ROC-AUC 0.9676. This step goes further: it trains a genuine Physics-Informed Neural
Network (PINN) on the same data and compares it against a 5-model ladder —
**Logistic Regression → Random Forest → XGBoost → plain MLP → PINN** — where the PINN
is the plain MLP's *identical architecture* plus one physics-derived input feature
and a physics-informed loss penalty. That design isolates what the physics
specifically contributes, rather than just showing "a neural net beats other models,"
which any sufficiently-tuned NN could claim.

## The physics constraint

True wildfire-susceptibility PINNs are essentially absent from the literature; the
closest same-data-shape precedent is a **landslide-susceptibility PINN**
(static per-pixel tabular table → binary hazard label — exactly this project's shape):
Dahal & Lombardo (2024), *JGR: Machine Learning and Computation*, arXiv:2407.06785.
Its key finding sets the framing here: the physics term's benefit showed up in
**spatial-generalization robustness** (AUC under spatial CV: 0.69 physics-informed vs.
0.58 plain NN), not average in-sample AUC (both ~0.87 under random CV). Since Step 7's
RF is already at 0.9676 on a random split, chasing a higher *average* AUC is a losing
framing — the defensible claim tested here is whether the physics constraint makes the
model generalize better to geographically unseen regions.

The physics equation comes from Rodrigues et al. (2024), *Agricultural and Forest
Meteorology* 346 — a mechanistic inverse-exponential relationship between dead
fine-fuel moisture and Vapor Pressure Deficit (VPD), VPD from the standard Tetens
psychrometric formula. This project's feature table only has **anomaly/trend**
climatic variables (no per-timestamp absolute temperature/RH), and the label is a
whole-period aggregate — so the constraint is reformulated in **anomaly/direction
space**: a "dryness proxy" built from standardized FLDAS anomaly + Mann-Kendall-trend
features, with signs **fixed by the Tetens/Clausius-Clapeyron VPD derivative**
(warmer/drier anomalies and trends push the proxy up), not fitted or learned. It's
enforced via a PyTorch-autograd monotonicity penalty: predicted fire-risk (as a logit)
should not *decrease* as this proxy increases.

## The 5-model ladder

| # | Model | Notes |
|---|---|---|
| 1 | Logistic Regression | `class_weight="balanced"` — floor for linear separability |
| 2 | Random Forest | Same hyperparameters as Step 7, retrained fresh here for a self-contained comparison table |
| 3 | XGBoost | `tree_method="hist", device="cpu"`, early stopping on a validation carve-out |
| 4 | Plain MLP | `52→128→64→32→1`, LayerNorm+ReLU+Dropout, `BCEWithLogitsLoss` |
| 5 | PINN | Identical to #4 + 1 physics input (dryness proxy) + autograd monotonicity penalty, `total_loss = data_loss + λ·physics_penalty` |

`λ` (`LAMBDA_PHYS`) is chosen from `{0, 0.01, 0.1, 1.0}` by a small ablation scored on
a **spatial** holdout (not random-split AUC) — consistent with the whole framing.

## Two-track evaluation

- **Track A — random 80/20 split**: identical to Step 7's split
  (`RANDOM_STATE=42`, stratified, median-fill on the full table pre-split — inherited
  from Step 7 as a disclosed, deliberate choice for direct comparability). All 5 models.
- **Track B1 — spatial block CV**: 2°×2° lon/lat grid blocks, `GroupKFold`, per-fold
  median-fill/scaling on the train portion only (the methodologically clean version —
  different from Track A's inherited shortcut). All 5 models.
- **Track B2 — leave-one-region-out**: `India_State_Boundary.shp` has 37 polygons and
  **no attribute table** (no `.dbf` anywhere in the repo, not even inside the source
  `.zip`) — there are no state names to key on. States are clustered by polygon
  centroid into KMeans-based geographic regions instead of run individually. RF, MLP,
  and PINN only (the three the headline claim needs).

**Compute-budget note (disclosed, not silent):** the full grid at the plan's original
fold/state counts (5-fold spatial CV, ~30+ individual states) would run for several
hours. Track B1 uses 3 folds (not 5) and Track B2 clusters into 6 regions (not 37
individual states) to keep a complete run tractable end-to-end. Track A — the
headline, directly-comparable-to-Step-7 numbers — uses the full model budget
throughout (200-tree RF, up to 50 NN epochs with patience 7).

## Results (run 2026-08-08, 64.5 min total wall time, zero errors)

### Track A — random 80/20 split (full budget)

| Model | ROC-AUC | Avg. Precision | F1@0.5 | Train time |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9460 | 0.5069 | 0.4534 | 9.6s |
| Random Forest | 0.9676 | 0.6765 | 0.5494 | 186s |
| XGBoost | **0.9678** | 0.6740 | 0.5179 | 43s |
| Plain MLP | 0.9614 | 0.6279 | 0.4808 | 265s |
| PINN (λ=0.1) | 0.9613 | 0.6278 | 0.4828 | 404s |

Fresh RF retrain matches Step 7's 0.9676 almost exactly (0.96759 vs. 0.9676). On
random-split accuracy, tree ensembles (RF/XGBoost) edge out both neural models
slightly; PINN and plain MLP are statistically indistinguishable here (Δ=0.0001) —
expected, since the physics term isn't designed to help average in-sample accuracy.

### Track B1 — spatial block CV (3 folds, mean AUC)

| Model | Mean AUC | vs. Track A |
|---|---:|---:|
| Logistic Regression | 0.9396 | −0.0064 |
| Random Forest | 0.9459 | −0.0217 |
| XGBoost | 0.9492 | −0.0186 |
| Plain MLP | 0.9499 | −0.0115 |
| PINN | 0.9494 | −0.0119 |

### Track B2 — leave-one-region-out (6 KMeans regions; RF/MLP/PINN)

| Model | Mean AUC | vs. RF |
|---|---:|---:|
| Random Forest | 0.8721 | — |
| Plain MLP | 0.8896 | +0.0175 |
| PINN | 0.8870 | +0.0149 |

### Seed-robustness check (`Step8b_PINN_Seed_Robustness_Check.ipynb`, run 2026-08-09)

The single run above left the PINN-vs-MLP question unresolved (deltas of
0.001-0.003 AUC, noise-level for one seed). A follow-up notebook reused Step 8's
*exact* data split, spatial folds, and regions (bit-identical, verified before
running) and retrained **only PlainMLP and PINN** across 5 random model-training
seeds (Track A/B1) / 3 seeds (Track B2, the most expensive per-seed), then computed
a bootstrap 95% CI on the PINN-minus-MLP AUC delta per track:

| Track | Mean Δ (PINN − MLP) | 95% CI | Significant? |
|---|---:|---:|---|
| A — random split | −0.00004 | [−0.00015, +0.00005] | **No** |
| B1 — spatial block CV | +0.00012 | [−0.00036, +0.00049] | **No** |
| B2 — leave-one-region-out | +0.00051 | [−0.00566, +0.00372] | **No** |

**Conclusion: no statistically significant difference between the PINN and a
same-capacity plain MLP, on any evaluation track.** All three 95% CIs include zero.
See `Model_Outputs/PINN_vs_MLP_Seed_Robustness_Summary.csv` (per-seed deltas) and
`Model_Outputs/PINN_vs_MLP_Seed_Robustness.png`.

### Honest read of these numbers

**The clearest finding isn't physics-specific**: both neural models generalize
better than Random Forest under leave-one-region-out (~+1.5-1.8 AUC points in the
single Step 8 run), and degrade less than RF under spatial-block CV. **The
physics-informed monotonicity penalty, as implemented here, does not produce a
measurable improvement over a same-capacity plain MLP** — this is now established
with proper multi-seed statistical testing, not just a single-run observation. The
λ-ablation's own internal comparison (Step 8) did show a small, non-monotonic effect
(spatial-holdout AUC 0.9504→0.9518 from λ=0 to λ=0.1), but that signal doesn't
survive being tested against seed-to-seed variance at production scale.

**What this means for the paper**: this is a genuine, rigorously-tested negative
result for this specific physics formulation — worth reporting as such rather than
reframing around a smaller, cherry-picked comparison. Two honest paths forward: (1)
publish the RF-vs-NN spatial-generalization finding as the paper's main contribution,
with this physics-constraint test reported as a disclosed negative result (multi-seed
statistical rigor on a negative finding is itself a defensible, relatively rare
contribution in PINN literature, which skews toward reporting only positive results);
or (2) revisit the physics constraint's design — a soft monotonicity penalty on an
engineered anomaly-space proxy may simply be too weak a signal at this data scale,
where RF/XGBoost/MLP already reach ~0.96-0.97 AUC from the raw features alone. A hard
architectural constraint (in the style of Dahal & Lombardo 2024, an intermediate
physical-transform layer rather than a soft loss penalty) is the more promising
untried alternative if a physics-specific claim is still wanted for the paper.

## How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=firerisk-anaconda3 --ExecutePreprocessor.timeout=7200 "Step8_PINN_FireRisk_Model.ipynb"
```

Requires Step 6's `Integrated_FireRisk_Pixels.parquet` to already exist
(`Integrated_Analysis/Integrated_Outputs/`) and `LST_analysis/India_State_Boundary.shp`
to be present. Neither is copied into this folder — both are read from their existing
locations. Note the longer timeout (7200s, not Step 7's 1800s) — 5 models across 3
evaluation tracks takes meaningfully longer than Step 7's single-model ~13 minutes.

## Outputs

```
Model_Outputs/
├── Model_Comparison_RandomSplit.csv           # Track A: 5 models x {AUC, AP, F1, train_sec}
├── Model_Comparison_SpatialBlockCV.csv         # Track B1: per-fold AUC, 5 models
├── Model_Comparison_LeaveOneStateOut.csv       # Track B2: per-region AUC, RF/MLP/PINN
├── PINN_Lambda_Ablation.csv                    # lambda selection sweep
├── ROC_PR_Curves_AllModels.png
├── RandomSplit_vs_SpatialCV_AUC_Comparison.png # headline "physics helps spatial robustness" figure
├── LeaveOneStateOut_Map.png
├── PINN_Physics_Diagnostic.png                 # partial dependence + gradient-sign histogram
├── XGBoost_Feature_Importance.png
├── Computational_Cost_Reproducibility_Report.json
└── Checkpoints/                                 # not tracked
    ├── mlp_state_dict.pt
    └── pinn_state_dict.pt
```

## Citation

- Biswas, S. et al. (2025). *[see other steps' notebook headers for full reference]*
- Dahal, A., & Lombardo, L. (2024). Physics-informed neural networks for spatial
  hazard prediction. *JGR: Machine Learning and Computation*. arXiv:2407.06785 —
  architecture precedent for a static-tabular hazard-susceptibility PINN.
- Rodrigues, M., Resco de Dios, V., Sil, Â., Cunill Camprubí, À., & Fernandes, P. M.
  (2024). *Agricultural and Forest Meteorology*, 346 — the VPD/dead-fuel-moisture
  relationship underlying the dryness-proxy physics constraint.

## License

No license has been chosen yet for this repository's code.
