# 🔥🧠 Physics-Informed Neural Operator (PINO) for Forest-Fire Spatial Mapping — India (2000–2022)

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
- **Headline results**: term-ablation AUC 0.602 (diffusion-only) → 0.941 (full CDR);
  temporal generalization (leave-years-out) AUC 0.897; all three of the reference
  paper's variable-understanding analyses (permutation importance, response curves,
  Jackknife) reproduced, all five independent methods converging on near-total
  elevation dominance. Full numbers: `Design_and_Paper/CDR_PINN_Full_Paper_Draft.md`.

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
