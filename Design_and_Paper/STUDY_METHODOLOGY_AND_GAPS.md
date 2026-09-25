# CDR-PINO Forest-Fire Susceptibility Study — Methodology and Gap Inventory (audited)

**Version 2, updated 2026-09-25 after the full end-to-end audit.** Every number below was recalculated
from raw data or reproduced by re-running the project's code. Sources are in `results/` (start at
`results/FULL_METHODOLOGY_AUDIT.md`), and paper-ready values are in
`results/FINAL_MANUSCRIPT_NUMBERS.md`. The pre-audit version of this file (compiled 2026-09-23) is in
`_Archive_Unwanted_2026-09-25/pre_audit_document_snapshots/STUDY_METHODOLOGY_AND_GAPS.md`.

"v1" means the historical pipeline as it was run (reproduced exactly). "v2" means the corrected
feature set and protocols built in the audit. v1 artefacts were not modified.

---

## PART I — METHODOLOGY (audited)

### 0. Study area, period, grid

- **Domain:** the dissolved `India_State_Boundary.shp` (EPSG:3857 set, then reprojected to 4326).
  State vs country vs GADM boundary sensitivity is quantified in
  `results/recalculated/R1_fire_labels/R1_report.json`.
- **Period:** 2000-11-01 – 2022-12-15 (266 months).
- **Grid:** EPSG:4326, 3,641 × 3,504, **1/120° (≈0.93 km)**; the earlier description as 0.01° was wrong.
  India mask 4,184,671 pixels, NDVI-valid 4,161,009, analysis population 4,160,768. The forest
  population (forest fraction 2001 > 0) is 1,197,538 pixels.
- **Fire-point rasterization (corrected):** `col = floor((lon − c)/a)`, `row = floor((lat − f)/e)`,
  where c, f are the top-left **edge** coordinates. The v1 rule, `round(...)` against the edge,
  displaced 405,723 of 541,545 points (74.9%) into a neighbouring pixel; the Jaccard overlap of the
  v1 and correct fire rasters is 0.50.

### Step 1 — Fire points (reproduced exactly)

| Stage | Rows |
|---|---:|
| Raw archive | 2,804,373 |
| India bbox | 2,801,347 |
| India polygon | 1,599,471 |
| Study period / dedup (lon, lat, date) | 1,599,471 / 1,599,466 |
| Forest LULC filter (same-year LCCS map) | **541,545** (identical point set) |

- **Forest cover of India** (13 forest codes): **18.3–19.3%** across 2000–2022. The earlier
  "9.86–10.43%" figure was computed over the rectangular download subset, which includes
  neighbouring countries and ocean.
- **Confidence < 30:** 4.29% (23,236 points).
- **Type ≠ 0:** 0.21% (type 2: 1,122; type 3: 2).
- **Filtering either** changes RF AUC by < 0.001 (null result).
- **Burned-area validation** (recomputed from the saved annual series; raw MCD64A1 not re-extracted):
  - All land cover: r = 0.9149, ρ = 0.8350.
  - Forest-masked: r = 0.9345, ρ = 0.7846 (n = 23).
  - The Biswas Fig. 7d values used for comparison were digitized from a chart.
- **Annual counts vs Biswas et al.** (derived from their published shares): +0.51% to +2.43%,
  r = 0.99996.
- **Fire-affected pixels:** 268,411 with the correct rule, versus 270,074 in v1. Prevalence is 6.45%
  over all pixels and 22.4% over forest pixels.

### Step 2 — NDVI (MOD13A3.061, 266 months)

- **Reproduced:** F1–F4 and F7 exactly.
- **2007-03 and 2007-04 have no pixel-reliability layer.** v1 used them unfiltered; v2 filters them
  with the MODLAND bits of the VI_Quality layer (F1 r = 0.999996).
- **QA Good-only vs Good + Marginal:** F1 r = 0.989, but Good-only loses 39.7% of pixel-months
  (vs 8.2%).
- **F3 anomaly mean is degenerate.** With a 2001–2020 baseline it equals the out-of-baseline residue.
  Dropped in v2.
- **F5 residual mean is identically about 0**, and F4 ≈ F1 (r = 0.99994). Both dropped in v2.
- **Trend test:**
  - v1 applied Mann–Kendall to the 2×12-MA trend series (lag-1 autocorrelation 0.975), so the test
    is invalid.
  - v2 uses **Seasonal Kendall + BH-FDR**: 3,552,278 greening / 72,305 browning pixels, median Sen
    slope +0.0026 yr⁻¹.
  - The v1 counts (147,206 / 3,731,210) reproduce but are not valid.
- **CVSI:** k\* = 8 under historical, corrected and training-only labels (MI 0.0125). Kept in v2.
- **Moran's I:** the v1 value, 0.8322, had 67% of its cells filled with row means outside India. With
  India cells only (8×8 block means), **I = 0.9456** (z = 494, p = 0.001). LISA was recomputed the
  same way.
- **θ\*:** 0.535 (v1 labels), 0.524 (corrected), 0.537 (training only). The θ\* indicator is dropped
  in v2 because it is a step function of NDVI mean fitted on labels; removing it changes AUC by 0.0000.

### Step 3 — LST (MOD11A2.061)

- **Composites:** 1,013 complete. 2001-06-26 and 2016-02-18 are incomplete and were excluded silently.
- **Reproduction:** MK τ reproduced exactly; anomaly means reproduced under the historical
  composite-weighted climatology.
- **v2 Seasonal Kendall + FDR:**
  - Day: **cooling** in 2,435,163 pixels (Sen −0.050 °C/yr).
  - Night: **warming** in 2,080,747 pixels (+0.024 °C/yr).
  - DTR: **narrowing** in 3,273,301 pixels (−0.077 °C/yr).
- **v1 raw-MK counts** (1,063,120 → 393,838; 234,318 → 17,935; 2,545,287 → 2,290,051) reproduce, but
  the test ignores seasonality. The directions hold; the significance counts do not.
- **DTR trend:** now a feature (v2); this closes gap A3.
- **Resolution relative to Biswas's MOD11C3 (0.05°):** 98.5% (day) and 99.4% (night) of the
  climatological-level variance is retained, and the effect on model AUC is about 0.

### Step 4 — FLDAS climate and land cover

- **Reproduction:** all 14 v1 FLDAS features reproduce exactly (r = 1.000000). The units were verified
  from the file attributes.
- **Specific humidity is a per-pixel feature** in v1 (`fldas_qair_anomaly`, `fldas_qair_mk_tau_monthly`)
  and in v2 (`v2_qair_level`, `v2_qair_sk_tau`). Derived relative humidity is an additional predictor.
- **Anomaly means are degenerate.** v1 used time-mean anomalies for all 7 variables. v2 replaces them
  with 2001–2020 climatological **levels** (the form Biswas et al. used) plus Seasonal-Kendall τ.
- **Trends:** the v1 statement "air-temperature trend is multiple-testing noise" (636 → 0) is an
  artefact of MK on seasonal data. With Seasonal Kendall, 9,988 of 29,056 pixels are FDR-significant
  (7,620 decreasing), and specific humidity increases in 26,373 pixels.
- **Land cover:**
  - The v1 22-class fractions come from the **2020** map, inside the label window. v2 uses 2001.
  - The measured post-fire reclassification signal is weak: 2001→2020 forest change is +0.019 at fire
    pixels vs +0.018 at non-fire forest pixels.

### Step 5 — Terrain and accessibility

- **Reproduced:** elevation, Horn slope and aspect (circular mean). Horn vs Zevenbergen–Thorne gives
  r = 0.9998 and ΔAUC 0.000.
- **Slope scale:** slope computed from a 0.25° DEM (Biswas's model resolution) has a mean of 0.29°,
  against 5.72° here (r = 0.70). The two are different quantities.
- **The −46.9 m minimum** is at the Neyveli open-cast lignite mines (11.56°N, 79.50°E), most likely a
  real excavation surface rather than an SRTM artefact.
- **Distances:** reproduced (r ≥ 0.999999). Validated against exact geodesic distances (n = 3,000):
  bias −0.15 / −0.22 / −0.14 km, RMSE 0.31 / 0.65 / 0.32 km (roads / railways / waterways), uniform
  across latitude.
- **National means** (analysis pixels): 5.60 / 37.44 / 6.68 km.
- **Fire coincidence** (corrected label):
  - Slope: +108%.
  - Road distance: −41.5%.
  - Waterway distance: −63.9%.
  - Railway distance: +1.4%.

### Step 6 — Integrated table

- **v1** (`Integrated_FireRisk_Pixels.parquet`): 4,161,009 × 61 columns, **57 features**. The earlier
  "55" was from before specific humidity was wired in.
- **v2** (`results/recalculated/FEATURE_TABLE_v2.parquet`): **55 features**. It uses levels and
  Seasonal-Kendall trends, 21 land-cover fractions from 2001 plus forest fraction 2001, aspect as
  sin/cos, the corrected label, and training-row median imputation.

### Step 7 — Classical models

Split: stratified 65/15/20 (v2), with the test set touched once and thresholds chosen on validation.

| Model | All pixels AUC / AP | Forest pixels AUC / AP |
|---|---|---|
| RF v2 | 0.975 / 0.720 | **0.897** / 0.721 |
| RF, Biswas-15 predictors | 0.970 / 0.692 | 0.885 / 0.698 |
| MaxEnt v2 (150k rows) | 0.967 / 0.663 | 0.866 / 0.664 |
| RF v2, 2° blocks (CDR-PINO fold geometry) | 0.959 | 0.838 |
| RF v2, leave one region out | 0.935 | 0.782 |
| RF v2, unseen years | 0.965 | 0.872 |
| Persistence null, unseen years | 0.807 | 0.744 |

- **v1 reproductions:** RF 0.9704 / 0.7011, MaxEnt 0.9598 / 0.6275, GroupKFold spatial CV
  0.9459 / 0.9527 / 0.9508 (all exact).
- **MaxEnt training size** (gap A2):

  | Training rows | AUC, all / forest | Fit time |
  |---|---|---|
  | 50k | 0.964 / 0.855 | 150 s |
  | 150k | 0.967 / 0.867 | 1,130 s |
  | 500k | 0.969 / 0.872 | 7,303 s |

  Fit time grows as about n^1.6. The RF–MaxEnt gap is not a subsampling artefact.
- **Calibration:** RF v2 ECE 0.064 (all) / 0.219 (forest); MaxEnt 0.015 / 0.054. No calibrated
  uncertainty estimate was established.

### Step 8 — CDR-PINO

**Implemented residual:** r = (u_{t+1} − u_t) − D∇²u + v·∇u − ρσ(u)(1 − σ(u)), evaluated at the
midpoint state.

- **Grid and population:** 256 × 256, about 12 km, 22,542 valid cells, 7 covariates.
- **Where the code and the text disagree:**
  - "Fisher–KPP" is a misnomer: the reaction acts on the logit.
  - The BC penalises |∇u|² on the boundary ring, not ∂u/∂n.
  - The IC loss is identically zero.
  - The loss-weight update is an EMA of mean-norm/norm, not the multiplicative formula in the text.
  - There is no size-matched negative sampling; the data loss is pos-weighted BCE over all training
    cells.
- **All historical results reproduce bit-exactly:** Track A 0.9398 / AP 0.9223; B1 0.7510 ± 0.0182;
  B2 0.6187 ± 0.0680; B3 0.8960. However, Track A and B1/B2/B3 were separate models trained under
  different protocols. The historical ablation's full-CDR checkpoint was overwritten by the Track A
  checkpoint.

**Unified protocol** (AdamW, ReduceLROnPlateau, early stopping on validation AUC, block-carved
validation for spatial tracks, 3 seeds); mean test AUC over all cells:

| Track | No physics | Diffusion | Diff + adv | Full CDR | RF on the same cells and covariates | Null |
|---|---|---|---|---|---|---|
| A (prev. 42%) | **0.945** | 0.924 | 0.939 | 0.939 | **0.980** | — |
| B1 spatial blocks | 0.724 | 0.642 | 0.730 | 0.719 | **0.974** | — |
| B2 regions | 0.614 | — | — | 0.570 | **0.959** | — |
| B3 unseen years (prev. 2.46%) | 0.904 | 0.886 | 0.894 | 0.893 | 0.972 (with month) | 0.908 / **0.930** |

- **Physics vs no physics (paired):** full − no physics is −0.006 on Track A (p < 0.02 for every seed)
  and −0.011 on B3 (replicated across seeds). On B1 and B2 the CIs include 0.
- **The B3 leakage fix changes AUC by +0.0006.**
- **Physics costs** 2.3–2.4× training time.
- **The trained field does not behave as a physical solution:** it is static in time (median
  |∂u/∂t| ≤ 0.002 per month), the PDE residual is 74–616× |∂u/∂t|, and diffusion is ≤ 0.3% of the
  right-hand side.

---

## PART II — GAP INVENTORY (status after the audit)

Full matrix: `results/audit/GAP_CLOSURE_MATRIX.csv`.

### II.A — Items from the previous version

| Gap | Previous claim | Status now |
|---|---|---|
| A1 | "Specific humidity should read relative humidity (derived)" | **Gap was incorrect.** Specific humidity is per-pixel; RH is an additional predictor. Do not apply the proposed fix |
| A2 | MaxEnt 150k subsample undisclosed | Quantified (sample-size sensitivity); disclose |
| A3 | DTR trend not in stack | Closed in v2 |

### II.B–II.D — Previously listed gaps

| Gap | Status |
|---|---|
| C1 / C2: FIRMS confidence and type | Tested; null (< 0.001 AUC) |
| C3: boundary sensitivity | Quantified |
| C4: burned-area script not tracked | **Out of date:** the scripts are tracked (commit 5b2c316) |
| C5: NDVI QA and climatology | Tested |
| C6: NDVI ablation | Partial (trend group +0.006 forest AUC; label-derived features 0.000) |
| C7: independent LST validation | Open |
| C8: MOD11C3 difference | Resolution effect negligible; product difference disclosed |
| C9: climate cross-check | Open; the precipitation-source difference is documented |
| C10: net LW coincidence | Open (low priority) |
| C11: static LULC vs dynamic climate | Documented; not leakage |
| C12: terrain algorithm and distance validation | Closed |
| C13: −46.9 m | Reinterpreted (Neyveli mines) |
| C14: dense grid and NaN | Quantified: CDR-PINO fills NaN with **0** (0 m elevation for 0.38% of cells; 2007-03/04 NDVI entirely 0; 270 cells without dryness) |
| C15: checkpoint identity | Closed: different models and protocols |
| C16: single-seed ablation | Closed (3 seeds) |

### II.E — New gaps found by the audit (all fixed in v2 or quantified)

1. Half-pixel label shift. Fixing it raises AUC by +0.006 (all) / +0.011 (forest).
2. In-window 2020 land-cover fractions.
3. Degenerate anomaly-mean features.
4. Invalid MK tests.
5. Label-fitted θ\* and CVSI (effect 0.000).
6. Moran's I dominated by filled cells.
7. Forest-mask dominance of all-pixel AUC.
8. B3 label leakage (+0.0006).
9. **B3 below persistence nulls.**
10. **CDR vs classical models on different populations**; the same-cells comparison favours RF on every track.
11. **The trained CDR field is static and violates its PDE.**
12. The equation/text mismatches listed in Step 8.
13. The RF map is not a calibrated probability, and the CDR-PINO map exists only as a PNG.
14. The `cdr_pinn_env` environment was undocumented.

### II.F — Open items (future work; do not claim)

- **Not evaluated:** resolution independence and zero-shot super-resolution.
- **Not implemented:** instance-wise fine-tuning.
- **Not established:** calibrated uncertainty.
- **Not computed:** SHAP (environment incompatibility; permutation importance and PDP were used instead).
- **Not tested:** CDR loss-weight and spectral-mode sensitivity beyond the historical scale-up test.
- **No cross-check exists:** independent LST/climate validation.

### II.G — What must change in the manuscript

See `results/FINAL_MANUSCRIPT_NUMBERS.md` ("Numbers to remove or reword") and
`results/audit/BISWAS_CLAIM_AUDIT.csv`. The headline changes:

1. Withdraw "temporal generalisation strong".
2. Withdraw "each physics term contributes measurable value".
3. Withdraw "beats Biswas (0.9576 vs 0.879)"; use the Biswas-style reimplementation, 0.893 ± 0.009,
   labelled moderately comparable.
4. Replace the "same fold scheme" comparison with the same-cells comparison.
5. Make forest pixels the primary population.
6. Replace all MK and Moran statistics.
7. Correct the forest-cover figure (18.3–19.3%).
8. Correct the physics-text details.
