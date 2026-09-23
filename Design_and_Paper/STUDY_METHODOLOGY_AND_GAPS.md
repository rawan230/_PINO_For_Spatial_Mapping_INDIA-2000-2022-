# CDR-PINO Forest-Fire Susceptibility Study — Complete Methodology and Gap Inventory

**Purpose.** This document has two parts. Part I is the complete, step-by-step methodology
of the underlying eight-stage pipeline, at a finer grain than the condensed version that
fits inside the TGRS manuscript's Section II–IV. Part II is the complete inventory of gaps,
limitations, and open items found across the project's own audit trail — including several
that are **not yet disclosed in `CDR_PINO_TGRS.tex`** and one **factual discrepancy between
the manuscript and the actual pipeline** that should be corrected before submission.

Every number below is traceable to a source file in `D:\Lab_Data_SCL\Forest Fire Modeling
(INDIA)\` — principally `Complete_Methodology_Section.md`, the eight `Step*_Audit_and_
Documentation.md` files, `TGRS_Submission_Plan_and_Reviewer_Guide.md`, `FULL_EXPERIMENT_
LOG.md`, and `CDR_PINN_Study_Clarifications_QA.md`. Nothing here is invented; where a
source document itself flagged a number as unresolved or superseded, that status is carried
over rather than silently picking one value.

Compiled 2026-09-23.

---

## PART I — COMPLETE METHODOLOGY

### 0. Study Area, Period, and the Common Analysis Grid

- **Domain.** The dissolved polygon of `India_State_Boundary.shp` (37 state/UT polygons
  merged via `union_all()`), *not* `India_Country_Boundary.shp`, whose ~60 degenerate
  sliver polygons near the Palk Strait (79–79.5°E, 9–9.3°N) corrupt point-in-polygon
  operations without materially changing the retained point set. Raw shapefile coordinates
  are EPSG:3857 (Web Mercator, metres), explicitly reprojected to EPSG:4326 before use.
- **Period.** 2000-11-01 to 2022-12-15 (266 months), fixed by the upper bound of the
  ESA-CCI/C3S land-cover archive (does not exist past 2022) — a correctness constraint,
  not a convenience, since a fire point cannot be forest-filtered without a matching
  land-cover epoch.
- **Grid.** Established by Step 2 (NDVI): **3,641 × 3,504 pixels, EPSG:4326, ≈0.01°
  (≈1 km)**, the native resolution of MOD13A3.061. Every other source reprojects onto this
  grid — bilinear interpolation for coarse-to-fine (e.g. FLDAS 0.1°→0.01°), area-weighted
  averaging for fine-to-coarse (e.g. ESA-CCI 300 m, SRTM 90 m). After boundary masking and
  NDVI-validity filtering: **4,161,009 in-India pixels** (32.6% of the 12,758,064-cell grid).
- **Fire-point rasterization**, used identically in every step: direct affine pixel lookup,
  not a nearest-neighbour spatial join. With the shear terms $b=d=0$ (confirmed on every
  raster in the pipeline: $a{=}0.01,\ c{=}68.20,\ e{=}{-}0.01,\ f{=}37.09$),
  $$\text{col} = \operatorname{round}\!\left(\frac{\text{lon}-c}{a}\right), \qquad
  \text{row} = \operatorname{round}\!\left(\frac{\text{lat}-f}{e}\right).$$

**Data sources at a glance:**

| Product | Sensor/model | Native res. | Coverage | Role |
|---|---|---|---|---|
| MODIS C6.1 FIRMS active fire | MODIS Terra/Aqua | point | 2000–2022 | Fire-occurrence ground truth (Step 1) |
| ESA-CCI/C3S Land Cover | multi-sensor CCI | 300 m, annual | 2000–2022 | Forest filter (Step 1); 22-class fractions (Steps 4, 6) |
| MOD13A3.061 NDVI | MODIS Terra | 1 km, monthly | Nov 2000–Dec 2022 | 9 features (Step 2); establishes the grid |
| MOD11A2.061 LST | MODIS Terra | 1 km, 8-day | 2000–2022 (1,013 composites) | Day/night LST, DTR (Step 3) |
| FLDAS_NOAH01_C_GL_M.001 | Noah LSM, MERRA-2+CHIRPS | 0.1° (~11 km), monthly | Nov 2000–Dec 2022 (266 files) | 7 climatic variables (Step 4) |
| SRTMGL3 DEM | SRTM radar | 90 m | static | Elevation/slope/aspect (Step 5a) |
| Geofabrik OpenStreetMap | OSM | vector | 2022 snapshot | Distance to roads/rail/water (Step 5b) |

---

### Step 1 — Fire-Point Extraction

Raw MODIS C6.1 FIRMS archive for India: **2,804,373** detections. Sequential filters:

| Stage | Rows remaining |
|---|---:|
| Raw archive | 2,804,373 |
| India bbox pre-filter (68.0–97.5°E, 6.5–37.5°N) | 2,801,347 |
| Exact India polygon clip (`shapely.contains_xy`) | 1,599,471 |
| Study-period clip | 1,599,471 |
| Exact-duplicate removal (lon, lat, acq\_date) | 1,599,466 |
| Forest-LULC pixel filter (per acquisition year) | **541,545** |

The polygon clip removes **1,201,876 points (42.9% of the bbox-passing set)** that lie
inside the rectangle but outside India — chiefly Sri Lanka, Nepal, Bangladesh, Myanmar,
southern Pakistan.

Forest classification uses the 13-code Sannigrahi et al. (2018) set,
$\text{FOREST\_CODES}=\{50,60,61,62,70,71,72,80,81,82,90,100,110\}$, applied **per
detection's own acquisition-year** raster (not one static mask). National forest cover from
the same yearly rasters holds within **9.86–10.43%** across all 23 years, confirming the
filter does not drift with epoch.

**Independent validation** (two series, both against MODIS MCD64A1.061 burned area):
- All-land-cover: $r=0.915$, $\rho=0.835$ ($p<0.0001$, $n=23$) — but this runs 6–10× larger
  in absolute magnitude than Biswas et al.'s forest-scoped Fig. 7d, because it is not
  forest-masked. Use only as the internal extraction-credibility check it was designed for.
- **Forest-masked** re-derivation from raw MCD64A1.061 GeoTIFFs: $r=0.9044$, magnitude gap
  closes from 6–10× to a residual **~4×**, disclosed and not forced to close. Correlates
  *more* tightly against this study's own fire-point counts ($r=0.9345$ vs. $0.9149$) than
  the all-land-cover series — the expected direction, since both measure the same
  population.
- Against Biswas et al.'s own published annual counts: this extraction runs **0.5–2.4%
  higher** across the 20 overlapping years — a small, external validation, not a
  discrepancy.

**Not applied**: FIRMS `confidence`-level filtering or spatial declustering/thinning (4.29%
of the 541,545 points carry `confidence<30`); the `type` field (0.21% carry
`type=2`, "other static land source"). Both fields are read and preserved but never used as
filters — a deliberate, disclosed scope decision (see Part II).

---

### Step 2 — NDVI Feature Engineering

Nine features derived from 266 months of MOD13A3.061 on the full grid.

| # | Feature | Definition | Key measured value |
|---|---|---|---|
| F1 | QA-filtered mean | Temporal mean over QA∈{Good, Marginal} months only | Range $[-0.1894, 0.9679]$ |
| F2 | Climatology | Per-pixel, per-calendar-month mean, 2001–2020 baseline | — |
| F3 | Anomaly | $\delta(t)=\text{NDVI}(t)-\bar\mu^{(m(t))}$, raw departure, not standardized | Modulates $D$ |
| F4/F5 | Trend / residual | Classical 2×12-MA additive decomposition, 13-tap symmetric window | 254/266 months valid |
| F6 | Mann–Kendall $\tau$ | Full pairwise-lag $S$-statistic on the F4 trend series, normal-approx. significance | 147,206 sig. browning; 3,731,210 sig. greening px |
| F7 | CVSI | $\sum_{\text{lag}=1}^{k}\max(-\delta_{t-\text{lag}},0)$, trailing dry-side accumulation | $k^{*}=8$ by max mutual information, $I=0.01257$ nats |
| F8 | LISA / global Moran | Queen contiguity, row-standardized weights, 8×-coarsened grid | Global $I=0.8322$, $z=742.1$ |
| F9 | NDVI–fire breakpoint | Two-regime logistic MLE fit jointly over $(a_1,b_1,a_2,b_2,\theta)$, multi-start Nelder–Mead | National $\theta^{*}=0.535$; per-zone values also fitted |
| F10 | Fire-occurrence raster | 541,545 points rasterized to the NDVI grid | 270,655 distinct burned pixels (2.12% of grid) |

F6/F3/F4's Mann–Kendall formula is reused identically by Steps 3 and 4. F9's Himalayan-zone
threshold once diverged to a non-physical value ($\theta^{*}=-0.613$) from MLE
non-identifiability on a flat likelihood plateau; fixed by bounding $\theta$ to the valid
NDVI range and rejecting multi-start winners leaving one regime with $<1\%$ of the sample.

---

### Step 3 — LST Analysis

MOD11A2.061 8-day composites (1,013 total). $\text{DTR}=\text{LST}_{\text{Day}}-\text{LST}_{\text{Night}}$,
with its own independently computed climatology/anomaly/trend (not derived by differencing
day/night $\tau$). Measured Mann–Kendall $\tau$: DTR $-0.104$ (narrowing), Day $-0.041$,
Night $+0.043$ — consistent with simultaneous day-cooling and night-warming.

**Benjamini–Hochberg FDR correction** applied to all trend tests (~4.17M independent
per-pixel tests otherwise producing many chance false positives):
$$k=\max\{i: p_{(i)}\le (i/m)\alpha\}, \quad \alpha=0.05.$$
Shrinkage is substantial and variable-specific: Day significant pixels fall
1,063,120→393,838 (−63%), Night 234,318→17,935 (−92%), DTR 2,545,287→2,290,051 (−10%).

Fire-coincidence: LST-Day anomaly at fire pixel-months is $+0.75^{\circ}\text{C}$ vs.
$-0.09^{\circ}\text{C}$ grid-wide; LST-Night $+0.31^{\circ}\text{C}$ vs. $+0.02^{\circ}\text{C}$
— anomalously warm conditions, not merely seasonally hot ones, coincide with fire.

---

### Step 4 — FLDAS Climatic Variables and Land Cover

Seven monthly variables from 266 FLDAS Noah-LSM files (0.1°, cropped to 335×315 px, 29,056
in-India pixels):

| Feature | Source | Transform |
|---|---|---|
| Wind speed | `Wind_f_tavg` | m/s, as-is |
| Air temperature | `Tair_f_tavg` | K, as-is |
| Precipitation | `Rainf_f_tavg` | kg/m²/s → mm/month |
| Surface soil moisture | `SoilMoi00_10cm_tavg` | → kg/m² |
| Net longwave radiation | `Lwnet_tavg` | W/m², as-is |
| Specific humidity | `Qair_f_tavg` | **national scalar only — never gridded, see Part II** |
| Relative humidity (derived) | `Qair_f_tavg`, `Tair_f_tavg`, `Psurf_f_tavg` | Magnus-type formula below |

$$e_s(\text{hPa})=6.112\exp\!\Big[\tfrac{17.67\,T_c}{T_c+243.5}\Big],\quad
e(\text{hPa})=\frac{qp}{0.622+0.378q},\quad \text{RH}(\%)=100\cdot e/e_s.$$

Six of the seven (all but specific humidity) get the full climatology/anomaly/Mann–Kendall
FDR treatment. FDR correction is decisive for air temperature: its 636 raw-significant
pixels collapse to **0** after correction — its apparent trend was multiple-testing noise.
Fire-affected pixel-months run drier (precipitation anomaly $-5.6$ mm vs. $+0.7$ mm
grid-wide) and slightly warmer.

**22-class ESA-CCI/C3S land cover**: Level-1 legend (22 global classes), each reprojected
300 m→~1 km by area-averaging. Five largest national mean fractions (2020): rainfed
cropland 35.00%, irrigated cropland 20.94%, broadleaved deciduous tree 8.02%, grassland
5.34%, mosaic natural vegetation 4.72%.

---

### Step 5 — Terrain and Accessibility

Closes 6 of Biswas et al.'s 15 predictor variables (elevation, slope, aspect, distance to
roads/rail/waterways) that the pipeline had zero coverage of before 2026-08-18.

**5a. Terrain.** SRTMGL3 (90 m) mosaicked from four latitude-band requests. Slope/aspect via
GPU-vectorized **Horn's (1981) method** (the algorithm ArcGIS/QGIS/`gdaldem` use internally),
computed at native 90 m *before* resampling to preserve gradient detail, latitude-corrected
for the ~20.2% longitude-spacing compression across India:
$$\frac{\partial z}{\partial x}=\frac{(z_3{+}2z_6{+}z_9)-(z_1{+}2z_4{+}z_7)}{8\Delta x},\qquad
\text{slope}=\arctan\sqrt{z_x^2+z_y^2},\qquad \text{aspect}=\operatorname{atan2}(z_y,-z_x).$$
Measured: elevation $-46.9$ to $8{,}169.0$ m (mean $737.2$ m); slope $0.00$–$77.31^{\circ}$
(mean $5.72^{\circ}$); circular-mean aspect $161.6^{\circ}$ (south-facing). Fire-coincidence:
fires sit at $12.3^{\circ}$ mean slope vs. $5.7^{\circ}$ nationally — **+115% enrichment**.

**5b. Accessibility.** Geofabrik OSM 2022, clipped and class-filtered (roads:
motorway/trunk/primary/secondary/tertiary+links, 884,940 of 10.7M features). Euclidean
distance transform in an India-centred equidistant conic projection (a flat
degree×111 km conversion is wrong by >19% across India's latitude range):
$D(p)=\min_{q\in S}\lVert p-q\rVert_2$. Measured national means: roads 5.69 km, railways
38.28 km, waterways 6.74 km. Fire-coincidence: roads $-40.1\%$, waterways $-64.7\%$
(strongest of the three), railways $+4.8\%$ (essentially no effect, matching Biswas et
al.'s own lowest-contribution ranking for railway distance).

---

### Step 6 — Integrated Feature Alignment

Assembles Steps 1–5 plus one feature built in-step — LULC forest fraction, binarized from
the same 13-code mask at three snapshot years (2001, 2020, 2022), area-averaged onto the
grid.

**2026-08-21 data-leakage fix**: `forest_frac_recent` (2020), `forest_frac_current` (2022),
and their difference feature were dropped. Both non-baseline snapshots fall *inside* the
2000–2022 fire-label window, and post-fire land-cover reclassification (burned forest →
shrubland/agriculture in the next epoch) is documented in the literature — a real
reverse-causality risk. Pre-fix, these three features were the model's **top-3
Gini-importance features** (combined importance ≈0.40). Only `forest_frac_baseline` (2001)
survives. This took the stack from 60→57 bands (55 trainable features; 57 after specific
humidity was wired in, see Part II).

**22-class land-cover as model features** (not just a filter, unlike Biswas et al.):
combined Gini importance 0.1529 (15.3%), concentrated in four classes accounting for 89.3%
of the group's own importance — tree broadleaved deciduous (0.0653), rainfed cropland
(0.0314), irrigated cropland (0.0201), mosaic tree and shrub (0.0197).

Final artifacts: `Integrated_FireRisk_Stack.tif` (57 bands), `Integrated_FireRisk_
Pixels.parquet` (4,161,009 rows × 59 columns → 55 trainable features after dropping
lon/lat/fire\_count/label).

---

### Step 7 — Classical Baselines: Random Forest and MaxEnt

**Split.** Genuine stratified 65/15/20 train/validation/test (`random_state=42`,
two-stage: test carved first, then validation from the remainder) for *hyperparameter
selection*; headline reported numbers use a separate 80/20 stratified split for
comparability with the project's earlier results — standard nested-CV practice.

**Random Forest.** `n_estimators=200`, `class_weight="balanced"`, `max_features='sqrt'`.
Validated search over `max_depth`/`min_samples_leaf` on the 65/15/20 split (train
2,704,655 / val 624,152 / test 832,202 rows):

| Config | Val AUC | Val AP |
|---|---:|---:|
| 20/5 (literature default) | 0.9679 | 0.6772 |
| **25/3 (winner)** | **0.9694** | **0.6934** |

Winner refit and scored once on the 65/15/20 test fold: 0.9698 AUC (ground-truth value
from `rf_hp_search_result.json`; one audit document's prose figure of 0.9701 is a
transcription slip, not the JSON's actual number). On the notebook's own 80/20 split with
the final 57-feature set: **test ROC-AUC 0.9704, AP 0.7011** — the headline figure.
Reproducibility: exact-seed refit differs by ≤$6.66\times10^{-16}$ (floating-point
summation order under multi-threading, not a seeding failure). 5-fold CV: $0.9698\pm0.0002$.

**MaxEnt** (`elapid.MaxentModel`, linear+hinge+product features, matching Biswas et al.).
**Trained on a 150,000-row stratified subsample of the training split** — a full-dataset
fit (≈450,000 rows) did not complete inside a 2-hour cell timeout (see Part II). Validated
search over `beta_multiplier`∈{0.5,1.0,1.5,2.5,4.0}:

| $\beta$ | Val AUC |
|---:|---:|
| 0.5 | 0.9589 |
| **4.0 (winner)** | **0.9592** |

Grid is essentially flat (0.9589–0.9592) — a genuine near-null tuning result. On the 80/20
split, 57-feature set: **test ROC-AUC 0.9598, AP 0.6275**, trailing RF by +0.0106 AUC.

**Spatial-block CV** (matching CDR-PINN's Track B1 exactly): 116 unique 2°×2° blocks,
`GroupKFold(n_splits=3)`, per-fold median imputation from that fold's training data only.
**RF $0.9498\pm0.0035$, MaxEnt $0.9465\pm0.0054$** — both fall only modestly from their
random-split scores (RF −2.1% relative, MaxEnt −1.4%), real evidence of spatial
autocorrelation inflating random-split scores, but not a collapse.

---

### Step 8 — CDR-PINN: Governing Equation, Well-Posedness, Architecture, Training

Full derivation, nomenclature, term justification, well-posedness proof, spherical
differential operators, architecture, and loss function are given in the manuscript
(`CDR_PINO_TGRS.tex`, Sections III–IV) and are not repeated here. This section lists only
the evaluation protocol and the numeric results that section condenses.

**Four generalisation tracks** (validated protocol: 65/15/20 split, AdamW, weight decay
selected by grid search, `ReduceLROnPlateau`, early stopping on validation AUC,
patience=4):

| Track | Description | AUC |
|---|---|---:|
| A | Random 65/15/20 split | **0.9398** (val 0.9351, AP 0.9223) |
| A′ | Track A, seeds 42–44 | $0.9391\pm0.0017$ |
| B1 | $2^{\circ}\times2^{\circ}$ spatial-block CV, 3 folds | $0.7510\pm0.0182$ (folds: 0.7768, 0.7395, 0.7368) |
| B2 | Leave-one-region-out, 6 $k$-means regions | $0.6187\pm0.0680$ (regions: 0.5387, 0.6805, 0.5506, 0.7301, 0.6157, 0.5970) |
| B3 | Leave-years-out (2000, 2008, 2009, 2015 held out) | **0.8960** (AP 0.1445, $n=856{,}596$, 2.46% positive) |

**Train-vs-validation diagnostic** (added 2026-09-02, localises the B1/B2 failure mode):
B1's three folds finish at train/val AUC pairs of (0.948, 0.939), (0.956, 0.936), (0.963,
0.936); B2's six regions show train/val gaps of only +0.009 to +0.027. Train and
validation AUC track closely and both stay high (0.93–0.96) throughout — **the model is
not overfitting the training pixels**. The collapse happens entirely at the
validation→test boundary (AUC falls from ~0.94 to 0.70–0.78 for B1, 0.54–0.73 for B2),
confirming this is an out-of-distribution transfer failure, not classical overfitting.
Track B3 shows no such gap: train/val/test are all near 0.896–0.899 throughout.

**Term ablation** (fixed-budget regime, 80 epochs, width 32, no validation set,
$n=4{,}508$ held-out, 42.06% positive):

| Configuration | ROC-AUC | AP |
|---|---:|---:|
| Diffusion only | 0.6017 | 0.6050 |
| + Advection | 0.9239 | 0.9014 |
| + Reaction (full CDR) | 0.9397¹ | 0.9233¹ |

¹ *Two values coexist in the source record for this row: the original 2026-08-20 run
reported 0.9406/0.9253 (used in `CDR_PINN_Methodology_Section.md`'s own ablation table and
several derived documents); the manuscript uses 0.9397/0.9233, from
`cdr_pinn_full_cdr_result.json`, the traceable artefact under the corrected
`forest_frac_baseline` feature set. The two are close (Δ0.0009 AUC) and both are real
measured runs — the manuscript's choice is the more recent, leakage-corrected one and
should be treated as canonical.*

**Jackknife variable importance** (mirrors Biswas et al.'s Fig. 10; 14 retrains + baseline,
40 epochs each, validated protocol, all-variables baseline AUC 0.9397):

| Covariate | Without-$X$ | Only-$X$ | Gain alone |
|---|---:|---:|---:|
| **Elevation** | **0.8027** | **0.9399** | **+0.4399** |
| Distance to roads | 0.9372 | 0.7880 | +0.2880 |
| Slope | 0.9365 | 0.7665 | +0.2665 |
| Forest fraction | 0.9402 | 0.7233 | +0.2233 |
| NDVI (baseline) | 0.9380 | 0.7177 | +0.2177 |
| NDVI anomaly | 0.9386 | 0.5911 | +0.0911 |
| Dryness proxy | 0.9400 | 0.5903 | +0.0903 |

A parallel train-vs-validation check across all 15 retrains confirms the ranking is not
an artefact of uneven convergence: every retrain's train−val gap falls in a narrow
+0.008 to +0.028 band regardless of which covariate is held out.

**Six independent lines of evidence** converge on elevation/terrain dominance: the term
ablation (advection +0.322 of +0.338 total gain), the field slope-coincidence statistic
(+115%), permutation importance (0.2268 drop, 24.13% of baseline), response curves
($\Delta=0.4611$), the Jackknife retraining, and the classical models' own Gini/MaxEnt
importance rankings.

**22-year vs. 20-year training-volume ablation** (controlled: same architecture, split,
seed; only the temporal training window varies, both scored against the *same* 2001–2020
target on the *same* held-out test pixels):

| Model | Trained on | Test AUC | Test AP |
|---|---|---:|---:|
| 22-year model, re-scored (no retrain) | 266 months | 0.9380 | 0.9103 |
| 20-year model, freshly trained | 240 months | **0.9404** | **0.9123** |
| *(reference)* 22-year model, own full target | 266 months | 0.9398 | 0.9223 |

No accuracy advantage from the extra 2 years ($\Delta=+0.0024$ *favouring* the 20-year
model, within the $\pm0.0017$–$0.002$ multi-seed noise floor) — a genuine null result for
"more months improves accuracy," disclosed rather than reframed. What the extra 2 years
*do* provide, independent of any model's accuracy: **+485 distinct fire-affected pixels
(+5.59%)** of India's real fire-prone geography captured in `fire_ever` that a
20-year-restricted study structurally cannot see.

**Physics-vs-no-physics** (matched architecture/budget, PDE and boundary losses removed,
data loss only):

| Track | With physics | No physics | $\Delta$ |
|---|---:|---:|---:|
| A | 0.9406 | 0.9461 | $-0.0055$ |
| B1 | 0.7595 | 0.7555 | $+0.0040$ |
| B2 | 0.5978 | 0.6368 | $-0.0390$ |
| B3 | 0.8935 | 0.9059 | $-0.0124$ |

Negative on three of four tracks; the one positive difference (B1) is smaller than that
track's own fold-to-fold spread ($\pm0.0182$).

**Computational cost**: physics constraint raises training time $2.50\times$ (146.7 s →
367.4 s) at identical architecture, data and epoch budget. Peak GPU memory 3.25 GB of 32
GB available.

---

## PART II — COMPLETE GAP INVENTORY

Organised by urgency for the TGRS submission: (A) a factual discrepancy needing a fix, not
just a disclosure; (B) gaps already disclosed in the current manuscript, listed for
completeness; (C) real, disclosed gaps found in the pipeline's own audit trail that are
**not yet in the manuscript**; (D) the project's current "genuinely open" research punch
list, suitable for the Future Work section.

### II.A — Needs a Fix, Not Just a Disclosure

**A1. The manuscript's Table I and its running text claim "specific humidity" as a feature
extending parity beyond Biswas et al.'s 15 predictor groups (`CDR_PINO_TGRS.tex` lines
144, 207, 241). The underlying pipeline does not do this.** Per Step 4's own
`Complete_Methodology_Section.md` §5: specific humidity (`Qair_f_tavg`) is computed and
exported **only as a national monthly scalar — never reprojected onto the working grid,
never used as a per-pixel model feature.** The humidity variable that actually reaches
the trained models is a *derived relative humidity* (`fldas_rh_anomaly`), computed via
the Magnus-type formula from specific humidity, air temperature, and surface pressure —
a related but genuinely different quantity from what Biswas et al. used, and a
**deliberate, disclosed substitution** in the source project's own documentation
(Step 4 audit, gap #1, flagged there as needing exactly this kind of explicit statement).
The manuscript currently states the wrong variable name for a claimed novelty item.
**Fix**: replace "specific humidity" with "relative humidity (derived)" everywhere it
appears, and add one sentence explaining the substitution and why (matching the source
project's own recommended framing — see Step 4 audit gap #1). This is a one-paragraph fix
that removes a factual error a domain-expert reviewer could catch.

**A2. MaxEnt is trained on a 150,000-row stratified subsample of the training split, not
the full ~2.7M-row training set** (`Complete_Methodology_Section.md` §8.3) — a
full-dataset fit did not complete inside a 2-hour cell timeout. The manuscript's Section
II-F describes the MaxEnt configuration (feature classes, regularisation search) but does
not disclose the subsampling. This is a real methodological detail a reviewer comparing
"RF trained on the full table" against "MaxEnt trained on 5.5% of it" would reasonably
want stated. **Fix**: one sentence in Section II-F or the MaxEnt paragraph of Section V-A.

**A3. Step 3's DTR Mann–Kendall trend feature is computed and exported but never actually
reaches the 57-feature stack** — a real, silent feature-parity gap flagged by the LST
audit (gap #8) and confirmed still open by the TGRS punch list (item 2). The manuscript's
Table I lists "diurnal temperature range" as a feature without qualification; strictly,
the DTR *anomaly* is carried forward but its own trend statistic is not, unlike LST
day/night, both of whose trend statistics are. **Fix**: either wire the feature in and
re-run Step 6/7 (a real code change + rerun, out of scope for a documentation pass), or
add a footnote scoping the DTR feature to its anomaly component only. Lowest-cost
immediate option is the footnote.

### II.B — Already Disclosed in the Manuscript (cross-reference, for completeness)

The manuscript's own Limitations section (`CDR_PINO_TGRS.tex`, Section VIII, 9 items)
already covers: the unclosed Track-A accuracy gap after six interventions; the negative
physics-vs-no-physics result and its $2.5\times$ training cost; the ~0.20 AUC spatial
blocking gap against classical baselines; partial hyperparameter validation and
single-seed B1/B2/B3/Jackknife coverage; the 2.3% positive-rate optimisation failure;
the two geometric approximations (symmetric extension, Neumann boundary convenience);
untested resolution independence; the point-estimate/transductive-exposure caveat; and
the research-stage deployment caveat. These do not need duplication — listed here only so
this document is a complete cross-reference, not a partial one.

### II.C — Real, Disclosed Gaps Not Yet in the Manuscript, by Pipeline Step

**Step 1 (fire points).**
1. No FIRMS confidence-level filtering or spatial declustering/thinning (4.29% of
   541,545 points carry `confidence<30`) — a deliberate, disclosed scope decision by the
   source project, but not mentioned anywhere in the manuscript. *Recommended: one
   sentence in Section II-B or the Limitations list.*
2. `type` field (vegetation fire vs. other static land source/offshore/volcano) also
   never filtered (0.21% of points, negligible volume, near-zero-cost fix if ever
   revisited alongside #1).
3. No quantitative sensitivity check on the state-vs-country boundary-polygon choice
   (qualitatively well-justified, never quantified).
4. The burned-area validation analysis script is not preserved in the tracked codebase —
   a reproducibility gap for a co-author or reviewer wanting to re-run it, though the
   output artefacts and their provenance (git commit messages) are.

**Step 2 (NDVI).**
5. No sensitivity analysis on the QA-reliability threshold (`{Good, Marginal}` vs.
   `{Good}` only) or the 2001–2020 climatology baseline window.
6. No controlled single-feature-vs-full-set ablation isolating NDVI's own marginal
   contribution (the current "decomposition adds value" claim is inferred from relative
   Gini importances within one already-trained model, not a direct controlled
   experiment).

**Step 3 (LST).**
7. No validation against an independent LST or ground-temperature product (ERA5-Land,
   AVHRR, IMD station data) — MOD11A2's own accuracy is taken as given.
8. No sensitivity check on the LST climatology baseline or against Biswas et al.'s actual
   product (MOD11C3, vs. this project's MOD11A2).

**Step 4 (FLDAS climatic + land cover).**
9. No independent climate-reanalysis or ground-station cross-check for any of the six
   climatic variables (Biswas et al. themselves used a different precipitation source,
   GPM 3IMERGHHL, not FLDAS/CHIRPS — a fact the manuscript does not mention).
10. The fire-coincidence check (Section covering conditioning-factor relationships) omits
    net longwave radiation from its variable set, with no stated reason — likely an
    oversight rather than a deliberate exclusion.
11. Land cover is a single static year while every climatic variable is time-varying
    across 2000–2022 — a temporal-resolution mismatch between predictor groups feeding
    the same model, worth one explicit caveat sentence (distinct from the leakage fix,
    which addresses a different issue).

**Step 5 (terrain/accessibility).**
12. No algorithm-choice sensitivity analysis: Horn's method is never compared against an
    alternative gradient algorithm (e.g. Zevenbergen & Thorne), and the Euclidean
    distance transform is never validated against a cost-distance surface or a
    hand-checked reference distance. Given slope is Biswas et al.'s second-highest
    contribution variable, this is worth a short methods-section sentence even if the
    expected result is "negligible difference."
13. A $-46.9$ m elevation-minimum artefact (likely an SRTM radar-return artefact over a
    lake or reservoir) is present in the terrain statistics and not masked out. Low
    priority — the source project's own recommendation is a one-line footnote in the
    paper, which the current manuscript does not yet carry (Table nomenclature reports
    the full measured range without qualification).

**Step 7 (classical baselines).** See II.A2 (MaxEnt subsampling) above — the single most
reviewer-relevant item in this group.

**Step 8 (CDR-PINN) — items beyond what Section VIII already covers.**
14. RF and MaxEnt can train on incomplete/NaN rows more flexibly than a spatial operator
    that requires a dense, complete grid at every training step — an inherited
    satellite-data-noise limitation (cloud cover, sensor gaps) that is compounded,
    not created, by the operator's dense-grid requirement. Not stated anywhere in the
    manuscript's Limitations list.
15. Whether the Track-A canonical checkpoint (`train_standard_protocol.py`) is the exact
    same checkpoint the B1/B2/B3 re-run was scored against is an internal
    cross-reference the source project flagged as unverified (two different scripts
    produced the numbers). This is an internal QA item rather than a manuscript
    limitation, but worth a direct code check before the numbers are treated as fully
    cross-consistent in a submission.
16. The term-ablation study (Table on diffusion/advection/reaction) is single-seed, a
    fact the manuscript's Section VIII item 4 states for B1/B2/B3/Jackknife but does not
    explicitly extend to the ablation table itself.

### II.D — CDR-PINN-Specific Limitations from the Source Project, Cross-Checked Against the Manuscript

All of the following, drawn from `CDR_PINN_Full_Paper_Draft.md` §5.7 and
`CDR_PINN_Study_Clarifications_QA.md` Q12, are **already covered** by the manuscript's
Limitations section, confirmed by direct comparison: data-hunger under sparse labels;
spectral truncation vs. sharp local features; rectangular-grid boundary approximation;
physics-loss computational overhead; optimisation sensitivity across the six
interventions; unverified resolution independence; elevation over-reliance; no calibrated
uncertainty; transductive information exposure; research-stage deployment scope. No
action needed beyond what Section VIII already states, **except item 14 above** (the
dense-grid-vs-incomplete-rows point), which is genuinely absent from the manuscript.

### II.E — Currently Open Research Items (candidates for the Future Work section)

From `TGRS_Submission_Plan_and_Reviewer_Guide.md` §4, current as of 2026-09-23 — every
item below is confirmed still open, not stale:

1. Multi-seed coverage for the B1/B2/B3 generalisation tracks and the term-ablation study
   (Track A alone has a 3-seed check). *Already implied by the manuscript's Section VIII
   item 4 for B1/B2/B3; the ablation study is the one piece not explicitly named there
   (see II.C16).*
2. Zero-shot super-resolution evaluation — a proven FNO architectural property, never
   exercised on a trained checkpoint. *Already in the manuscript's Future Work.*
3. Instance-wise fine-tuning (PINO's own prescribed second training phase) — not
   attempted. *Already in the manuscript's Future Work.*
4. Extending Step 1's fire-point filtering to include FIRMS confidence/spatial
   declustering — flagged as high-impact but deliberately deferred, since it would
   cascade through every downstream step (2 through CDR-PINN) and is a scoped decision
   to make explicitly, not a quick fix.
5. Direct verification that the Track-A checkpoint and the B1/B2/B3 checkpoint are the
   same artefact (see II.C15).

### II.F — Summary: What Needs Action Before Submission

| Priority | Item | Action | Effort |
|---|---|---|---|
| **High** | A1 — "specific humidity" should read "relative humidity (derived)" | Fix in Table I + 2 prose mentions + 1-sentence rationale | ~15 min |
| **High** | A2 — MaxEnt's 150k-row training subsample undisclosed | Add one sentence to Section II-F/V-A | ~10 min |
| Medium | A3 — DTR trend feature scope | Footnote scoping DTR to its anomaly component | ~10 min |
| Medium | II.C1 — FIRMS confidence/type filtering undisclosed | One Limitations-list sentence | ~10 min |
| Medium | II.C9 — no independent climate cross-check; different precipitation source than Biswas et al. | One Limitations-list sentence | ~10 min |
| Medium | II.C14 — dense-grid requirement vs. classical models' NaN tolerance | One Limitations-list sentence | ~10 min |
| Low | II.C12, II.C13 — terrain algorithm sensitivity, elevation-min artefact | One combined footnote | ~10 min |
| Low | II.C2, C3, C4, C5, C6, C7, C8, C10, C11 | Optional, lowest reviewer relevance | — |

The two **High** items are the only ones that change what the manuscript actually claims
(rather than adding a caveat to something already correctly stated) and should be fixed
before any submission draft is finalised.
