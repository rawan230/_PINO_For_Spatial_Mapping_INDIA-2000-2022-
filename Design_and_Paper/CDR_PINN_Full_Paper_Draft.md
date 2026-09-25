# Does a Governing Equation Help? A Controlled Evaluation of a Convection–Diffusion–Reaction Physics-Informed Neural Operator for Forest-Fire Susceptibility Mapping in India (2000–2022)

> **Manuscript draft v2, 2026-09-25 (post-audit rewrite).** This version replaces the
> 2026-08-20/22 draft in full. It is written as a controlled test of whether physics
> helps, following the end-to-end audit of 2026-09-24/25.
>
> - **Sources.** Every number is taken from `results/FINAL_MANUSCRIPT_NUMBERS.md` or from the
>   audit result files it cites (`results/final/*.csv`, `results/FULL_METHODOLOGY_AUDIT.md`).
> - **Earlier drafts.** The previous draft is in the git history and in
>   `_Archive_Unwanted_2026-09-25/pre_audit_document_snapshots/`.
> - **Citation tags.** `[cite-confirmed]` marks references whose records were independently
>   checked in an earlier literature pass. `[cite-verify]` marks standard references added in
>   this rewrite whose bibliographic details still need a final check before submission.
>
> **Alternative titles:**
> 1. *Physics-Informed Neural Operators for National-Scale Forest-Fire Susceptibility:
>    A Controlled Test over India*.
> 2. *When Physics Does Not Help: Controlled Evidence from a Convection–Diffusion–Reaction
>    Neural Operator for Forest-Fire Susceptibility in India*.

---

## Abstract

Physics-informed machine learning is widely expected to improve environmental prediction
by embedding a governing equation in the learning objective. This advantage is expected to
be largest under distribution shift. For forest-fire susceptibility mapping, the claim has
not been tested under controlled conditions. We build a convection–diffusion–reaction
(CDR) physics-informed Fourier neural operator (CDR-PINO) for India, trained on 22 years
(2000–2022) of monthly MODIS fire observations. Its diffusion, advection and reaction
terms are tied to vegetation–moisture, terrain and human-access covariates.

We test it against three controls:
- the identical network trained without physics terms;
- classical models trained on exactly the same cells, splits and covariates;
- covariate-free persistence baselines.

The comparison uses four evaluation tracks (random, spatial-block, leave-one-region-out and
unseen-year), three seeds and paired tests. No physics configuration outperformed the same
network without physics. On the random split, the full CDR model scored ROC-AUC 0.939
against 0.945 without physics (DeLong p < 0.02 in every seed); on spatial blocks and held-out
regions the differences were indistinguishable from zero. On the same 12 km cells and
covariates, a Random Forest reached 0.980 / 0.974 / 0.959 (random / spatial / regional),
against 0.939 / 0.719 / 0.570 for CDR-PINO. For unseen years, CDR-PINO (0.893) scored below
climatological (0.908) and seasonal (0.930) fire-frequency baselines. Diagnostics of the
trained field showed an almost static solution whose PDE residual is 74–616 times its own
temporal change. The trained operator therefore does not act as a solution of the equation it
was regularised towards.

These negative results rest on a rebuilt, fully reproducible national pipeline:
- 541,545 forest-fire points;
- 55 predictors on a 1/120° grid;
- a corrected fire label and Seasonal Kendall trend statistics;
- forest pixels as the primary evaluation population.

On this pipeline a Random Forest reaches ROC-AUC 0.975 over all pixels and 0.897 over forest
pixels. A reimplementation of the reference MaxEnt protocol reaches 0.893 ± 0.009. Accuracy
differences are explained by the model family and the predictor set, not by physics.

**Keywords:** forest fire; susceptibility mapping; physics-informed machine learning; Fourier
neural operator; convection–diffusion–reaction; negative result; spatial cross-validation;
India; MODIS.

---

## 1. Introduction

### 1.1 Motivation

Forest fires are an escalating ecological and economic hazard in India. About 36% of
the country's forest cover is classified as fire-prone (ISFR 2021, cited in Biswas, Mahato &
Joshi, 2025 `[cite-confirmed]`). National susceptibility maps guide early warning, patrol
allocation and fuel management. The most recent national study, Biswas et al. (2025), fits a
MaxEnt presence–background model to 15 static predictors at 0.25° resolution and reports a
test AUC of 0.879.

### 1.2 Susceptibility mapping in India

A substantial regional literature exists alongside the national reference study, using a
range of statistical and machine-learning methods:
- **Western Ghats:** AHP with RF, SVM and XGBoost for Goa (Uthappa et al., 2025, *J. Environ.
  Manage.*, 379:124777 `[cite-confirmed]`); an ANN/RF/MaxEnt/GLM/MARS/GBM ensemble (Kanda Naveen
  Babu et al., 2023, *For. Ecol. Manage.*, 540:121057 `[cite-confirmed]`).
- **Tamil Nadu:** MaxEnt (Meraj et al., 2025, *Risk Anal.*, 45(11):3604–3625 `[cite-confirmed]`).
- **Similipal, Odisha:** tree ensembles (Guria et al., 2025, *Environ. Sci. Pollut. Res.*,
  32(59):31375–31396 `[cite-confirmed]`).
- **Southern Mizoram:** six ML methods (Gupta, Shukla & Shukla, 2025, *Environ. Sci. Pollut.
  Res.*, 32(59):31433–31454 `[cite-confirmed]`).
- **Northeast India:** ML ensembles (Sarkar et al., 2024, *Ecol. Inform.*, 81:102598
  `[cite-confirmed]`).
- **Western Himalaya:** ensemble ML with explainable AI (Hang et al., 2024, *Environ. Technol.
  Innov.*, 35:103655 `[cite-confirmed]`).
- **Jammu & Kashmir:** fuzzy-AHP (Malik et al., 2025, *Discover Forests*, 1(1):4
  `[cite-confirmed]`).

All of these treat susceptibility as a static, per-location classification problem. Almost all
evaluate on a single random split.

### 1.3 Susceptibility mapping internationally

The same paradigm dominates internationally:
- CNNs in Yunnan (Zhang, Wang & Liu, 2019, *Int. J. Disaster Risk Sci.*, 10(3):386–403
  `[cite-confirmed]`);
- RF/ANN in north-east Türkiye (Kantarcioglu, Schindler & Kocaman, 2023, *ISPRS Archives*,
  XLVIII-M-1-2023:161–167 `[cite-confirmed]`);
- SHAP-explained ML in İzmir (İban & Aksu, 2024, *Remote Sens.*, 16(15):2842 `[cite-confirmed]`);
- XGBoost in New South Wales (Zakari, Malik & Ong, 2025, *Nat. Hazards*, 121(13):15331–15357
  `[cite-confirmed]`);
- gradient-boosting ensembles in Greece (Symeonidis et al., 2025, *Earth*, 6(3):75
  `[cite-confirmed]`);
- Dempster–Shafer uncertainty-aware ML in Iran (Gholamnia et al., 2026, *Spat. Inf. Res.*,
  34(4):35 `[cite-confirmed]`);
- driver analysis in Portugal (Santana Neto et al., 2025, *J. Nat. Conserv.*, 86:126956
  `[cite-confirmed]`).

Spatially structured validation is recommended precisely because random splits overstate
transferability for autocorrelated data (Roberts et al., 2017 `[cite-verify]`). It remains the
exception.

### 1.4 Physics-informed learning, and the claim this paper tests

Physics-informed neural networks (PINNs) embed a governing equation as a soft constraint
(Raissi, Perdikaris & Karniadakis, 2019, *J. Comput. Phys.*, 378:686–707 `[cite-confirmed]`).
Neural operators such as the Fourier Neural Operator (FNO) and its physics-informed variant
(PINO) learn solution operators on grids (Li et al., 2023, arXiv:2111.03794 `[cite-confirmed]`).
FNOs have been used for climate downscaling (Jiang et al., 2023, *JAMES*, 15(7):e2023MS003800
`[cite-confirmed]`), hydrological ensembles (Sun et al., 2024, *Water Resour. Res.*,
60(10):e2024WR037555 `[cite-confirmed]`) and global weather forecasting (Kurth et al., 2023,
*PASC'23* `[cite-confirmed]`).

The case made for physics-informed learning has three parts (Karniadakis et al., 2021,
*Nat. Rev. Phys.*, 3(6):422–440 `[cite-confirmed]`):
- better generalisation when data are scarce or shifted;
- physically plausible outputs;
- mechanistic interpretability.

Process-guided models have shown the first property for lake temperature (Read et al., 2019,
*Water Resour. Res.*, 55(11):9173–9190 `[cite-confirmed]`). For fire, physics-informed methods
have so far addressed spread and data assimilation, not susceptibility:
- a PINN has been used to learn rate-of-spread parameters for a single event (Vogiatzoglou et al.,
  2025, *CMAME*, 434:117545 `[cite-confirmed]`);
- a Bayesian PINN has been used for spatio-temporal assimilation (Dabrowski et al., 2023,
  *Spat. Stat.*, 55:100746 `[cite-confirmed]`).

A physics-informed susceptibility model is therefore an open proposition. Susceptibility maps
already exist and are useful, so the real question is not whether such a model can be built. It
is whether the governing equation **adds** anything once confounders are removed:
- network capacity and architecture;
- predictor choice;
- the evaluation grid and population;
- the spatial or temporal structure of the test split.

Answering this needs controls that are usually absent:
- the identical network without physics;
- classical models on identical inputs;
- baselines that use no covariates at all.

### 1.5 Research questions

- **RQ1.** Does adding diffusion, advection and reaction terms to a neural operator improve
  held-out discrimination over the same network trained without them, on random, spatial,
  regional and temporal splits?
- **RQ2.** How does the operator compare with classical models trained on exactly the same
  cells, splits and covariates?
- **RQ3.** On unseen years, does it beat baselines that use no covariates?
- **RQ4.** Does the trained operator actually behave as a solution of its governing equation?

### 1.6 Contributions

1. **A controlled test of physics-informed learning for fire susceptibility.** It covers four
   physics configurations × four evaluation tracks × three seeds, with paired DeLong and
   block-bootstrap tests. It includes same-cell classical comparisons and persistence nulls.
   The answer to RQ1–RQ3 is negative, and the test identifies where the accuracy differences
   actually come from (§4.7).
2. **A physical-consistency diagnostic for trained PINOs.** It measures the magnitude of each
   term and the PDE residual. It shows that a trained CDR operator can score well while its
   field is effectively static and violates its own equation (RQ4). This is a check that
   physics-informed hazard models should report.
3. **A corrected and reproducible national susceptibility pipeline for India.**
   - It uses 541,545 forest-fire points, 55 predictors and a 1/120° grid.
   - Every statistic is recalculated from raw data.
   - Five defects common in this literature are corrected: half-pixel label misregistration,
     degenerate anomaly-mean predictors, Mann–Kendall tests on seasonal or smoothed series,
     Moran's I over filled non-study cells, and in-window land-cover predictors.
   - Forest pixels are evaluated as the primary population, because non-forest pixels are
     negative by construction.
4. **A like-for-like position relative to the reference study.** A reimplementation of Biswas
   et al.'s protocol on this pipeline's data places the reference result in context. It also
   makes explicit which pixel-level AUCs are, and are not, comparable with it.

---

## 2. Study area and data

**Domain and period.** The study covers India, using the dissolved state-boundary polygon (EPSG:3857 in the source file,
reprojected to EPSG:4326). The period is 2000-11-01 to 2022-12-15 (266 months), capped by the
availability of ESA-CCI/C3S land cover.

**Grid.** EPSG:4326, 3,641 × 3,504 pixels at 1/120° (≈ 0.93 km). It contains 4,184,671
India-mask pixels, of which **4,160,768** analysis pixels have valid NDVI. The **forest
population** has 2001 forest fraction > 0 (**1,197,538** pixels).

**Fire points.** The source is the MODIS Collection 6.1 active-fire archive (Giglio, Schroeder &
Justice, 2016 `[cite-verify]`). The filter stages are:
- bounding box: 2,804,373 → 2,801,347;
- India polygon: 1,599,471;
- deduplication on (lon, lat, date): 1,599,466;
- same-year ESA-CCI/C3S forest classes (13 codes, following Sannigrahi et al., 2018): **541,545**.

Of these, 4.29% have confidence < 30 and 0.21% have type ≠ 0. Filtering on either changes
ROC-AUC by < 0.001, so all points are kept. Annual counts agree with those derived from Biswas
et al.'s published shares within +0.5% to +2.4% (r = 0.99996, 2001–2020). India's forest cover
over the period is 18.3–19.3%.

**Label.** A pixel is positive if it contains at least one forest-fire point over the whole
period. Points are assigned to their **containing** pixel, `col = ⌊(lon − c)/a⌋`,
`row = ⌊(lat − f)/e⌋`, where (c, f) is the top-left pixel edge. This gives **268,411** positive
pixels, with prevalence 6.45% over all pixels and 22.4% over forest pixels. The commonly used
rule `round((·)/a)` measures against the pixel *edge*. It displaces 74.9% of points into a
neighbouring pixel, and correcting it raises Random-Forest AUC by +0.006 (all) and +0.011
(forest).

**Predictors.** Table 1 lists the 55 predictors. They cover all 15 of Biswas et al.'s variable
groups. Five of those groups come from different products: MOD11A2 instead of MOD11C3 for LST,
MOD13A3 instead of MOD13C2 for NDVI, and FLDAS instead of GPM/GLDAS for precipitation, soil
moisture and net longwave radiation. The resolution effect of the LST product was tested and is
negligible (§4.8).

**Table 1.** v2 predictor set (55 features; dictionary in `results/final/FINAL_FEATURE_DICTIONARY.csv`).

| Group | Source | Features |
|---|---|---|
| Vegetation (6) | MOD13A3.061, 1 km monthly | QA-masked mean NDVI; 2001–2020 climatological June NDVI; Seasonal Kendall τ; seasonal Sen slope; CVSI at the MI-optimal lag k* = 8; LISA cluster (India-only, 8 × 8 block means) |
| Land-surface temperature (6) | MOD11A2.061, 8-day 1 km | 2001–2020 climatological levels of day LST, night LST and DTR; Seasonal Kendall τ of each |
| Climate (14) | FLDAS Noah (MERRA-2/CHIRPS), 0.1° monthly | Climatological levels and Seasonal Kendall τ of air temperature, specific humidity, relative humidity, wind, precipitation, net longwave radiation and soil moisture |
| Terrain (4) | SRTMGL3 90 m | Elevation; Horn slope; aspect sin and cos |
| Accessibility (3) | OpenStreetMap 2022 (Geofabrik) | Euclidean distance to roads, railways and waterways (India-centred equidistant conic) |
| Land cover (22) | ESA-CCI/C3S 2001 | 21 class fractions + forest fraction |

Three design choices in this table are corrections:
- **Climate and LST levels.** Climate and LST enter as climatological *levels*, the form Biswas
  et al. used, not as time-mean anomalies. A time-mean anomaly over a 2001–2020 baseline equals
  the residue of the 26 out-of-baseline months. It is degenerate: the spatial SD of each level is
  40–431× that of its anomaly mean.
- **Land-cover year.** Land-cover fractions come from 2001, before most of the label window.
- **Distance accuracy.** Distances were checked against exact geodesic distances (n = 3,000):
  RMSE 0.31, 0.65 and 0.32 km for roads, railways and waterways.

---

## 3. Methods

### 3.1 Trend and spatial statistics

**Trends.** Per-pixel trends use the Seasonal Kendall test (Hirsch, Slack & Smith, 1982
`[cite-verify]`), tie-corrected with at least 30 pairs. They use the seasonal Sen slope (Sen,
1968 `[cite-verify]`) and Benjamini–Hochberg false-discovery-rate control at q < 0.05
(Benjamini & Hochberg, 1995 `[cite-verify]`). The Mann–Kendall test is not used: it is invalid
on seasonal monthly series, and more so on a moving-average-smoothed series (here with lag-1
autocorrelation 0.975).

**Spatial autocorrelation.** Global Moran's I and LISA (Anselin, 1995 `[cite-verify]`) are
computed on 8 × 8-pixel block means over India cells only, with 999 permutations.

### 3.2 Classical susceptibility models (1 km)

**Models.**
- **Random Forest** (Breiman, 2001 `[cite-verify]`): 200 trees, `max_depth = 25`,
  `min_samples_leaf = 3` (validated), balanced class weights.
- **MaxEnt** (Phillips, Anderson & Schapire, 2006 `[cite-verify]`): linear, quadratic, hinge
  and product features, with β = 4.0 (validated), fitted on 150,000 training rows. A sample-size
  sensitivity from 50k to 500k rows is reported (§4.8).

**Protocol.**
- Stratified 65/15/20 train/validation/test split.
- Imputation fitted on training rows only.
- Decision thresholds chosen on validation (max-F1).
- The test set is scored once.

The same models are also trained on Biswas et al.'s 15 predictors (as levels) to separate the
predictor-set effect from the model-family effect.

### 3.3 The CDR physics-informed neural operator

**Governing equation.** A latent logit field u(x, y, t) evolves as

```
∂u/∂t = D(x,y,t) ∇²u − v(x,y)·∇u + ρ(x,y,t) σ(u)(1 − σ(u)),     in Ω × (0, T]
```

The susceptibility is s = σ(u). The three coefficients come from physics heads:
- **Diffusion.** D = softplus(D_net[NDVI, forest fraction] − softplus(w)·NDVI anomaly) links
  diffusion to vegetation and moisture.
- **Advection.** v = softplus(c)·∇E is upslope transport along the elevation gradient.
- **Reaction.** ρ = softplus(ρ_net[dryness, NDVI, slope, distance to roads]) links the reaction
  to ignition pressure.

Because the reaction acts on the logit, it is **not** a Fisher–KPP term in s: in s it reads
ds/dt = ρ s²(1 − s)². For the continuous equation with bounded coefficients, the design
documents show global-in-time existence and uniqueness:
- uniform parabolicity from D ≥ D_min > 0;
- Gårding's inequality for the advective perturbation;
- a globally bounded (≤ ρ_max/4) and globally Lipschitz (constant ρ_max/(6√3)) reaction.

These results concern the equation. §4.6 tests whether the *trained network* satisfies it.

**Discretisation and architecture.**
- **Grid.** The operator runs on a 256 × 256 grid (≈ 12 km) with 22,542 valid cells and 266
  months.
- **Covariates (7).** NDVI mean, monthly NDVI anomaly, 2001 forest fraction, monthly dryness,
  slope, distance to roads and elevation.
- **Backbone.** An FNO with width 32, 4 spectral layers and 16 × 16 retained modes, 1,054,613
  parameters in total.
- **Training scheme.** One-step-ahead monthly training with truncated back-propagation over
  24-month windows.

**Losses, as implemented.**
- **PDE residual.** The residual is
  r = (u_{t+1} − u_t) − D∇²u + v·∇u − ρσ(u)(1 − σ(u)), evaluated at the midpoint state, with
  spectral derivatives on a symmetric (Neumann-type) extension.
- **Data loss.** The data loss is a positive-weighted binary cross-entropy over all training
  cells. It is mixed 0.5/0.5 with a log-sum-exp-pooled terminal loss over the last window.
- **Boundary and initial conditions.** The boundary term penalises |∇u|² on the boundary ring,
  as a proxy for ∂u/∂n = 0. The initial condition u(·, 0) = 0 is imposed exactly, so its loss is
  zero.
- **Loss weights.** The weights follow w_i ← 0.9 w_i + 0.1 · mean‖∇L‖/‖∇L_i‖ every five windows
  (after Wang, Teng & Perdikaris, 2021 `[cite-verify]`).

### 3.4 Controlled evaluation design

**Physics configurations.** All four configurations use the same network, data, budget and
protocol:
- **none:** data and initial-condition losses only;
- **diffusion:** adds the PDE residual with the diffusion term only, plus the boundary term;
- **diffusion + advection:** adds advection to the residual;
- **full CDR:** adds the reaction term, giving the full equation.

**Tracks.**

| Track | Held-out unit | Construction |
|---|---|---|
| A | random cells | 65/15/20 cell split |
| B1 | spatial blocks | 2° × 2° blocks, 3 folds |
| B2 | regions | 6 K-means regions, leave one out |
| B3 | years | test years 2000, 2008, 2009, 2015; validation years 2001, 2012, 2013, 2020; scored on test cell-months |

**Unified protocol.** One protocol is applied to every track:
- AdamW, learning rate 10⁻³, validated weight decay 0;
- ReduceLROnPlateau on validation loss;
- up to 80 epochs, with early stopping on validation AUC (checked every 5 epochs, patience 4
  checks);
- validation carved from whole 2° blocks inside the training region on the spatial tracks;
- for B3, a terminal label built from training months only (the historical label pooled over
  all years, including test years).

Seeds 42, 43 and 44 vary initialisation and training noise; the fold, region and year
assignments are fixed.

**Same-cell classical comparison.** RF, MaxEnt and logistic regression are fitted on
CDR-PINO's own 12 km cells, partitions and 7 covariates. A second set of classical models also
uses 12 km aggregates of the 55 v2 predictors. The two sets separate predictor effects from
model effects.

**Persistence nulls for B3.** Two baselines use no covariates:
- **climatological frequency:** each cell's fire frequency over the training years;
- **seasonal frequency:** its fire frequency in the same calendar month over the training years.

An RF on monthly covariates, with and without the month, is also reported.

**Statistics.**
- Paired differences on identical test units use DeLong's test (DeLong, DeLong & Clarke-Pearson,
  1988 `[cite-verify]`) on Tracks A and B3.
- On B1 and B2, where test cells are spatially clustered, they use a spatial block bootstrap
  (123 blocks, 300 resamples).
- Calibration is reported as the Brier score and the expected calibration error (ECE).

### 3.5 Physical-consistency diagnostic

For trained checkpoints (Track A, seed 42, all four configurations, plus the historical full
model), we evaluate over all valid cell-months:
- the mean magnitude of |∂u/∂t|;
- the mean magnitude of each right-hand-side term;
- the RMS PDE residual.

A model that has learned a solution of its equation should have a residual small relative to
|∂u/∂t|. Its temporal change should also be non-trivial, since fire occurrence is strongly
seasonal.

### 3.6 Reimplementation of the reference protocol

Biswas et al.'s protocol is reimplemented on this pipeline's data as follows:
- a 0.25° grid of 4,630 cells;
- the 15 predictors as levels;
- presences from 2020 forest fires (1,292 cells);
- MaxEnt with linear, quadratic, hinge and product features and β = 1;
- 10 random 75/25 presence splits (Biswas et al.'s own counts, 1,830/609, imply 75/25 although
  the text states 70/30).

This is a reimplementation, not the reference result.

### 3.7 Evaluation populations

Non-forest pixels can never be positive under a forest-fire label. Over all pixels, forest
fraction alone therefore reaches AUC 0.91. We report every classical metric over all pixels and
over forest pixels, and treat the forest population as primary. We always state the grid and
the prevalence next to an AUC.

---

## 4. Results

### 4.1 National patterns (supporting results)

**Trends (Seasonal Kendall, FDR q < 0.05).**
- **NDVI:** greening in 3,552,278 pixels and browning in 72,305 (median Sen slope +0.0026 yr⁻¹).
- **Day LST:** cooling in 2,435,163 pixels (median −0.050 °C yr⁻¹).
- **Night LST:** warming in 2,080,747 pixels (+0.024 °C yr⁻¹).
- **Diurnal temperature range:** narrowing in 3,273,301 pixels (−0.077 °C yr⁻¹).
- **FLDAS climate:**
  - specific humidity increases in 26,373 of 29,056 FLDAS pixels;
  - air temperature changes significantly in 9,988 (7,620 decreasing).

Mann–Kendall on seasonal data had reported the air-temperature trend as noise. That was an
artefact of the test.

**Spatial structure.** NDVI's global Moran's I over India cells is 0.9456 (z = 494, p = 0.001).

**Fire–covariate associations.** Fire points lie on steeper slopes than India as a whole
(12.35° vs 5.72°). They are 38.9% closer to roads and 64.4% closer to waterways than the
national mean.

### 4.2 Classical models at 1 km

**Table 2.** Classical models, 1 km test set (all pixels | forest pixels). Sources:
`results/final/FINAL_METRICS.csv`, `CLASSICAL_METRICS.csv`.

| Model / track | ROC-AUC | Average precision |
|---|---|---|
| RF v2, random split | **0.975 \| 0.897** | 0.720 \| 0.721 |
| MaxEnt v2 (150k rows), random split | 0.967 \| 0.866 | 0.663 \| 0.664 |
| RF, Biswas-15 predictors, random split | 0.970 \| 0.885 | 0.692 \| 0.698 |
| RF v2, 2° spatial blocks (3 folds) | 0.959 ± 0.014 \| 0.838 ± 0.039 | 0.582 \| 0.583 |
| MaxEnt v2, 2° spatial blocks | 0.956 ± 0.012 \| 0.828 ± 0.029 | 0.561 \| 0.563 |
| RF v2, leave one region out (6) | 0.935 ± 0.036 \| 0.782 ± 0.034 | 0.412 \| 0.413 |
| RF v2, unseen years | 0.965 \| 0.872 | 0.282 \| 0.282 |
| Persistence null, unseen years | 0.807 \| 0.744 | 0.131 \| 0.146 |

**The forest population matters.**
- Accuracy drops by 0.08–0.15 AUC on forest pixels, and the forest-pixel ranking is sharper.
- RF beats MaxEnt by +0.008 over all pixels but by +0.031 over forest pixels (paired DeLong).
- Spatial and regional hold-out lower forest AUC to 0.838 and 0.782.

**Calibration.** MaxEnt is better calibrated than RF (ECE 0.015 | 0.054 vs 0.064 | 0.219). The
RF score is therefore a relative ranking, not a probability.

### 4.3 RQ1: physics terms vs the same network without them

**Table 3.** CDR-PINO physics ablation (mean ± SD over seeds × folds; 12 km cells; all cells).
Source: `results/final/ABLATION_RESULTS.csv`, `SEED_LEVEL_RESULTS.csv`.

| Track | No physics | Diffusion | Diffusion + advection | Full CDR | Full − none (paired) |
|---|---|---|---|---|---|
| A: random (prevalence 42%) | **0.945 ± 0.002** | 0.924 ± 0.005 | 0.939 ± 0.001 | 0.939 ± 0.002 | −0.005 / −0.007 / −0.006; DeLong p < 0.02 each seed |
| B1: spatial blocks | 0.724 ± 0.010 | 0.642 ± 0.013 | **0.730 ± 0.004** | 0.718 ± 0.007 | block-bootstrap CIs include 0 (all seeds) |
| B2: regions | **0.614 ± 0.021** | — | — | 0.570 ± 0.007 | CIs include 0 (p ≈ 0.06–0.87) |
| B3: unseen years (prevalence 2.46%) | **0.904 ± 0.001** | 0.886 ± 0.006 | 0.894 ± 0.001 | 0.893 ± 0.003 | −0.010 / −0.009 / −0.014 (replicated) |

**Findings (Fig. B):**
- **No configuration beats no physics on any track.** Where differences are resolvable (A, B3),
  the physics terms are *costly*.
- **Diffusion alone is the worst configuration** on every track where it was run.
- **Adding advection recovers most of the loss.** This matches its interpretation as a pathway
  for elevation information (§4.6), not as a spread mechanism.
- **The same ordering holds on forest cells** (Track A: 0.910 without physics vs 0.897 full;
  B3: 0.848 vs 0.833).
- **Cost.** Physics raises training time by 2.3–2.4×.
- **Leakage check.** Rebuilding the historical leaky B3 label changes AUC by only +0.0006.

### 4.4 RQ2: classical models on identical cells, splits and covariates

**Table 4.** Same 12 km cells, same partitions, same 7 covariates (all cells | forest cells;
B1/B2 are fold/region means). Source: `bridge_predictions/`, `results/generalization/PAIRED_cdr_vs_classical_same_cells.csv`.

| Track | CDR-PINO full | CDR-PINO none | Logistic regression | MaxEnt | Random Forest |
|---|---|---|---|---|---|
| A | 0.939 \| 0.897 | 0.945 \| 0.910 | 0.924 \| 0.874 | 0.958 \| 0.927 | **0.980 \| 0.950** |
| B1 | 0.719 \| 0.701 | 0.724 \| 0.702 | 0.919 \| 0.869 | 0.950 \| 0.917 | **0.974 \| 0.936** |
| B2 | 0.570 \| 0.569 | 0.614 \| 0.635 | 0.872 \| 0.835 | 0.845 \| 0.824 | **0.959 \| 0.916** |

**Findings (Fig. C):**
- **The gap is structural.** On the random split, RF leads CDR-PINO by 0.041. Under spatial and
  regional hold-out the gap widens to 0.26–0.39.
- **Even a linear model wins.** Logistic regression on the same seven covariates beats CDR-PINO
  by 0.20 (B1) and 0.30 (B2).
- **The weak spatial transfer is the operator's, not the data's.** Adding the other v2 predictors
  (12 km aggregates) changes the classical results by at most ±0.01 (RF: 0.981, 0.976, 0.953).
  The covariates are therefore not what limits spatial transfer.

### 4.5 RQ3: unseen years against covariate-free baselines

On the four held-out years, scored per cell-month (prevalence 2.46%):
- a covariate-free **climatological fire frequency** reaches **0.908** (forest cells 0.857);
- the **seasonal frequency** (same calendar month) reaches **0.930** (0.920);
- CDR-PINO reaches 0.893 with full physics and 0.904 without;
- an RF on the monthly covariates reaches 0.902, and **0.972** when the month is included.

CDR-PINO's unseen-year skill is therefore below what fire persistence alone provides (Fig. D).
The earlier interpretation of strong temporal generalisation is not supported. Most of the
temporal signal is seasonality plus location, and a model that encodes the calendar month
captures it better.

### 4.6 RQ4: does the trained operator solve its equation?

**Table 5.** Term magnitudes of trained Track A checkpoints (seed 42). Source:
`results/cdr_pino/term_magnitudes_*.json`.

| Configuration | median \|∂u/∂t\| | Advection share of RHS | Reaction share | Diffusion share | RMS residual / mean \|∂u/∂t\| |
|---|---:|---:|---:|---:|---:|
| none | 0.0016 | 0.98 | 0.02 | 0.003 | 616 |
| diffusion | 0.0003 | 0.83 | 0.16 | 0.002 | 224 |
| diffusion + advection | 0.0002 | 0.83 | 0.17 | 0.0004 | 74 |
| full CDR | 0.0003 | 0.91 | 0.09 | 0.0004 | 78 |

Every trained configuration shows three features (Fig. G):
- **An effectively static field.** The median |∂u/∂t| is ≤ 0.002 per month, although fire
  occurrence is strongly seasonal.
- **A large residual.** The PDE residual is 74–616× the field's own temporal change.
- **Negligible diffusion.** The diffusion term is ≤ 0.3% of the right-hand side.

The physics loss therefore does not produce a field that evolves by the equation. The optimiser
satisfies the data loss with a near-static susceptibility map and absorbs the residual. The
advection head is the channel through which elevation enters the operator. Within CDR-PINO's
7-covariate model, removing elevation had the largest effect in the historical single-seed
Jackknife test (not re-run in the audit), but with the full predictor set
the terrain group is redundant (−0.0001 AUC, §4.7). The earlier "elevation dominance" was a
property of the restricted covariate set, not of fire.

### 4.7 Where accuracy differences come from

**Table 6.** Paired ΔAUC decomposition (all | forest pixels unless stated). Source:
`results/final/CONTRIBUTION_DECOMPOSITION.csv`.

| Source of difference | ΔAUC |
|---|---|
| Model family: RF − MaxEnt (Biswas-15 predictors) | +0.025 \| +0.067 |
| Model family: RF − MaxEnt (v2 predictors) | +0.008 \| +0.031 |
| Added predictors and engineering: v2 − Biswas-15 (RF) | +0.005 \| +0.012 |
| Added predictors: v2 − Biswas-15 (MaxEnt) | +0.022 \| +0.048 |
| Predictor restriction: v2 − CDR's 5 static covariates (RF) | +0.017 \| +0.071 |
| Trend group (Seasonal Kendall / Sen) | +0.001 \| +0.006 |
| Land-cover group | +0.003 \| +0.008 |
| Terrain group | −0.0001 \| −0.0006 (redundant) |
| Physics terms in CDR-PINO (full − none) | −0.006 (A); CI includes 0 (B1, B2); −0.011 (B3) |
| Model structure: full CDR-PINO − RF, same cells and covariates | −0.041 (A) |

Accuracy differences in this problem come from the **model family** and the **predictor set**.
Neither the physics terms nor the operator structure contribute positively (Fig. A).

### 4.8 Sensitivity analyses

Most sensitivities are null or small (`results/final/SENSITIVITY_RESULTS.csv`).

**Null results:**
- FIRMS confidence and type filters;
- NDVI QA level (Good-only vs Good + Marginal);
- LST aggregated to 0.05° (the MOD11C3 resolution): ΔAUC ≤ 0.0002;
- slope algorithm (Horn vs Zevenbergen–Thorne);
- slope computed from a 0.25° DEM;
- the B3 label leakage.

**MaxEnt training size.** Performance rises monotonically but only slightly, from 0.964 | 0.855
at 50k rows to 0.969 | 0.872 at 500k (Fig. E). Fit time grows as about n^1.6. The RF–MaxEnt gap
is therefore not a subsampling artefact.

**Slope from a 0.25° DEM.** This has a mean of 0.29° vs 5.72° at 90 m (r = 0.70). It is a
different quantity, which limits comparability with slope-based findings at coarse resolution.

### 4.9 Relation to the reference study

**Reimplementation (§3.6).** The reimplementation of Biswas et al.'s protocol reaches test AUC
**0.893 ± 0.009** (range 0.877–0.912; train 0.902 ± 0.003), against their 0.879 test and 0.894
train. The two are moderately comparable. Their top permutation-importance variables are
reproduced (Fig. F):

| Variable | This reimplementation | Biswas et al. |
|---|---:|---:|
| NDVI | 19.0% | 22.3% |
| Night LST | 14.3% | 10.1% |
| Air temperature | 13.2% | 13.1% |
| Day LST | 8.1% | 9.6% |
| Slope | 5.8% | 5.6% |
| Specific humidity | 3.1% | 13.0% (not reproduced) |
| Elevation | 10.7% | 2.4% |

Biswas et al.'s group totals (33.9/26.1/10.8/9.7%) are permutation importance, although they are
labelled as contribution.

**Comparability of the pixel-level results.** The pixel-level AUCs in Table 2 are **not**
comparable with the reference AUC. They differ in:
- population: all or forest pixels vs presence–background cells;
- resolution: 1 km vs 0.25°;
- label: pooled 2000–2022 vs 2020 presences;
- evaluation design.

Earlier claims of "beating" the reference are withdrawn.

### 4.10 National susceptibility map

The final map (`results/final/maps/Susceptibility_RF_v2_score.tif`; Fig. H) is the Track-A RF v2
score at 1/120°. Five classes use quantile breaks over forest pixels (0.043 / 0.210 / 0.582 /
0.893). The GeoTIFF is tagged as a *relative score, not a calibrated probability*. The map's mean
score (0.135) exceeds the prevalence (0.065), and no calibrated uncertainty estimate was
established.

---

## 5. Discussion

### 5.1 Why the governing equation did not help

Four observations together explain the negative result.

1. **The target is static, but the equation is dynamic.** Susceptibility, as mapped here and in
   the reference study, is a long-run property of a location. The data loss rewards a stable
   ranking of cells, so the easiest optimum is a near-static field. The trained fields are
   exactly that: median |∂u/∂t| ≤ 0.002 per month. The PDE residual cannot then be small: for a
   static u, the right-hand side must itself vanish, and nothing in the data enforces that. The
   residual is absorbed rather than satisfied.
2. **The terms map onto covariates, not onto processes observed at this scale.** Monthly MODIS
   detections at 12 km record where ignitions and detectable fires occur. They do not record how
   fire spreads between cells within a month. Diffusion and advection describe spread, and a
   susceptibility label contains little information about spread. The diffusion term's ≤ 0.3%
   share of the right-hand side reflects this.
3. **Advection works as an elevation channel.** Adding advection recovers most of what diffusion
   costs. The term magnitudes show that advection carries 83–98% of the right-hand side in every
   configuration. The model uses v = c·∇E to route terrain information, and in the full predictor
   set terrain is redundant (§4.7).
4. **Spatial transfer is limited by the operator, not the inputs.** On identical cells and
   covariates, even logistic regression transfers far better across blocks and regions than the
   operator does (Table 4). The global Fourier representation, trained on one national domain,
   may couple distant regions and fit domain-specific structure that does not transfer. The
   network without physics shows the same weakness. The physics terms neither cause it nor cure
   it.

### 5.2 Implications for physics-informed hazard mapping

This study was designed to find a physics benefit if one exists. It includes the conditions
under which such benefits are usually expected to appear: spatial, regional and temporal shift,
sparse positives and multiple seeds. The benefit did not appear. Physics-informed models are not
ruled out for fire. Spread prediction, where the equation describes the observed process, is a
different problem.

The negative result does, however, set a reporting standard for susceptibility applications:
- **the same network without physics:** without this arm, the accuracy of a PINO cannot be
  attributed to its physics;
- **classical models on identical cells and covariates:** a different grid, population or
  predictor set makes comparisons meaningless (our own earlier, unmatched comparison was one such
  case);
- **covariate-free persistence baselines for temporal claims:** fire recurrence alone is a strong
  predictor;
- **a physical-consistency check of the trained field:** a physics-regularised model can score
  well while violating its equation.

### 5.3 Methodological lessons from the pipeline

Several defects that were corrected here are generic and easy to miss:
- **Half-pixel misregistration.** Rounding against a raster edge instead of taking the floor
  moved three quarters of the fire points into a neighbouring pixel. Correcting it raises AUC,
  because the label becomes spatially consistent with the predictors.
- **Degenerate predictors.** A time-mean anomaly over its own baseline is degenerate. Such
  predictors add dimensions but no information.
- **Invalid trend tests.** Mann–Kendall on seasonal or smoothed series produces significance
  counts that are wrong by an order of magnitude or more. Here Seasonal Kendall found 6.7× (day
  LST) to 117× (night LST) more significant pixels. For air temperature it reversed the
  conclusion.
- **Population mismatch.** Reporting all-pixel AUC for a forest-only label inflates accuracy,
  because non-forest pixels are trivially negative. Forest-pixel AUC is the informative number.

None of these corrections changed any model's AUC by more than 0.012. The substantive
conclusions above therefore do not depend on them. However, the intermediate statistics that
underpinned earlier descriptive claims were invalid.

### 5.4 Limitations

1. **Model family.** One operator family (FNO) and one equation (CDR) were tested. The results do
   not rule out other architectures, local operators, or equations fitted to spread data.
2. **Hyperparameters.** The operator's loss weights, spectral modes and window length were not
   tuned beyond the validated weight decay and early stopping. A full nested search was not run,
   and a larger search could narrow, but is unlikely to reverse, a 0.26–0.39 AUC gap under spatial
   hold-out.
3. **Transductive covariates.** The FNO's global receptive field means the network sees the
   covariates (never the labels) of test cells, and the dryness index is standardised over all
   months. Both are disclosed. Both would, if anything, favour the operator.
4. **Grid mismatch.** The operator runs at 12 km and the classical models at 1 km. All operator
   comparisons are therefore made on the operator's own 12 km cells (Table 4), never across grids.
5. **Label definition.** The label is pooled detection of fire, not burned area or severity. The
   burned-area series (MCD64A1) correlates with annual forest-fire counts (r = 0.93, forest-masked),
   but it was not used as a label.
6. **Calibration and uncertainty.** The RF map is a relative score, not a calibrated probability.
   No calibrated uncertainty estimate was established.
7. **Untested theoretical properties.** Resolution independence, zero-shot super-resolution and
   instance-wise fine-tuning of the operator were not evaluated. They are not claimed.
8. **Explainability.** SHAP values were not computed. Permutation importance and partial
   dependence were used instead.

---

## 6. Conclusions

We asked whether embedding a convection–diffusion–reaction equation in a neural operator improves
forest-fire susceptibility mapping over India. The design controlled for architecture, inputs,
grid and evaluation split. The answers are:
- **RQ1 (physics vs no physics):** the physics terms did not improve discrimination on any of four
  evaluation tracks, and reduced it where the difference was resolvable.
- **RQ2 (classical models on identical inputs):** they outperformed the operator on every track,
  by 0.04 on random splits and by 0.26–0.39 under spatial and regional hold-out.
- **RQ3 (unseen years):** the operator scored below covariate-free persistence baselines.
- **RQ4 (physical consistency):** the trained field was effectively static and did not satisfy its
  own equation.

The best susceptibility model in this study is a Random Forest on a corrected 55-predictor
pipeline (forest-pixel AUC 0.897; 0.838 under 2° spatial hold-out). Its advantage comes from the
model family and the predictor set.

The two positive contributions are the controlled-evaluation design and the physical-consistency
diagnostic. We recommend both as minimum reporting practice for physics-informed hazard models.
The corrected national pipeline, the Biswas-style reimplementation (0.893 ± 0.009 vs the reported
0.879) and the full set of predictions, checkpoints and scripts are released for reuse.

---

## Data and code availability

- **Step repositories.** The pipeline (Steps 1–8) is organised as independent repositories:
  - fire points;
  - NDVI;
  - LST;
  - FLDAS and land cover;
  - terrain;
  - accessibility;
  - integration and classical models;
  - CDR-PINO.

  Each repository's notebook or build script produces the v2 outputs reported here.
- **Integration and CDR-PINO code.** `Integrated_Analysis/step7_models.py` refits every classical
  experiment. `Physics_Informed_FireRisk_Model/cdr_pinn/run_unified_protocol.py` retrains all 102
  CDR-PINO runs, and `analyze_unified.py` rebuilds Tables 3–5 from the saved predictions.
- **Audit.** The complete audit is in `results/`: recalculations, the v2 feature table, run
  registry, figures, and the reproducibility report with the run order.
- **Raw data.** The raw data are public (NASA LP DAAC / FIRMS, GES DISC, ESA-CCI/C3S, USGS SRTM,
  Geofabrik OSM). Download instructions are given in each repository's README.

## Figures (generated from result files; `results/final/figures/`)

- **Fig. A** (`FigA_contribution_1km.png`): paired ΔAUC decomposition (Table 6).
- **Fig. B** (`FigB_cdr_ablation_seeds.png`): physics ablation, seed-level AUC per track (Table 3).
- **Fig. C** (`FigC_same_cells_cdr_vs_classical.png`): CDR-PINO vs classical models on identical
  cells (Table 4).
- **Fig. D** (`FigD_B3_vs_nulls.png`): unseen years, CDR-PINO vs persistence baselines (§4.5).
- **Fig. E** (`FigE_maxent_sample_size.png`): MaxEnt training-size sensitivity (§4.8).
- **Fig. F** (`FigF_biswas_importance_comparison.png`): permutation importance, reimplementation vs
  Biswas et al. (§4.9).
- **Fig. G** (`FigG_cdr_term_magnitudes.png`): term magnitudes and PDE residual of trained
  checkpoints (Table 5).
- **Fig. H** (`FigH_final_map_RF_v2.png`): national susceptibility score, RF v2 (§4.10).

---

## References

Anselin, L. (1995). Local indicators of spatial association—LISA. *Geographical Analysis*, 27(2),
93–115. `[cite-verify]`

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and
powerful approach to multiple testing. *Journal of the Royal Statistical Society B*, 57(1),
289–300. `[cite-verify]`

Biswas, U., Mahato, S., & Joshi, P.K. (2025). Spatial prediction of forest fires in India: a
machine learning approach for improved risk assessment and early warning systems. *Environmental
Science and Pollution Research*, 32(8), 4856–4878. DOI: 10.1007/s11356-025-35982-8.
`[cite-confirmed]`

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. `[cite-verify]`

Dabrowski, J.J., Pagendam, D.E., Hilton, J., Sanderson, C., MacKinlay, D., Huston, C., Bolt, A., &
Kuhnert, P. (2023). Bayesian Physics Informed Neural Networks for data assimilation and
spatio-temporal modelling of wildfires. *Spatial Statistics*, 55, 100746.
DOI: 10.1016/j.spasta.2023.100746. `[cite-confirmed]`

DeLong, E.R., DeLong, D.M., & Clarke-Pearson, D.L. (1988). Comparing the areas under two or more
correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*,
44(3), 837–845. `[cite-verify]`

Gholamnia, K., Tahmasebi Moghaddam, H., Einali, G., Akbari Monfared, B., Lorestani, G.,
Ghorbanzadeh, O., & Einali, J. (2026). Uncertainty-aware machine learning via Dempster–Shafer
theory for wildfire susceptibility mapping. *Spatial Information Research*, 34(4), 35.
DOI: 10.1007/s41324-026-00692-x. `[cite-confirmed]`

Giglio, L., Schroeder, W., & Justice, C.O. (2016). The collection 6 MODIS active fire detection
algorithm and fire products. *Remote Sensing of Environment*, 178, 31–41. `[cite-verify]`

Guria, R., Mishra, M., Mohanta, S., & Paul, S. (2025). Forest fire probability zonation using dNBR
and machine learning models: a case study at the Similipal Biosphere Reserve (SBR), Odisha,
India. *Environmental Science and Pollution Research*, 32(59), 31375–31396.
DOI: 10.1007/s11356-025-35976-6. `[cite-confirmed]`

Gupta, P., Shukla, A.K., & Shukla, D.P. (2025). Machine learning-based forest fire susceptibility
mapping of Southern Mizoram, a part of Indo-Burma Biodiversity Hotspot. *Environmental Science and
Pollution Research*, 32(59), 31433–31454. DOI: 10.1007/s11356-025-36621-y. `[cite-confirmed]`

Hang, H.T., Mallick, J., Alqadhi, S., Bindajam, A.A., & Abdo, H.G. (2024). Exploring forest fire
susceptibility and management strategies in Western Himalaya: Integrating ensemble machine
learning and explainable AI. *Environmental Technology & Innovation*, 35, 103655.
DOI: 10.1016/j.eti.2024.103655. `[cite-confirmed]`

Hirsch, R.M., Slack, J.R., & Smith, R.A. (1982). Techniques of trend analysis for monthly water
quality data. *Water Resources Research*, 18(1), 107–121. `[cite-verify]`

Horn, B.K.P. (1981). Hill shading and the reflectance map. *Proceedings of the IEEE*, 69(1),
14–47. DOI: 10.1109/PROC.1981.11918. `[cite-confirmed]`

İban, M.C., & Aksu, O. (2024). SHAP-Driven Explainable Artificial Intelligence Framework for
Wildfire Susceptibility Mapping Using MODIS Active Fire Pixels: A Case Study in Izmir, Türkiye.
*Remote Sensing*, 16(15), 2842. DOI: 10.3390/rs16152842. `[cite-confirmed]`

Jiang, P., Yang, Z., Wang, J., Huang, C., Xue, P., Chakraborty, T.C., Chen, X., & Qian, Y.
(2023). Efficient Super-Resolution of Near-Surface Climate Modeling Using the Fourier Neural
Operator. *Journal of Advances in Modeling Earth Systems*, 15(7), e2023MS003800.
DOI: 10.1029/2023MS003800. `[cite-confirmed]`

Kanda Naveen Babu, Gour, R., Kurian Ayushi, Ayyappan, N., & Parthasarathy, N. (2023).
Environmental drivers and spatial prediction of forest fires in the Western Ghats biodiversity
hotspot, India: An ensemble machine learning approach. *Forest Ecology and Management*, 540,
121057. DOI: 10.1016/j.foreco.2023.121057. `[cite-confirmed]`

Kantarcioglu, O., Schindler, K., & Kocaman, S. (2023). Forest Fire Susceptibility Assessment with
Machine Learning Methods in North-East Türkiye. *ISPRS Archives*, XLVIII-M-1-2023, 161–167.
DOI: 10.5194/isprs-archives-xlviii-m-1-2023-161-2023. `[cite-confirmed]`

Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021).
Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422–440.
DOI: 10.1038/s42254-021-00314-5. `[cite-confirmed]`

Kurth, T., Subramanian, S., Harrington, P., Pathak, J., Mardani, M., Hall, D., Miele, A.,
Kashinath, K., & Anandkumar, A. (2023). FourCastNet: Accelerating Global High-Resolution Weather
Forecasting Using Adaptive Fourier Neural Operators. *Proceedings of PASC '23*.
DOI: 10.1145/3592979.3593412. `[cite-confirmed]`

Li, Z., Zheng, H., Kovachki, N., Jin, D., Chen, H., Liu, B., Azizzadenesheli, K., & Anandkumar,
A. (2023). Physics-informed neural operator for learning partial differential equations.
arXiv:2111.03794. `[cite-confirmed]`

Malik, F.A., Mushtaq, F., Farooq, M., Guite, L.T.S., Kanga, S., Meraj, G., Singh, S.K., & Kumar,
P. (2025). Assessing forest fire vulnerability with fuzzy-AHP: insights from Poonch forest
division, Jammu and Kashmir. *Discover Forests*, 1(1), 4. DOI: 10.1007/s44415-025-00004-5.
`[cite-confirmed]`

Meraj, G., Hashimoto, S., Dasgupta, R., & Mitra, B.K. (2025). Ecological Risk Assessment and
Management of Forest Fires in Tamil Nadu, India: A MaxEnt Model-Based Approach for Strategic
Resource Allocation and Fire Mitigation. *Risk Analysis*, 45(11), 3604–3625.
DOI: 10.1111/risa.70098. `[cite-confirmed]`

Phillips, S.J., Anderson, R.P., & Schapire, R.E. (2006). Maximum entropy modeling of species
geographic distributions. *Ecological Modelling*, 190(3–4), 231–259. `[cite-verify]`

Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Physics-informed neural networks: A deep
learning framework for solving forward and inverse problems involving nonlinear partial
differential equations. *Journal of Computational Physics*, 378, 686–707. `[cite-confirmed]`

Read, J.S., Jia, X., Willard, J., Appling, A.P., Zwart, J.A., Oliver, S.K., Karpatne, A.,
Hansen, G.J.A., Hanson, P.C., Watkins, W., Steinbach, M., & Kumar, V. (2019). Process-Guided Deep
Learning Predictions of Lake Water Temperature. *Water Resources Research*, 55(11), 9173–9190.
DOI: 10.1029/2019WR024922. `[cite-confirmed]`

Roberts, D.R., Bahn, V., Ciuti, S., Boyce, M.S., Elith, J., Guillera-Arroita, G., et al. (2017).
Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic
structure. *Ecography*, 40(8), 913–929. `[cite-verify]`

Sannigrahi, S., et al. (2018). ESA-CCI/C3S forest land-cover class mapping (forest-class
definition used for the fire filter). `[cite-verify — full record in METHODOLOGY.md]`

Santana Neto, V.P., Nunes, A.J.N., Torres, F.T.P., Gleriani, J.M., & Cosenza, D.N. (2025).
Assessing Wildfire Susceptibility and Driving Variables in Portugal Using Machine Learning
Approach. *Journal for Nature Conservation*, 86, 126956. DOI: 10.1016/j.jnc.2025.126956.
`[cite-confirmed]`

Sarkar, M.S., Majhi, B.K., Pathak, B., Biswas, T., Mahapatra, S., Kumar, D., Bhatt, I.D.,
Kuniyal, J.C., & Nautiyal, S. (2024). Ensembling machine learning models to identify forest
fire-susceptible zones in Northeast India. *Ecological Informatics*, 81, 102598.
DOI: 10.1016/j.ecoinf.2024.102598. `[cite-confirmed]`

Sen, P.K. (1968). Estimates of the regression coefficient based on Kendall's tau. *Journal of the
American Statistical Association*, 63(324), 1379–1389. `[cite-verify]`

Sun, A.Y., Jiang, P., Shuai, P., & Chen, X. (2024). Bridging Hydrological Ensemble Simulation and
Learning Using Deep Neural Operators. *Water Resources Research*, 60(10), e2024WR037555.
DOI: 10.1029/2024WR037555. `[cite-confirmed]`

Symeonidis, P., Vafeiadis, T., Ioannidis, D., & Tzovaras, D. (2025). Wildfire Susceptibility
Mapping in Greece Using Ensemble Machine Learning. *Earth*, 6(3), 75. DOI: 10.3390/earth6030075.
`[cite-confirmed]`

Uthappa, A.R., Das, B., Raizada, A., Kumar, P., Jha, P., & Prasad, P.V.V. (2025). Forest Fire
Susceptibility Mapping Using Multi-Criteria Decision Making and Machine Learning Models in the
Western Ghats of India. *Journal of Environmental Management*, 379, 124777.
DOI: 10.1016/j.jenvman.2025.124777. `[cite-confirmed]`

Vogiatzoglou, K., Papadimitriou, C., Bontozoglou, V., & Ampountolas, K. (2025). Physics-informed
neural networks for parameter learning of wildfire spreading. *Computer Methods in Applied
Mechanics and Engineering*, 434, 117545. DOI: 10.1016/j.cma.2024.117545. `[cite-confirmed]`

Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies
in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5), A3055–A3081.
`[cite-verify]`

Zakari, R.Y., Malik, O.A., & Ong, W.-H. (2025). Machine learning-driven wildfire susceptibility
mapping in New South Wales, Australia using remote sensing and explainable artificial
intelligence. *Natural Hazards*, 121(13), 15331–15357. DOI: 10.1007/s11069-025-07395-w.
`[cite-confirmed]`

Zhang, G., Wang, M., & Liu, K. (2019). Forest Fire Susceptibility Modeling Using a Convolutional
Neural Network for Yunnan Province of China. *International Journal of Disaster Risk Science*,
10(3), 386–403. DOI: 10.1007/s13753-019-00233-1. `[cite-confirmed]`
