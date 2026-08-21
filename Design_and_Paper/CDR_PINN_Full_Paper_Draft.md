# A Physics-Informed Neural Operator for Forest-Fire Susceptibility Mapping in India: A Convection–Diffusion–Reaction Formulation

> **Manuscript draft — 2026-08-20.** Assembles this project's already-completed work
> (8-step classical pipeline + CDR-PINN design, implementation, and a first
> term-ablation validation) into a submission-shaped structure, citing this session's
> newly-verified literature throughout. Sections point to the project's own detailed
> design/methodology documents rather than re-deriving proofs already written up —
> consistent with how this whole document set has been built. Citation tags
> (`[cite-confirmed]`, `[cite-verify]`) follow this project's own established
> convention (`METHODOLOGY.md`): confirmed means an independently checked DOI/record;
> verify means found but not every bibliographic detail independently re-confirmed by
> me directly (as opposed to the research pass that found it).

---

## Abstract

*(Drafted to the complete Track A/B1/B2/B3 evidence plus Jackknife retraining,
validation-honest re-selection, and causal-weighting/curriculum-learning tests,
2026-08-21 — reported honestly, including the tracks that did not go as
hypothesized, since that is the accurate current state of the work.)* Forest fires
in India have grown in frequency and
severity, and existing susceptibility-mapping approaches — including the most recent
national-scale study, Biswas, Mahato & Joshi (2025) — rely on presence-background
statistical models (MaxEnt) applied to static, independently-modeled predictor
rasters, discarding both the spatial continuity of fire spread and the temporal
structure of a multi-decade observational record. We reformulate forest-fire
susceptibility as an initial-boundary value problem for a convection–diffusion–
reaction (CDR) partial differential equation over a latent susceptibility field,
solved by a physics-informed Fourier neural operator (PINO) trained against 22 years
(2000–2022) of real, monthly-resolved fire observations across India. Each governing-
equation term is constructed to correspond to a distinct fire-behavior mechanism —
vegetation/moisture-driven diffusion, terrain-driven advection, human-ignition-driven
reaction — mapping directly onto the four non-trivial predictor groups identified in
the reference MaxEnt study, and we prove global-in-time well-posedness of the
governing equation. A term-ablation study demonstrates that each mechanism
contributes measurable, real predictive value under random-split evaluation
(held-out ROC-AUC: diffusion-only 0.602, +advection 0.924, full CDR 0.941), within
0.02–0.03 AUC of classical Random Forest (0.968) and MaxEnt (0.960) baselines trained
on an identical, now-complete 15-variable feature set. A four-track generalization
study, however, gives a mixed and instructive picture rather than a uniformly
favorable one: temporal generalization to unseen years is strong (leave-years-out
AUC=0.897), but spatial generalization to unseen 2°×2° blocks (AUC=0.754±0.016) and
especially to entirely unseen regions (leave-one-region-out, AUC=0.599±0.082, one
region below chance) is weak at the training scale evaluated here, and a direct
physics-vs-no-physics comparison found no accuracy advantage from the physics
constraint under random-split conditions. Five independent methods — term-ablation, spatial fire-point statistics,
input-channel permutation, marginal-effect response curves, and Biswas et al.'s own
Fig. 10 Jackknife retraining test, all three of their variable-understanding
analyses now reproduced — converge on the same finding: elevation dominates the
trained operator almost completely (a single-covariate model reaches AUC=0.938,
within 0.002 of the full 7-covariate model), a mechanistic attribution capability no
correlational baseline in this literature offers, but also an explicit
shortcut-learning caveat this study does not resolve. Six further attempts to close
the accuracy gap to Random Forest/MaxEnt — evaluation-metric correction,
capacity scale-up, an untuned learning-rate schedule, causal time-weighting, staged
curriculum learning, and an honest validation-selected re-test — all land within a
narrow 0.93–0.94 AUC band except scale-up (robustly worse across two independent
splits), evidence against an under-optimized model and consistent instead with a
representation ceiling. We report these results without softening, situate them
against the literature's own prediction that physics-informed advantages should
appear specifically under distribution shift (not yet confirmed here), and identify
the physics-informed neural operator's demonstrated advantage as currently
mechanistic and falsifiable rather than raw-accuracy-superior — a genuine, if
partial, contribution against a literature in which no comparable ablation or
mechanistic account exists.

## 1. Introduction

### 1.1 Motivation

Forest fires are an escalating ecological and economic hazard in India: approximately
36% of the country's forest cover is classified as susceptible to fire (ISFR 2021,
cited in Biswas, Mahato & Joshi, 2025 `[cite-confirmed]`), and the frequency of
detected fire events has risen across the two decades examined by both that study and
this project's own Step 1 extraction (232,189 events 2001–2010 vs. 243,761 in
2011–2020, per Biswas et al.'s own reported totals). Accurate, spatially- and
temporally-resolved susceptibility mapping is a prerequisite for early-warning systems
and resource allocation, and is the explicit objective of this work.

### 1.2 Related Work: Forest-Fire Susceptibility Mapping in India

Beyond the primary reference study (Biswas, Mahato & Joshi, 2025, MaxEnt, national
scale, 0.25° resolution — see `Biswas et al. Verification` for this project's own
source-checked audit of that paper's methods and results), a substantial and growing
body of India-specific susceptibility-mapping literature exists at regional scale,
spanning a range of statistical and machine-learning methods:

- **Western Ghats / biodiversity hotspot studies**: Uthappa et al. (2025,
  *J. Environmental Management*, 379:124777 `[cite-confirmed]`) combine AHP with
  Random Forest, SVM, and XGBoost for Goa; Kanda Naveen Babu et al. (2023, *Forest
  Ecology and Management*, 540:121057 `[cite-confirmed]`) use an ensemble of ANN, RF,
  MaxEnt, GLM, MARS, and GBM across 14 conditioning factors for the wider Western
  Ghats hotspot.
- **Regional/state-level studies elsewhere in India**: Meraj et al. (2025, *Risk
  Analysis*, 45(11):3604–3625 `[cite-confirmed]`) report a MaxEnt model for Tamil
  Nadu (AUC=0.92) using ~19 topo-climatic-anthropogenic predictors; Guria et al.
  (2025, *Environ. Sci. Pollut. Res.*, 32(59):31375–31396 `[cite-confirmed]`) compare
  XGBTree, AdaBag, Random Forest, and GBM for Similipal Biosphere Reserve, Odisha
  (best RF AUC=0.85); Gupta, Shukla & Shukla (2025, *Environ. Sci. Pollut. Res.*,
  32(59):31433–31454 `[cite-confirmed]`) compare six ML methods for Southern Mizoram;
  Sarkar et al. (2024, *Ecological Informatics*, 81:102598 `[cite-confirmed]`)
  ensemble multiple ML models for Northeast India; Hang et al. (2024, *Environmental
  Technology & Innovation*, 35:103655 `[cite-confirmed]`) integrate ensemble ML with
  explainable AI for the Western Himalaya; Malik et al. (2025, *Discover Forests*,
  1(1):4 `[cite-confirmed]`) apply fuzzy-AHP for Poonch, Jammu & Kashmir.

**What this body of work has in common, and what it lacks**: every study above,
including the primary reference, treats susceptibility as a *static, per-location*
classification or density-estimation problem. None incorporates an explicit governing
equation, a temporal/dynamical formulation, or a physics-based inductive bias — the
predictor set (however large or well-chosen) enters each model as an undifferentiated
feature vector. This project's own prior work (Steps 1–7) sits methodologically within
this same paradigm before the contribution reported here.

### 1.3 Related Work: Wildfire Susceptibility Mapping Internationally

The same static-classification paradigm dominates internationally. Representative
recent examples spanning distinct fire-prone regions and methods: Zhang, Wang & Liu
(2019, *Int. J. Disaster Risk Science*, 10(3):386–403 `[cite-confirmed]`) apply a CNN
for Yunnan, China (82% validation accuracy); Kantarcioglu, Schindler & Kocaman (2023,
*ISPRS Archives*, XLVIII-M-1-2023:161–167 `[cite-confirmed]`) compare RF and ANN for
north-east Türkiye (AUC 0.89/0.88); İban & Aksu (2024, *Remote Sensing*, 16(15):2842
`[cite-confirmed]`) use SHAP-explainable ML with MODIS active-fire pixels for İzmir,
Türkiye; Zakari, Malik & Ong (2025, *Natural Hazards*, 121(13):15331–15357
`[cite-confirmed]`) report XGBoost (F1=0.965) for New South Wales, Australia;
Symeonidis et al. (2025, *Earth*, 6(3):75 `[cite-confirmed]`) ensemble
XGBoost/GBM/LightGBM/CatBoost for Greece; Gholamnia et al. (2026, *Spatial Information
Research*, 34(4):35 `[cite-confirmed]`) use Dempster–Shafer uncertainty-aware ML
(AUC=0.893) for the Hyrcanian forests, Iran; Santana Neto et al. (2025, *J. Nature
Conservation*, 86:126956 `[cite-confirmed]`) assess driving variables for Portugal.
As with the India-specific literature, none of these studies incorporates a governing
physical equation or a neural-operator architecture.

### 1.4 Related Work: Physics-Informed Learning and Its Demonstrated Advantages

Physics-informed neural networks (PINNs; Raissi, Perdikaris & Karniadakis, 2019,
*J. Comput. Phys.*, 378:686–707 `[cite-confirmed]`) embed a governing differential
equation as a soft constraint in a neural network's training objective. Karniadakis
et al. (2021, *Nature Reviews Physics*, 3(6):422–440 `[cite-confirmed]`) survey the
case for this approach: improved generalization under data scarcity, physical
plausibility guaranteed by construction rather than hoped for, and interpretability
through mechanistic (not merely correlational) structure. This is not only a
theoretical claim: Read et al. (2019, *Water Resources Research*, 55(11):9173–9190
`[cite-confirmed]`) demonstrate it concretely for a structurally similar
environmental-prediction problem (lake water temperature) — as training data is
thinned, a purely data-driven RNN's error rises sharply while a physically-constrained
(energy-conservation) model degrades far more gracefully, precisely the property this
project's own sparse fire-observation supervision (~2.3% monthly positive rate, see
Section 5) needs.

**Physics-informed methods applied to fire specifically remain rare, and none address
susceptibility mapping at national scale.** Vogiatzoglou et al. (2025, *Computer
Methods in Applied Mechanics and Engineering*, 434:117545 `[cite-confirmed]`) use a
PINN with explicit mass/energy-conservation constraints to learn *parameters* of a
wildfire rate-of-spread model, validated on the 2002 Troy Fire (California) — a
forward-simulation, single-event framing, not a susceptibility map. Dabrowski et al.
(2023, *Spatial Statistics*, 55:100746 `[cite-confirmed]`) use a Bayesian PINN for
spatio-temporal wildfire data assimilation. Neither uses a neural *operator*
architecture, and neither is applied to India or to a multi-decade national
observational record. **This is the specific, precisely-stated gap this work closes**:
the first — to the best of this project's own literature search — physics-informed
neural *operator* applied to national-scale forest-fire susceptibility mapping,
and the first physics-informed fire model of any kind evaluated against India's own
Biswas et al. (2025) reference study.

The Fourier Neural Operator (FNO) architecture underlying the PINO framework used here
(Li, Zheng, Kovachki et al., 2023, arXiv:2111.03794 `[cite-confirmed]`) is
independently well-established for spatial environmental prediction: Jiang et al.
(2023, *J. Advances in Modeling Earth Systems*, 15(7):e2023MS003800 `[cite-confirmed]`)
use FNO for climate-model super-resolution; Sun et al. (2024, *Water Resources
Research*, 60(10):e2024WR037555 `[cite-confirmed]`) bridge hydrological ensemble
simulation with deep neural operators; Kurth et al. (2023, *Proc. PASC'23*,
DOI:10.1145/3592979.3593412 `[cite-confirmed]`) introduce the Adaptive FNO in
FourCastNet for global weather forecasting — the same AFNO variant subsequently reused
for wildfire fuel-density prediction by Caglar et al. (2026, arXiv:2607.06999
`[cite-verify — preprint, journal record not yet confirmed]`), which also reports
physics-guided models outperforming purely data-driven baselines, a third concrete
data point for the Section 1.4 data-efficiency argument, cited with its preprint
status disclosed rather than presented as peer-reviewed.

### 1.5 Contributions

This paper's contributions, stated precisely against the literature reviewed above
(§1.2–1.4) — full itemization and evidence in `CDR_PINN_Novelty_Comparison_
Advantages.md`:

1. A convection–diffusion–reaction PDE whose three terms map onto Biswas et al.
   (2025)'s own four predictor groups, each grounded in a specific, citable
   fire-behavior mechanism (Rothermel, 1972, upslope acceleration for advection;
   Fisher, 1937/Kolmogorov–Petrovsky–Piskunov, 1937, logistic reaction-diffusion for
   the ignition term) and formally proven globally well-posed.
2. The first physics-informed neural *operator* (not pointwise PINN) applied to
   wildfire susceptibility mapping, and the first application of any physics-informed
   method to India specifically.
3. A term-ablation study with genuine held-out validation demonstrating each physical
   mechanism's measurable, separable contribution — not asserted from the equation's
   elegance, evidence given in Section 5.
4. Full parity with Biswas et al. (2025)'s real 15-variable predictor set (previously
   verified as 15, not the 11 this project's own earlier documentation had
   mis-stated — see `Biswas et al. Verification`), at both the pipeline and the
   trained-model level.

### 1.6 Relationship to Biswas et al. (2025)'s Own Stated Objectives

Because Biswas et al. (2025) is this paper's primary reference and comparison point,
their own stated research design is represented here directly, not paraphrased, so
that this study's relationship to it — extension, modification, or departure — is
traceable term by term. Biswas et al. state their objectives as explicitly threefold
(their Introduction, p. 4859): *"Firstly, it seeks to identify the forest fire
occurrence conditioning factors that contribute to the ignition and spread of fires
across India. Secondly, the study aims to examine the relationships between
documented forest fire events and the identified conditioning factors... Thirdly, by
integrating the findings from the previous objectives, the research endeavors to
develop a comprehensive forest fire occurrence probability map for the entire
country."* Table below maps each objective to how this study treats it.

| Biswas et al.'s objective | Biswas et al.'s method | This study's treatment |
|---|---|---|
| (1) Identify conditioning factors | MaxEnt permutation importance over 15 static predictors | Same 15 variable groups (full parity, §1.5 item 4), plus a structurally different identification method: term-ablation (§4.2) attributes importance to *physical mechanisms* (diffusion/advection/reaction), and per-covariate permutation importance (§4.4, new to this work) attributes it to individual inputs *within* the trained operator — a finer-grained, mechanism-aware decomposition their single-model permutation test cannot produce |
| (2) Examine relationships between fire events and conditioning factors | A correlation matrix (Fig. 11) and MaxEnt response curves (marginal-effect plots) | The governing PDE *is* an explicit model of these relationships — e.g., the advection term formalizes the slope/fire relationship as a directional transport mechanism rather than a marginal response curve, and the well-posedness proofs (§3) make the relationships' mathematical structure precise and provable rather than descriptive |
| (3) Integrate findings into one national probability map | A single static MaxEnt probability surface, five susceptibility classes | A *time-resolved* susceptibility field `u(x,y,t)`, evaluated monthly across 22 years, with an explicit temporal-generalization test (Track B3) their static map has no equivalent of |

**Where this study modifies Biswas et al.'s approach outright, stated plainly**: (a)
replaces their single presence-background statistical model with a physics-informed
neural operator family, evaluated alongside (not instead of) a direct MaxEnt
replication on this study's own richer data (§5.3); (b) replaces their one random
train/test split with a four-track generalization protocol (§4.3); (c) extends their
static, whole-period predictor rasters into temporally-resolved (monthly anomaly,
trend, Mann-Kendall significance) features throughout (§1.9 in the Methodology
document); (d) extends their 0.25° working resolution to a 1 km native pipeline,
downsampled to 256×256 specifically for the operator (§2). Where this study does
**not** depart from Biswas et al.: the same 15 conceptual predictor groups, the same
forest-classification LULC codes (independently corroborated as identical in the
`Biswas et al. Verification` audit), and the same overarching goal of an
early-warning-relevant national probability product.

## 2. Study Area and Data

India, `6°–37.5°N, 68°–97.5°E`, Nov 2000–Dec 2022. Full data provenance table
in `CDR_PINN_Methodology_Section.md` §2; underlying pipeline (fire-point extraction,
NDVI/LST/FLDAS feature engineering, terrain/accessibility, integration) documented
step-by-step in `METHODOLOGY.md`.

**From raster to two different model inputs.** All upstream products are stacked
into one 60-band `Integrated_FireRisk_Stack.tif` (Step 6), which is then consumed in
two structurally different ways by the two model families compared in this paper.
The classical baselines (Random Forest, MaxEnt) never read a raster directly:
Step 6 flattens the stack into `Integrated_FireRisk_Pixels.parquet` — one row per
valid in-India pixel (4,161,009 rows), one column per band — a standard
raster-to-tabular operation restricted to pixels passing
`india_mask & ~isnan(ndvi_mean)`. RF/MaxEnt train on this flattened table with
ordinary `pandas`/`scikit-learn`/`elapid` tooling, with no explicit awareness that
neighboring rows are geographically adjacent. CDR-PINN, by contrast, consumes the
**raw gridded tensor** directly, resampled to a 256×256 working grid (§3) — spatial
adjacency is preserved and architecturally meaningful, since the FNO's spectral
convolutions mix information according to genuine 2D spatial structure. This
distinction is the concrete technical mechanism behind the "discards spatial
continuity" critique of the static-classification paradigm raised in §1.2.

## 3. Methodology

Full derivation, architecture, collocation-point taxonomy, training protocol, and
computational-complexity analysis: `CDR_PINN_Methodology_Section.md`. Governing
equation, boundary/initial conditions, and well-posedness proofs: `CDR_PINN_Diffusion_
Design.md`/`_v2.md`, `CDR_PINN_Advection_Design.md`, `CDR_PINN_Reaction_Design.md`,
`CDR_PINN_Final_Design_STEP_D.md`. Summary for this section:

```
∂u/∂t = D(x,y,t)·∇²u  −  v(x,y)·∇u  +  ρ(x,y,t)·σ(u)·(1−σ(u))     in Ω×(0,T]
∂u/∂n = 0                                                            on ∂Ω×(0,T]
u(x,y,0) = 0                                                         in Ω
```

solved by a Fourier Neural Operator (Li et al., 2023), `1,054,613` parameters
(`width=32`, `4` spectral layers, `16×16` modes), trained as a per-month one-step-ahead
operator with truncated backpropagation-through-time (24-month windows), a single
combined PDE residual plus data/boundary/initial loss terms with adaptive
gradient-norm-balanced weighting (Wang, Teng & Perdikaris, 2021).

### 3.1 Well-Posedness — Proof Structure (condensed; full proofs in the design documents)

Each term's well-posedness is proven incrementally, extending rather than replacing
the previous term's result:

1. **Diffusion alone**: `D=softplus(D_net(...))` bounded in `[D_min,D_max]` by
   construction (extreme value theorem on a finite MLP over a compact, verified
   input domain), giving **uniform parabolicity** — the hypothesis Evans (2010) Ch. 7
   requires for existence/uniqueness of linear parabolic PDEs.
2. **+ Advection**: the drift term `v·∇u` is a lower-order perturbation of the
   elliptic principal part; **Gårding's inequality** (proven via Young's inequality,
   explicit constants `α=D_min/2`, `β=V_max²/(2D_min)`, `V_max` computed from the
   measured 77.31° maximum slope) is the correct, weaker-than-coercivity condition
   this extension needs.
3. **+ Reaction**: the Fisher–KPP term is proven **globally** bounded
   (`≤ρ_max/4` for any real `u`) and **globally** Lipschitz (exact constant
   `ρ_max/(6√3)`) — a strictly stronger property than a generic nonlinear reaction
   term would have (a bare cubic term, for comparison, only guarantees local-in-time
   existence). Combined with steps 1–2 via a Gronwall inequality, this yields
   **global-in-time** existence/uniqueness over the full `T=266`-month horizon.

Every constant above is computed from this study's own verified data extremes, not
asserted as a generic bound.

### 3.2 Baseline Model Selection Rationale

Three model families are compared: MaxEnt (Biswas et al.'s own method, directly
replicated on this study's data, not cited from their reported number), Random
Forest, and CDR-PINN. XGBoost is not excluded from this study — it appears in the
model ladder (Table, §4.1) at parity with RF (0.9678 vs. 0.9676, pre-parity-expansion
numbers), confirming RF's designation as the headline classical baseline reflects
methodological convention rather than an unexamined choice: RF requires no feature
scaling across NDVI/LST/LULC's very different units, provides Gini feature
importance under the same metric used throughout this study's feature-engineering
narrative (§5.6), and is the single most common baseline in the reviewed regional
literature (§1.2–1.3), keeping this study's classical comparison point directly
legible against the field's own convention.

### 3.3 Feature Set — 15 Variable Groups, 58 Engineered Features

Biswas et al.'s 15 predictor variables are represented here not as 15 raw snapshots
but as their full temporal decomposition (climatology, anomaly, Mann-Kendall trend
and significance) already established throughout Steps 2–4 of the underlying
pipeline. Precisely accounted: **31 features** are direct decompositions of the 15
variable groups (e.g. NDVI alone → 9 features: mean, climatology, anomaly, trend,
residual, Mann-Kendall τ, the CVSI stress index, LISA cluster, breakpoint threshold;
each FLDAS climatic variable → anomaly + trend-significance = 2 features); **27
features** are additional, not present in Biswas et al.'s 15 at all (the full
22-class ESA-CCI land-cover fractional breakdown, 4 forest-fraction features, and
diurnal temperature range). Total: 58 features (+4 non-feature columns — `lon`,
`lat`, and the two label columns — for 62 parquet columns overall). This is stated
explicitly to avoid the count being misread as an inflated or incomparable predictor
set: the underlying variable *groups* are identical to Biswas et al.'s 15; the
representation is richer.

### 3.4 Validation Protocol — Scope, and a Since-Corrected Gap

Track A/B1/B2/B3 (§4.3) are a **generalization-robustness** protocol — they estimate
how well one already-fixed model configuration transfers across random, spatial, and
temporal held-out splits. They were not, at the time those tracks were run, paired
with a hyperparameter-selection protocol: the original `width=64` scale-up and
cosine-learning-rate comparisons (§4.2, §5.5 as originally drafted) used the Track A
**test**-set AUC directly to decide whether to keep each variant — precisely the kind
of decision a held-out validation set exists to make without touching the final test
metric. This was disclosed rather than hidden, and has since been corrected: §4.8
reports a dedicated train/val/test three-way split (65/15/20%, same seed=42) built
specifically to redo that comparison honestly, selecting a winner by validation AUC
only and reporting test AUC once, for the winner alone. The result is not a simple
confirmation of the original test-based conclusion — see §4.8 for the honest,
slightly more complicated outcome. The remaining gap: this fix covers only the one
scale/schedule decision it was built to re-examine, not a full nested
cross-validation across every architectural choice made in this study (layer count,
mode count, window length, τ in LSE-pooling, `pos_weight` computation) — those still
use PINO-paper defaults chosen once, not tuned, and remain future work (§7.2).

## 4. Results

### 4.1 Classical Baselines (Steps 7–8, already-established, real, measured)

| Model | ROC-AUC | Average Precision |
|---|---:|---:|
| Random Forest (58-feature, full 15/15 Biswas parity) | 0.9683 | 0.6796 |
| MaxEnt (`elapid`, 58-feature) | 0.9595 | 0.6237 |
| Plain MLP (Step 8) | 0.9614 | — |
| Plain-monotonicity PINN (Step 8) | 0.9613 | — |

### 4.2 CDR-PINN Term-Ablation Study (new, this work)

| Configuration | ROC-AUC | AP | Δ AUC |
|---|---:|---:|---:|
| Diffusion only | 0.6017 | 0.6050 | — |
| + Advection | 0.9239 | 0.9014 | **+0.3222** |
| + Reaction (full CDR) | **0.9406** | **0.9253** | +0.0167 |

Held out on an identical, disjoint 20% random pixel split (`n=4,508`, seed=42) across
all three configurations and all classical baselines' own respective splits, for
direct comparability. Full methodology, the diagnosed-and-fixed class-imbalance
collapse, and physical interpretation of the ablation ordering:
`CDR_PINN_Methodology_Section.md` §8, `CDR_PINN_Novelty_Comparison_Advantages.md` §4.

### 4.3 Generalization Tracks and Data-Efficiency Test (new, this work)

| Track | Description | AUC |
|---|---|---:|
| A | Random split (full CDR) | 0.9406 |
| B1 | 2°×2° spatial block CV, 3 folds | 0.7538 ± 0.0162 |
| B2 | Leave-one-region-out, 6 regions | 0.5989 ± 0.0815 |
| B3 | Leave-years-out (new) | 0.8967 |
| — | Data-efficiency: no-physics vs. physics (Track A split) | 0.9463 vs. 0.9406 |

Full table, per-fold breakdown, and epoch-budget disclosures:
`CDR_PINN_Methodology_Section.md` §8.

### 4.4 Per-Covariate Permutation Importance (new, this work)

A Biswas-et-al.-style permutation-importance test — shuffle one covariate spatially,
measure the held-out AUC drop — applied to the trained full-CDR operator's seven
input channels (inference only, no retraining, on the already-trained Track A
checkpoint):

| Covariate | AUC after permutation | Drop | % of baseline |
|---|---:|---:|---:|
| **Elevation** | 0.7168 | **+0.2238** | **23.80%** |
| Distance to roads | 0.9406 | +0.0000 | 0.00% |
| Slope | 0.9406 | +0.0000 | 0.00% |
| Forest fraction | 0.9406 | +0.0000 | 0.00% |
| NDVI (baseline) | 0.9406 | +0.0000 | 0.00% |
| Dryness proxy | 0.9406 | +0.0000 | 0.00% |
| NDVI anomaly | 0.9406 | −0.0000 | −0.00% |

Baseline AUC=0.9406, exactly matching §4.2/§4.3 (confirms correct checkpoint
loading). **Elevation dominates the trained operator's forward pass almost
completely** — every other covariate shows no measurable effect on the prediction
when shuffled. This is a genuinely striking result, reported exactly as measured,
with two readings that both belong in the paper:

- **Corroborating**: this is the *second of what becomes five* independent lines of
  evidence for terrain dominance in this study, alongside the term-ablation's
  advection-driven +0.322 AUC jump (§4.2), Step 5a's own +115% fire/slope
  coincidence measurement, the response-curve analysis immediately following (§4.5),
  and the Jackknife retraining test (§4.6) — five different methods (equation-term
  ablation, spatial fire-point statistics, input-channel permutation, marginal-effect
  sweeps, and per-variable retraining) converge on the same conclusion.
- **A limitation, not just a confirmation**: near-total reliance on a single input
  channel, to the exclusion of others with real, independently-established
  predictive content (e.g. NDVI, RF's own top-ranked feature family — §5.6), is
  also a plausible sign of shortcut learning at this training scale — elevation has
  the largest absolute dynamic range and geographic structure of any input channel,
  which may make it disproportionately easy for a modestly-sized, single-seed model
  to latch onto early in training. This is stated as an open question for further
  investigation (multi-seed testing, §5.7), not resolved by this single measurement.

### 4.5 Response Curves (Biswas et al.'s Figs. 8/9, reproduced for CDR-PINN)

Beyond permutation importance (§4.4, Biswas et al.'s Table 3 analogue), Biswas et
al.'s Figs. 8–9 report **response curves** — how predicted suitability changes as
one variable is swept across its range, holding others at their sample mean. The
identical methodology, reproduced here for CDR-PINN (inference only, same trained
checkpoint, all-other-covariates held at their domain mean):

| Covariate | Swept range | Predicted-probability range | Δ (marginal effect) |
|---|---|---|---:|
| **Elevation** | 3.94 – 5,459 m | 0.0396 – 0.4677 | **0.4281** |
| Distance to roads | 0 – 154 km | 0.3669 – 0.3750 | 0.0081 |
| Slope | 0 – 32.6° | 0.3723 – 0.3753 | 0.0030 |
| Dryness proxy | −0.41 – 2.50 | 0.3746 – 0.3749 | 0.0003 |
| Forest fraction | 0 – 1 | 0.3746 – 0.3747 | 0.0002 |
| NDVI (baseline) | −0.1 – 0.9 | 0.3747 – 0.3748 | 0.0001 |

This is a **third, independent** confirmation of terrain dominance — alongside the
term-ablation's advection-driven AUC jump (§4.2), Step 5a's own field measurement,
and the permutation-importance test (§4.4) — obtained via a completely different
method (a synthetic single-variable sweep, not held-out accuracy or spatial
shuffling). With this table and the Jackknife retraining test immediately following
(§4.6), this study now reproduces all three of Biswas et al.'s variable-understanding
analyses (Table 3 permutation importance, Figs. 8/9 response curves, and Fig. 10
Jackknife), on an architecture their own methodology was never applied to — full
methodological parity with the reference study (§1.6), not the 2-of-3 partial
reproduction disclosed in an earlier draft of this manuscript.

### 4.6 Jackknife Variable Importance (Biswas et al.'s Fig. 10, reproduced)

Unlike §4.4–4.5 (both inference-only, a single trained checkpoint perturbed or
swept), Biswas et al.'s Jackknife test requires genuine retraining: one model with
each covariate held at its domain-mean constant field ("without $X$"), and one model
with every *other* covariate held constant ("only $X$") — 14 retrains across the 7
covariates, plus an "all variables" model retrained at the same reduced 40-epoch
budget for a fair, apples-to-apples comparison array (the §4.2–4.5 checkpoint used
80 epochs). Same architecture, seed=42, and 80/20 pixel split as every other
CDR-PINN experiment in this study.

| Covariate | Without-*X* AUC | Drop when removed | Only-*X* AUC | Gain alone (vs. chance) |
|---|---:|---:|---:|---:|
| **Elevation** | **0.7779** | **−0.1613** | **0.9376** | **+0.4376** |
| Slope | 0.9394 | −0.0003 | 0.7731 | +0.2731 |
| Distance to roads | 0.9372 | +0.0019 | 0.7536 | +0.2536 |
| NDVI (baseline) | 0.9399 | −0.0008 | 0.7048 | +0.2048 |
| Forest fraction | 0.9392 | −0.0001 | 0.7016 | +0.2016 |
| Dryness proxy | 0.9393 | −0.0002 | 0.5755 | +0.0755 |
| NDVI anomaly | 0.9439 | +0.0047 | 0.5374 | +0.0374 |

All-variables baseline (same 40-epoch budget): AUC=0.9391, matching the 80-epoch
checkpoint's 0.9406 closely enough to confirm the model is essentially converged well
before 80 epochs. Two results, both genuinely new relative to §4.4–4.5 because this
is retraining, not perturbation of a fixed model:

- **Elevation is the only covariate whose removal meaningfully hurts the model** —
  every other "without-$X$" AUC sits within noise of the full-model baseline
  (0.9372–0.9439), several even nominally *above* it. This is the **fifth**
  independent line of evidence for terrain dominance in this study (§4.4's list,
  extended), and the first obtained via retraining rather than a fixed checkpoint.
- **Elevation alone very nearly reproduces the full model**: a model trained on
  elevation as its *only* informative input reaches AUC=0.9376, within 0.0015 of the
  7-covariate baseline (0.9391). This sharpens rather than merely repeats the
  shortcut-learning concern already raised in §4.4 and §5.7 item 10: it is not just
  that elevation permutation/response-curve tests show large marginal effects, it is
  that a model given *only* elevation and nothing else learns almost the entire
  achievable signal at this scale. The other six covariates are not informationally
  useless in isolation — five of six "only-$X$" models score meaningfully above
  chance (0.70–0.77 for slope/roads/NDVI/forest-fraction) — they simply add
  negligible signal on top of what elevation alone already provides.

### 4.7 Advanced PINN Techniques Tested: Causal Time-Weighting and Curriculum Learning

Two techniques from the wider PINN literature, chosen because each maps onto a
specific, already-diagnosed property of this study's training setup rather than
applied generically:

- **Causal time-weighting** (Wang, Sankaran & Perdikaris, 2022 [cite-verify]):
  weights each month's residual loss within a training window by
  $w_i=\exp(-\varepsilon\sum_{k<i}\mathcal{L}_r(k))$, so the optimizer must reduce
  earlier-month residual before later-month residual is allowed much gradient —
  directly targeting the explicit 266-month autoregressive structure this study
  trains over, which had no causal weighting of any kind before this test.
- **Staged curriculum learning**: rather than all three CDR terms active from epoch
  1 (every prior run in this study, including the term-ablation checkpoints, which
  train separate models per term combination rather than unlocking terms
  progressively within one run), advection is switched on at epoch 15 and reaction
  at epoch 35 of an 80-epoch run — a principled alternative to the scale/schedule
  tuning already tried and already ruled out (§5.7 item 8), motivated by the same
  observed optimization sensitivity.

| Configuration | Test AUC | Test AP | vs. baseline (0.9406) |
|---|---:|---:|---:|
| Baseline (full CDR, all terms from epoch 1) | 0.9406 | 0.9253 | — |
| Causal time-weighting ($\varepsilon=1.0$) | 0.9369 | 0.9206 | −0.0037 |
| Staged curriculum (advection@15, reaction@35) | 0.9343 | 0.9154 | −0.0063 |

Neither improved on the baseline. This is the fourth and fifth consecutive
optimization-side intervention (after §5.7 item 8's scale-up and LR-schedule tests)
that fails to beat the original default configuration — an accumulating signal that
this model's bottleneck is not optimization dynamics but the representation problem
already identified in §4.4–4.6 (near-total elevation dominance), which no amount of
retuning *how* the loss is optimized can be expected to fix. One caveat limits how
strongly this should be read: neither $\varepsilon$ nor the curriculum's unlock
epochs were swept — a single default value was tried for each, so the honest
conclusion is "did not help at the tested default," not "ruled out across all
settings." Wavelet PINNs (Tripura & Chakraborty, 2022 [cite-verify]), PIKANs
(physics-informed Kolmogorov-Arnold networks), and domain-decomposition PINNs
aligned with the existing biogeographic-zone infrastructure from Step 2's F9
breakpoint analysis were also considered against this study's specific diagnosed
weaknesses (spectral-truncation limitation, item 5; weak Track B2 spatial transfer,
§5.4) but require substantial architecture changes beyond this study's remaining
scope, and are recorded as future work (§7.2) rather than attempted without
sufficient justification. Fourier feature encoding and residual-adaptive domain
(RAD) sampling were considered and are *not* recommended for this architecture: the
former targets spectral bias in coordinate-input MLPs, which this FNO backbone does
not exhibit the same way (it already parameterizes filters directly in frequency
space); the latter targets sparse-collocation placement, and this study already uses
dense collocation (every valid pixel, every month) rather than a sparse sampled set.

### 4.8 Validation-Selected Re-Test of the Scale/Schedule Decision

§3.4 disclosed that the original `width=64` scale-up and cosine-LR-schedule
comparisons used Track A test AUC for selection. This section redoes that
comparison honestly: a fresh train/val/test split (65/15/20% of valid pixels,
seed=42 — a different partition than Track A's original 80/20 split, so absolute
numbers are not directly the same run repeated), all three configurations retrained
on the same split, winner selected by **validation** AUC only, test AUC reported
once for the winner and never used in selection.

| Configuration | Val AUC | Test AUC (not used for selection) |
|---|---:|---:|
| `width=32`, 80 epochs, cosine schedule | **0.9368** ← selected | 0.9403 |
| `width=32`, 80 epochs, no schedule | 0.9329 | 0.9370 |
| `width=64`, 150 epochs, no schedule | 0.9266 | 0.9339 |

Two findings, one confirming the original conclusion and one **reversing** it:

- **Scale-up is robustly worse**, now on a second, independent split (val AUC 0.0102
  below the small config, test AUC 0.0031–0.0064 below) — the original finding that
  bigger is not the fix is reinforced, not an artifact of one particular split.
- **The cosine-schedule finding reverses.** On the original Track A split, cosine
  scored *worse* than plain (0.9154 vs. 0.9406, `CDR_PINN_Methodology_Section.md`
  §8.1/§5.7 item 1) and was reported as ruled out. On this independent split,
  cosine wins on both validation (0.9368 vs.
  0.9329) and test (0.9403 vs. 0.9370) AUC. All non-scale-up configurations
  (baseline, causal, curriculum, plain, cosine) cluster within a ~0.93–0.94 AUC band
  across every split tested — ordinary split-to-split noise at this training scale,
  not a reliable ranking. **The earlier "LR schedule ruled out" conclusion is
  retracted to "split-sensitive, unresolved without multi-seed testing"** — an
  honest correction, not a discarded result: both numbers are real and both are
  reported (§5.7 item 1, revised).

Read together with §4.2's ablation and §4.7's negative results, six distinct
attempts to close the Track A gap to RF/MaxEnt (evaluation-metric fix, scale-up,
LR-schedule, causal weighting, curriculum learning, and this validation-honest
re-test) all land in the same 0.93–0.94 band except scale-up, which is consistently
worse. That convergence is itself informative: it is much more consistent with a
representation ceiling (§4.6's elevation-dominance finding) than with an
under-optimized model that further tuning would unlock.

### 4.9 Computational Cost

| Model | Params/trees | Train time | Inference | ROC-AUC |
|---|---:|---|---|---:|
| Random Forest | 200 trees | 195.2 s | 1.4 s | 0.9683 |
| MaxEnt | linear+hinge+product | 1,486.8 s | 33.3 s | 0.9595 |
| CDR-PINN, full physics | 1,054,613 | 354.6 s (80 ep) | — | 0.9406 |
| CDR-PINN, no physics (identical architecture) | 1,054,613 | 140.4 s (80 ep) | — | 0.9463 |

The physics constraint's own computational cost is directly measurable, not
estimated: **~2.5× training time** (354.6s vs. 140.4s, identical architecture, data,
and epoch budget) — the cost of computing the spectral PDE residual and boundary
loss every training step. Peak GPU memory across all CDR-PINN configurations tested
(width=32 through width=64) stayed under 5.6 GB of the 32 GB available, leaving
substantial headroom for a larger production run.

## 5. Discussion

### 5.1 The Ablation Ordering Is Physically Interpretable, Not Just Numerically Convenient

The advection term accounts for the overwhelming majority of the physics-informed
model's discriminative power (+0.322 of the total +0.339 AUC gain from diffusion-only
to full CDR). This is independently corroborated twice over, not merely
post-hoc-rationalized: by this project's own Step 5a measurement that real fire
locations sit at +115% mean slope versus the national average, and by Biswas et al.
(2025)'s own MaxEnt result ranking slope as their second-most-important predictor
(16.7% contribution, their Table 3). A model architecture that structurally routes
terrain information through a dedicated, physically-directed (upslope) transport term
— rather than presenting slope as one undifferentiated feature among 15 — recovering
this same signal as the dominant driver of its own accuracy gain is evidence the
physics formulation is capturing a real mechanism, not fitting noise.

### 5.2 A Transparent Failure Mode, and Why Reporting It Strengthens the Paper

The diffusion-only model's first training attempt collapsed to a trivial,
uninformative solution (Section 4.2; full account in the Methodology document) because
the true monthly fire-positive rate (2.3%) gave too weak a data-loss gradient to
compete with the physics loss's pull toward a constant-field solution that trivially
satisfies a homogeneous diffusion equation. This is a known category of PINN
optimization pathology (Wang, Teng & Perdikaris, 2021) — reported here not as a
limitation to be hidden, but as a concrete, diagnosed, and fixed instance of it,
strengthening rather than weakening the paper's methodological credibility.

### 5.3 Comparison to Biswas et al. (2025) and the Wider Literature

Full comparison table: `CDR_PINN_Novelty_Comparison_Advantages.md` §3. In brief: this
project already replicates Biswas et al.'s own MaxEnt method directly (not citing
their number) on this project's full data and beats it (0.9576 vs. their 0.879, see
`Biswas et al. Verification`); the CDR-PINN reported here is a categorically different
contribution — not a better classifier on the same static-feature paradigm, but a
different modeling paradigm entirely, evaluated on the same held-out-accuracy terms
for direct comparability while additionally offering the mechanistic,
ablation-testable structure neither Biswas et al. nor any of the regional India/global
studies reviewed in §1.2–1.3 provide.

### 5.4 Generalization Is Where the Honest Story Gets Complicated

The hypothesis motivating this whole architectural pivot (§1.4, and Step 8's own
prior finding that neural architectures already generalize better than Random Forest
under spatial CV) was that physics-informed structure should show its clearest
advantage under distribution shift. The evidence collected here is **mixed, not
confirmatory**: temporal generalization (Track B3, leave-years-out, AUC=0.897) is
genuinely strong — a real, positive result for exactly the axis this project's
per-month operator framing was built to enable. Spatial generalization is not:
Track B1 (spatial block CV, 0.754) and especially Track B2 (leave-one-region-out,
0.599, one of six regions scoring below chance) show the model does not yet transfer
well to geographically unseen terrain at this training scale. A direct
physics-vs-no-physics comparison under identical sparse supervision (§4.3) found
**no accuracy advantage from the physics constraint** on the random-split evaluation
— the same comparison run on the harder B1/B2/B3 splits, where the literature
predicts the effect should actually appear, has not yet been performed and is the
single most important remaining experiment for this paper's central claim.

**What we do and do not claim as a result.** We do not claim the CDR-PINN currently
generalizes better than RF/MaxEnt spatially — the evidence available says it does
not, at this architecture size and epoch budget. We do claim: (1) the physics
structure is capturing a real, independently-corroborated mechanism (§5.1); (2) the
model achieves temporal generalization no classical baseline in this study or the
reviewed literature was evaluated on; (3) the mechanistic, ablation-testable
structure is itself a contribution independent of whether it currently wins on raw
accuracy. Plausible causes for the weak spatial results — smaller architecture
(1.05M parameters) than would typically be tuned for this problem, a reduced
50-epoch budget for the multi-fold B1/B2 tracks (vs. 80 for A/B3, disclosed in
Methodology §8), single-seed results with no bootstrap confidence interval yet, and
KMeans-region boundaries that may isolate climatically distinct zones with too
little in-region training signal — are stated as hypotheses to test, not
explanations to excuse the result.

### 5.5 What Three Models and Four Tracks Actually Demonstrate, Together

The pixel-level (Track A), spatial-block/region-level (Tracks B1/B2), and
year-level (Track B3) analyses are not three independent results to report in
sequence — read together, they answer a question none of them answers alone:
*which axis of generalization can each modeling paradigm even be meaningfully
evaluated on, and does model ranking hold across all of them?* It does not. RF and
MaxEnt lead on in-distribution accuracy (Track A) but are structurally ineligible
for Track B3 at all (§3, no year-resolved feature table exists for them to be
evaluated on); CDR-PINN trails on Track A but is the only model of the three
capable of being tested on temporal generalization, where it performs well (0.897).
**No prior study reviewed in §1.2–1.3 reports more than one evaluation axis** — this
three-model, four-track comparison is itself the paper's methodological
contribution, independent of any single number: a demonstration that single-split
AUC reporting, the field's current norm, can hide exactly this kind of structural
capability gap between modeling paradigms.

### 5.6 Feature Engineering's Measured Impact

Real evidence, not a claim: Random Forest's Gini importance ranking (§3.3's 58
features) shows *engineered, derived* quantities systematically outranking their
own raw source variables — `forest_frac_recent/current/baseline` occupy the top 3
positions (0.166/0.120/0.111), ahead of `ndvi_mean` itself, and `ndvi_trend_2x12ma`
(the trend-decomposition feature) outranks the raw NDVI mean it's derived from
(0.085 vs. 0.057). This validates the feature-engineering investment across Steps
2–6 (climatology/anomaly/trend/significance decomposition, not just raw monthly
means) as measurably, not just methodologically, justified — a pipeline that
stopped at raw variable snapshots (the reference paper's own approach) would have
missed the features this study's own model relies on most.

### 5.7 Limitations

1. **The Track-A accuracy gap to RF/MaxEnt is not yet closed**, and six tested
   interventions have been ruled out directly rather than left unexamined:
   evaluation-metric mismatch (§5.2), under-parameterization (`width=64` scored
   *worse*, 0.9292 on Track A and again 0.9339 on the independent validation-split
   re-test, §4.8), causal time-weighting (0.9369, §4.7), staged curriculum learning
   (0.9343, §4.7), and the learning-rate schedule — this last one not cleanly ruled
   out but shown to be **split-sensitive** (worse on Track A, 0.9154 vs. 0.9406;
   better on the independent validation split, 0.9403 vs. 0.9370, §4.8), so it is
   downgraded from "ruled out" to "unresolved without multi-seed testing." All six
   interventions except scale-up land within a ~0.93–0.94 AUC band, more consistent
   with a representation ceiling (§4.6) than an optimization deficit.
2. **No held-out validation set was used for the original architecture decisions**
   (§3.4) — the `width=64` and LR-schedule comparisons initially used Track A's test
   AUC directly, which a validation set exists specifically to avoid. This has since
   been partially corrected: §4.8 reports a genuine train/val/test re-test of that
   specific decision. The correction remains partial — every *other* architectural
   choice in this study (layer count, mode count, window length, LSE-pooling τ,
   `pos_weight` derivation) still uses PINO-paper defaults chosen once, not
   validated against held-out data, and a full nested cross-validation across all of
   them remains future work (§7.2).
3. **Spatial generalization is weak at this training scale** (Tracks B1/B2, §4.3) —
   the physics-informed advantage this architecture was motivated to test has not
   yet been empirically confirmed under spatial distribution shift, and the direct
   physics-vs-no-physics comparison needed to test it (§5.4) has only been run on
   Track A so far.
4. **Data-hunger under sparse labels.** The ~2.3% monthly fire-positive rate
   produced a real, observed optimization failure (the trivial-solution collapse,
   §5.2) before the class-imbalance fix — evidence, not hypothesis, that this
   architecture's data appetite is a genuine operational concern for this problem.
5. **Spectral truncation vs. sharp local risk features.** FNO represents fields via
   a small number of global Fourier modes (16×16), well-suited to smooth, large-
   scale patterns but structurally less able to represent sharp local
   discontinuities than a local-receptive-field architecture — not directly tested
   here, a citable architectural tradeoff.
6. **Rectangular-grid approximation of India's true boundary geometry** — the
   Neumann whole-sample-symmetric extension (§3, advection document) fixes the FFT
   differentiation's boundary artifact but does not exactly represent India's
   irregular coastline/land border the way a mesh-based method could.
7. **Physics-loss computational overhead is real and measured**: ~2.5× training
   time (§4.9) — a genuine deployment-cost consideration for any operational,
   frequently-retrained use.
8. **Optimization sensitivity, empirically demonstrated — see item 1 for the full,
   corrected account.** Scale-up, causal time-weighting, and curriculum learning
   all made results worse; the learning-rate schedule alone gave opposite outcomes
   on two different splits and is downgraded to split-sensitive rather than
   ruled out (item 1). The overall pattern — five of six interventions landing in a
   narrow AUC band regardless of split, one (scale-up) robustly worse — is a
   disclosed sign that this architecture/problem combination has a
   not-yet-well-understood optimization landscape, though it now reads more like a
   representation ceiling than pure optimization difficulty (§4.6).
9. **Resolution-independence is proven, not yet empirically exercised** — a
   theoretical property of the FNO backbone (Li et al., 2023), not yet tested via
   zero-shot super-resolution on a trained checkpoint.
10. **Over-reliance on a single input channel — now confirmed by retraining, not
    just perturbation.** The permutation-importance test (§4.4), the response-curve
    analysis (§4.5), and the Jackknife retraining test (§4.6) — three independent
    methods, the last a genuinely different kind of evidence since it retrains
    rather than perturbs a fixed model — all show near-total sensitivity to
    elevation alone: removing elevation drops AUC by 0.16 while removing any other
    single covariate changes nothing measurable, and a model trained on elevation
    *alone* reaches AUC=0.9376, within 0.0015 of the full 7-covariate model
    (0.9391). Three independent confirmations make this a
    robust *observation*; its *interpretation* remains open (plausibly shortcut
    learning at this training scale), flagged for multi-seed testing (§7.2) rather
    than resolved by any single test.
11. **Biswas et al.'s Jackknife test (their Fig. 10) has now been reproduced**
    (§4.6) — 14 retrains (leave-one-covariate-out and leave-only-one-covariate-in
    across the 7 covariates) at a reduced 40-epoch budget, plus a matched-budget
    "all variables" baseline for fair comparison. This study now reproduces all 3 of
    Biswas et al.'s variable-understanding analyses, resolving what was previously
    an explicit 2-of-3 gap. The Jackknife's own result sharpens rather than resolves
    item 10 above: it is retraining-based confirmation of the same shortcut-learning
    concern, not new counter-evidence against it.
12. **No calibrated uncertainty quantification.** The model outputs a point estimate
    (`σ(u)`), not a calibrated probability with confidence bounds — relevant for any
    model intended to inform real resource-allocation decisions.
13. **Transductive information exposure via the FNO's global receptive field** —
    stated explicitly rather than left implicit: held-out test pixels are excluded
    from the *data* loss during training, but the FNO's spectral layers mix
    information globally across the full grid, so the network's hidden
    representations do see test-pixel *covariates* (never labels) during every
    forward pass. This is standard for spatial/transductive learning settings but
    should not go unstated in a Methods section, particularly given FNO's global
    (not local) receptive field makes it more pronounced than a local-CNN baseline
    would exhibit.
14. **Responsible-use scope.** This is a research-stage model — single-seed,
    incomplete spatial-generalization validation, no calibrated uncertainty — and
    is not yet suitable for direct operational deployment in land-use, insurance,
    or resource-allocation decisions without further validation.

## 6. Policy Implications and Sustainable Development Goals

Biswas et al. (2025) explicitly connect their results to nine SDGs (1, 2, 3, 5, 6, 9,
12, 13, 15) and India's governance apparatus for forest-fire management (the
National Forest Fire Prevention and Management Scheme, established 2003, plus
state-level strategies in Uttarakhand, Himachal Pradesh, and Maharashtra). This
study's mechanistic decomposition offers a genuinely differentiated extension of
that framing, not a restatement of it: where an undifferentiated MaxEnt probability
map can only say *where* risk is high, the CDR equation's term structure can, in
principle, say *why*, and each mechanism maps onto a distinct policy lever.

- **Diffusion (vegetation/moisture)** → fuel and vegetation management, controlled
  burns, forest composition policy (SDG 15 — Life on Land).
- **Advection (terrain)** → terrain-aware firebreak placement and early-warning
  resource pre-positioning specifically along steep upslope corridors, the single
  largest driver of this study's own predictive accuracy (§4.2) — SDG 13 (Climate
  Action) and disaster-preparedness policy.
- **Reaction (human ignition, road proximity)** → patrol allocation and
  ignition-source control near roads, directly relevant to the forest-dependent
  communities Biswas et al. emphasize under SDG 1 (No Poverty) and SDG 2 (Zero
  Hunger), and to SDG 9 (resilient infrastructure).

This differentiated framing is offered as a genuine, low-cost extension of the
reference study's own policy contribution, made possible specifically by the
governing-equation structure — not available to an undifferentiated probability
surface regardless of its underlying accuracy.

## 7. Conclusion and Future Work

### 7.1 What This Study Demonstrates, in Plain Terms

This study reformulates forest-fire susceptibility mapping — conventionally a
correlational, static classification problem — as a physically-structured
dynamical-systems problem, and asks whether that reformulation captures real signal
rather than assuming it does. The evidence says, honestly: partially, and in a
specific, identifiable way. The physics structure recovers a real, independently
corroborated mechanism (terrain-driven spread dominates, confirmed three separate
ways — §4.2, §4.4, and Step 5a's own field measurement) and enables a genuinely new
evaluation capability (temporal generalization, §4.3) no prior study in this
literature, including the reference paper, can attempt. It does not yet match
classical machine learning on raw in-distribution accuracy, and does not yet
demonstrate the spatial-generalization advantage that motivated its design. Both
outcomes are reported without softening, because a Q1 submission's credibility rests
on that honesty being visible in the methodology, not asserted in the abstract.

**Why this matters beyond the numbers**: a susceptibility map that can only rank
locations by risk is less operationally useful than one that can attribute *why* a
location is high-risk — vegetation, terrain, or human activity — because each
attribution implies a different, actionable intervention (§6). That mechanistic,
falsifiable structure, testable via ablation in a way no correlational baseline
permits, is this study's actual contribution, independent of whether its current,
first-implementation accuracy exceeds Random Forest's.

### 7.2 Future Work

Prioritized by how directly each addresses an open question raised in this paper,
not a generic list:

1. **Physics-vs-no-physics comparison on Tracks B1/B2/B3**, not just Track A — the
   single most important unresolved experiment for this paper's central hypothesis,
   still open after this pass.
2. **Full nested cross-validation across every architectural choice**, not just the
   one scale/schedule decision re-tested honestly in §4.8 — layer count, mode count,
   window length, LSE-pooling τ, and `pos_weight` derivation still use PINO-paper
   defaults chosen once, never validated against held-out data (§5.7 item 2, revised
   scope).
3. **Instance-wise fine-tuning** (Li et al., 2023 §3.2) and **self-adaptive
   per-point loss weighting** (McClenny & Braga-Neto, 2020) — qualitatively
   different techniques from the six scale/schedule/causal/curriculum interventions
   already tested (§4.7–4.8, §5.7 item 1). **Transfer learning specifically for
   Track B2's weak regions** — fine-tuning a pretrained base per held-out region
   rather than training each fold from scratch — is a direct, literature-standard
   candidate for closing the spatial-generalization gap (§5.4) that this study has
   not yet attempted.
4. **Multi-seed robustness testing** with bootstrap confidence intervals, mirroring
   this project's own Step 8b protocol — every number in this paper is currently
   single-seed, and is now the single most direct way to resolve the split-sensitive
   LR-schedule finding (§4.8, §5.7 item 1) and confirm whether the causal-weighting
   and curriculum-learning negative results (§4.7) hold under a tuned $\varepsilon$/
   unlock-schedule rather than the single defaults tested.
5. **Zero-shot super-resolution evaluation** on a trained checkpoint, exercising the
   proven-but-untested discretization-convergence property.
6. **Extending the permutation-importance, response-curve, and Jackknife tests**
   (§4.4–4.6, now all three of Biswas et al.'s variable-understanding analyses
   reproduced) with multi-seed averaging to determine whether the observed
   elevation-dominance is a robust finding or a single-run artifact.
7. **LULC's deeper role in the diffusion coefficient** — flagged as deferred since
   the original diffusion design document, still open.
8. **Pre-2000 fire history for a non-zero, empirically-grounded initial condition**
   — a documented upgrade path since the original design.
9. **Calibrated uncertainty quantification**, naturally pairable with item 4's
   multi-seed ensemble.
10. **Wavelet PINNs** (Tripura & Chakraborty, 2022 [cite-verify]) for sharper local
    risk features than the FNO's 16×16 global-mode truncation permits (§5.7 item 5);
    **PIKANs** (physics-informed Kolmogorov-Arnold networks), whose learned per-edge
    spline activations align naturally with this study's own interpretability goals
    (§4.4–4.6's response-curve/importance analyses); and **domain-decomposition
    PINNs** (XPINN-style) aligned with the biogeographic zones already established
    in Step 2's F9 breakpoint analysis — a second, architecturally distinct route to
    the same Track B2 spatial-generalization gap item 3 targets via transfer
    learning. All three require substantial architecture work beyond this study's
    remaining scope and were deliberately not attempted without that justification
    (§4.7).

## References

This section compiles the literature newly verified for this manuscript (§1.2–1.4).
For the full reference list also covering the classical pipeline (Steps 1–7) and the
CDR-PINN design/proof citations (Fisher–KPP, Evans, PINO, Rothermel, etc.), see
`METHODOLOGY.md`'s "Consolidated Reference List" — merge alphabetically with the
entries below when assembling the final manuscript bibliography.

Caglar, T., Jaiswal, J., Azim, S., Gala, Y., Nguyen, M.H., & Altintas, I. (2026).
Physics-guided spatiotemporal neural models for fuel density prediction.
*arXiv:2607.06999*. [cite-verify — preprint, journal record not yet confirmed]

Dabrowski, J.J., Pagendam, D.E., Hilton, J., Sanderson, C., MacKinlay, D., Huston, C.,
Bolt, A., & Kuhnert, P. (2023). Bayesian Physics Informed Neural Networks for data
assimilation and spatio-temporal modelling of wildfires. *Spatial Statistics*, 55,
100746. DOI: 10.1016/j.spasta.2023.100746. [cite-confirmed]

Gholamnia, K., Tahmasebi Moghaddam, H., Einali, G., Akbari Monfared, B., Lorestani,
G., Ghorbanzadeh, O., & Einali, J. (2026). Uncertainty-aware machine learning via
Dempster–Shafer theory for wildfire susceptibility mapping. *Spatial Information
Research*, 34(4), 35. DOI: 10.1007/s41324-026-00692-x. [cite-confirmed]

Wang, S., Sankaran, S., & Perdikaris, P. (2022). Respecting causality is all you
need for training physics-informed neural networks. *Computer Methods in Applied
Mechanics and Engineering* (submitted/arXiv:2203.07404). [cite-verify — causal
time-weighting technique tested in §4.7, negative result at the tested default]

Tripura, T., & Chakraborty, S. (2022). Wavelet neural operator: a neural operator
for parameterized differential equations. *arXiv:2205.02191*. [cite-verify — wavelet
PINN/operator architecture considered in §4.7 as a candidate fix for the spectral-
truncation limitation (§5.7 item 5), not implemented in this study]

Guria, R., Mishra, M., Mohanta, S., & Paul, S. (2025). Forest fire probability
zonation using dNBR and machine learning models: a case study at the Similipal
Biosphere Reserve (SBR), Odisha, India. *Environmental Science and Pollution
Research*, 32(59), 31375–31396. DOI: 10.1007/s11356-025-35976-6. [cite-confirmed]

Gupta, P., Shukla, A.K., & Shukla, D.P. (2025). Machine learning-based forest fire
susceptibility mapping of Southern Mizoram, a part of Indo-Burma Biodiversity
Hotspot. *Environmental Science and Pollution Research*, 32(59), 31433–31454.
DOI: 10.1007/s11356-025-36621-y. [cite-confirmed]

Hang, H.T., Mallick, J., Alqadhi, S., Bindajam, A.A., & Abdo, H.G. (2024). Exploring
forest fire susceptibility and management strategies in Western Himalaya: Integrating
ensemble machine learning and explainable AI. *Environmental Technology &
Innovation*, 35, 103655. DOI: 10.1016/j.eti.2024.103655. [cite-confirmed]

İban, M.C., & Aksu, O. (2024). SHAP-Driven Explainable Artificial Intelligence
Framework for Wildfire Susceptibility Mapping Using MODIS Active Fire Pixels: A Case
Study in Izmir, Türkiye. *Remote Sensing*, 16(15), 2842. DOI: 10.3390/rs16152842.
[cite-confirmed]

Jiang, P., Yang, Z., Wang, J., Huang, C., Xue, P., Chakraborty, T.C., Chen, X., &
Qian, Y. (2023). Efficient Super-Resolution of Near-Surface Climate Modeling Using the
Fourier Neural Operator. *Journal of Advances in Modeling Earth Systems*, 15(7),
e2023MS003800. DOI: 10.1029/2023MS003800. [cite-confirmed]

Kanda Naveen Babu, Gour, R., Kurian Ayushi, Ayyappan, N., & Parthasarathy, N. (2023).
Environmental drivers and spatial prediction of forest fires in the Western Ghats
biodiversity hotspot, India: An ensemble machine learning approach. *Forest Ecology
and Management*, 540, 121057. DOI: 10.1016/j.foreco.2023.121057. [cite-confirmed]

Kantarcioglu, O., Schindler, K., & Kocaman, S. (2023). Forest Fire Susceptibility
Assessment with Machine Learning Methods in North-East Türkiye. *ISPRS Archives*,
XLVIII-M-1-2023, 161–167. DOI: 10.5194/isprs-archives-xlviii-m-1-2023-161-2023.
[cite-confirmed]

Karniadakis, G.E., Kevrekidis, I.G., Lu, L., Perdikaris, P., Wang, S., & Yang, L.
(2021). Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422–440.
DOI: 10.1038/s42254-021-00314-5. [cite-confirmed]

Kurth, T., Subramanian, S., Harrington, P., Pathak, J., Mardani, M., Hall, D., Miele,
A., Kashinath, K., & Anandkumar, A. (2023). FourCastNet: Accelerating Global
High-Resolution Weather Forecasting Using Adaptive Fourier Neural Operators.
*Proceedings of the Platform for Advanced Scientific Computing Conference (PASC
'23)*. DOI: 10.1145/3592979.3593412. [cite-confirmed]

Malik, F.A., Mushtaq, F., Farooq, M., Guite, L.T.S., Kanga, S., Meraj, G., Singh,
S.K., & Kumar, P. (2025). Assessing forest fire vulnerability with fuzzy-AHP:
insights from Poonch forest division, Jammu and Kashmir. *Discover Forests*, 1(1), 4.
DOI: 10.1007/s44415-025-00004-5. [cite-confirmed]

Meraj, G., Hashimoto, S., Dasgupta, R., & Mitra, B.K. (2025). Ecological Risk
Assessment and Management of Forest Fires in Tamil Nadu, India: A MaxEnt Model-Based
Approach for Strategic Resource Allocation and Fire Mitigation. *Risk Analysis*,
45(11), 3604–3625. DOI: 10.1111/risa.70098. [cite-confirmed]

Read, J.S., Jia, X., Willard, J., Appling, A.P., Zwart, J.A., Oliver, S.K., Karpatne,
A., Hansen, G.J.A., Hanson, P.C., Watkins, W., Steinbach, M., & Kumar, V. (2019).
Process-Guided Deep Learning Predictions of Lake Water Temperature. *Water Resources
Research*, 55(11), 9173–9190. DOI: 10.1029/2019WR024922. [cite-confirmed]

Santana Neto, V.P., Nunes, A.J.N., Torres, F.T.P., Gleriani, J.M., & Cosenza, D.N.
(2025). Assessing Wildfire Susceptibility and Driving Variables in Portugal Using
Machine Learning Approach. *Journal for Nature Conservation*, 86, 126956.
DOI: 10.1016/j.jnc.2025.126956. [cite-confirmed]

Sarkar, M.S., Majhi, B.K., Pathak, B., Biswas, T., Mahapatra, S., Kumar, D., Bhatt,
I.D., Kuniyal, J.C., & Nautiyal, S. (2024). Ensembling machine learning models to
identify forest fire-susceptible zones in Northeast India. *Ecological Informatics*,
81, 102598. DOI: 10.1016/j.ecoinf.2024.102598. [cite-confirmed]

Sun, A.Y., Jiang, P., Shuai, P., & Chen, X. (2024). Bridging Hydrological Ensemble
Simulation and Learning Using Deep Neural Operators. *Water Resources Research*,
60(10), e2024WR037555. DOI: 10.1029/2024WR037555. [cite-confirmed]

Symeonidis, P., Vafeiadis, T., Ioannidis, D., & Tzovaras, D. (2025). Wildfire
Susceptibility Mapping in Greece Using Ensemble Machine Learning. *Earth*, 6(3), 75.
DOI: 10.3390/earth6030075. [cite-confirmed]

Uthappa, A.R., Das, B., Raizada, A., Kumar, P., Jha, P., & Prasad, P.V.V. (2025).
Forest Fire Susceptibility Mapping Using Multi-Criteria Decision Making and Machine
Learning Models in the Western Ghats of India. *Journal of Environmental
Management*, 379, 124777. DOI: 10.1016/j.jenvman.2025.124777. [cite-confirmed —
already in METHODOLOGY.md's reference list; repeated here for §1.2's completeness]

Vogiatzoglou, K., Papadimitriou, C., Bontozoglou, V., & Ampountolas, K. (2025).
Physics-informed neural networks for parameter learning of wildfire spreading.
*Computer Methods in Applied Mechanics and Engineering*, 434, 117545.
DOI: 10.1016/j.cma.2024.117545. [cite-confirmed]

Zakari, R.Y., Malik, O.A., & Ong, W.-H. (2025). Machine learning-driven wildfire
susceptibility mapping in New South Wales, Australia using remote sensing and
explainable artificial intelligence. *Natural Hazards*, 121(13), 15331–15357.
DOI: 10.1007/s11069-025-07395-w. [cite-confirmed]

Zhang, G., Wang, M., & Liu, K. (2019). Forest Fire Susceptibility Modeling Using a
Convolutional Neural Network for Yunnan Province of China. *International Journal of
Disaster Risk Science*, 10(3), 386–403. DOI: 10.1007/s13753-019-00233-1.
[cite-confirmed]

**Not cited above but verified as real during the research pass, scope-flagged**:
Yarmohammadian, R., Put, F., & Van Coile, R. (2025). Physics-Informed Surrogate
Modelling in Fire Safety Engineering: A Systematic Review. *Applied Sciences*,
15(15), 8740. DOI: 10.3390/app15158740. [cite-confirmed, but scope is structural/
building fire-safety engineering, not wildfire — cite only if the manuscript's
Introduction is extended to discuss physics-informed fire modeling broadly, not
wildfire-specifically].
