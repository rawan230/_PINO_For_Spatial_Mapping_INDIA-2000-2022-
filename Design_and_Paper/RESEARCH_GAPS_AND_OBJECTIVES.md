# Research Gaps, Objectives and How This Study Fills Them

**Study:** *A Physics-Informed Neural Operator for Forest-Fire Susceptibility Mapping in India: A Convection–Diffusion–Reaction Formulation* (CDR-PINO), prepared for IEEE TGRS.
**Reference study:** Biswas, Mahato & Joshi (2025), *Environmental Science and Pollution Research* 32(8):4856–4878 [R1].
**Literature window:** 2010 to 2026. Pre-2010 classics (Rothermel, Breiman, Phillips, Fisher–KPP and others) are left out of the gap evidence on purpose.
**Results:** every number below comes from the current `CDR_PINO_TGRS.tex` and `STUDY_METHODOLOGY_AND_GAPS.md`, using the leakage-corrected 57-feature stack. Compiled 2026-09-24.

> **Venue-tier note.** Tiers are given as JCR/SJR quartile or CORE rank. Quartiles change every year, so check each one against the latest JCR/SJR release before submission. A few venues are Q1 in SJR only (for example *Remote Sensing*, *ESPR*, *Spatial Statistics*), and they are labelled that way.

---

## At a Glance: Research Gaps → Research Objectives

### Research Gaps (RG)

- **RG1.** Forest-fire susceptibility studies in India [R1, R3, R4] and elsewhere [R5–R7] treat susceptibility as static per-pixel classification. None uses a governing equation for fire behaviour [R2].
- **RG2.** Predictors are collapsed into whole-period layers that produce one map. Susceptibility therefore has no temporal state and is never tested on unseen years [R1, R8–R11].
- **RG3.** Physics-informed ML in fire science covers only single-event spread [R15, R16] and pointwise hazard models [R17]. No physics-informed model exists for national fire susceptibility, or for India [R12–R14].
- **RG4.** Neural operators are benchmarked on synthetic PDE families on flat grids [R18–R21]. They have not been applied to multi-decade observational archives on the sphere.
- **RG5.** Physics-informed models give no well-posedness guarantee and leave coefficient signs to a soft loss penalty, which is prone to failure [R22–R24].
- **RG6.** Variable importance is post-hoc and correlational (SHAP, permutation, Jackknife) [R25–R27]. Biswas et al. note that MaxEnt contributions depend on the algorithm's path [R1].
- **RG7.** Models are evaluated on one random split (Biswas et al.: 70/30, test AUC 0.879). Spatial autocorrelation inflates scores from such splits [R28–R31].
- **RG8.** National maps use coarse 0.25° snapshot predictors, and land cover serves only as a fire-point filter [R1, R32].
- **RG9.** Fire-point labels are not validated against an independent burned-area product, and predictor stacks are not audited for leakage [R35–R37].
- **RG10.** Physics constraints are claimed to improve generalisation [R13, R38], but this has not been tested against tuned classical baselines under distribution shift [R21, R22].

### Research Objectives (RO)

- **RO1 (fills RG1).** To derive, from a susceptibility balance law, a convection–diffusion–reaction (CDR) equation whose diffusion, advection and reaction terms each represent a named fire mechanism and one of Biswas et al.'s predictor groups.
- **RO2 (fills RG5).** To prove that this equation is globally well-posed over the 266-month record using constants measured from data, and to fix the coefficient signs through the model architecture.
- **RO3 (fills RG2, RG3, RG4, RG6).** To train the first physics-informed Fourier neural operator for fire susceptibility on 22 years of monthly MODIS data for India, using exact spherical derivatives, and to test each mechanism by term ablation.
- **RO4 (fills RG2, RG7, RG10).** To evaluate the model on four tracks (random, spatial-block, leave-region-out, leave-years-out) against tuned Random Forest and MaxEnt baselines, and to test whether the physics constraint helps under distribution shift.
- **RO5 (fills RG8, RG9).** To build a 1 km, month-resolved, leakage-audited predictor stack matching Biswas et al.'s 15 variables, and to validate the fire points against independent burned-area data.

### Key Outcome per Objective

| Objective | Outcome |
|---|---|
| RO1 | CDR equation derived; Moran's I = 0.8322 supports the diffusion term; slope is 115% higher at fire locations, which supports the advection term |
| RO2 | Well-posedness theorem proved; D > 0, ρ ≥ 0 and upslope **v** are guaranteed by the architecture |
| RO3 | Term ablation AUC: 0.6017 (diffusion) → 0.9239 (+ advection) → **0.9397** (full CDR) |
| RO4 | Unseen years **0.8960**; spatial-block 0.7510 vs. RF 0.9498 and MaxEnt 0.9465; physics effect negative on 3 of 4 tracks (reported in full) |
| RO5 | 25× finer grid than Biswas et al.; fire points match burned area (r = 0.915); leakage audit removed the top-3 features |

---

## 1. Research Objectives of This Study

**Aim.** To replace static, classifier-based forest-fire susceptibility mapping for India with a susceptibility field governed by a derived, provably well-posed physical equation. The field should be resolved in time, decomposable into named mechanisms, and evaluated honestly against the reference study and tuned classical baselines.

To achieve this aim, the study has five research objectives. O1–O4 are the objectives stated in the manuscript (§I-E). O5 is the data objective behind them; it appears in the manuscript's contributions paragraph but not yet as a numbered objective.

| Objective | Statement | Gaps addressed |
|---|---|---|
| **O1 Formulate** | To derive, from a susceptibility balance law, a convection–diffusion–reaction (CDR) equation in which each term (diffusion, advection, reaction) represents a named fire-behaviour mechanism and absorbs one of the reference study's predictor groups, with an explicit reason stated for every term. | G1 |
| **O2 Guarantee** | To prove that the resulting initial–boundary value problem has a unique global-in-time weak solution over the full 266-month record, with every constant computed from measured data ranges, and to guarantee the physically required coefficient signs by architecture rather than by loss penalty. | G5 |
| **O3 Solve and test** | To train a physics-informed Fourier neural operator, with exact spherical differential operators, on 22 years (2000–2022) of monthly forest-filtered MODIS fire observations, and to measure each mechanism's own contribution by term ablation. | G2, G3, G4, G6 |
| **O4 Evaluate honestly** | To evaluate the model on four generalisation tracks (random, spatial-block, leave-region-out and leave-years-out), to apply the same spatial blocking to tuned Random Forest and MaxEnt baselines, and to test whether the physics constraint improves robustness under distribution shift, reporting the answer whichever way it falls. | G2, G7, G10 |
| **O5 Build and validate the data** | To build a 1 km, month-resolved, leakage-audited national predictor stack that keeps parity with the reference study's 15 conditioning variables, and to validate the extracted forest-fire points against an independent burned-area product. | G8, G9 |

---

## 2. Summary Table: Gap → Objective → Evidence

| # | Research gap | Key evidence (tier) | What Biswas et al. [R1] did | Research objective that fills it | Updated result |
|---|---|---|---|---|---|
| G1 | Susceptibility is modelled as static per-pixel classification with no governing equation | R1, R2, R3, R4, R5, R6, R7 (Q1) | MaxEnt density estimation over 15 static rasters | **O1:** To derive a CDR susceptibility equation whose terms are named mechanisms | CDR equation derived from a balance law; each term maps to one predictor group; Moran's I = 0.8322 supports the diffusion term |
| G2 | No explicit temporal state, so nothing can be tested on unseen years | R8, R9, R10, R11 (Q1) | Predictors aggregated over 2001–2020; one probability map | **O3 + O4:** To model susceptibility as a monthly field over 266 months and test it on held-out years | Leave-years-out Track B3 AUC **0.8960**; the classical models cannot be evaluated on this axis |
| G3 | Physics-informed ML has not reached wildfire *susceptibility*, and nothing exists for India | R12, R13, R14, R15, R16, R17 (Q1) | Purely data-driven | **O3:** To build the first physics-informed neural operator for fire susceptibility, applied to India | First PINO for susceptibility and first physics-informed treatment for India (541,545 fire points) |
| G4 | Neural operators are tested on synthetic PDEs on flat grids, not on observational archives | R18, R19, R20, R21 (A*/Q1) | — | **O3:** To train a neural operator on a real 22-year archive using exact spherical derivatives | FNO with 1,054,613 parameters; Laplace–Beltrami operator removes a 24.5% metric distortion |
| G5 | Learned-physics models give no well-posedness guarantee, and coefficient signs are left to the loss | R22, R23, R24, R14 (A*/Q1) | Not applicable | **O2:** To prove well-posedness from measured constants and fix coefficient signs by architecture | Global well-posedness theorem (e.g. slope ≤ 77.31°); D > 0, ρ ≥ 0, **v** upslope by construction |
| G6 | Interpretability is post-hoc and correlational (SHAP, permutation, Jackknife) | R25, R26, R27, R1 (Q1) | Percent contribution, permutation importance, Jackknife; R1 itself notes contributions are path-dependent | **O3:** To test each mechanism by term ablation instead of post-hoc ranking | Ablation AUC 0.6017 → 0.9239 → **0.9397** |
| G7 | Evaluation uses one random split, which spatial autocorrelation inflates | R28, R29, R30, R31 (Q1) | 70/30 random split, AUC 0.894 train / 0.879 test | **O4:** To evaluate on four tracks with identical spatial blocking for all models | Spatial block: RF 0.9498, MaxEnt 0.9465, CDR-PINO 0.7510; leave-region-out 0.6187 |
| G8 | National maps are coarse; predictors enter as whole-period snapshots; land cover is only a filter | R1, R32, R33, R34 (Q1) | 0.25° grid; land cover used only to mask fire points | **O5:** To build a 1 km, month-resolved predictor stack with land cover as a predictor | 25× finer grid; temporal decompositions with FDR control; 22-class land cover carries **15.3%** of RF importance |
| G9 | Fire points are not independently validated and data leakage is not audited | R35, R36, R37 (Q1) | Fire-point extraction not independently validated | **O5:** To validate fire points independently and audit the stack for leakage | r = 0.915 / ρ = 0.835 against burned area; leakage audit removed the top-3 features (≈ 0.40 importance) |
| G10 | The claimed physics benefit is asserted rather than tested against tuned baselines under shift | R13, R38, R22, R21 (A*/Q1) | Single model, no baseline comparison | **O4:** To test physics vs. no physics on every track, against tuned baselines | Physics effect negative on 3 of 4 tracks at 2.50× training cost; tuned RF 0.9704 and MaxEnt 0.9598 beat CDR-PINO's 0.9398 in-distribution |

---

## 3. Detailed Gap Statements

### G1. Susceptibility is modelled as classification, not as a governed physical field

**Gap.** Indian and international susceptibility studies from 2012 to 2026 frame the problem the same way: a per-pixel classifier or density estimator over independently derived rasters. The model varies (Random Forest [R6], ensembles [R3, R4], MaxEnt [R1], CNN [R7], global ML [R5]), but none states a governing equation. None represents spread between neighbouring pixels, and none imposes a physical inductive bias on the predictor–response mapping. The Q1 review by Jain et al. [R2] surveys the whole field and does not identify any susceptibility formulation built on a governing equation.

**In Biswas et al. [R1].** MaxEnt estimates the relative likelihood of fire per pixel from 15 predictors, and each pixel is treated independently.

**Research objective (O1).** To derive susceptibility as the solution of a convection–diffusion–reaction equation, obtained from a susceptibility balance law, in which each term is a named fire-behaviour mechanism that absorbs one of the reference study's predictor groups.

**How it is filled.** The equation is built up from an integral balance, through a constitutive flux law (Fickian fuel term plus terrain transport) and a Fisher–KPP production term, to the operational CDR form ∂u/∂t = D∇²u − **v**·∇u + ρσ(u)(1−σ(u)). Each term takes over one of R1's predictor groups:
- **Diffusion:** biophysical and climatic group.
- **Advection:** topographic group.
- **Reaction:** human-activity group.

The diffusion term has direct data support: global Moran's I = **0.8322** (z = 742.1) on baseline NDVI rejects the pixel-independence assumption behind static classifiers.

---

### G2. There is no temporal state, and models are never tested on unseen years

**Gap.** Fire activity in India shows a statistically significant increasing trend (p = 0.004) [R8], yet national susceptibility maps collapse the record into one static surface. Spatiotemporal deep-learning work does exist, but it targets next-day spread [R9], daily danger [R10] or sub-seasonal global forecasting [R11], not a long-horizon susceptibility field. It also does not use a mechanistic state equation.

**In Biswas et al. [R1].** Conditioning factors cover 2001–2020 and are used as whole-period layers. The output is a single probability map, so temporal generalisation cannot be evaluated.

**Research objective (O3 + O4).** To represent susceptibility as a latent field that evolves monthly across the full 266-month record (O3), and to test it on years withheld from training (O4, Track B3).

**How it is filled.**
- The latent field u(x,y,t) evolves over **266 months** (Nov 2000–Dec 2022).
- **Track B3** (leave-years-out: 2000, 2008, 2009 and 2015 held out) reaches **AUC 0.8960** (AP 0.1445, n = 856,596, 2.46% positive). This is an axis on which RF and MaxEnt cannot be evaluated at all.
- A controlled test compared a 22-year and a 20-year training window. The extra years give no accuracy gain (Δ = +0.0024 in favour of the 20-year model, within seed noise). They do add **+485 fire-affected pixels (+5.59%)** of real fire geography.

---

### G3. Physics-informed ML has not been applied to wildfire susceptibility, and never for India

**Gap.** PINNs [R12] and physics-informed ML more broadly [R13, R14] have been recommended for Earth-system problems where labels are scarce [R14]. In fire science, however, they are confined to *event-scale* problems:
- rate-of-spread parameter learning for a single event [R15];
- Bayesian PINN data assimilation for individual fires [R16].

The closest hazard analogue is a pointwise PINN for landslide susceptibility [R17]. A 2025–26 search found only event-scale spread emulators and fuel-density models, not national susceptibility.

**In Biswas et al. [R1].** Purely data-driven, with no physical constraint.

**Research objective (O3).** To develop and train the first physics-informed neural operator for wildfire susceptibility, and to apply it to India for the first time.

**How it is filled.** This is the first physics-informed neural *operator* for wildfire susceptibility and the first physics-informed treatment of the problem for India. The operator is trained on **541,545** forest-filtered MODIS C6.1 detections (reduced from 2,804,373 raw).

---

### G4. Neural operators are tested on synthetic problems on flat grids

**Gap.** Fourier and DeepONet operators [R18, R19, R20] learn maps between function spaces, but the standard benchmarks (PDEBench, NeurIPS A* [R21]) are made up entirely of synthetic PDE families. Environmental uses such as climate super-resolution exist, but they do not learn a governing equation from a multi-decade *observational* archive. Most also apply planar derivatives to latitude–longitude grids.

**Research objective (O3).** To train a neural operator on a real, multi-decade observational archive rather than a synthetic PDE family, with spatial derivatives evaluated exactly on the sphere.

**How it is filled.**
- An FNO backbone (1,054,613 parameters) is amortised across 265 monthly instances of a real 22-year archive.
- Spatial derivatives use the **exact Laplace–Beltrami operator on the sphere**. Across the domain (6.75°–37.09° N), one degree of longitude shrinks from 110.4 km to 88.7 km, a **24.5%** compression that a flat Laplacian would silently get wrong.

---

### G5. Learned-physics models give no well-posedness guarantee and leave coefficient signs to the loss

**Gap.** PINN training has documented failure modes (A* [R22]; Q1 [R23, R24]). A soft PDE penalty does not guarantee that the learned coefficients are physically admissible, for example non-negative diffusivity or upslope-directed transport. The fire PINN studies [R14, R15] also do not prove existence and uniqueness for the equation they fit.

**Research objective (O2).** To prove that the governing equation has a unique global-in-time weak solution over the full record, with every constant computed from measured data, and to guarantee the physically required coefficient signs by architecture.

**How it is filled.**
- **Theorem (global well-posedness):** the problem has a unique global-in-time weak solution over the full 266-month record. The constants are computed from measured data ranges, including India's maximum slope of **77.31°**.
- Coefficient signs are guaranteed **by architecture, not by loss pressure**:
  - D > 0;
  - **v** = softplus(c_raw)·∇E, which always points upslope;
  - ρ ≥ 0.

---

### G6. Interpretability is post-hoc and correlational

**Gap.** Susceptibility studies explain models after training, using SHAP [R25, R26, R27], permutation importance or Jackknife tests. These show that a variable *correlates* with the output inside an already-fitted model. They cannot test whether a physical *mechanism* is needed. Biswas et al. themselves note that MaxEnt percent contributions "can vary depending on the specific algorithmic path" [R1].

**Research objective (O3).** To measure each mechanism's own contribution by term ablation, so that every term of the equation can be falsified directly rather than ranked after training.

**How it is filled.** Each mechanism is removed in turn and the model retrained (term ablation). This is a falsification test, not a ranking:

| Configuration | ROC-AUC | AP |
|---|---:|---:|
| Diffusion only | 0.6017 | 0.6050 |
| + Advection | 0.9239 | 0.9014 |
| + Reaction (full CDR) | **0.9397** | **0.9233** |

Six independent analyses agree that elevation/terrain dominates. These are the ablation, the field slope excess (**+115%** at fire locations), permutation importance (drop 0.2268, 24.13%), response curves (Δ 0.4611), Jackknife "only-X" AUC **0.9399**, and the classical models' own rankings. This mirrors and extends R1's Jackknife test (Fig. 10).

---

### G7. Evaluation relies on one random split that spatial autocorrelation inflates

**Gap.** Random hold-out splits of spatially autocorrelated data overstate predictive skill. Large-scale ecological maps that look strong on random validation perform poorly under spatial validation [R29]. Blocked cross-validation is the recommended fix [R28, R31], and predictions outside the training feature space need explicit assessment [R30]. Susceptibility studies still mostly report one random split.

**In Biswas et al. [R1].** 70/30 random split of occurrences, AUC **0.894** (train) and **0.879** (test).

**Research objective (O4).** To evaluate the model on four generalisation tracks (random, spatial-block, leave-region-out and leave-years-out), applying the same spatial blocking to tuned Random Forest and MaxEnt baselines.

**How it is filled.** Four tracks, with identical spatial blocking applied to the baselines:

| Track | Protocol | RF | MaxEnt | CDR-PINO |
|---|---|---:|---:|---:|
| A | Random split | 0.9704 | 0.9598 | 0.9398 (seeds 42–44: 0.9391 ± 0.0017) |
| B1 | 2°×2° spatial blocks, 3 folds | 0.9498 ± 0.0035 | 0.9465 ± 0.0054 | 0.7510 ± 0.0182 |
| B2 | Leave-one-region-out (6 regions) | n/a | n/a | 0.6187 ± 0.0680 |
| B3 | Leave-years-out | n/a | n/a | **0.8960** |

A train-vs-validation diagnostic shows the B1/B2 drop is an out-of-distribution transfer failure, not overfitting. The train–validation gap stays between +0.009 and +0.027.

---

### G8. National maps are coarse, predictors are snapshots, and land cover is only a filter

**Gap.** National susceptibility products for India run at coarse resolution [R1]. Predictors enter as whole-period snapshots rather than being decomposed into climatology, anomaly and trend. Land cover is used to mask fire points instead of being tested as a predictor [R1, R32]. The MODIS products that support 1 km monthly work already exist [R33, R34].

**In Biswas et al. [R1].** Conditioning rasters at **0.25° × 0.25°**. ESA-CCI land cover is used only to confine fire points to forest.

**Research objective (O5).** To build a 1 km, month-resolved national predictor stack that keeps parity with the reference study's 15 conditioning variables, decomposes each into climatology, anomaly and trend, and admits 22-class land cover as a predictor.

**How it is filled.**
- Common grid of **3,641 × 3,504** pixels at about 1 km: **4,161,009** in-India pixels, a **25×** linear refinement over R1.
- NDVI is split into 9 features (climatology, anomaly, trend, Mann–Kendall τ, CVSI with k* = 8 months, LISA, breakpoint θ* = 0.535). LST and FLDAS variables get the same treatment, with **Benjamini–Hochberg FDR** control (for example, air temperature's 636 "significant" pixels fall to 0).
- The **22 ESA-CCI classes are used as predictors** and carry **15.3%** of RF importance.

---

### G9. Fire points are not independently validated and leakage is not audited

**Gap.** Susceptibility labels from active-fire products are rarely checked against an independent burned-area record [R35, R36]. Leakage, meaning a feature that encodes the outcome, is a documented cause of irreproducible ML results [R37]. Susceptibility pipelines seldom audit for it.

**In Biswas et al. [R1].** Burned area is described, but the fire-point extraction is not validated against it quantitatively.

**Research objective (O5).** To validate the extracted forest-fire points against an independent burned-area product, and to audit the predictor stack for label leakage before any model is trained.

**How it is filled.**
- Against MCD64A1.061 burned area: **r = 0.915, ρ = 0.835** (p < 0.0001, n = 23). The forest-masked re-derivation gives r = 0.9044.
- Against R1's own annual counts: this extraction runs **0.5–2.4% higher** across the 20 overlapping years.
- The **leakage audit** removed the 2020 and 2022 forest-fraction snapshots, which sit inside the label window and so carry reverse-causality risk. Before removal they held the **top-3 Gini ranks (≈ 0.40 combined)**.

---

### G10. The physics benefit is asserted, not tested against tuned baselines under shift

**Gap.** Physics constraints are said to improve generalisation and data efficiency [R13, R38]. In practice, physics-informed models are rarely compared against *tuned* classical baselines under *matched* distribution shift. Their known optimisation pathologies [R22] and synthetic-only benchmarks [R21] leave the question open.

**In Biswas et al. [R1].** A single MaxEnt model with no comparison model.

**Research objective (O4).** To test whether the physics constraint improves robustness under distribution shift, through a matched physics-vs-no-physics comparison on all four tracks against tuned classical baselines, and to report the answer whichever way it falls.

**How it is filled.**
- **Tuned baselines:** RF (depth 25, leaf 3) reaches 0.9704 AUC; MaxEnt (β = 4.0, replicating R1's configuration) reaches 0.9598. Both beat the operator in-distribution (by about 0.02 AUC) and under spatial blocking (by about 0.20 AUC). This is reported plainly.
- **Physics vs. no physics** (matched architecture and budget):

| Track | With physics | No physics | Δ |
|---|---:|---:|---:|
| A | 0.9406 | 0.9461 | −0.0055 |
| B1 | 0.7595 | 0.7555 | +0.0040 |
| B2 | 0.5978 | 0.6368 | −0.0390 |
| B3 | 0.8935 | 0.9059 | −0.0124 |

- The difference is negative on 3 of 4 tracks. The one positive difference is smaller than that track's fold spread. The physics loss also costs **2.50×** training time (146.7 s → 367.4 s). This negative result is disclosed in full.

---

## 4. Gap → Objective Coverage Matrix

| Gap | O1 Formulate | O2 Guarantee | O3 Solve and test | O4 Evaluate honestly | O5 Build and validate data |
|---|:-:|:-:|:-:|:-:|:-:|
| G1 Static classification, no governing equation | ✔ | | | | |
| G2 No temporal state / no unseen-year test | | | ✔ | ✔ | |
| G3 No physics-informed susceptibility model | | | ✔ | | |
| G4 Operators on synthetic, flat-grid problems | | | ✔ | | |
| G5 No well-posedness / unconstrained signs | | ✔ | | | |
| G6 Post-hoc, correlational interpretability | | | ✔ | | |
| G7 Single random-split evaluation | | | | ✔ | |
| G8 Coarse, snapshot predictors | | | | | ✔ |
| G9 Unvalidated labels / unaudited leakage | | | | | ✔ |
| G10 Physics benefit untested | | | | ✔ | |

Every gap is addressed by at least one objective, and every objective addresses at least one gap. **O5 is new:** add it to §I-E of `CDR_PINO_TGRS.tex` (one sentence) so the manuscript's objectives match this document.

> **Consistency flag (still open in the manuscript):** the contributions paragraph and §II-D of `CDR_PINO_TGRS.tex` list "specific humidity" as an added feature. The pipeline actually uses **derived relative humidity** (Magnus formula). Any G8 text copied into the paper should say "relative humidity (derived)".

---

## 5. References (2010–2026)

| ID | Reference | Tier |
|---|---|---|
| R1 | Biswas, U., Mahato, S., & Joshi, P. K. (2025). Spatial prediction of forest fires in India: a machine learning approach for improved risk assessment and early warning systems. *Environ. Sci. Pollut. Res.*, 32(8), 4856–4878. doi:10.1007/s11356-025-35982-8 | Q1 (SJR) |
| R2 | Jain, P., Coogan, S. C. P., Subramanian, S. G., Crowley, M., Taylor, S., & Flannigan, M. D. (2020). A review of machine learning applications in wildfire science and management. *Environ. Rev.*, 28(4), 478–505. doi:10.1139/er-2020-0019 | Q1 |
| R3 | Kanda Naveen Babu, Gour, R., Kurian Ayushi, Ayyappan, N., & Parthasarathy, N. (2023). Environmental drivers and spatial prediction of forest fires in the Western Ghats biodiversity hotspot, India: an ensemble machine learning approach. *For. Ecol. Manage.*, 540, 121057. | Q1 |
| R4 | Sarkar, M. S., et al. (2024). Ensembling machine learning models to identify forest fire-susceptible zones in Northeast India. *Ecol. Inform.*, 81, 102598. | Q1 |
| R5 | Shmuel, A., & Heifetz, E. (2022). Global wildfire susceptibility mapping based on machine learning models. *Forests*, 13(7), 1050. | Q1 (SJR) |
| R6 | Oliveira, S., Oehler, F., San-Miguel-Ayanz, J., Camia, A., & Pereira, J. M. C. (2012). Modeling spatial patterns of fire occurrence in Mediterranean Europe using Multiple Regression and Random Forest. *For. Ecol. Manage.*, 275, 117–129. doi:10.1016/j.foreco.2012.03.003 | Q1 |
| R7 | Zhang, G., Wang, M., & Liu, K. (2019). Forest fire susceptibility modeling using a convolutional neural network for Yunnan province of China. *Int. J. Disaster Risk Sci.*, 10(3), 386–403. | Q1 |
| R8 | Vadrevu, K. P., Lasko, K., Giglio, L., Schroeder, W., Biswas, S., & Justice, C. (2019). Trends in vegetation fires in South and Southeast Asian countries. *Sci. Rep.*, 9, 7422. doi:10.1038/s41598-019-43940-x | Q1 |
| R9 | Huot, F., Hu, R. L., Goyal, N., Sankar, T., Ihme, M., & Chen, Y.-F. (2022). Next day wildfire spread: a machine learning dataset to predict wildfire spreading from remote-sensing data. *IEEE Trans. Geosci. Remote Sens.*, 60, 4412513. | Q1 |
| R10 | Kondylatos, S., Prapas, I., Ronco, M., Papoutsis, I., Camps-Valls, G., Piles, M., Fernández-Torres, M.-Á., & Carvalhais, N. (2022). Wildfire danger prediction and understanding with deep learning. *Geophys. Res. Lett.*, 49(17), e2022GL099368. doi:10.1029/2022GL099368 | Q1 |
| R11 | Prapas, I., Bountos, N. I., Kondylatos, S., Michail, D., Camps-Valls, G., & Papoutsis, I. (2025). SeasFire cube: a multivariate dataset for global wildfire modeling. *Sci. Data*, 12, 41. | Q1 |
| R12 | Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *J. Comput. Phys.*, 378, 686–707. doi:10.1016/j.jcp.2018.10.045 | Q1 |
| R13 | Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. *Nat. Rev. Phys.*, 3(6), 422–440. doi:10.1038/s42254-021-00314-5 | Q1 |
| R14 | Reichstein, M., Camps-Valls, G., Stevens, B., Jung, M., Denzler, J., Carvalhais, N., & Prabhat (2019). Deep learning and process understanding for data-driven Earth system science. *Nature*, 566, 195–204. doi:10.1038/s41586-019-0912-1 | Q1 |
| R15 | Vogiatzoglou, K., Papadimitriou, C., Bontozoglou, V., & Ampountolas, K. (2025). Physics-informed neural networks for parameter learning of wildfire spreading. *Comput. Methods Appl. Mech. Eng.*, 434, 117545. | Q1 |
| R16 | Dabrowski, J. J., et al. (2023). Bayesian physics informed neural networks for data assimilation and spatio-temporal modelling of wildfires. *Spat. Stat.*, 55, 100746. | Q1 (SJR) |
| R17 | Dahal, A., & Lombardo, L. (2025). Towards physics-informed neural networks for landslide prediction. *Eng. Geol.*, 344, 107852. | Q1 |
| R18 | Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. *ICLR*. | CORE A* |
| R19 | Kovachki, N., et al. (2023). Neural operator: learning maps between function spaces with applications to PDEs. *J. Mach. Learn. Res.*, 24(89), 1–97. | Q1 |
| R20 | Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nat. Mach. Intell.*, 3, 218–229. doi:10.1038/s42256-021-00302-5 | Q1 |
| R21 | Takamoto, M., et al. (2022). PDEBench: an extensive benchmark for scientific machine learning. *NeurIPS Datasets and Benchmarks*, 35. | CORE A* |
| R22 | Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R., & Mahoney, M. W. (2021). Characterizing possible failure modes in physics-informed neural networks. *NeurIPS*, 34. | CORE A* |
| R23 | Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM J. Sci. Comput.*, 43(5), A3055–A3081. | Q1 |
| R24 | Wang, S., Sankaran, S., & Perdikaris, P. (2024). Respecting causality for training physics-informed neural networks. *Comput. Methods Appl. Mech. Eng.*, 421, 116813. doi:10.1016/j.cma.2024.116813 | Q1 |
| R25 | Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nat. Mach. Intell.*, 2(1), 56–67. doi:10.1038/s42256-019-0138-9 | Q1 |
| R26 | Iban, M. C., & Aksu, O. (2024). SHAP-driven explainable artificial intelligence framework for wildfire susceptibility mapping using MODIS active fire pixels: a case study in İzmir, Türkiye. *Remote Sens.*, 16(15), 2842. | Q1 (SJR) |
| R27 | Hang, H. T., Mallick, J., Alqadhi, S., Bindajam, A. A., & Abdo, H. G. (2024). Exploring forest fire susceptibility and management strategies in Western Himalaya: integrating ensemble machine learning and explainable AI. *Environ. Technol. Innov.*, 35, 103655. | Q1 |
| R28 | Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. doi:10.1111/ecog.02881 | Q1 |
| R29 | Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nat. Commun.*, 11, 4540. doi:10.1038/s41467-020-18321-y | Q1 |
| R30 | Meyer, H., & Pebesma, E. (2021). Predicting into unknown space? Estimating the area of applicability of spatial prediction models. *Methods Ecol. Evol.*, 12(9), 1620–1633. doi:10.1111/2041-210X.13650 | Q1 |
| R31 | Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G. (2019). blockCV: an R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. *Methods Ecol. Evol.*, 10(2), 225–232. doi:10.1111/2041-210X.13107 | Q1 |
| R32 | Uthappa, A. R., et al. (2025). Forest fire susceptibility mapping using multi-criteria decision making and machine learning models in the Western Ghats of India. *J. Environ. Manage.*, 379, 124777. | Q1 |
| R33 | Giglio, L., Schroeder, W., & Justice, C. O. (2016). The Collection 6 MODIS active fire detection algorithm and fire products. *Remote Sens. Environ.*, 178, 31–41. doi:10.1016/j.rse.2016.02.054 | Q1 |
| R34 | Wan, Z. (2014). New refinements and validation of the Collection-6 MODIS land-surface temperature/emissivity product. *Remote Sens. Environ.*, 140, 36–45. | Q1 |
| R35 | Giglio, L., Boschetti, L., Roy, D. P., Humber, M. L., & Justice, C. O. (2018). The Collection 6 MODIS burned area mapping algorithm and product. *Remote Sens. Environ.*, 217, 72–85. doi:10.1016/j.rse.2018.06.016 | Q1 |
| R36 | Rodrigues, M., & de la Riva, J. (2014). An insight into machine-learning algorithms to model human-caused wildfire occurrence. *Environ. Model. Softw.*, 57, 192–201. doi:10.1016/j.envsoft.2014.03.003 | Q1 |
| R37 | Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. doi:10.1016/j.patter.2023.100804 | Q1 |
| R38 | Read, J. S., et al. (2019). Process-guided deep learning predictions of lake water temperature. *Water Resour. Res.*, 55(11), 9173–9190. | Q1 |

**New relative to `references.bib`** (verified online 2026-09-24, not yet added to the .bib): R6 Oliveira 2012, R8 Vadrevu 2019, R10 Kondylatos 2022, R14 Reichstein 2019, R29 Ploton 2020, R30 Meyer & Pebesma 2021, R36 Rodrigues & de la Riva 2014. R20 Lu 2021, R31 Valavi 2019 and R37 Kapoor & Narayanan 2023 are standard references, but check their DOIs before adding them. All other entries are already in the manuscript's verified bibliography.
