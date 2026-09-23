# Study Clarifications — Q&A for Manuscript Finalization

> Answers your 13 questions directly, in plain language first then technical detail,
> each grounded in what's actually been built/measured this project (no invented
> numbers) — flagged clearly wherever something still needs verification or further
> work rather than presented as settled. Four additional questions are added at the
> end that reviewers are likely to ask, based on the same standard this document
> applies throughout.

---

## Q1. Forest-fire prevention policy and SDG impacts — currently missing

**You're right, this is missing, and it's a real gap** — Biswas et al. (2025)
explicitly ties their results to 9 SDGs (1, 2, 3, 5, 6, 9, 12, 13, 15) and India's
actual governance apparatus (NFFPMS — National Forest Fire Prevention and Management
Scheme, established 2003; state-level strategies for Uttarakhand, Himachal Pradesh,
Maharashtra, Northeast states). This costs no new computation, only writing — and it
should be written, because reviewers comparing you directly to Biswas et al. will
notice its absence.

**What to add, and why this study's version can be genuinely richer than a generic
restatement of Biswas's framing**: your model's term-decomposition (diffusion/
advection/reaction) gives *differentiated* policy relevance that an undifferentiated
MaxEnt probability map cannot:

- **Diffusion (vegetation/moisture)** → actionable as fuel/vegetation management,
  controlled burns, forest composition policy (SDG 15 — Life on Land).
- **Advection (terrain)** → actionable as terrain-aware firebreak placement and
  early-warning resource pre-positioning specifically on steep upslope corridors
  (SDG 13 — Climate Action, SDG 11-adjacent disaster preparedness).
- **Reaction (human-ignition, road-proximity)** → actionable as patrol allocation
  and ignition-source control near roads (SDG 1/2 — the socioeconomic
  forest-dependent-community angle Biswas et al. emphasize, and SDG 9 — resilient
  infrastructure).

A single paragraph in the Discussion mapping each PDE term to a distinct
intervention type is a genuine novelty point (Biswas et al.'s single undifferentiated
map cannot support this kind of targeted framing) and costs nothing to add now.

---

## Q2. Computational cost / time-complexity comparison — expand what exists

**You already have real, measured numbers** — `CDR_PINN_Methodology_Section.md`
Section 7 has a cross-model table (RF 195.2s train/1.4s inference, MaxEnt 1,486.8s
train/33.3s inference) and an analytic FNO complexity argument. What's missing is
pulling it into the paper's own Results/Discussion narrative explicitly, and adding
the CDR-PINN numbers now that you have them:

**Table to add** (all real, measured this project):

| Model | Params/trees | Train time | Inference | Notable |
|---|---:|---|---|---|
| Random Forest | 200 trees | 195.2 s | 1.4 s | fastest |
| MaxEnt | linear+hinge+product | 1,486.8 s | 33.3 s | slowest to train |
| CDR-PINN (full physics) | 1,054,613 | 354.6 s (80 ep) | not yet benchmarked separately | — |
| CDR-PINN (no physics) | 1,054,613 | 140.4 s (80 ep) | — | **same architecture, ~2.5× faster without the PDE/BC loss terms** |

That last row is a genuinely useful, already-measured number: **the physics
constraint itself costs ~2.5× training time** (354.6s vs. 140.4s, identical
architecture/data/epochs) — a real, quotable computational-cost-of-physics number for
your Discussion, not an estimate.

**On A*-venue references specifically**: I don't want to invent citations for this
section — the PINO paper itself (already in your reference list) reports concrete
compute comparisons (400× speedup vs. a GPU pseudo-spectral solver) and FourCastNet
(Kurth et al., 2023, PASC'23, already cited) is built entirely around a compute/
accuracy tradeoff argument for its AFNO architecture — both are legitimate models for
how to frame this section. If you want additional A*-venue papers specifically
benchmarking neural-operator computational efficiency (NeurIPS/ICML/ICLR), I'd need
to run a dedicated, verified literature search rather than guess — say the word and
I'll do that the same way I sourced the 15 citations already in the paper draft.

---

## Q3. Why RF specifically, when XGBoost and other ML methods exist

**Your study already contains the answer — it's just not stated as a defense.**
XGBoost isn't absent from this work: it's in Step 8's model ladder
(LR → RF → XGBoost → plain MLP → PINN), where it scored **0.9678** on Track A —
essentially tied with RF's 0.9676 (the pre-Step-5-expansion number). So the honest,
correct framing is not "we chose RF instead of XGBoost," it's "we tested both, they
perform equivalently, and RF was designated the headline classical baseline for three
stated reasons":

1. **In-code rationale already on record** (Step 7): robust to the very different
   scales/units across NDVI, LST, and LULC features with no scaling needed, gives
   feature importances for free, parallelizes cleanly.
2. **Literature precedent** — RF is the single most common baseline across the
   newly-reviewed literature (Uthappa et al. 2025 combine it with AHP/SVM/XGBoost;
   Kantarcioglu et al. 2023 compare it directly to ANN; Guria et al. 2025 compare it
   against XGBTree/AdaBag/GBM) — using it as the primary classical reference point
   keeps this study directly comparable to the field's own convention, while XGBoost
   still appears in the full ladder for readers who want the stronger-gradient-
   boosting comparison too.
3. **Interpretability match** — RF's Gini importance is the metric already used
   throughout this study's own feature-engineering narrative (Q8 below); keeping one
   consistent importance metric across the classical-baseline story avoids mixing
   incompatible importance definitions (Gini vs. XGBoost's gain-based importance vs.
   MaxEnt's permutation importance) in the same paper.

**Why PINO specifically, not just "a neural network"** — this is the Introduction's
own stated gap (§1.4 of the paper draft): no prior wildfire-susceptibility study, in
India or internationally, uses a physics-informed *operator* architecture. RF/XGBoost/
MaxEnt are all correlational; PINO is the only one of the model families tested here
whose architecture can be constrained by a governing physical equation at all — the
comparison isn't "which black box scores highest," it's "does adding a mechanistic
constraint change what's learnable, and can we test that directly" (which the
term-ablation study does).

---

## Q4. Is the CDR equation itself novel, and do you need a component-comparison test?

**Be precise about what's actually claimed as novel — the PDE terms themselves are
not.** Diffusion (Fick, 1855), advection, and Fisher–KPP reaction (Fisher, 1937;
Kolmogorov–Petrovsky–Piskunov, 1937) are all 90–170-year-old classical forms. Claiming
*those* as novel would be an easy, damaging reviewer objection. What's actually novel,
stated precisely:

1. The **specific construction** of each coefficient from real Indian remote-sensing
   data via small learned sub-networks (`D_net`, the advection scalar, `ρ_net`) — no
   existing wildfire study builds a diffusion/advection/reaction coefficient this way.
2. The **deliberate mapping** of each term onto one of Biswas et al.'s own four
   predictor groups — a structuring choice, not a mathematical one, but a genuine
   contribution to how the equation is motivated and interpreted.
3. Embedding this specific construction in a **neural operator** (not a classical
   numerical PDE solver, not a pointwise PINN) — the combination is what's new, not
   either ingredient alone.
4. **Construction-specific well-posedness proofs** — not citing that "diffusion
   equations are well-posed" generically, but proving it for *this* softplus-bounded,
   data-derived `D`/`v`/`ρ` construction, with constants derived from your own
   verified data extremes (NDVI range, measured max slope). This is real, original
   mathematical work, and the most defensible "novel" claim in the whole document.

**You've already done the component-comparison test you're asking about** — the
term-ablation study (diffusion-only → +advection → +reaction, AUC 0.602 → 0.924 →
0.941) *is* exactly this analysis: it isolates each component's measured contribution
rather than asserting the equation's value from its construction alone. Point
reviewers to that table directly when this question comes up. If you want to go
deeper, a natural extension is comparing your specific reaction form (Fisher–KPP)
against an alternative (e.g., a simple linear source term) to show *that specific
choice*, not just "having a reaction term," matters — not yet done, a reasonable
future-work item (see Q13).

---

## Q5. Can you do a Biswas-style variable-importance test for the CDR-PINN?

Biswas et al. use two things: MaxEnt's own permutation importance (shuffle a
variable, measure the AUC drop) and a Jackknife test (train with only one variable,
and separately with all-but-one). Your RF and MaxEnt baselines already have directly
equivalent numbers (Gini importance for RF, permutation importance for MaxEnt — both
already computed in Step 7).

**Update 2026-08-21 — all three now run, with real results.** All three of Biswas's
variable-understanding analyses have been reproduced for CDR-PINN, exactly the
per-covariate-per-physics-head adaptation described below, applied for real:

1. **Permutation importance** (inference-only, trained checkpoint): shuffling
   elevation spatially drops AUC by 0.2238 (23.80% of baseline); every other
   covariate shows ~0.0000 effect.
2. **Response curves** (Biswas Figs. 8/9 analogue, inference-only): sweeping
   elevation across its range swings predicted probability by 0.428; every other
   covariate's swing is 0.0001–0.008.
3. **Jackknife** (Biswas Fig. 10 analogue, genuine retraining — 14 runs, leave-one-
   covariate-out and leave-only-one-covariate-in across the 7 covariates, plus a
   matched-budget baseline; re-run 2026-08-23 with genuine validation-set-driven
   early stopping, patience=4): removing elevation drops AUC by 0.1370 (every other
   removal is noise-level, 0.0004–0.0032); elevation *alone* reaches AUC=0.9399,
   within 0.0002 of the full 7-covariate model (0.9397).

All three converge on the same finding: shuffling slope drops CDR-PINN's AUC by
~0% (not 16.7% as in Biswas's own MaxEnt), because CDR-PINN's advection head is
already structurally routing terrain information in a way that makes elevation,
not slope in isolation, the dominant learned signal — a genuinely different
(and stronger) finding than the literature-mirrored "match their number" comparison
originally proposed here. Full tables: `CDR_PINN_Full_Paper_Draft.md` §4.4–4.6.

---

## Q6. Plain-language explanation: what did this study actually find, and why does it matter?

**In one paragraph**: You built a computer model that predicts forest-fire risk
across India, but instead of the model being a "black box" that just learns
correlations (like every prior study, including the reference paper), you gave it a
simplified physical theory of *how* fire risk spreads — vegetation dries out and
carries risk between neighboring areas (diffusion), fire preferentially climbs
uphill (advection), and human activity near roads adds new risk (reaction). You then
tested whether each of these three physical ideas actually helps the model predict
real fire locations it never saw during training. The answer: yes, especially the
"fire climbs uphill" idea, which produced the single biggest jump in accuracy of
anything tested. You also tested whether the model could predict fire risk in years
it had never seen (2000, 2008, 2009, 2015) — it did reasonably well (89.6% accuracy
on a 0–100% scale where 50% is a coin flip) — something none of the classical models
in this study, or in the prior published literature, can even attempt, because they
don't track time at all.

**Why it matters beyond the numbers**: a model that says "this area is high-risk"
is less useful to a fire-management agency than one that says "this area is
high-risk *because of terrain*, so build firebreaks here" vs. "*because of road
proximity*, so increase patrols here." Classical models (RF, MaxEnt) cannot make
that distinction — they only rank importance, they don't decompose cause. Your model
can, in principle, once fully validated. That mechanistic, actionable distinction —
not raw accuracy — is the actual scientific contribution, and it's why the honest
reporting of where the model currently falls short (spatial generalization, Q1 vs.
Q2 not yet closed) doesn't undermine the contribution; it's a normal, expected part
of introducing a genuinely new modeling paradigm to a field that has only ever used
correlational methods.

---

## Q7. Three models, spatial/temporal/pixel-level analysis — impact not explained

This is a fair gap to close explicitly in the Discussion. What the three model
*families* (MaxEnt, RF, CDR-PINN) and the pixel-level, spatial-block, region-level,
and year-level analyses collectively demonstrate, stated as a single coherent
argument rather than separate results:

- **Pixel-level (Track A)**: establishes a fair accuracy baseline — do all three
  paradigms roughly agree on *in-distribution* prediction? Yes (RF 0.968, MaxEnt
  0.960, CDR-PINN 0.941) — meaning the physics-informed reformulation doesn't
  sacrifice much accuracy where classical methods already work.
- **Spatial-block / region-level (Tracks B1/B2)**: tests whether that agreement
  survives when the model must generalize to geography it never saw. This is where
  real operational risk lives — a fire-management agency needs risk maps for regions
  with sparse or no historical fire data. The current answer (weak, 0.7510/0.6187) is a
  genuine, actionable finding: **none of the models in this study, including the
  physics-informed one at its current scale, are yet reliable for truly novel
  regions** — a caution worth stating plainly for anyone deploying this kind of
  model operationally, not just a limitation of your specific implementation.
- **Temporal / year-level (Track B3)**: the one axis where CDR-PINN has no
  competitor to lose to, because RF/MaxEnt structurally cannot be tested here (Q9
  below explains why). This is the study's clearest evidence that a
  dynamically-formulated model captures something a static one cannot.

State this as the paper's actual "depth of analysis" contribution: not just "we got
these numbers," but "we identified *which axis of generalization* each modeling
paradigm can and cannot be meaningfully evaluated on" — itself a methodological
lesson for the field, independent of who wins.

---

## Q8. Feature engineering (e.g., NDVI) — what's the actual impact?

Real, measured impact, not a claim: Step 7's Random Forest Gini importance ranking
(55-feature model, tuned via validation, 2026-08-22 retrain — updated after a
2026-08-21 data-leakage fix removed `forest_frac_recent`/`current`) is the direct
evidence.

**Top features, and what they tell you about which engineering choices paid off**:

| Rank | Feature | Importance | What this validates |
|---|---|---:|---|
| 1 | `forest_frac_baseline` | 0.2066 | LULC forest-fraction engineering (Step 6) — the single most informative feature in the whole 55-feature set, now that the leakage-risk `recent`/`current` snapshots have been removed |
| 2 | `ndvi_trend_2x12ma` | 0.0886 | the trend-decomposition feature engineering (Step 2, classical 2×12-MA), not the raw NDVI mean, is what matters most from the NDVI feature family |
| 3 | `ndvi_mean` | 0.0858 | the simplest NDVI feature still contributes, close behind its own derived trend |
| 4 | `ndvi_below_threshold` | 0.0749 | the fire-data-driven breakpoint feature (Step 2) — a genuinely novel, data-derived threshold, not an assumed constant |
| 5 | `ndvi_clim_june` | 0.0629 | seasonal climatology matters, not just the annual mean |
| 6 | `terrain_slope` | 0.0456 | validates Step 5a's terrain engineering — directly corroborates Biswas et al.'s own 16.7% slope contribution |

**The general lesson, worth stating explicitly**: engineered *derived* features
(forest fraction, NDVI trend, the breakpoint feature) outrank raw/simpler source
variables — this justifies the feature-engineering investment across Steps 2–6 as
more than methodological thoroughness; it's measurably what the model actually
relies on. The 9 raw-NDVI-only features would have missed this if the pipeline had
stopped at `ndvi_mean` alone. **Note on the leakage fix's own impact on this
table**: with `forest_frac_recent`/`current` removed, `forest_frac_baseline` alone
now carries roughly what those three features carried combined (was
0.166+0.120+0.111=0.397 vs. now 0.2066) — a real, measurable redistribution, not
just a relabeling, though total accuracy barely moved (§Q-headline numbers).

---

## Q9. 15 variables (Biswas) vs. 59 columns (this study) — reconciled precisely

**Real, exact accounting** (verified against the actual parquet schema, not
approximated; updated 2026-08-21 after a data-leakage fix — see below): the parquet
has **59 columns** total — 4 are not features at all (`lon`, `lat`, `fire_count`,
`fire_ever` — georeferencing and labels, dropped before training), leaving **55
features**. Of those 55:

- **31 features** are richly-engineered decompositions of Biswas's original 15
  variable *groups* — e.g., Biswas's single "air temperature" raster becomes this
  study's `fldas_airtemp_anomaly` + `fldas_airtemp_mk_tau_monthly` (2 features:
  anomaly + trend-significance, not just one raw snapshot); NDVI alone becomes 9
  features (mean, climatology, anomaly, trend, residual, Mann-Kendall τ, the novel
  CVSI index, LISA cluster, breakpoint threshold).
- **24 features** are genuinely additional, not part of Biswas's 15 at all: the full
  22-class ESA-CCI land-cover fractional breakdown (`landcover_frac_LC22_*`) and DTR
  (diurnal temperature range, a natural derived quantity from day/night LST that
  Biswas doesn't separately track).

**2026-08-21 correction**: this table originally listed 58 features/62 columns,
including `forest_frac_recent` (2020) and `forest_frac_current` (2022) as 2 of the
"4 forest-fraction features." Both were dropped as a data-leakage fix — they
overlapped the pooled 2000–2022 fire label's own time window, a real
reverse-causality risk (published literature documents burned forest commonly gets
reclassified to shrubland/agriculture in later land-cover products). Only
`forest_frac_baseline` (2001) survives, now counted among the 31 Biswas-group
decompositions rather than as a separate "additional" bonus family.

**Impact of this choice, stated honestly**: this is *not* a claim of having 59
independent Biswas-equivalent predictors — it's the same 15 conceptual variable
groups represented with the temporal-decomposition machinery (climatology/anomaly/
trend/significance) this whole pipeline is built around, plus a genuinely useful
bonus (land cover class detail) Biswas's own MaxEnt input never had access to at
this granularity. State this distinction explicitly in the paper's data section —
"15 variable groups, richly feature-engineered into 55 total predictors" — so a
reviewer doesn't misread it as an inflated or incomparable predictor count.

---

## Q10. RF/MaxEnt preprocessing — how do `.tif` files actually become RF training data?

**Real pipeline, traced precisely, not guessed**: RF and MaxEnt (Step 7) never touch
raw `.tif` files directly. The flattening happens entirely in **Step 6**:

1. Every upstream step's output (NDVI, LST, FLDAS, land cover, terrain,
   accessibility — all GeoTIFFs on the shared ~3641×3504 NDVI grid) is stacked into
   one 57-band `Integrated_FireRisk_Stack.tif` (was 60-band before the 2026-08-21
   leakage fix removed 3 forest-fraction bands, Q9 above).
2. Step 6 then **flattens** this stack: for every pixel that passes the validity
   mask (`india_mask & ~isnan(ndvi_mean)`), each of the 57 bands' value at that
   pixel becomes one column value in one row of `Integrated_FireRisk_Pixels.parquet`
   — a standard raster-to-tabular "stack and ravel" operation, restricted to
   in-India, non-NaN pixels only (4,161,009 rows survive out of the full grid).
3. **Step 7 (RF/MaxEnt) reads only this parquet** — plain tabular data, `pandas`/
   `sklearn`-compatible, no raster libraries involved at model-training time at all.

**This is a genuinely useful contrast to add to the Methods section explicitly**:
RF/MaxEnt consume a *flattened, spatially-decontextualized* table (each row is an
independent pixel, with no explicit awareness that neighboring rows are
geographically adjacent) — whereas CDR-PINN consumes the *raw gridded tensor*
directly (256×256 spatial structure preserved, adjacency is architecturally
meaningful via the FNO's spectral convolutions). This is precisely the "discards
spatial continuity" critique leveled at the classical paradigm in your own
Introduction (§1.1) — worth citing this exact mechanism as the concrete technical
reason for that critique, not just an abstract claim.

---

## Q11. Mathematical proof of the CDR equation's construction — consolidated summary

Full, complete proofs live across four documents (`CDR_PINN_Diffusion_Design_v2.md`
§4, `CDR_PINN_Advection_Design.md` §5–6, `CDR_PINN_Reaction_Design.md` §5) — here is
the condensed logical chain for your own understanding, in the order the proof
actually builds:

1. **Diffusion alone**: `D(x,y,t) = softplus(D_net(...))` is bounded in
   `[D_min, D_max]`, `0 < D_min`, by construction — `D_net` is a finite MLP on a
   compact input domain (verified NDVI/forest-fraction ranges), so by the extreme
   value theorem its output is bounded, and `softplus` maps any bounded input to a
   bounded, strictly-positive output. This gives **uniform parabolicity** — the
   standard hypothesis needed for Evans (2010) Ch. 7's existence/uniqueness theorem
   for linear parabolic PDEs.
2. **Adding advection**: the drift term `v·∇u` is a *lower-order* perturbation of the
   diffusion operator's principal part — it doesn't change ellipticity, so the same
   theorem family applies, but now needs **Gårding's inequality** (a weaker,
   sufficient condition than full coercivity) rather than direct coercivity. Proven
   via Young's inequality: the drift term's contribution to the energy estimate is
   absorbed into half the diffusion term's own coercivity, leaving explicit constants
   `α = D_min/2`, `β = V_max²/(2D_min)` (both computed from real data: `D_min` from
   NDVI's range, `V_max` from the measured 77.31° maximum slope).
3. **Adding reaction**: the Fisher–KPP term `ρ·σ(u)(1−σ(u))` is proven **globally**
   (not just locally) bounded (`≤ρ_max/4` for *any* real `u`) and **globally**
   Lipschitz (exact constant `ρ_max/(6√3)`, from bounding `σ''(u)`). This is the
   strongest link in the chain: most reaction terms (e.g. a plain cubic) only
   guarantee local-in-time existence and can blow up; this specific form rules that
   out by construction. Combined with steps 1–2 via a **Gronwall inequality**, this
   gives **global-in-time** existence and uniqueness over the entire `T=266`-month
   horizon — the complete result.

Each step **extends** the previous one's proof rather than replacing it — worth
stating this explicitly in the paper as "the proof is built incrementally, term by
term, matching the equation's own incremental design," which is itself a clean,
pedagogically defensible structure for a Methods section.

---

## Q12. Limitations of PINO for satellite-driven wildfire mapping — not yet stated

Real, specific limitations, several with measured evidence behind them (not
generic caveats):

1. **Data-hunger tension with sparse labels.** Deep operator learning inherits
   standard deep learning's appetite for data; real fire observations are only
   ~2.3% positive per month. This was directly observed, not hypothesized — the
   first diffusion-only training run collapsed to a trivial solution before the
   class-imbalance fix (Q-what-happened, documented in Methodology §8).
2. **Spectral truncation vs. sharp local risk features.** FNO represents fields via
   a small number of global Fourier modes (16×16 here) — well-suited to smooth,
   large-scale patterns, structurally less able to represent sharp, highly local
   discontinuities (e.g., risk concentrated right at one village's edge) than a
   local-receptive-field CNN would be. Not directly tested here, but a real,
   citable architectural tradeoff.
3. **Rectangular-grid approximation of India's true boundary.** The Neumann
   whole-sample-symmetric extension fixes the FFT differentiation's boundary
   artifact (verified, §2 of the advection document), but the underlying grid is
   still rectangular — India's true coastline/border geometry is only approximated,
   not exactly represented, unlike a mesh-based finite-element method could achieve.
4. **Physics-loss computational overhead is real and measured**: ~2.5× training
   time versus an identical no-physics architecture (Q2 above) — a genuine
   deployment-cost consideration if this were ever scaled to operational,
   frequently-retrained use.
5. **Optimization sensitivity, empirically demonstrated, not just asserted — updated
   2026-08-21.** Six tuning-side interventions have now been tested: scale-up
   (0.9406→0.9292, worse, confirmed again on an independent split at 0.9339),
   causal time-weighting (0.9369, worse), staged curriculum learning (0.9343,
   worse), and the learning-rate schedule, tested twice with **opposite outcomes**
   on two different splits — worse on the original Track A split (0.9154), but
   *best* of three configurations on an independent, honest validation-selected
   split (0.9403 test AUC). The LR-schedule result is therefore **split-sensitive,
   not ruled out** — the only clean, split-independent finding is that scale-up is
   robustly worse. Five of six interventions land in a ~0.93–0.94 AUC band
   regardless of split, more consistent with a representation ceiling (elevation
   dominance, confirmed via Jackknife retraining, Q5) than an unresolved
   optimization landscape — though multi-seed testing (item 3 above) is still
   needed to fully settle the split-sensitivity itself.
6. **Resolution-independence is proven, not yet verified operationally.**
   Discretization convergence is a property of the FNO architecture (Li et al.,
   2023) but has not been empirically exercised on a trained CDR-PINN checkpoint —
   until tested, this remains a theoretical rather than demonstrated capability for
   this specific model.
7. **Inherited satellite-data limitations, compounded by the dense-grid
   requirement.** Cloud cover, sensor gaps, and QA-filtering (already handled via
   masking upstream) introduce noise any model would inherit — but RF/MaxEnt can
   train on incomplete rows more flexibly than a spatial operator that expects a
   dense, complete grid at every training step.

---

## Q13. Future work — not yet stated

Concrete, prioritized, each tied to a specific open question raised in this
document or the paper draft, not a generic wish list:

1. **Physics-vs-no-physics comparison on Tracks B1/B2/B3**, not just Track A — the
   single most important unresolved experiment for the paper's central claim
   (flagged repeatedly in the Methodology and Novelty documents).
2. **Instance-wise fine-tuning** (Li et al., 2023 §3.2) and **self-adaptive
   per-point loss weighting** (McClenny & Braga-Neto, 2020) — the two
   literature-prescribed, qualitatively-different techniques not yet tried, given
   that six scale/schedule/causal/curriculum interventions have now all been tested
   (Q12 item 5, updated 2026-08-21) without closing the gap.
3. **Multi-seed robustness testing** with bootstrap confidence intervals, mirroring
   Step 8b's own established protocol — every number in this study is currently
   single-seed, and would resolve the split-sensitive LR-schedule finding (Q12
   item 5) and confirm whether causal-weighting/curriculum-learning hold under a
   tuned hyperparameter rather than the single default each was tested at.
4. **Multi-seed averaging of the permutation-importance/response-curve/Jackknife
   tests** (Q5, all three now run 2026-08-21) — to confirm the observed
   elevation-dominance is robust and not a single-run artifact.
5. **Zero-shot super-resolution evaluation** — test the proven-but-unexercised
   discretization-convergence property on a trained checkpoint at native 1 km
   resolution.
6. **STEP C (LULC's deeper role in D)** — flagged as deferred since the original
   diffusion design document, still open.
7. **Pre-2000 fire history for a non-zero, empirically-grounded initial condition**
   — flagged as a documented upgrade path since the original v1 design document.
8. **SDG/policy narrative** (Q1) — low-effort, high-value addition before
   submission.

---

## Additional questions worth addressing (not in your list, but likely reviewer questions)

**A. Transductive leakage via the FNO's global receptive field — needs an explicit
statement.** Track A's held-out pixels are excluded from the *data* loss during
training, but the FNO's spectral layers mix information *globally* across the whole
256×256 grid — meaning the network's hidden representations do see test-pixel
*covariates* (inputs) during every forward pass, even though test-pixel *labels*
never enter the loss. This is standard in spatial/transductive ML settings (and
arguably unavoidable for any architecture with global receptive field, including
Biswas's own MaxEnt if it used spatial smoothing), but it should be **stated
explicitly** in the Methods section — a reviewer familiar with FNOs will ask, and an
unstated omission reads worse than a disclosed, well-understood modeling choice.

**B. No calibrated uncertainty quantification.** The model outputs a point estimate
(`σ(u)`), not a calibrated probability with confidence bounds — a real gap for any
model intended to inform actual resource-allocation decisions, where knowing *how
confident* a prediction is matters as much as the prediction itself. Worth flagging
as future work (could pair naturally with the multi-seed testing in Q13.3 — an
ensemble's spread is a cheap first uncertainty estimate).

**C. Responsible-use framing.** A national fire-susceptibility map is the kind of
output that could inform real land-use, insurance, or resource-allocation decisions
— worth a brief, explicit statement that this is a research-stage model (single
seed, incomplete spatial-generalization validation, per Track B1/B2's honest
results) and not yet suitable for direct operational deployment without further
validation — the kind of statement Q1 methodology papers increasingly expect.

**D. Why the terminal aggregate label (`fire_ever`) is fractional, not strictly
binary, at the 256×256 working resolution — worth one clarifying sentence.** Each
coarse cell aggregates many native ~1km pixels; `fire_ever_frac` is the *fraction*
of those sub-pixels that ever burned, and this study binarizes it (`>0`) for AUC
computation. A reviewer may ask whether using the fractional value directly (as a
soft regression target) instead of binarizing was considered — worth a sentence
explaining the binarization choice was for direct comparability with RF/MaxEnt's
binary classification framing, not a default/unconsidered choice.
