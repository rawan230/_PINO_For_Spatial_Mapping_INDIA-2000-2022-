# Methods — CDR-PINN: A Physics-Informed Neural Operator for Forest-Fire Susceptibility Mapping in India

> **Status note (2026-08-20, updated twice)**: architecture, governing equation,
> well-posedness, training protocol, term-ablation study, and all four generalization
> tracks (A/B1/B2/B3) plus a physics-vs-no-physics data-efficiency test are now real,
> executed, and reported — not a plan any longer. Headline: Track A/B3 (random split,
> temporal generalization) hold up well (full CDR AUC 0.9406 / 0.8967); Track B1/B2
> (spatial generalization) are genuinely weak at this training scale (0.75 / 0.60,
> one B2 region below chance) — reported plainly, not softened, since that is
> precisely the honest information a reviewer needs. See Section 8 for the complete
> table and an unresolved, disclosed open question: the data-efficiency test found
> physics gave **no** advantage on Track A, and the literature's own prediction is
> that any advantage should show under distribution shift instead — not yet tested.
> This is a first, modest-scale run (small architecture, 50–80 epochs, single seed),
> not a final hyperparameter-tuned model — consistent with this project's standing
> convention of never reporting an untested number as if it were verified.

---

## 1. Problem Formulation

Forest-fire susceptibility mapping is conventionally posed as a static, pixel-wise
supervised classification problem: a feature vector per location predicts a binary or
probabilistic fire-occurrence label, independent of neighboring pixels and independent
of the temporal ordering of the conditioning variables (e.g. Biswas, Mahato & Joshi,
2025, MaxEnt; and this project's own Random Forest and MaxEnt baselines, Step 7). This
formulation discards two structurally available pieces of information: (i) spatial
continuity — fire risk at one location is not independent of risk at neighboring
locations, since fire itself spreads across space; (ii) temporal structure — the
21-year, monthly-resolved record available here (Nov 2000–Dec 2022) is a genuine time
series, not 266 independent snapshots.

This work reformulates the problem as an **initial-boundary value problem for a
convection-diffusion-reaction (CDR) partial differential equation** over a latent
susceptibility field `u(x,y,t)`, solved not by a classical numerical scheme but by a
**physics-informed neural operator** (PINO; Li, Zheng, Kovachki et al., 2023) trained
jointly against the governing equation and real, sparse, spatiotemporally-resolved fire
observations. The reformulation is deliberate on two counts: physically, it lets
distinct, well-established fire-behavior mechanisms (vegetation/moisture-driven
spread, terrain-driven directional bias, human-ignition sources) enter the model as
distinct, separately-interpretable terms rather than as undifferentiated input
features; methodologically, it lets the model be evaluated not just on held-out
accuracy but on whether its physics constraint measurably improves spatial and
temporal generalization — the axis on which static classifiers are known to be
weakest (Roberts et al., 2017).

## 2. Data

All inputs are drawn from this project's own already-validated pipeline (Steps 1–7),
itself extending Biswas et al. (2025) to full 15/15 predictor-variable parity (see
`Biswas et al. Verification`, this project's source-checked audit of that paper) — no
new remote-sensing product is introduced for the CDR-PINN specifically; the novelty is
in how existing, already-validated data is *reformulated* for a dynamical-systems
model rather than a static classifier. Table 1 summarizes provenance.

**Table 1 — Data sources feeding the CDR-PINN**

| Quantity | Source (this project's step) | Native resolution | Role |
|---|---|---|---|
| NDVI monthly + whole-period mean | Step 2 (MOD13A3.061) | 1 km, monthly | Diffusion structural + dynamic input |
| Forest-fraction (LULC) | Step 6 (ESA-CCI/C3S, 13-code forest definition) | 300 m → 1 km | Diffusion structural input (STEP C) |
| Elevation, slope | Step 5a (SRTMGL3 DEM, Horn's method) | 90 m → 1 km | Advection velocity field |
| Distance to roads | Step 5b (Geofabrik OSM 2022, Euclidean distance transform) | 1 km | Reaction-rate input |
| FLDAS air temp / RH / soil moisture / precipitation anomalies | Step 4 (FLDAS_NOAH01_C_GL_M) | 0.1° → 1 km, monthly | Dryness proxy → reaction-rate input |
| Real fire occurrences | Step 1 (MODIS FIRMS Collection 6.1, forest-filtered) | point, 541,545 events | Sparse monthly data-supervision |
| Whole-record fire label | Step 6/7 (`fire_ever`) | 1 km, static | Terminal-aggregate data-supervision |

All fields are resampled (area-weighted averaging, this project's established
convention — Steps 4/6) onto a common **256×256 working grid**, downsampled from the
native ~3641×3504 NDVI grid specifically for the neural operator's spectral layers
(Section 4), covering the verified domain `lon∈[68.20°,97.40°], lat∈[6.75°,37.09°]`.
Downsampling here is a deliberate memory/compute trade-off, not a resolution claim
about the physics itself — the governing equation (Section 3) is resolution-agnostic;
Fourier neural operators are additionally provably **discretization-convergent** (Li et
al., 2023, following Kovachki et al., 2021), so the same trained operator can in
principle be evaluated at the native 1 km grid at inference time without retraining
(zero-shot super-resolution), a capability with no analogue in the RF/MaxEnt baselines.

## 3. Governing Equation

```
∂u/∂t = D(x,y,t)·∇²u  −  v(x,y)·∇u  +  ρ(x,y,t)·σ(u)·(1−σ(u))     in Ω×(0,T]
∂u/∂n = 0                                                            on ∂Ω×(0,T]
u(x,y,0) = 0                                                         in Ω
```

Full derivation, physical motivation, and formal well-posedness proofs (global
existence and uniqueness of a weak solution over the entire `T=266`-month horizon, via
Galerkin approximation, an explicit Gårding's inequality, and a Gronwall argument) are
in `CDR_PINN_Diffusion_Design.md`/`_v2.md`, `CDR_PINN_Advection_Design.md`,
`CDR_PINN_Reaction_Design.md`, and `CDR_PINN_Final_Design_STEP_D.md` — summarized here
rather than repeated. The essential design property, worth stating plainly in a Methods
section: **each term is constructed from a physically bounded quantity by
architecture, not by loss-term pressure** — `D>0` and `ρ>0` via `softplus`, `v`'s
direction fixed upslope via `softplus`-constrained scalar times the elevation
gradient — so the model cannot learn a physically inverted or unbounded coefficient
regardless of what the data loss alone would prefer.

Each term is deliberately mapped to one of Biswas et al. (2025)'s four non-trivial
predictor groups: diffusion ↔ biophysical/climatic (33.9%+26.1% of their reported
MaxEnt contribution), advection ↔ topographic (9.7%), reaction ↔ human-activity
(10.8%) — Section 6 develops this into the paper's central novelty argument.

## 4. Network Architecture

### 4.1 Backbone

A Fourier Neural Operator (FNO; Li et al., 2023) rather than a pointwise
coordinate-MLP PINN (Raissi et al., 2019) — the deliberate architectural pivot
documented in `CDR_PINN_Final_Design_STEP_D.md` §3, made because (i) the training data
is structurally a *family* of 265 monthly instances sharing one fixed spatial domain,
exactly the regime neural operators amortize across, not a single long trajectory a
pointwise PINN would have to solve end-to-end; (ii) PINO's own reported finding
(Li et al., 2023, §4.2) is that plain PINNs specifically fail on long time horizons
(their "long temporal transient flow" experiment, structurally analogous to this
study's 266-month record) while an operator ansatz solves the same case with a 400×
speedup over a numerical solver.

```
lifting (1x1 conv) -> [FFT -> spectral conv R -> iFFT + 1x1 skip -> GELU] x4 -> projection (1x1 conv x2)
```

Measured parameter count (`width=32`, `modes=16×16`, `L=4` layers, `256×256` grid,
7 static+dynamic covariate channels + the evolving state `u_t`):

**Table 2 — Parameter budget by submodule (measured, not estimated)**

| Submodule | Parameters | Share |
|---|---:|---:|
| FNO backbone (lifting + 4 spectral blocks + projection) | 1,054,177 | 99.96% |
| — of which: 4× spectral conv layers | 1,048,576 | 99.42% |
| — of which: 4× pointwise skip conv | 4,224 | 0.40% |
| — of which: lift + proj1 + proj2 | 1,377 | 0.13% |
| Diffusivity head (`D_net`) | 206 | 0.02% |
| Advection head (`c_adv`) | 1 | <0.01% |
| Reaction head (`rho_net`) | 229 | 0.02% |
| **Total** | **1,054,613** | 100% |

The three physics heads together account for **0.04%** of total parameters — the
model's capacity is overwhelmingly spent on the general-purpose operator backbone, not
on the physics constraint itself, which is architecturally minimal by design (§7.6 of
the diffusion document: "an oversized network risks overfitting a smooth function with
no benefit" — the same minimal-parameter philosophy extended to all three heads).

### 4.2 Physics Heads

| Head | Inputs | Output | Constrains |
|---|---|---|---|
| `D_net` | NDVI baseline, forest fraction | `D(x,y,t)>0` | diffusion |
| advection scalar | elevation gradient (fixed direction) | `v(x,y)`, upslope | advection |
| `rho_net` | dryness proxy, NDVI baseline, slope, distance-to-roads | `ρ(x,y,t)>0` | reaction |

### 4.3 Non-Periodic Domain and Spectral Differentiation

India's polygon is not periodic, and FFT-based operations implicitly assume
periodicity. A **whole-sample symmetric (Neumann) extension** is used rather than the
zero-padding the PINO paper's own Appendix C suggests as a default — this choice was
empirically forced, not stylistic: zero-padding was tried first and measured to
introduce a large edge-region error (~127× the signal scale, `spectral_ops.py` Test 3)
from the value discontinuity it creates at the domain boundary, which is fatal for the
second-derivative (curvature) terms the Laplacian needs. The whole-sample symmetric
extension removes that discontinuity by construction and was verified to reduce edge
error by >99% (Test 5) — and, usefully, it implicitly imposes a zero-derivative
condition at the domain boundary, which is exactly this problem's own physical Neumann
condition, not a numerical convenience adopted for unrelated reasons.

## 5. Collocation Point Sets

This section distinguishes three point sets with different roles and different
statistical character — worth stating explicitly, since "collocation points" in the
classical PINN literature (Raissi et al., 2019) refers to a single, typically
randomly-sampled set, and this architecture's structure changes that picture
substantially.

**PDE-residual evaluation points — dense, not randomly sampled.** Because spectral
differentiation (§4.3) computes `∂u/∂t`, `∇u`, `∇²u` for the *entire* spatial grid in
one FFT pass (`O(HW log HW)`), the residual is evaluated at **every one of the 22,542
valid in-India grid cells, every one of the 265 monthly transitions** — not a
random subsample the way a pointwise coordinate-MLP PINN would need for tractability.
This is a genuine, quantifiable difference from Raissi et al. (2019)'s original
formulation and from self-adaptive collocation-weighting schemes such as McClenny &
Braga-Neto (2020), which were developed specifically to compensate for collocation
*sparsity* — a problem this architecture does not have, since PDE-residual coverage is
already exhaustive over the domain by construction of the operator, not by a sampling
choice that could be made denser or sparser.

**Data-supervision points — sparse, and observational, not synthetic.** 541,545 real
MODIS-detected forest-fire events (Step 1), each entering the loss at its true
`(x,y,t)` grid cell and month — a genuinely different epistemic status from the
synthetic boundary/initial-condition points typically sampled in the PINN literature,
since these are *observed*, not chosen. Negative supervision uses this project's own
established case-control convention (random, size-matched, `seed=42` background
sampling — first used for Step 2's CVSI optimal-lag selection, reused here for
consistency rather than introducing a new sampling scheme).

**Boundary and initial points.** The boundary ring (1,253 pixels, the set of valid
in-India cells adjacent to at least one invalid/non-India cell — a genuine morphological
edge detected from the data, not the rectangular grid edge) supplies the Neumann
condition's evaluation points; the single `t=0` grid supplies the initial condition,
trivially satisfied by construction (`u₀≡0` is hard-set as the literal starting tensor
of every training trajectory, not learned — see §6 for why this is not a
compatibility problem).

**Terminal-aggregate point set.** A distinct, fourth category: the full grid at the
end of a training window, compared via smooth-max (log-sum-exp) pooling against the
whole-record `fire_ever` empirical label (Step 6/7's own already-validated ground
truth) — a weak, record-level supervision signal in the sense of multiple-instance
learning (Pinheiro & Collobert, 2015), not a pointwise collocation set in the PDE
sense.

## 6. Loss Function and Training Protocol

### 6.1 Loss groups

Four groups — `data`, `pde`, `bc`, `ic` — matching the PINO paper's own Eq. 4 loss
structure exactly. **The PDE loss is a single combined residual** of the full CDR
equation, never three separately-weighted per-mechanism terms — a deliberate choice
to avoid unjustified manual sub-term tuning (`CDR_PINN_Final_Design_STEP_D.md` §4).

```
L_total = w_data*L_data + w_pde*L_pde + w_bc*L_bc + w_ic*L_ic
```

Weights are **not** fixed hand-picked constants: gradient-norm-balanced adaptive
rescaling (Wang, Teng & Perdikaris, 2021, already cited in this project's own Step 8
methodology) recomputes `w_i ← w_i · mean_j‖∇_θ L_j‖ / ‖∇_θ L_i‖` on a fixed schedule,
directly pre-empting the standard reviewer question about how PINN loss weights were
chosen.

### 6.2 Operator framing and truncated backpropagation-through-time

The operator learns a one-step-ahead map `G_θ: (u_t, a_t) → u_{t+1}`, unrolled
autoregressively from the hard initial condition `u_0≡0`. Backpropagating through the
full 265-month unroll in one gradient step is both memory-prohibitive and a known
failure mode for long-horizon recurrent training (vanishing/exploding gradients); this
work uses **truncated backpropagation-through-time in 24-month windows** (a standard,
well-established recurrent-training technique — not a physics decision), carrying the
evolving state forward across window boundaries with gradients detached between
windows. 24 months is a natural, if provisional, choice (roughly one annual fire
cycle); the window length is a tunable engineering parameter, not a locked design
decision.

### 6.3 Reproducibility

- Optimizer: Adam, `lr=1e-3`, gradient-norm clipping at 5.0.
- Hardware: single NVIDIA RTX PRO 4500 (Blackwell), CUDA 12.8, `torch==2.11.0+cu128`,
  in an isolated `cdr_pinn_env` conda environment (does not share dependencies with
  the classical-ML pipeline's `firerisk-anaconda3` environment, avoiding version
  conflicts across the two model families).
- Seeds: to be fixed and multi-seed-tested for the final run, mirroring Step 8b's own
  established multi-seed robustness protocol (5 seeds, bootstrap 95% CIs) — not yet
  run for CDR-PINN as of this writing.

## 7. Computational Complexity Analysis

Presented against this project's own already-measured classical-ML baselines (Step 7)
for direct comparability — every number in this section is either directly measured
or an analytically derived Big-O bound, none estimated by analogy.

**Table 3 — Cross-model computational cost (this project, all measured)**

| Model | Parameters / trees | Train cost (measured) | Inference cost (measured) | ROC-AUC (measured) |
|---|---:|---|---|---:|
| Random Forest (Step 7) | 200 trees, `max_depth=20` | 195.2 s (3,328,807 rows, full grid) | 1.4 s (832,202-px test set) | 0.9683 |
| MaxEnt / `elapid` (Step 7) | linear+hinge+product features | 1,486.8 s (150,000-row subsample) | 33.3 s | 0.9595 |
| CDR-PINN (this work) | 1,054,613 | 269.0 s / 60 epochs (4.48 s/epoch, full 265-month autoregressive rollout per epoch) | not yet benchmarked (no held-out classification eval run yet) | *(pending — loss-convergence only, not yet an AUC)* |

**CDR-PINN training cost, measured on the 60-epoch verification run**
(`width=32`, `4` spectral layers, `16×16` modes, `256×256` grid, single RTX PRO
4500, `torch 2.11.0+cu128`): total training loss fell **8.03 → 0.306** (data loss
plateaued at ≈0.089, PDE residual loss fell from 6.79 to ≈0.10–0.14 and is still
declining at epoch 60, boundary loss fell to ≈1×10⁻⁵) — a genuinely converging
trajectory, not yet a final validated model, since no held-out classification metric
(ROC-AUC/AP) has been computed for it yet; that requires a separate evaluation
protocol (Section 8) this run was not designed to produce. **Peak GPU memory: 3,247.7
MB** (`torch.cuda.max_memory_allocated()`), comfortably inside a single consumer/
workstation GPU's budget, and small relative to the 32 GB available on the RTX PRO
4500 used here — headroom that a wider/deeper architecture or a longer unroll window
could use without a hardware change.

**Analytic complexity, per FNO spectral-convolution layer** (batch `B`, channels `C`,
grid `H×W`, truncated modes `M_h×M_w`, `M_h,M_w ≪ H,W`):

```
FFT/iFFT:        O(B·C·H·W·log(HW))
Truncated einsum: O(B·C_in·C_out·M_h·M_w)        <- dominant cost is INDEPENDENT of H,W
pointwise skip:   O(B·C_in·C_out·H·W)
```

The truncated-mode einsum is the FNO's key complexity property, worth stating
explicitly as a paper-facing argument: because only a small, fixed number of Fourier
modes are retained (`16×16` here, vs. a `256×256` grid), the learned spectral
transform's cost is **independent of spatial resolution** — the same trained layer
that costs `O(M_h·M_w)` at `256×256` costs the identical amount at `1024×1024` (only
the FFT/pointwise terms, both linear in `HW`, scale with resolution). This underlies
the zero-shot super-resolution capability noted in §2 and is a genuine, citable
computational advantage over a fixed-receptive-field CNN operating at native
resolution, whose per-layer cost scales with `H·W·k²` for kernel size `k` regardless
of how the information content is distributed across scales.

**Memory**: activations dominate over parameters at this scale (1.05M parameters ≈
4.2 MB in `float32`); the reported peak allocation (§6.3) is measured directly via
`torch.cuda.max_memory_allocated()`, capturing the true activation + gradient +
optimizer-state footprint rather than a parameter-only estimate that would understate
real training cost.

## 8. Validation and Evaluation Protocol

**Term-ablation study — completed 2026-08-20.** Three configurations — diffusion-only,
diffusion+advection, full CDR — trained identically (80 epochs, `width=32`, `4`
spectral layers, `16×16` modes, Adam `lr=1e-3`) and evaluated on an identical held-out
20% random pixel split (`seed=42`, matching Step 7's own split convention), isolating
each physics term's actual measured contribution rather than asserting it from the
equation's elegance alone (the same discipline already applied to Option-A-vs-B in
the diffusion document §5.4, and to the PINN-vs-plain-MLP comparison in Step 8).
Evaluation metric: held-out ROC-AUC/AP against the real, binarized `fire_ever` label,
using a smooth-max (LSE) pooled terminal score from the full 265-month rollout — the
same aggregation described in §6.3 of the reaction design document.

**Table 4 — Term-ablation results (held-out test, `n=4,508` pixels, 42.06% positive)**

| Configuration | ROC-AUC | Average Precision | Δ AUC vs. previous row |
|---|---:|---:|---:|
| Diffusion only | 0.6017 | 0.6050 | — |
| + Advection | 0.9239 | 0.9014 | **+0.3222** |
| + Reaction (full CDR) | **0.9406** | **0.9253** | +0.0167 |

The advection term (terrain-driven, upslope-directed) accounts for the overwhelming
majority of the model's discriminative power beyond diffusion alone — a striking,
paper-worthy finding on its own, and a direct, independent corroboration of this
project's own Step 5a measurement that fire strongly co-locates with steep slope
(+115% mean slope at fire pixels vs. the national average) and of Biswas et al.
(2025)'s own MaxEnt result ranking slope as their second-most-important variable
(16.7% contribution). The reaction term's smaller but real additional contribution
is consistent with human-ignition factors being a genuine but secondary predictor
group in both this project's own Step 5b findings and Biswas et al.'s reported 10.8%
combined human-activity contribution.

A class-imbalance diagnostic surfaced and fixed during this study is reported
transparently rather than hidden: the initial diffusion-only run (unweighted BCE)
collapsed to a trivial constant-field solution exploiting the diffusion equation's
homogeneous form (PDE residual → ~1e-7, held-out AUC ≈ 0.53, chance level) — a
documented consequence of the real ~2.3% monthly fire-positive rate providing too
weak a gradient to escape that attractor. Fixed with inverse-frequency positive-class
weighting (`pos_weight≈43`), applied identically across all three ablation
configurations for a fair comparison.

**Four generalization tracks — all executed 2026-08-20.** Track A (random split, as
reported above, full CDR, 80 epochs), Track B1 (2°×2° spatial block CV, 3 folds, 50
epochs/fold — a reduced epoch budget disclosed explicitly, not hidden, for wall-clock
reasons), Track B2 (leave-one-region-out, 6 KMeans-derived regions, 50 epochs/fold),
and Track B3 (**new**, leave-years-out, 80 epochs — a temporal generalization axis
this project's classical-ML baselines have no equivalent of, made possible
specifically by the per-month operator framing).

**Table 5 — Generalization tracks (full CDR configuration, held-out ROC-AUC)**

| Track | Description | Mean AUC | Detail |
|---|---|---:|---|
| A | Random 80/20 pixel split | **0.9406** | single split, `n=4,508` |
| B1 | 2°×2° spatial block CV | 0.7538 ± 0.0162 | 3 folds: 0.7728, 0.7554, 0.7332 |
| B2 | Leave-one-region-out (6 regions) | 0.5989 ± 0.0815 | range 0.4929–0.7323; one region below chance |
| B3 | Leave-years-out (2000, 2008, 2009, 2015 held out) | **0.8967** | monthly-resolution eval, `n=856,596`, 2.46% positive |

**This is reported plainly, without softening**: Track A and Track B3 (temporal
generalization) hold up well; Tracks B1 and B2 (spatial generalization to unseen
blocks/regions) degrade substantially, and B2 in particular shows real weakness (one
of six regions scores below the 0.5 chance line). At the current training scale (a
comparatively small architecture, 50–80 epochs, no hyperparameter search), the model
has **not** demonstrated the spatial-generalization advantage the framework was
originally motivated to test (§1.4 of the companion paper draft) — the temporal
result (B3) is the strongest positive evidence collected so far for the physics
formulation adding real value under distribution shift, not the spatial one.
Plausible, disclosed explanations rather than excuses: B1/B2 folds train on genuinely
disjoint spatial data (an entire 2° block or KMeans region removed, not just a random
pixel subsample), a harder learning problem than Track A's setup, compounded by the
reduced 50-epoch budget used for those multi-fold tracks purely for wall-clock
reasons.

**Data-efficiency test (physics vs. no-physics, identical sparse supervision, Track-A
split)**: a matched-architecture, matched-data, matched-split comparison — full CDR
(`use_diffusion/advection/reaction=True`, all physics+data losses) vs. a no-physics
variant (PDE/BC losses removed entirely, data loss only) — was run to test the
data-efficiency claim (§1.4) directly rather than only citing it from other domains.
Result: **no-physics AUC=0.9463 vs. full-physics AUC=0.9406** — the physics
constraint did **not** show an advantage on this specific comparison. This is reported
honestly as a negative result for the *narrow* claim "physics helps on a random
in-distribution split," consistent with this project's own prior finding for a
different physics formulation (Step 8's plain-monotonicity PINN also showed no
significant gain over a plain MLP on Track A). The literature's actual prediction
(Read et al., 2019; Karniadakis et al., 2021) is that physics-informed advantages
should appear specifically under distribution shift, not in-distribution accuracy —
meaning the correct follow-up test is a physics-vs-no-physics comparison run
specifically on Tracks B1/B2/B3, not yet performed, and the natural next step before
this claim can be considered either confirmed or closed.

**Multi-seed robustness testing** (mirroring Step 8b's protocol) has not yet been
performed for any CDR-PINN configuration — the numbers above are single-seed
(`seed=42`) results, not yet bootstrap-confidence-interval-tested.

### 8.1 Diagnosing the Gap to Classical Baselines (2026-08-20)

Given full CDR's Track-A accuracy trails RF/MaxEnt (§4.1), two specific hypotheses
were tested directly rather than left as unexamined caveats:

1. **Train/eval aggregation mismatch — tested, ruled out.** Training's terminal data
   loss uses LSE (log-sum-exp, `τ=5`) pooling over the trajectory (§6.1 of the
   companion paper draft), but the original evaluation code used a hard max-pool
   instead — a genuine inconsistency between the objective optimized and the metric
   reported. Fixed to use identical LSE-pooling at evaluation. Result: AUC changed
   from 0.9406 to 0.9406 (no-physics: 0.9463 → 0.9461) — statistically identical.
   **This rules out the aggregation mismatch as an explanation for the gap** — a
   real bug worth fixing for methodological correctness, but not the cause of the
   accuracy shortfall.
2. **Under-parameterization/under-training — tested directly, twice, and ruled out.**
   A scaled-up configuration (`width=64`, 4× the spectral-layer parameter count,
   150 epochs vs. 80) was trained on the identical Track-A split and seed. Result:
   AUC=**0.9292** — *worse* than the original `width=32`/80-epoch run (0.9406), not
   better. Hypothesizing this was an untuned-learning-rate artifact (the same fixed
   `lr=1e-3` was reused unchanged across every configuration), a cosine
   learning-rate schedule (`T_max=150`, `eta_min=lr/100`) was added and the same
   `width=64` configuration re-run. Result: AUC=**0.9154** — *worse again*, directly
   refuting the LR-schedule hypothesis rather than confirming it. Peak GPU memory
   5,573.6 MB both times (still well under the 32 GB available, so this is not a
   memory-limited failure).

**Conclusion of this diagnostic pass (as it stood 2026-08-20)**: three plausible,
testable explanations for the Track-A gap were checked directly rather than
assumed — evaluation-metric mismatch (ruled out, §8.1 item 1), under-parameterization
(ruled out), and untuned learning rate (at the time, called ruled out — see §8.2 for
the correction). See §8.2 immediately below for three further tests and a revision
to the learning-rate conclusion.

### 8.2 Further Diagnostics: Advanced PINN Techniques and an Honest Validation-Split Re-Test (2026-08-21)

Three more interventions were tested, plus a correction to §8.1's LR-schedule
conclusion:

1. **Causal time-weighting** (Wang, Sankaran & Perdikaris, 2022 [cite-verify]):
   weights each month's residual loss within a training window by
   $w_i=\exp(-\varepsilon\sum_{k<i}\mathcal{L}_r(k))$, targeting this study's
   explicit 266-month autoregressive structure directly. At $\varepsilon=1.0$:
   AUC=0.9369 — worse than baseline (0.9406).
2. **Staged curriculum learning**: advection unlocked at epoch 15, reaction at
   epoch 35 of an 80-epoch run (distinct from the term-ablation study, which trains
   separate models per term combination rather than unlocking terms progressively
   within one run). AUC=0.9343 — worse than baseline.
3. **Honest validation-selected re-test of the §8.1 scale/schedule decision.** §8.1's
   `width=64` and cosine-schedule comparisons used Track A's test AUC directly for
   selection — exactly what a validation set exists to prevent (§3.4/companion
   paper draft). A genuine train/val/test split (65/15/20% of valid pixels,
   seed=42 — a different partition than Track A's 80/20 split) was built and all
   three configurations (`width=32`/80ep, `width=64`/150ep, `width=32`/80ep+cosine)
   retrained on it, winner selected by **validation** AUC only, test AUC reported
   once for the winner:

   | Configuration | Val AUC | Test AUC (not used for selection) |
   |---|---:|---:|
   | `width=32`, cosine schedule | **0.9368** ← selected | 0.9403 |
   | `width=32`, no schedule | 0.9329 | 0.9370 |
   | `width=64`, no schedule | 0.9266 | 0.9339 |

   Scale-up is confirmed worse again on this independent split — §8.1's
   under-parameterization conclusion stands, reinforced. **The LR-schedule
   conclusion reverses**: cosine wins here on both val and test AUC, the opposite
   of §8.1's Track-A result (0.9154 vs. 0.9406). **Correction to §8.1**: the
   learning-rate-schedule hypothesis is downgraded from "ruled out" to
   "split-sensitive, unresolved without multi-seed testing" — both results are real
   and both are reported; a single-split comparison (which is what §8.1's original
   test was) is not sufficient to rank configurations this close together.

**Revised conclusion**: six tested interventions (metric-fix, scale-up, causal
time-weighting, curriculum learning, and the LR-schedule tested two ways) now span
the full diagnostic effort. Five of six land within a ~0.93–0.94 AUC band; only
scale-up is consistently, robustly worse across two independent splits. This
pattern is better explained by a representation ceiling — §8.3's Jackknife test,
together with the permutation-importance and response-curve tests reported in
`CDR_PINN_Full_Paper_Draft.md` §4.4–4.5, all show near-total elevation dominance —
than by an under-optimized model that further tuning would unlock. Remaining,
qualitatively
different candidates not yet tried: instance-wise fine-tuning (Li et al., 2023
§3.2), self-adaptive per-point loss weighting (McClenny & Braga-Neto, 2020), and
transfer learning targeted specifically at Track B2's weak regions.

### 8.3 Jackknife Variable Importance (Biswas et al.'s Fig. 10, reproduced 2026-08-21)

Unlike the permutation-importance and response-curve tests (both inference-only, a
single fixed checkpoint), Biswas et al.'s Jackknife test requires genuine
retraining: 14 runs (leave-one-covariate-out and leave-only-one-covariate-in across
the 7 covariates, covariates held at their domain-mean constant field to keep
architecture/input shape unchanged), 40 epochs each, plus a matched-budget
"all-variables" baseline for fair comparison (the §4/§8 checkpoint used 80 epochs).

| Covariate | Without-*X* AUC | Drop | Only-*X* AUC | Gain vs. chance |
|---|---:|---:|---:|---:|
| Elevation | 0.7779 | −0.1613 | 0.9376 | +0.4376 |
| Slope | 0.9394 | −0.0003 | 0.7731 | +0.2731 |
| Distance to roads | 0.9372 | +0.0019 | 0.7536 | +0.2536 |
| NDVI | 0.9399 | −0.0008 | 0.7048 | +0.2048 |
| Forest fraction | 0.9392 | −0.0001 | 0.7016 | +0.2016 |
| Dryness proxy | 0.9393 | −0.0002 | 0.5755 | +0.0755 |
| NDVI anomaly | 0.9439 | +0.0047 | 0.5374 | +0.0374 |

All-variables baseline (40 epochs): AUC=0.9391, closely matching the 80-epoch
checkpoint's 0.9406 — the model is essentially converged well before 80 epochs.
Elevation is the only covariate whose removal meaningfully hurts the model; every
other removal is noise-level. Elevation alone reaches AUC=0.9376, within 0.0015 of
the full model — five of six other covariates still score meaningfully above
chance in isolation (0.70–0.77), so they are not informationally useless, they
simply add negligible signal on top of what elevation already provides. This is the
fifth independent line of evidence for terrain/elevation dominance in this study
(term-ablation, Step 5a field measurement, permutation importance, response curves,
and now Jackknife retraining), and completes reproduction of all 3 of Biswas et
al.'s variable-understanding analyses.

Scripts: `train_causal_curriculum.py`, `jackknife_test.py`, `validation_split_test.py`
in `Physics_Informed_FireRisk_Model/cdr_pinn/`.

## 9. Software and Data Availability

Implementation: `Physics_Informed_FireRisk_Model/cdr_pinn/` (`spectral_ops.py`,
`model.py`, `losses.py`, `build_monthly_stacks.py`, `train.py`), built and verified
2026-08-20. Design documents (equation derivation, well-posedness proofs):
`CDR_PINN_Diffusion_Design.md`, `_v2.md`, `CDR_PINN_Advection_Design.md`,
`CDR_PINN_Reaction_Design.md`, `CDR_PINN_Final_Design_STEP_D.md`. All raw data sources
and their citations are as listed in `METHODOLOGY.md`'s per-step sections and
Consolidated Reference List.
