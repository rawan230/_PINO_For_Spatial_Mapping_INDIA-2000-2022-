# CDR-PINN: Diffusion Equation Design — v2 (Rigor Pass)
## Forest Fire Susceptibility Mapping — India (2000–2022)

> **Provenance**: this document extends `CDR_PINN_Diffusion_Design.md` (the
> researcher's original design, archived unmodified). Everything in that document —
> the base equation form, the state-variable convention, the boundary/initial
> condition choices, the softplus construction, the exclusion of LULC/fire-points/
> slope from D, the novelty framing — is carried forward **unchanged**. This v2 adds:
> (1) domain-bounds reconciliation against verified data, (2) exact coordinate-system
> correctness via the spherical Laplacian (superseding an earlier finite-difference
> approximation once the architecture below made exact derivatives available), (3) a
> formal well-posedness statement (STEP B), (4) a locked time-varying construction
> for D (Option B, §5), and (5) the network architecture itself — single-output,
> collocation-based, resampling-free (§3, locked 2026-08-10).

---

## 1. Domain Reconciliation (Fixed)

**What was found**: the originally-stated domain `x∈[68°,97.5°], y∈[6°,37.5°]` does
not exactly match the actual NDVI grid that `D(x,y)` is built from. Verified directly
against `F1_NDVI_QA_mean.tif` (the literal file `D_net` will read):

```
lon: [68.2000°, 97.4000°]      (stated: [68°, 97.5°])
lat: [6.7500°,  37.0917°]      (stated: [6°,  37.5°])
```

**Why the fix is "correct the stated domain," not "re-derive the grid"**: the
rectangle bounding India's irregular polygon has no physical significance in itself
— what matters for a well-posed PDE is that the *stated* computational domain
exactly matches the domain the network can actually evaluate `D` on. The NDVI grid
above is the output of Step 2's already-executed, already-validated pipeline (GPU
Mann-Kendall sweep, CVSI mutual-information optimization, LISA clustering, etc., all
run against this exact grid). Re-deriving Steps 1–6 to hit `68.0/97.5/6.0/37.5`
precisely would mean re-running months of prior GPU computation for a fractional-
degree shift with no scientific benefit — the "fix" that actually improves rigor is
stating the **true, verified** domain everywhere it's referenced, not chasing round
numbers. I made this call rather than triggering a pipeline-wide re-run; flag it if
you disagree and want the grid itself re-derived instead.

**Corrected domain**:

| Term | Corrected value | Original v1 value |
|---|---|---|
| x (longitude) | **[68.20°, 97.40°]** | [68°, 97.5°] |
| y (latitude) | **[6.75°, 37.09°]** | [6°, 37.5°] |
| t (time) | 0–266 months (Nov 2000 – Dec 2022) | *unchanged — already exact* |

`t` needed no fix: 266 months against the study period Nov 2000–Dec 2022 is exact and
was independently re-verified as the convention used consistently across every other
step in this pipeline.

**Ω itself is still the India polygon, not the rectangle.** Per the original design,
`Ω` = the dissolved `India_State_Boundary.shp` polygon; the corrected lon/lat ranges
above are the bounding rectangle of that polygon *as it actually sits on the data
grid*, used for stating axis limits and building the numerical mesh — not a claim
that the PDE is solved on the full rectangle.

---

## 2. Coordinate-System Correctness — Exact Spherical Laplacian (Fixed, upgraded)

**The issue**: `∂²u/∂x² + ∂²u/∂y²` in the base equation is written with `x,y` in
degrees, but `D` carries units of km²/month. Computed directly for this domain: at
6.75°N, 1° of longitude ≈ 110.4 km; at 37.09°N, 1° of longitude ≈ 88.7 km — a
**24.5% compression** across the domain's latitude range (1° of latitude is
constant at ≈111.19 km throughout, since meridians don't converge that way).

**This section originally proposed a finite-difference metric correction**
(`dx=R·cos(lat)·dlon`, `dy=R·dlat`, applied per grid cell). That fix is superseded
here by an exact version, made possible by the architecture decision in §3: since
`u_net` is a continuous, differentiable function of `(lon,lat,t)` and **autograd
computes exact derivatives**, not finite differences, there is no reason to settle
for the flat-tangent-plane approximation implicit in the original correction — the
exact 2D Laplace–Beltrami operator on a sphere (standard in geophysical fluid
dynamics, e.g. for the horizontal Laplacian in spherical-coordinate PDE solvers) can
be used directly:

```
∇²u = 1/(R²cos²(lat)) · ∂²u/∂lon²  +  1/R² · ∂²u/∂lat²  −  tan(lat)/R² · ∂u/∂lat
```
(angles in radians; `R = 6371 km`). The flat-tangent-plane version implicitly drops
the last term (`−tan(lat)/R² · ∂u/∂lat`), which is small near the equator but not
negligible across this domain's full latitude range (6.75°–37.09°N) — the exact
form removes that approximation error entirely rather than bounding it.

**How this gets computed in practice**: `u_net` takes raw `(lon, lat, t)` (in
degrees/months, matching every other step's convention — no reprojected copies of
NDVI/LST/FLDAS needed) as input. `∂u/∂lon`, `∂u/∂lat`, and the second derivatives are
obtained via `torch.autograd.grad` with `create_graph=True` (the same double-backward
mechanism already used and validated in this project's Step 7 PINN). These raw
degree-space derivatives are then combined via the **known, non-learned, exact**
analytical formula above — a fixed chain-rule composition, not a network or a
learned correction — to produce the physical `∇²u`. The **same exact-metric
treatment applies to the Neumann boundary condition**: `∂u/∂n=0` is evaluated using
the true physical outward normal (computed from the boundary polygon's local
tangent, scaled through the same `cos(lat)` factor), not a naive degree-space normal.

**Alternative not chosen (noted for completeness)**: reprojecting to an equal-area
projection (e.g. Albers Equal-Area centered on India) before solving would avoid
needing any metric correction at derivative-evaluation time, at the cost of losing
direct pixel-correspondence to the rest of the pipeline's EPSG:4326 outputs. Not
needed now — the exact spherical-Laplacian correction achieves the same rigor without
that cost, and composes naturally with §3's collocation-based, interpolation-fed
architecture.

---

## 3. Network Architecture — Single-Output, Resampling-Free (Locked 2026-08-10)

**Decision**: single-output network `u_net: (lon, lat, t) → u`, with every physical
covariate (NDVI, FLDAS variables) entering as an *interpolated value at the query
point*, never pre-resampled onto a shared grid. This directly answers the
researcher's stated requirement — no downsampling the 1km NDVI product, no
fabricating sub-11km detail in FLDAS by upsampling it — while staying the standard,
literature-aligned PINN pattern (Raissi, Perdikaris & Karniadakis, 2019): coordinates
(and covariates) in, solution out, autograd differentiates the network directly.

### 3.1 Why single-output, not multi-output

A multi-output architecture (shared backbone, separate heads predicting e.g. u and
an explicit temperature/moisture field) was considered and **not adopted** — it
doesn't address the resampling concern (that's solved by *how inputs are sampled*,
not by *how many things the network predicts*), and it reopens a materially
different, less standard research question (jointly learning continuous
representations of physical fields the way a neural-field/multi-fidelity-fusion
model would) that hasn't been scoped or justified. The single-output pattern already
supports the stated goal — autograd computing gradients of `u` with respect to any
input, including interpolated covariates — without the added training/validation
risk of a second class of learned field.

### 3.2 Per-source differentiable interpolation (the resampling fix)

Each physical driver is read from **its own native-resolution raster** via a
differentiable interpolator (e.g. `torch.nn.functional.grid_sample`, bilinear,
fully differentiable with respect to the query coordinates) evaluated at the exact
`(lon,lat,t)` query point:

- `NDVI_interp(lon,lat,t)` — read from the 1km NDVI grid (`F1_NDVI_QA_mean.tif` for
  the static baseline, the monthly stack for `NDVI_anomaly`) at full native
  resolution. No downsampling anywhere.
- `FLDAS_interp(lon,lat,t)` — read from FLDAS's own native ~0.1°/11km grid directly,
  **not** the bilinearly-upsampled-to-1km copy Step 4 currently produces for the
  grid-based pipeline. That upsampled product remains valid for Steps 5/6/7's
  raster-based feature table (a separate, already-validated use case) but should
  **not** be the source this network reads from — reading FLDAS at its own native
  resolution here is exactly the "no fabricated detail" requirement.

`D_net` (§5.2) is unchanged in construction — it still maps `(NDVI_F1, NDVI_anomaly)`
to a diffusivity value — only the *source* of those two numbers changes, from a grid
lookup to an interpolated query at the same point `u_net` is being evaluated at.

### 3.3 Collocation-point training (replaces grid coverage)

Consistent with standard PINN practice, `Ω` is no longer covered by a fixed
computational mesh — training samples **collocation points**:
- **Interior points** `(lon,lat,t) ∈ Ω×(0,T]`, sampled for the PDE residual loss
  `r = ∂u_net/∂t − D(lon,lat,t)·∇²u_net` (spherical Laplacian, §2).
- **Boundary points** on `∂Ω×(0,T]`, sampled for the Neumann loss `(∂u_net/∂n)²`.
- **Initial points** on `Ω×{0}`, sampled for the IC loss `(u_net(lon,lat,0))²`
  (trivially near-zero given `u₀≡0`, but still sampled/enforced as a loss term in
  standard PINN training rather than hard-coded).

Sampling density/strategy (uniform random vs. importance-weighted toward fire-dense
regions, batch size, etc.) is an open training-design detail, not yet specified —
flag when ready to define the training loop itself.

### 3.4 Well-posedness is unaffected

The proof in §4 depends only on `D` being uniformly bounded over `Ω×[0,T]` via the
compactness argument (NDVI values bounded ⇒ `D_net`'s output bounded ⇒ `softplus`
output bounded away from 0 and ∞) — nothing about *how* points are sampled from that
compact domain (grid vs. collocation) affects that argument. The proof carries over
unchanged; only the mechanism for *enforcing* the boundary/initial conditions during
training changes (loss terms at sampled points, not values fixed at grid edges).

---

## 4. Well-Posedness Statement (STEP B — Complete)

**Claim**: the initial-boundary value problem
```
∂u/∂t = D(x,y)·∇²u    in Ω×(0,T],   T=266
∂u/∂n = 0              on ∂Ω×(0,T]
u(x,y,0) = 0            in Ω
```
with `D(x,y) = softplus(D_net(NDVI_F1(x,y)))` as constructed in §7 of the original
document, admits a unique weak solution, by the standard theory for linear parabolic
equations with variable, uniformly-bounded coefficients (Evans, *Partial
Differential Equations*, 2010, Ch. 7, Theorem on existence/uniqueness of weak
solutions for uniformly parabolic operators).

**Verifying the required hypotheses against this specific construction:**

1. **Uniform parabolicity (`0 < D_min ≤ D(x,y) ≤ D_max < ∞` for all `(x,y)∈Ω`)**:
   `NDVI_F1(x,y) ∈ [-0.1894, 0.9679]` is a compact set (verified directly from data,
   §1 above; unaffected by §3's switch to interpolated queries — interpolation within
   a bounded raster cannot produce out-of-range values). `D_net` is a finite-depth MLP with continuous activations
   (tanh/ReLU, per §7.6 of the original design) — a continuous function `ℝ→ℝ` for
   any fixed, finite set of trained weights. A continuous function on a compact
   domain attains its minimum and maximum (extreme value theorem), so
   `z(x,y) = D_net(NDVI_F1(x,y)) ∈ [z_min, z_max]` for some finite `z_min, z_max`.
   `softplus` is a strictly increasing continuous bijection `ℝ→(0,∞)`, so
   `D(x,y) = softplus(z(x,y)) ∈ [softplus(z_min), softplus(z_max)]`, both finite and
   strictly positive — **uniform parabolicity holds by construction**, not by
   assumption. (Caveat: this holds for any fixed set of finite trained weights; if
   training were to diverge to unbounded weight magnitudes the bound would degrade
   in the limit — worth a one-line footnote in the paper, not a practical concern for
   a converged model.)
2. **Domain regularity**: `Ω` (the dissolved India state boundary) has no degenerate/
   self-intersecting geometry — confirmed in the original project's own data
   conventions (this is precisely why `India_State_Boundary.shp` is used instead of
   `India_Country_Boundary.shp`, which has ~60 degenerate sliver polygons). A
   dissolved, simplified polygon boundary is regular enough (Lipschitz boundary, in
   the PDE-theory sense) for the standard existence theorem to apply.
3. **Compatibility of IC and BC**: `∂u₀/∂n = 0` on `∂Ω` — trivially satisfied since
   `u₀≡0` (already confirmed in the original document, §6.2, carried forward
   unchanged).
4. **Coefficient regularity**: `D` inherits the smoothness of `D_net` composed with
   NDVI_F1's own spatial smoothness — `D_net` (an MLP with smooth activations) is
   `C^∞` in its input, and NDVI_F1 is a bounded, measurable (though not necessarily
   smooth pixel-to-pixel) field. This is sufficient for existence of a **weak**
   solution in the standard Sobolev-space sense (Evans Ch. 7 requires only bounded
   measurable coefficients for weak existence/uniqueness — full classical
   smoothness of `D` is not required at this level of the well-posedness claim).

**Conclusion**: existence and uniqueness of a weak solution `u ∈ L²(0,T; H¹(Ω))` with
`∂u/∂t ∈ L²(0,T; H¹(Ω)')` follows directly from Evans (2010), Ch. 7, given (1)–(4)
above. This is now a complete, citable well-posedness statement for the diffusion
term as designed — nothing about the equation's form, boundary condition, or
initial condition needed to change to establish it; the original design was already
well-posed, this section just proves it formally with the specific compactness
argument for softplus∘MLP that makes uniform parabolicity automatic rather than
assumed.

---

## 5. Time-Invariance of D — Resolved (Option B, locked)

**Decision (2026-08-09): Option B — static baseline + anomaly modulation.** This is
now the locked construction for `D` and should be treated as consistent going
forward in every later step of this study (advection/reaction terms, the assembled
equation in STEP D, and the eventual training notebook) — not re-opened per-step.

### 5.1 Physical rationale

`D` is split into two timescales, matching an established distinction in fire
behavior science between fuel *load/continuity* (structural, slow-changing) and fuel
*moisture/stress state* (fast, seasonal):

- **Structural component** (unchanged from the original design): `NDVI_F1(x,y)`, the
  whole-period mean — "what kind of vegetation/fuel bed is here."
- **Dynamic component** (new): `NDVI_anomaly(x,y,t)`, Step 2's F3 feature (already
  computed, already on this exact grid, `t` indexed by month) — "how stressed/dry is
  it this particular month, relative to that pixel's own climatology." A
  below-normal (negative) anomaly indicates drought-stressed vegetation.

**Citation for the qualitative direction of this relationship**: the idea that fuel
moisture state modulates fire spread is established fire-behavior science, not a new
claim — Rothermel, R.C. (1972). *A mathematical model for predicting fire spread in
wildland fuels.* USDA Forest Service Research Paper INT-115. **[cite-verify]**. This
work is being cited for the qualitative moisture–spread relationship, adapted here
into a diffusion-coefficient modulation appropriate for a PDE-based susceptibility
framework — **not** as a claim that Rothermel's semi-empirical rate-of-spread model
is being implemented directly. State this distinction explicitly in the paper to
avoid over-claiming.

### 5.2 Exact formulation

```
z(x,y,t) = D_net(NDVI_F1(x,y)) − softplus(w_raw) · NDVI_anomaly(x,y,t)
D(x,y,t) = softplus(z(x,y,t))
```

- `w_raw` is a **single additional learnable scalar** (not a network) — kept minimal
  and consistent with the original design's stated philosophy (§7.6 of the base
  document: "an oversized network risks overfitting a smooth 1D function with no
  benefit"). One new scalar parameter is the smallest possible extension that makes
  `D` time-varying.
- `softplus(w_raw) ≥ 0` is a **hard architectural constraint**, not left to training
  to discover — this guarantees the physically-expected direction (drier-than-normal
  ⇒ higher diffusivity, wetter-than-normal ⇒ lower diffusivity) can't be inverted by
  the optimizer, the same way `D>0` itself is architecturally guaranteed rather than
  hoped for. This mirrors the original document's own design philosophy (physics
  enforced by construction, not by loss-term pressure alone) and is a stronger,
  more defensible claim than Option A's plain re-indexing would have been.
- Sign check: positive `NDVI_anomaly` (greener/wetter than normal) → `z` decreases →
  `D` decreases (lower diffusivity, correct direction). Negative anomaly (drought
  stress) → `z` increases → `D` increases (higher diffusivity, correct direction).

### 5.3 Well-posedness under Option B (extends §4, does not replace it)

The uniform-parabolicity argument in §4 (item 1) extends directly to this
time-dependent form — it only requires re-confirming that `z(x,y,t)` stays within a
compact range for **all** `(x,y,t) ∈ Ω×[0,T]`, not just all `(x,y)`:

- `NDVI_F1(x,y) ∈ [-0.1894, 0.9679]` — compact, unchanged from §4.
- `NDVI_anomaly(x,y,t) = NDVI_monthly(x,y,t) − climatology(x,y,month(t))`. Since both
  terms are themselves bounded within the valid NDVI range `[-0.2, 1.0]`, the anomaly
  is bounded within `[-1.2, 1.2]` in the worst case (in practice the empirical range
  is far tighter, since climatology is itself an average of the same underlying NDVI
  values) — compact, for all `t`.
- `w_raw` is a single fixed scalar for any given trained model, so `softplus(w_raw)`
  is a fixed finite positive constant.
- Therefore `z(x,y,t) = D_net(NDVI_F1(x,y)) − softplus(w_raw)·NDVI_anomaly(x,y,t)` is
  a continuous function of two compact-domain quantities, hence itself bounded within
  a compact interval `[z_min, z_max]` for all `(x,y,t) ∈ Ω×[0,T]` — the same extreme-
  value-theorem argument as §4, just over a domain that now includes `t`.
- `D(x,y,t) = softplus(z(x,y,t)) ∈ [softplus(z_min), softplus(z_max)]`, uniformly
  bounded away from `0` and `∞` over the **entire** space-time domain.

**Conclusion**: uniform parabolicity still holds under Option B, by the identical
mechanism as the time-invariant case — the well-posedness proof in §4 (existence and
uniqueness of a weak solution via Evans 2010, Ch. 7) carries over without
modification to the hypotheses, only to noting that boundedness must (and does) hold
jointly over `(x,y,t)` rather than just `(x,y)`.

### 5.4 Recommended validation (for defensibility, not yet run)

Because this is now a claim ("time-varying, moisture-modulated diffusivity improves
the model") rather than just an architectural choice, it should be validated with the
same statistical rigor already established in this project (Step 7/7b's multi-seed
bootstrap testing), not asserted from the equation alone. Suggested three-rung
ablation once the training notebook exists: **static-D** (original design) vs.
**Option B (locked here)**, compared on both random-split and spatial-generalization
tracks, seed-repeated, bootstrap CI on the AUC/skill-metric delta — mirroring exactly
the evaluation harness Step 7b already built. This turns "we made D time-varying"
into a tested, citable result rather than an assumption.

---

## 6. Updated Decision Table

| Component | Definition | Status |
|---|---|---|
| PDE form | ∂u/∂t = D(x,y,t)·∇²u | LOCKED (v1, unchanged) |
| u/S convention | unbounded u, S=σ(u) | LOCKED (v1, unchanged) |
| Domain Ω | India polygon; bounding rect **corrected** to [68.20°,97.40°]×[6.75°,37.09°] | **FIXED (v2)** |
| t range | 0–266 months | LOCKED (v1, re-verified exact) |
| Boundary condition | Neumann, no-flux — **now with exact spherical-metric evaluation** | **FIXED (v2)** |
| Initial condition | u₀=0 everywhere | LOCKED (v1, unchanged) |
| **Network architecture** | **Single-output `u_net(lon,lat,t)→u`; NDVI/FLDAS entered via per-source differentiable interpolation, never resampled onto a shared grid; collocation-point training** | **LOCKED (v2, 2026-08-10)** |
| D(x,y,t) construction | `softplus(D_net(NDVI_F1(x,y)) − softplus(w_raw)·NDVI_anomaly(x,y,t))` — Option B, both terms now interpolated at the query point rather than grid-read | **LOCKED (v2, 2026-08-09)** |
| Coordinate/unit correctness | degree-space Laplacian was physically inconsistent with D's km² units | **FIXED (v2)** — exact spherical Laplacian via autograd + analytical metric (upgraded from an earlier finite-difference approximation) |
| Well-posedness | existence/uniqueness of weak solution | **PROVEN (v2)** — Evans (2010) Ch. 7, uniform parabolicity shown via compactness argument, extended to the time-dependent Option B form in §5.3, unaffected by the collocation-based architecture (§3.4) |
| Time-invariance of D | resolved — structural (baseline NDVI) + dynamic (anomaly-modulated) two-timescale construction | **LOCKED (v2, Option B)** — apply consistently in all later steps, do not re-open per-step |
| Novelty framing | learned spatial-*and-temporal* D from Indian NDVI at 1km, embedded via Raissi et al. (2019) PINN framework | LOCKED (v1) + **extended (v2)**: two-timescale diffusivity, resampling-free multi-resolution data fusion, and exact spherical-metric autograd are all now part of the novelty claim |

---

## 7. What Still Remains

- STEP C (deferred): LULC inclusion in D, encoding TBD.
- STEP D: full assembled diffusion equation — every piece (domain, BC, IC, D(x,y,t),
  network architecture, coordinate correction, well-posedness) is now locked, so this
  can proceed directly: assemble `∂u/∂t = D(x,y,t)·∇²u` (exact spherical Laplacian,
  §2) with the Option B diffusivity (§5.2), evaluated through the single-output,
  interpolation-fed architecture (§3), into one complete, citable equation statement.
- Training-loop specifics (§3.3): collocation-point sampling density/strategy, batch
  size, loss-term weighting between data/PDE/BC/IC terms — open, to be defined
  alongside STEP D or the training notebook itself.
- Advection and reaction terms — not yet started, same one-piece-at-a-time process.
  When these are designed, keep the Option B two-timescale pattern in mind for
  consistency (e.g. if a reaction/ignition term also needs a moisture-driving input,
  reuse `NDVI_anomaly` rather than introducing a third ad hoc dryness signal), and
  keep them single-output/interpolation-fed, consistent with §3.
