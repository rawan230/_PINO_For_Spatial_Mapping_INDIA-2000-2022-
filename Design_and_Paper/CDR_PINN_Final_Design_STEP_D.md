# CDR-PINN: Final Assembled Design (STEP D)
## Forest Fire Susceptibility Mapping — India (2000–2022)

> **Provenance**: consolidates `CDR_PINN_Diffusion_Design.md` (v1),
> `CDR_PINN_Diffusion_Design_v2.md` (rigor pass), `CDR_PINN_Advection_Design.md`, and
> `CDR_PINN_Reaction_Design.md` — all four carried forward **unchanged**. This
> document closes the three items left open at the end of the reaction document's
> §7: STEP C (LULC in `D`), the operator-learning framing, and loss-term weighting —
> implementing the recommendations given and agreed on in this conversation. With
> this document, the CDR equation, its architecture, and its training methodology
> are all fully specified.

---

## 1. The Complete Equation

```
∂u/∂t = D(x,y,t)·∇²u  −  v(x,y)·∇u  +  ρ(x,y,t)·σ(u)·(1−σ(u))     in Ω×(0,T],  T=266
∂u/∂n = 0                                                            on ∂Ω×(0,T]
u(x,y,0) = 0                                                         in Ω
```

with:
```
D(x,y,t) = softplus( D_net([NDVI_F1(x,y), forest_frac(x,y)])
                      − softplus(w_raw)·NDVI_anomaly(x,y,t) )        [§2 below — STEP C resolved]
v(x,y)   = softplus(c_raw) · ∇E(x,y)
ρ(x,y,t) = softplus( ρ_net([dryness_proxy, NDVI_F1, slope, dist_to_roads]) )
```

`Ω`, its corrected bounds, the exact spherical Laplacian, and every well-posedness
argument (diffusion §4/§5.3, advection §5–6, reaction §5) carry forward unchanged —
this document does not reopen any of them except where STEP C's extension is noted
below.

---

## 2. STEP C, Resolved — LULC's Role in `D`

**Decision**: fold `forest_frac(x,y)` (Step 6's reconciled 13-code LULC
forest-fraction feature) into `D_net`'s input alongside `NDVI_F1`, rather than as a
separate multiplicative gate on `D`.

**Why not a multiplicative gate**: `D_total = D(x,y,t)·forest_frac(x,y)` would let
`D→0` wherever `forest_frac=0` (a real, frequently-occurring value), breaking the
uniform-parabolicity assumption every well-posedness proof in this design series
depends on — it would need an artificial floor constant patched in ad hoc. Folding
`forest_frac` into `D_net`'s input instead needs no new machinery at all.

**Well-posedness extension (trivial)**: `forest_frac(x,y) ∈ [0,1]` by construction
(a fractional cover value). `NDVI_F1(x,y) ∈ [-0.1894, 0.9679]` (verified, diffusion
document §4). The joint input domain `[-0.1894,0.9679]×[0,1]` is compact (product of
two compact intervals). `D_net`, a finite MLP with continuous activations, is
continuous on this compact domain, so by the extreme value theorem its output lies
in some finite `[z_min,z_max]` — **the identical argument already used for the
1D case**, extended to 2D with no new proof technique required. `D(x,y,t)` remains
bounded away from `0` and `∞` everywhere, exactly as before.

**Implementation note**: use a single recent snapshot (`forest_frac_recent`, Step
6's 2020 estimate) as a static input rather than interpolating across the three
available snapshot years (2001/2020/2022) — the measured drift (10.2%→10.5%→10.7%
national mean) is small enough that this is a defensible simplification, stated
explicitly rather than silently assumed.

**Reviewer-facing justification**: `forest_frac_recent/current/baseline` are Step
7's own **top-3 Gini-importance features** — this isn't an untested addition, it's
promoting the variable this project's own results already flagged as most
informative into the physics term that most naturally fits it (fuel
type/continuity, which raw NDVI greenness alone conflates across genuinely
different land-cover types).

---

## 3. Operator-Learning Framing

**Decision**: per-month, one-step-ahead operator structure —
`G_θ: (u_t, a_t) → u_{t+1}` — with a **hybrid data-supervision scheme** that is
honest about what ground truth actually exists at monthly resolution versus only at
the aggregate level.

### 3.1 Why not pure per-month supervision, and why not pure single-trajectory

A literal per-month framing would need a dense monthly ground-truth susceptibility
field, which doesn't exist — only sparse monthly fire points (Step 1) and one
validated **terminal aggregate label** (`fire_ever`, Step 6/7's pixel table) exist.
A pure single-trajectory framing, on the other hand, forfeits the genuine
operator-learning benefit (amortization across real instances, zero-shot resolution
generalization) that motivated the FNO/PINO pivot in the first place. The hybrid
below keeps the per-month *operator structure* (for the architectural reasons
already established) while matching *supervision* honestly to what data actually
exists.

### 3.2 Data supervision — two components

**(a) Sparse monthly signal.** Wherever a real Step 1 fire point exists at
`(x,y,t)`, it supplies a positive label at that exact pixel-month. Negative
samples are drawn using the **same case-control pattern already established in
Step 2's CVSI optimal-lag selection** — a random, size-matched sample of
never-burned background pixel-months (`RandomState(seed=42)`, consistent with the
rest of this project's convention, not a new sampling scheme):
```
L_data,monthly = BCE( σ(u(x,y,t)), monthly_fire_indicator(x,y,t) )
```
evaluated only over the sparse set of sampled `(x,y,t)` triples.

**(b) Terminal-aggregate anchor.** A smooth-max (log-sum-exp) pooling of the
trajectory is compared against the already-validated `fire_ever` label:
```
L_data,terminal = BCE( LSE_τ[σ(u(x,y,·))], fire_ever(x,y) )
LSE_τ[f(t)] = (1/τ)·log( (1/T)·Σ_t exp(τ·f(t)) )
```
`LSE_τ` is a smooth approximation to `max_t` (a hard max would be non-smooth for
backprop); as `τ→∞` it approaches the true max, matching the semantics of
`fire_ever` itself (a whole-record "did this pixel ever burn" label — the natural
aggregation is peak risk over the record, not average risk). This kind of smooth
pooling under a weak, image/record-level label is standard in weakly-supervised /
multiple-instance learning (e.g. Pinheiro & Collobert (2015), *From image-level to
pixel-level labeling with convolutional networks*, CVPR. **[cite-verify — recalled
with reasonable confidence, confirm before citing]**) — our situation is
structurally the same problem (weak, record-level label; want a pixel-and-time
resolved field).

`L_data := L_data,monthly + L_data,terminal` — combined as **one** data-loss group
(no separate top-level weight between the two; see §4).

### 3.3 New validation axis this enables

Leave-years-out cross-validation (train on a subset of years' months, test on
held-out years) — genuinely new, made possible only by the per-month operator
structure, and a direct temporal analogue of the spatial-block/leave-region-out
tracks Step 8 already established. See §5.

---

## 4. Loss-Term Weighting

**Correction to the originally-posed framing**: four weight groups, not six —
`data`, `pde`, `bc`, `ic` — where `pde` is a **single combined residual** of the
full CDR equation, not three separately-weighted diffusion/advection/reaction
terms. This matches the PINO paper's own loss formula exactly (Eq. 4: one
`L_pde`, one boundary weight `α`, one initial-condition weight `β`) — splitting the
equation's internal terms into separately-tuned weights would read as
under-justified knob-turning to a reviewer; one combined residual is the standard,
defensible unit.

```
L_total = w_data·L_data  +  w_pde·L_pde  +  w_bc·L_bc  +  w_ic·L_ic

L_pde = ‖∂u/∂t − D∇²u + v·∇u − R(u)‖²   (spectral evaluation, §2 of the advection doc's derivative method)
L_bc  = ‖∂u/∂n‖²  on ∂Ω  (exact spherical-metric normal, unchanged throughout)
L_ic  = ‖u(·,·,0)‖²
```

**Weights are not fixed hand-picked constants** — adaptive, gradient-norm-balanced
rescaling (Wang, Teng & Perdikaris (2021), *Understanding and mitigating gradient
flow pathologies in physics-informed neural networks*, SIAM J. Sci. Comput.,
43(5):A3055–A3081 — already cited in this project's own Step 8 methodology, reused
here rather than introduced fresh):
```
w_i  ←  w_i · ( mean_j ‖∇_θ L_j‖ ) / ‖∇_θ L_i‖         recomputed every epoch
```
This directly preempts the most predictable reviewer question about any PINN/PINO
loss ("how were these weights chosen, and how sensitive are the results to them?")
with "adaptive, literature-standard scheme," rather than "tuned by trial and
error."

---

## 5. Validation Plan — Term-Ablation, Consistent With Existing Rigor

Three configurations, trained and evaluated identically:

| Configuration | Terms active |
|---|---|
| D-only | diffusion |
| D+A | diffusion + advection |
| Full CDR | diffusion + advection + reaction |

Evaluated on four tracks — the first three already established in Step 8, the
fourth new (enabled by §3):

- **Track A** — random split (parity check against existing baselines)
- **Track B1** — 2°×2° spatial block CV
- **Track B2** — leave-one-region-out
- **Track B3 (new)** — leave-years-out, temporal generalization

Bootstrap 95% CIs on pairwise AUC/skill-metric deltas between configurations,
mirroring Step 8b's existing multi-seed methodology exactly (5 seeds for A/B1/B3, 3
for B2, 10,000 resamples, percentile method). This turns "we added three physics
terms" into a measured, citable result — consistent with the discipline already
applied to Option-A-vs-B in the diffusion document (§5.4) and to the
PINN-vs-plain-MLP comparison in Step 8 itself.

---

## 6. Master Decision Table (consolidated across all four documents)

| Component | Definition | Status |
|---|---|---|
| Domain `Ω` | India polygon; bounding rect `[68.20°,97.40°]×[6.75°,37.09°]` | LOCKED |
| `t` range | 0–266 months (Nov 2000–Dec 2022) | LOCKED |
| Laplacian | Exact spherical `∇²`, autograd + analytical metric | LOCKED |
| Network architecture | FNO/PINO backbone, padded+masked FFT (Fourier continuation) for the non-periodic domain, downsampled working grid (~256×256) | LOCKED |
| `D(x,y,t)` | `softplus(D_net([NDVI_F1, forest_frac]) − softplus(w_raw)·NDVI_anomaly)` | LOCKED (STEP C resolved here) |
| `v(x,y)` | `softplus(c_raw)·∇E(x,y)`, terrain-driven, upslope | LOCKED |
| `ρ(x,y,t)` | `softplus(ρ_net([dryness, NDVI_F1, slope, dist_to_roads]))` | LOCKED |
| Reaction form | Fisher–KPP, `ρ·σ(u)(1−σ(u))` | LOCKED |
| Boundary condition | Homogeneous Neumann, whole `∂Ω` — valid under uniform parabolicity even with drift; reaction contributes no boundary term | LOCKED |
| Initial condition | `u₀≡0`; nonzero `∂u/∂t|_{t=0}` from reaction is expected, not a compatibility issue | LOCKED |
| Well-posedness | Global existence/uniqueness, full `T=266` horizon — Galerkin + Gårding (diffusion+advection) + Gronwall (reaction), explicit constants throughout | PROVEN |
| Operator framing | Per-month one-step-ahead `G_θ`, hybrid sparse-monthly + terminal-aggregate supervision | LOCKED (this document) |
| Loss weighting | 4 groups (data/pde/bc/ic), single combined PDE residual, adaptive gradient-norm balancing | LOCKED (this document) |
| Validation | Term-ablation (D / D+A / full CDR) × 4 generalization tracks (A/B1/B2/B3-new), bootstrap CIs | LOCKED (this document) |

---

## 7. What Remains

Design is now complete end-to-end — equation, BC, IC, well-posedness proof,
architecture, operator framing, loss function, and validation plan. What's left is
implementation:

- The actual FNO/PINO training notebook (Step 8's successor) — network dimensions,
  optimizer schedule, epoch budget, `τ` (LSE pooling temperature), gradient-norm
  rebalancing interval — none of these are physics decisions, all engineering
  defaults to propose and confirm when the notebook is scoped.
- Still waiting on the user's own pending action: the Jan/Feb burned-area download,
  unrelated to this design track but blocking a separate re-run.
