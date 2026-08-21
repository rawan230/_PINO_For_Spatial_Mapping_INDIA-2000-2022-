# CDR-PINN: Advection Term Design
## Forest Fire Susceptibility Mapping — India (2000–2022)

> **Provenance**: extends `CDR_PINN_Diffusion_Design_v2.md` (locked diffusion term,
> domain, coordinate treatment, network architecture, well-posedness proof for the
> pure-diffusion equation). Everything there — `Ω`, the corrected domain bounds, `t`
> range, `D(x,y,t)` construction (Option B, two-timescale), the exact spherical
> Laplacian, the PINO/FNO operator-learning pivot — is carried forward **unchanged**.
> This document adds the **advection term only**; reaction is deferred to a follow-up
> document. The equation treated here is diffusion + advection, reaction-free.

---

## 1. Physical Motivation

Fire spread accelerates **upslope**, not downslope — convective and radiative
preheating of fuel above an advancing flame front is well established in
fire-behavior science (Rothermel, R.C. (1972). *A mathematical model for predicting
fire spread in wildland fuels.* USDA Forest Service Research Paper INT-115.
**[cite-verify]**, already in this project's reference list). The advection velocity
field is therefore constructed to point in the direction of **increasing elevation**
(upslope), scaled by local steepness:

```
v(x,y) = c_adv · ∇E(x,y)
```

- `E(x,y)` — elevation, from Step 5a's SRTMGL3 (90m) DEM. `∇E` (the elevation
  gradient) is used directly rather than reading a separately-conventioned "aspect"
  raster — Step 5a's Horn's-method pipeline already computes this gradient as an
  intermediate quantity before decomposing it into slope magnitude / aspect angle, so
  using `∇E` directly avoids any ambiguity about whether a given aspect convention
  points upslope or downslope-facing.
- `c_adv = softplus(c_raw)` — a single learnable scalar, architecturally constrained
  positive so the model **cannot** invert the physically-required upslope direction
  during training. Same minimal-parameter, physics-by-construction philosophy as
  `w_raw` in the diffusion document's Option B (§5.2 there) — one new scalar, not a
  network.

**Sign check**: for the pure-advection limit `∂u/∂t = −v·∇u`, characteristics move in
the direction of `v`. With `v = c_adv·∇E` (`c_adv>0`), `u` is transported toward
higher elevation — the physically correct direction.

---

## 2. The Equation (diffusion + advection, reaction deferred)

```
∂u/∂t = D(x,y,t)·∇²u  −  v(x,y)·∇u     in Ω×(0,T],  T=266
∂u/∂n = 0                                on ∂Ω×(0,T]
u(x,y,0) = 0                             in Ω
```

`D(x,y,t)` and `∇²u` (exact spherical Laplacian) are exactly as locked in the
diffusion document — unchanged.

---

## 3. Boundary Condition — Neumann Retained, With Justification

**Decision**: homogeneous Neumann (no-flux), `∂u/∂n = 0` on the whole of `∂Ω`,
unchanged from the diffusion-only design — **not** split into separate inflow/outflow
conditions.

**Why a single condition on the whole boundary remains valid once advection is
added**: for a *purely* hyperbolic transport equation (`D≡0`), only the inflow portion
of the boundary (where `v·n < 0`) can carry a condition — the outflow portion is
determined entirely by the interior solution, and imposing data there would
over-determine the problem. That is not the regime here. `D(x,y,t)` is **uniformly
bounded away from zero** everywhere on `Ω×[0,T]` (proven in the diffusion document,
§4 item 1 and §5.3) — the operator `D∇² − v·∇` is therefore **uniformly parabolic**,
with advection entering only as a bounded lower-order (first-order) perturbation of
the elliptic principal part. Uniformly parabolic operators regularize the entire
boundary regardless of local flow direction, so a single condition on all of `∂Ω` is
both sufficient and standard (Evans, *PDE*, 2010, Ch. 7, general form with a drift
term `b·∇u`) — exactly the same boundary-condition class as the pure-diffusion case,
not a new one. Physically: no fire-susceptibility signal is allowed to enter or leave
across India's mapped border, consistent with why the diffusion-only design chose
no-flux in the first place — that reasoning is unaffected by adding a directional
drift inside the domain.

`∂u/∂n=0` is evaluated with the same exact spherical-metric outward normal already
established in the diffusion document, §2 — unchanged, orthogonal to the interior
equation's form.

---

## 4. Initial Condition — Unchanged

`u(x,y,0) = 0` everywhere, carried forward from the original design. Compatibility
with the boundary condition (`∂u₀/∂n = 0`) is trivially satisfied since `u₀≡0` — this
holds regardless of what the interior operator looks like, so it needs no new
argument.

---

## 5. Well-Posedness — Statement

**Claim**: the initial-boundary value problem in §2, with `D(x,y,t)` as already
constructed (diffusion document §5.2) and `v(x,y) = c_adv·∇E(x,y)` as constructed in
§1 above, admits a unique weak solution `u ∈ L²(0,T; H¹(Ω))` with
`∂u/∂t ∈ L²(0,T; H¹(Ω)′)`.

This extends (does not replace) the diffusion document's Evans Ch. 7 citation: that
document invoked the theorem's *pure-diffusion* special case; here the *general*
form — linear parabolic operators with both a diffusion coefficient and a bounded
first-order drift term — is invoked, which is the same theorem family, proved via the
same Galerkin-method energy estimate, just requiring one additional step (Gårding's
inequality with a drift term, §6 below) that the diffusion-only case didn't need,
since there `D_min>0` alone gave coercivity directly.

### 5.1 Verifying the hypotheses

1. **Uniform parabolicity of the principal part** — `D(x,y,t) ∈ [D_min, D_max]`,
   `0<D_min≤D_max<∞`, for all `(x,y,t)∈Ω×[0,T]`. Already proven (diffusion document
   §5.3); unaffected by adding advection, since this concerns only the second-order
   coefficient.

2. **Boundedness of the drift coefficient `v(x,y)`** — needs `|∇E(x,y)|` bounded on
   `Ω`. Verified directly from Step 5a's own measured data: slope angle (which
   `|∇E|` is proportional to, `|∇E| = tan(slope angle)` in the appropriate physical
   units) ranges **0.00°–77.31°** across India (Step 5a's own measured extremes,
   `Terrain_Outputs` statistics) — bounded strictly away from the 90° singularity
   where a gradient bound would fail. This gives an explicit, data-verified constant:
   ```
   G_max = tan(77.31°) ≈ 4.39   (finite, not asymptotic)
   ```
   `c_adv = softplus(c_raw)` is a fixed finite positive scalar for any given trained
   model (same argument as `w_raw` in the diffusion document, §5.2). Therefore
   ```
   |v(x,y)| = c_adv · |∇E(x,y)| ≤ c_adv · G_max =: V_max < ∞
   ```
   — uniformly bounded, by construction plus verified data, not by assumption.

3. **Domain regularity** — `Ω` (dissolved India state boundary) has no
   degenerate/self-intersecting geometry. Unchanged from the diffusion document, §4
   item 2.

4. **IC/BC compatibility** — `∂u₀/∂n=0` on `∂Ω`, trivially satisfied since `u₀≡0`.
   Unchanged from the diffusion document, §4 item 3 (see §4 above).

5. **Coefficient regularity** — `D` inherits the smoothness of `D_net`∘NDVI as
   before. `v(x,y) = c_adv·∇E(x,y)`: `E` is a real DEM raster — bounded, measurable,
   but not smooth pixel-to-pixel (real terrain has genuine local variation). This is
   exactly the same regularity class already accepted for `D`'s dependence on
   `NDVI_F1` in the diffusion document (§4 item 4: *"sufficient for existence of a
   weak solution in the standard Sobolev-space sense — full classical smoothness...
   is not required at this level of the well-posedness claim"*). Evans Ch. 7's
   general theorem for operators with a drift term requires only bounded, measurable
   coefficients — the same hypothesis class, not a stronger one.

---

## 6. Proof — Gårding's Inequality With an Explicit Drift Bound

This is the one genuinely new piece of mathematical content advection requires (the
diffusion-only case got coercivity directly from `D_min>0`; a drift term needs one
extra absorption step).

**Weak formulation.** Multiply the PDE by a test function `w ∈ H¹(Ω)` and integrate
over `Ω`. Integrating the diffusion term by parts and using `∂u/∂n=0` kills the
boundary term:
```
∫_Ω (∂u/∂t) w dx  =  −∫_Ω D∇u·∇w dx  −  ∫_Ω (v·∇u) w dx
```
Define the associated bilinear form:
```
B[u,w;t]  :=  ∫_Ω D∇u·∇w dx  +  ∫_Ω (v·∇u) w dx
```

**Claim (Gårding's inequality)**: there exist `α>0`, `β≥0` such that
```
B[u,u;t]  ≥  α‖∇u‖²_{L²(Ω)}  −  β‖u‖²_{L²(Ω)}      for all u ∈ H¹(Ω)
```

**Proof.**
```
B[u,u;t] = ∫_Ω D|∇u|² dx  +  ∫_Ω (v·∇u) u dx
         ≥ D_min‖∇u‖²_{L²}  +  ∫_Ω (v·∇u) u dx                     [item 1, §5.1]
```
Bound the drift term with Young's inequality (`|ab| ≤ (ε/2)a² + (1/2ε)b²`), using the
uniform bound `|v(x,y)| ≤ V_max` from item 2:
```
|∫_Ω (v·∇u) u dx|  ≤  ∫_Ω |v||∇u||u| dx
                    ≤  (ε/2)∫_Ω|∇u|² dx  +  (1/2ε)∫_Ω |v|²u² dx
                    ≤  (ε/2)‖∇u‖²_{L²}  +  (V_max²/2ε)‖u‖²_{L²}
```
Choose `ε = D_min` (spends exactly half the diffusion coercivity absorbing the drift
term, leaving half in reserve):
```
B[u,u;t]  ≥  D_min‖∇u‖²_{L²}  −  (D_min/2)‖∇u‖²_{L²}  −  (V_max²/2D_min)‖u‖²_{L²}
          =  (D_min/2)‖∇u‖²_{L²}  −  (V_max²/2D_min)‖u‖²_{L²}
```
This is Gårding's inequality with
```
α = D_min/2 > 0          β = V_max²/(2·D_min) ≥ 0
```
— both **explicit, finite constants derived from this problem's own verified data**
(`D_min` from the diffusion document's NDVI-range compactness argument; `V_max` from
the measured `77.31°` maximum slope above), not asserted to exist abstractly. ∎

**Completing existence/uniqueness**: Gårding's inequality (rather than full
coercivity — the `−β‖u‖²` term is the standard obstruction) is handled by the usual
substitution `u = e^{λt}ũ` for `λ>β`, which converts `B` into a fully coercive form
for the transformed problem (Evans 2010, Ch. 7, §7.1.2). With Gårding's inequality
established, the Galerkin method (finite-dimensional approximating subspaces of
`H¹(Ω)`, energy estimates uniform in the approximation, weak-* compactness to pass to
the limit) gives existence of a weak solution; uniqueness follows from linearity of
the equation plus the same energy estimate applied to the difference of two
hypothetical solutions. This is the identical proof mechanism the diffusion-only case
used, now correctly invoked in its general (drift-inclusive) form. ∎

---

## 7. Updated Decision Table (advection additions only)

| Component | Definition | Status |
|---|---|---|
| Advection velocity | `v(x,y) = softplus(c_raw)·∇E(x,y)` — terrain-driven, upslope-directed by construction | **LOCKED** |
| Boundary condition | Homogeneous Neumann, unchanged — justified as still sufficient under uniform parabolicity (§3) | **LOCKED, re-justified** |
| Initial condition | `u₀≡0`, unchanged | **LOCKED (unchanged)** |
| Well-posedness | Existence/uniqueness of weak solution for diffusion+advection | **PROVEN** — Gårding's inequality with explicit `α=D_min/2`, `β=V_max²/(2D_min)`, both derived from verified data extremes |

---

## 8. What Still Remains

- Reaction term — deferred per the user's request, to be designed next as its own
  document, extending this one (diffusion+advection well-posedness carries forward
  unchanged into that document, exactly as this document carried the diffusion
  document's results forward).
- Operator-learning framing (per-month instance vs. single-trajectory) — flagged as
  open in the prior conversation turn, not yet decided.
- Training-loop specifics (collocation/instance sampling, loss-term weighting between
  data/PDE/BC/IC) — still open, unchanged from the diffusion document's own §7.
