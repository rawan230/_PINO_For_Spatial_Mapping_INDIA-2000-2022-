# CDR-PINN: Reaction Term Design
## Forest Fire Susceptibility Mapping — India (2000–2022)

> **Provenance**: extends `CDR_PINN_Diffusion_Design_v2.md` (diffusion term, domain,
> coordinate treatment, architecture) and `CDR_PINN_Advection_Design.md` (advection
> term, boundary-condition re-justification, Gårding's inequality with an explicit
> drift bound). Both carried forward **unchanged**. This document adds the
> **reaction term**, completing the full CDR equation.

---

## 1. Physical Motivation

Diffusion (vegetation/moisture-driven spread) and advection (terrain-driven
directional bias) together still leave one of Biswas et al.'s four predictor groups
untouched: **human-activity / ignition factors** — distance to roads, railways,
waterways. A reaction term is the natural place for a **local, algebraic (no
spatial derivative) ignition-generation mechanism** — something diffusion and
advection, both spatial-transport operators, structurally cannot express.

**Form chosen: Fisher–KPP logistic reaction**, with a learned, multi-source
ignition-rate coefficient:
```
R(x,y,t,u) = ρ(x,y,t) · σ(u) · (1 − σ(u))
ρ(x,y,t)   = softplus(ρ_net([dryness_proxy, NDVI_F1, slope, dist_to_roads]))
```

- `σ(u)(1−σ(u)) = σ'(u)` — a useful identity: the logistic factor is literally the
  derivative of the sigmoid, so `R` grows fastest where `u` is near its
  "undecided" middle range (`u≈0`, `σ≈0.5`) and shrinks toward the extremes (`σ→0`
  or `σ→1`) — the standard Fisher–KPP behavior (population genetics/combustion:
  Fisher, R.A. (1937). *The wave of advance of advantageous genes.* Annals of
  Eugenics, 7(4), 355–369; Kolmogorov, A., Petrovsky, I., & Piskunov, N. (1937).
  *Study of the diffusion equation with growth of the quantity of matter.* Moscow
  Univ. Bull. Math., 1, 1–25. **[cite-confirmed — both foundational, standard]**),
  now driving local risk growth rather than population/allele growth.
- `ρ_net` takes the same kind of bounded, verified-range inputs already used for
  `D_net` — the dryness proxy and `NDVI_F1` (diffusion's own inputs, reused rather
  than a third ad hoc dryness signal, consistent with the diffusion document's own
  stated preference), plus `slope` and `dist_to_roads` (Step 5a/5b) — folding the
  human-activity predictor group into the equation for the first time.
- `softplus` again architecturally guarantees `ρ>0` by construction, same philosophy
  as `c_adv` and `w_raw`.

**Narrative payoff** (worth stating explicitly for the paper): diffusion carries
Biswas et al.'s biophysical/climatic group, advection carries the topographic group,
reaction carries the human-activity group — all four non-trivial predictor groups
from the reference paper now map onto a distinct, physically-motivated mechanism in
one governing equation, not just four groups of features concatenated into a
network's input layer.

---

## 2. The Full CDR Equation

```
∂u/∂t = D(x,y,t)·∇²u  −  v(x,y)·∇u  +  ρ(x,y,t)·σ(u)·(1−σ(u))     in Ω×(0,T]
∂u/∂n = 0                                                            on ∂Ω×(0,T]
u(x,y,0) = 0                                                         in Ω
```

`D`, `∇²`, `v` unchanged from the prior two documents.

---

## 3. Boundary Condition — Unchanged, Trivially

Reaction is a **zeroth-order (algebraic) term** — no spatial derivative of `u`
appears in it. When the weak form is derived (multiply by a test function, integrate
over `Ω`), only the diffusion term produces a boundary integral via integration by
parts (killed by Neumann, as in the diffusion document); advection and reaction
contribute directly as volume integrals with no boundary term generated at all. So
reaction requires **no new boundary argument** — a more elementary situation than
advection's case, which at least needed the uniform-parabolicity argument (§3 of the
advection document) to justify keeping a single condition on the whole boundary.
Neumann, `∂u/∂n=0`, stands unchanged.

---

## 4. Initial Condition — Unchanged, One Clarification

`u(x,y,0)=0` is carried forward unchanged. Compatibility with the Neumann BC
(`∂u₀/∂n=0`) is trivial since `u₀≡0`, exactly as before — this is a **spatial**
derivative compatibility and is unaffected by what the reaction term does.

**Worth noting explicitly** (not a well-posedness issue, but a modeling behavior a
reviewer might ask about): at `t=0`, `u≡0` everywhere makes the diffusion term
(`∇²u=0`) and advection term (`∇u=0`) both vanish, but the reaction term does
**not** — `σ(0)(1−σ(0)) = 0.25`, so `∂u/∂t|_{t=0} = 0.25·ρ(x,y,0) > 0`. The
solution begins growing immediately from the zero baseline, driven purely by the
reaction/ignition mechanism, with diffusion and advection only becoming active once
`u` develops spatial structure. This is physically sensible — risk doesn't start
from nothing and stay nothing; something has to "switch it on" — and it is not a
compatibility problem: the IC only constrains `u(·,0)`, never `∂u/∂t(·,0)`, which is
determined by the equation itself, exactly as in any standard parabolic IVP.

---

## 5. Well-Posedness — Statement

**Claim**: the IBVP in §2 admits a unique **global-in-time** (not just local) weak
solution `u ∈ L²(0,T; H¹(Ω)) ∩ C([0,T]; L²(Ω))`, for the full 266-month horizon.

This is a genuinely stronger claim than a generic semilinear-parabolic existence
result would guarantee by default — global existence for semilinear equations is
not automatic (a badly-behaved nonlinearity, e.g. a cubic reaction term, can blow up
in finite time). It holds here specifically because of a property of the *sigmoid*
construction: **`R` is globally bounded in `u`**, not merely locally Lipschitz.

### 5.1 Two properties of `R(x,y,t,u) = ρ(x,y,t)·σ(u)(1−σ(u))`

**(a) Global boundedness, uniform in `u`.** `σ(u)(1−σ(u)) ∈ (0, 0.25]` for every
`u∈ℝ` (maximized at `u=0`, `σ=0.5`; → 0 as `u→±∞`) — this holds for *any* real
value of `u`, not just a bounded range. `ρ(x,y,t) = softplus(ρ_net(·))` is bounded
exactly as `D` was (§4 item 1 of the diffusion document): `ρ_net` is a finite MLP on
a compact input domain (dryness proxy, `NDVI_F1`, slope, distance-to-roads — all
verified-bounded fields), so by the extreme value theorem `ρ_net`'s output lies in a
compact `[z_min,z_max]`, and `softplus` (strictly increasing, continuous) maps this
to `ρ(x,y,t) ∈ [ρ_min,ρ_max]`, both finite and strictly positive. Therefore:
```
|R(x,y,t,u)| ≤ ρ_max · 0.25 =: F_max < ∞    for ALL u ∈ ℝ, all (x,y,t) ∈ Ω×[0,T]
```
No blow-up is possible at any finite time, **regardless of how large `u` becomes** —
this is a strictly stronger guarantee than a naive polynomial reaction term (e.g.
Allen–Cahn's `u−u³`) would give, since those are only locally bounded and require a
separate a priori bound to rule out blow-up. Here, `R` itself supplies that bound.

**(b) Global Lipschitz continuity in `u`.** Write `R = ρ(x,y,t)·g(u)` with
`g(u):=σ(u)(1−σ(u))=σ'(u)`. Then `g'(u)=σ''(u)=σ(u)(1−σ(u))(1−2σ(u))`. Maximizing
`|σ''(u)|` over `u∈ℝ` (equivalently over `s=σ(u)∈(0,1)`, a standard calculus
exercise: let `s=0.5+x`, maximize `|(0.25−x²)(−2x)|`) gives the exact bound
```
sup_{u∈ℝ} |σ''(u)| = 1/(6√3) ≈ 0.0962
```
so `g` is globally Lipschitz with constant `1/(6√3)`, and therefore
```
|R(x,y,t,u₁) − R(x,y,t,u₂)| ≤ ρ_max·(1/(6√3))·|u₁−u₂| =: L_f·|u₁−u₂|
```
— globally Lipschitz in `u`, uniformly in `(x,y,t)`, with an explicit constant.

### 5.2 Proof — Galerkin energy estimate + Gronwall

This extends the advection document's Galerkin/Gårding framework directly (same
weak form, same bilinear form `B[u,w;t]` for the linear diffusion+advection part),
adding the standard technique for a globally-Lipschitz, globally-bounded reaction
term.

**Weak form**: for `w∈H¹(Ω)`,
```
(∂u/∂t, w) + B[u,w;t] = (R(·,·,t,u), w)
```
Take `w=u`:
```
(1/2) d/dt‖u‖²_{L²(Ω)} + B[u,u;t] = ∫_Ω R(x,y,t,u)·u dx
```
Bound the right-hand side using the global bound from §5.1(a) and Young's inequality:
```
∫_Ω R·u dx ≤ ‖R‖_{L²}‖u‖_{L²} ≤ F_max·|Ω|^{1/2}‖u‖_{L²} ≤ (1/2)‖u‖²_{L²} + (1/2)F_max²|Ω|
```
Combine with the advection document's Gårding's inequality
(`B[u,u;t] ≥ α‖∇u‖²_{L²} − β‖u‖²_{L²}`, `α=D_min/2`, `β=V_max²/(2D_min)`):
```
(1/2) d/dt‖u‖²_{L²} + α‖∇u‖²_{L²} ≤ (β + 1/2)‖u‖²_{L²} + (1/2)F_max²|Ω|
⟹  d/dt‖u‖²_{L²} ≤ (2β+1)‖u‖²_{L²} + F_max²|Ω|
```
**Gronwall's inequality** (with `u(0)=0`):
```
‖u(t)‖²_{L²(Ω)} ≤ F_max²|Ω|·t·e^{(2β+1)t}      for all t ∈ [0,T]
```
— a **finite, explicit, a priori bound** on the solution over the *entire* 266-month
horizon, computable from constants already derived in the prior two documents
(`β` from the advection document) plus `F_max=ρ_max·0.25` from §5.1(a) here. This a
priori bound, together with the same Galerkin-approximation / weak-* compactness
machinery used for the linear case, gives **existence** of a global weak solution
(standard method: energy bound independent of Galerkin truncation dimension ⇒
bounded sequence in `L²(0,T;H¹)` ⇒ weakly convergent subsequence ⇒ limit satisfies
the weak form, using continuity of `R` in `u` to pass to the limit in the nonlinear
term). Reference for this class of argument: Evans, L.C. (2010). *Partial
Differential Equations* (2nd ed.), Ch. 7 (energy methods extend directly to
semilinear equations with Lipschitz nonlinearities); Pazy, A. (1983). *Semigroups of
Linear Operators and Applications to Partial Differential Equations.* Springer.
**[cite-verify — standard reference for this exact class of semilinear
global-existence result via Lipschitz nonlinearity + linear-part coercivity; exact
chapter/theorem number should be confirmed before citing in the paper]**.

**Uniqueness**: let `u₁,u₂` both solve the IBVP, `w:=u₁−u₂`. Subtracting the two
weak forms and taking the test function `w` itself:
```
(1/2) d/dt‖w‖²_{L²} + B[w,w;t] = (R(·,·,t,u₁) − R(·,·,t,u₂), w)
                                ≤ L_f‖w‖_{L²}‖w‖_{L²}      [§5.1(b), Cauchy–Schwarz]
                                = L_f‖w‖²_{L²}
```
Using Gårding (`B[w,w;t] ≥ −β‖w‖²_{L²}`, dropping the non-negative `α‖∇w‖²` term):
```
d/dt‖w‖²_{L²} ≤ (2β + 2L_f)‖w‖²_{L²}
```
`w(0) = u₁(0)−u₂(0) = 0−0 = 0` (both satisfy the same IC), so Gronwall gives
`‖w(t)‖²_{L²} ≤ 0·e^{(2β+2L_f)t} = 0` for all `t∈[0,T]` — hence `u₁≡u₂`. ∎

---

## 6. Updated Decision Table (reaction additions only)

| Component | Definition | Status |
|---|---|---|
| Reaction term | `R(x,y,t,u) = softplus(ρ_net(·))·σ(u)(1−σ(u))`, Fisher–KPP form | **LOCKED** |
| `ρ_net` inputs | dryness proxy, `NDVI_F1`, slope (Step 5a), distance-to-roads (Step 5b) | **LOCKED** |
| Boundary condition | Neumann, unchanged — reaction is zeroth-order, contributes no boundary term | **LOCKED (trivial)** |
| Initial condition | `u₀≡0`, unchanged; `∂u/∂t|_{t=0}≠0` from reaction is expected, not a compatibility issue | **LOCKED, clarified** |
| Well-posedness | Global (not just local) existence/uniqueness, `t∈[0,266]` | **PROVEN** — Gronwall with explicit constants `F_max=ρ_max/4`, `L_f=ρ_max/(6√3)`, extending the advection document's Gårding inequality |
| Novelty framing | diffusion↔biophysical/climatic, advection↔topographic, reaction↔human-activity — all four Biswas et al. predictor groups mapped onto one governing equation | **LOCKED** |

---

## 7. What Still Remains

- **STEP C** (from the original diffusion document, still open): LULC's role, if
  any, in `D` — not resolved by adding advection/reaction, still a separate
  decision.
- **Operator-learning framing** (per-month instance vs. single-trajectory) —
  flagged as open two turns ago, still not decided.
- **Training-loop specifics**: collocation/instance sampling strategy, batch size,
  and — now with three physics-loss terms (diffusion residual, advection, reaction)
  plus data/BC/IC losses — the relative loss weighting between all of them. This is
  a real open design question now that the equation is complete; worth deciding
  before writing the training notebook.
- The full CDR equation (§2 here) is now complete and ready for STEP D (final
  assembled equation statement) — all three terms designed, all boundary/initial
  conditions justified, well-posedness proven end-to-end.
