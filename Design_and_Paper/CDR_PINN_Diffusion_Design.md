# CDR-PINN: Diffusion Equation Design
## Forest Fire Susceptibility Mapping — India (2000–2022)
## Working Document — Locked Decisions So Far (Diffusion Term Only)

> Archived verbatim from the researcher's own design document, 2026-08-09. This is
> the researcher's own work and is not to be modified, extended, or redesigned by
> Claude — held here as project reference until the researcher advances it further
> (STEP B/C/D, advection, reaction terms).

## 1. Purpose of This Document

This document consolidates every decision made so far in the step-by-step design of
the diffusion term for the CDR-PINN (Convection–Diffusion–Reaction Physics-Informed
Neural Network) forest fire susceptibility model for India. It is intended as a saved
reference point to resume design of the advection and reaction terms, and the full
assembled equation, in future sessions.

Design principle followed throughout: build one physical concept at a time, confirm
with the researcher before adding the next layer, and separate classical/established
physics from the researcher's own novel contribution at every step.

## 2. Base Governing Equation (Classical Form)

The starting point is the classical diffusion equation (Fick's Second Law), a
well-established PDE used across physics, biology, and ecology (Fick, 1855; Crank,
*The Mathematics of Diffusion*, 1975).

```
∂u/∂t = D · ∇²u
```

Expanded 2D form:

```
∂u(x,y,t)/∂t = D(x,y) · [ ∂²u/∂x² + ∂²u/∂y² ]
```

This base form is classical — not novel on its own. The novelty of this work enters
through (a) how D(x,y) is constructed from Indian remote-sensing data, and (b) how
the equation is embedded as a soft physics-informed loss term in a trained neural
network (Raissi, Perdikaris & Karniadakis, 2019, *J. Comput. Phys.*, 378:686–707).

## 3. State Variable: u(x,y,t) and S(x,y,t)

### 3.1 Decision: Unbounded Latent Field

u(x,y,t) is defined as an unbounded latent field, with the final bounded
susceptibility obtained via a sigmoid readout:

```
u(x,y,t) ∈ ℝ          S(x,y,t) = σ(u(x,y,t)) = 1/(1+e⁻ᵘ) ∈ [0,1]
```

Reasoning: the planned reaction term uses a logistic growth form
R(u) = r·u(1−u), which is only well-behaved when applied to a variable already
confined to [0,1]. Keeping u itself unbounded and applying the sigmoid only as a
final readout avoids boundary-clipping gradient problems during training and matches
standard convention in the reaction-diffusion literature (Fisher, 1937, *Annals of
Eugenics* — origin of this diffusion + logistic-reaction pattern; also standard in
PINN literature).

### 3.2 Why u / S Are Dimensionless

Susceptibility is a constructed probability/likelihood index, not a physically
measured quantity (unlike temperature or mass). This mirrors NDVI's own dimensionless
status (NDVI = (NIR−Red)/(NIR+Red), a ratio with units cancelling). Both u and S
therefore carry no physical unit.

## 4. Domain and Coordinate Ranges

Domain Ω = India, represented by the dissolved `India_State_Boundary.shp` (NOT
`India_Country_Boundary.shp`, which has ~60 degenerate slivers — this choice was
already established in the project's data conventions).

| Term | Range | Units |
|---|---|---|
| x (longitude) | [68°, 97.5°] | degrees |
| y (latitude) | [6°, 37.5°] | degrees |
| t (time) | 0–266 (months) | months (Nov 2000 – Dec 2022) |

## 5. Boundary Condition

### 5.1 Decision: Neumann (No-Flux) Everywhere

```
∂u/∂n = 0     for all (x,y) ∈ ∂Ω, all t
```

Physical meaning: susceptibility does not artificially leak into or out of India's
border. Chosen over a Dirichlet (u=0 at boundary) or mixed condition because most of
India's boundary is a mix of true physical edge (coastline) and artificial study-area
cutoff (land borders with neighboring countries), and Neumann is the standard, more
physically honest default in spatial ecology/epidemiology PDE models for this
situation. A mixed condition (Dirichlet on coast, Neumann on land border) was
considered but not chosen, to avoid the added preprocessing complexity of classifying
boundary segments.

## 6. Initial Condition

### 6.1 Decision: Zero/Neutral Initialization

```
u₀(x,y) = u(x,y,0) = 0     for all (x,y) ∈ Ω
```

Chosen over two alternatives that were evaluated:

- NDVI-informed initialization (u₀ = f(NDVI at Nov 2000)) — rejected for now due to
  leakage risk, since NDVI also drives D(x,y); reusing it in the initial condition
  risks the network conflating initialization with physics.
- Fire-history-informed initialization (u₀ = f(pre-2000 fire occurrence)) — the
  architecturally preferred option once available, but MODIS/FIRMS (the project's
  fire data source) only begins Nov 2000, so no directly comparable pre-2000 record
  currently exists in the pipeline. Researcher plans to download and integrate
  pre-2000 fire data later, at which point this option should be revisited as an
  upgrade to the initial condition.

Documented upgrade path for presentation: "Initial condition currently set to zero
(neutral prior); planned extension once pre-2000 historical fire records are
integrated into a modified Step 1 will replace this with fire-history-informed
initialization, providing a direct empirical rather than proxy-based starting state."

### 6.2 Neumann–Initial Condition Compatibility Check

Necessary Condition 3 (compatibility between initial and boundary conditions,
required for parabolic PDE well-posedness per Evans, 2010) requires ∂u₀/∂n = 0 on
∂Ω. Since u₀(x,y) = 0 everywhere, this is trivially satisfied — confirmed, no
further work needed.

## 7. Diffusion Coefficient D(x,y) — Where the Novelty Begins

### 7.1 Why D Cannot Be Constant

A single constant D would claim fire spreads through Western Ghats rainforest at the
same rate as fragmented Rajasthan scrubland — physically false. D must vary
spatially based on local fuel/vegetation structure.

### 7.2 Positivity Requirement (Necessary Condition 1)

Diffusion coefficients must satisfy D(x,y) > 0 everywhere; negative D produces
backward diffusion, which is numerically unstable and amplifies noise without bound
(Evans, *Partial Differential Equations*, 2010, Ch. 7). This must be enforced by
network architecture, not assumed.

### 7.3 Decision: Softplus Transform

```
D(x,y) = softplus(z(x,y)) = ln(1 + e^z)          where z(x,y) = D_net(NDVI_F1(x,y))
```

Softplus was chosen over exponential (risk of unstable gradient blowup) and squared
(can hit exactly zero, flat gradient at z=0) transforms, as the standard, gentlest,
most training-stable option in the PINN/physics-ML literature for enforcing strictly
positive physical coefficients.

### 7.4 Input to D: NDVI Feature Choice

Input: NDVI_F1(x,y) — QA-filtered NDVI mean (reliability ∈ {Good, Marginal} pixels
kept; Snow/Ice/Cloud/Fill dropped), from the project's Step 2 pipeline
(`NDVI_ANALYSIS_WITH_FFP.ipynb`). This is a genuinely quality-controlled signal, not
raw/noisy NDVI.

### 7.5 LULC and Fire Points — Explicitly Excluded From D (For Now)

Two candidate additional inputs to D were considered and rejected for this stage:

- LULC (22-class land cover): dropped for now — no decision yet on encoding (one-hot
  vs. grouped fuel classes); can be added back later as a separate, clean extension.
- Forest fire points: explicitly rejected as an input to D. Fire points are the
  supervision label used in the data loss term (L_data = BCE(S, fire_observed)).
  Feeding the same fire points into D would let the model's diffusion coefficient
  directly encode the answer it is being trained to predict — a data leakage problem
  that would produce deceptively high accuracy without any real physical learning,
  and would undermine the physics-informed novelty claim of the work.
- Slope: not currently available in the project pipeline (no SRTM/terrain data
  confirmed in any of the four working notebooks); dropped from D pending a
  confirmed data source.

### 7.6 D_net Architecture Guidance

Since D_net maps a single scalar (NDVI_F1) to a single scalar (z), a small MLP is
sufficient — e.g., 1 input → hidden layer (8–16 neurons, tanh or ReLU) → 1 output. An
oversized network risks overfitting a smooth 1D function with no benefit.

### 7.7 Range/Units for D

| Term | Range | Units |
|---|---|---|
| NDVI_F1(x,y) | [-0.2, 1.0] (valid NDVI range) | dimensionless |
| z(x,y) = D_net(NDVI_F1) | ℝ (unbounded raw output) | dimensionless |
| D(x,y) = softplus(z) | (0, ∞) strictly positive | km²/month |

## 8. Novelty Summary (For Presentation / Publication Framing)

The base diffusion equation, boundary/initial condition machinery, and well-posedness
theory are all classical (Fick 1855; Crank 1975; Evans 2010; Fisher 1937 for the
bounded/unbounded convention). The researcher's specific novel contributions are:

- D(x,y) as a learned, data-driven function rather than a constant — no existing
  PINN-based wildfire study builds a diffusion coefficient this way. The closest
  related work, Vogiatzoglou et al. (2024), applies PINNs to rate-of-spread physics,
  not susceptibility mapping at country scale.
- D(x,y) derived specifically from Indian remote-sensing data (NDVI) at 1 km
  resolution — no PINN-based fire susceptibility study exists for India. The only
  comparable study, Biswas, S. et al. (2025), *Environ. Sci. Pollut. Res.*,
  32:4856–4878, uses MaxEnt (a non-physics, purely statistical method) at
  approximately 5 km resolution, making this work 25× finer.
- Embedding this equation as a soft physics-informed loss term in a trained neural
  network, following the general PINN framework of Raissi, Perdikaris & Karniadakis
  (2019, *J. Comput. Phys.*, 378:686–707), which has never been applied to wildfire
  susceptibility in India.

## 9. What Remains (Next Steps)

- STEP B: Final well-posedness statement combining domain, boundary condition,
  initial condition, and D>0 together, citing Evans (2010).
- STEP C (deferred): Revisit LULC inclusion in D, if desired, with an explicit
  encoding decision (one-hot vs. grouped fuel classes).
- STEP D: Assemble the complete diffusion equation, all pieces combined into final
  form.
- Future upgrade: once pre-2000 fire data is downloaded and integrated into a
  modified Step 1 notebook, revisit the initial condition (Section 6) to switch from
  zero-init to fire-history-informed init.
- Not yet started: advection equation (wind-driven transport) and reaction equation
  (ignition kinetics) — to follow the same step-by-step, decision-confirmed process
  used here for diffusion.

## 10. Full Consolidated Decision Table

| Component | Definition | Status |
|---|---|---|
| PDE form (general) | ∂u/∂t = D(x,y)·∇²u | LOCKED |
| u(x,y,t) range | ℝ (unbounded latent field) | LOCKED |
| S(x,y,t) = σ(u) | [0,1] (bounded susceptibility, final output) | LOCKED |
| Domain Ω | India, dissolved state boundary (India_State_Boundary.shp) | LOCKED |
| x (longitude) range | [68°, 97.5°] | LOCKED |
| y (latitude) range | [6°, 37.5°] | LOCKED |
| t range | 0–266 months (Nov 2000 – Dec 2022) | LOCKED |
| Boundary condition | Neumann, no-flux: ∂u/∂n = 0 on ∂Ω, everywhere | LOCKED |
| Initial condition | u₀(x,y) = 0 everywhere (zero/neutral init) | LOCKED (upgrade path noted) |
| D(x,y) construction | D(x,y) = softplus(D_net(NDVI_F1(x,y))) | LOCKED |
| Necessary Condition 1 (D>0) | Enforced by softplus construction | CLOSED |
| Necessary Condition 2 (clean domain) | Satisfied by India_State_Boundary.shp (dissolved, no slivers) | CONFIRMED |
| Necessary Condition 3 (IC-BC compatibility) | ∂u₀/∂n = 0 trivially satisfied (u₀=0) | CONFIRMED |
| Well-posedness (existence/uniqueness) | Follows from Evans (2010) parabolic PDE theory once D>0 confirmed | READY — STEP B pending |
| Full assembled diffusion equation | Not yet combined into final form | PENDING — STEP D |
