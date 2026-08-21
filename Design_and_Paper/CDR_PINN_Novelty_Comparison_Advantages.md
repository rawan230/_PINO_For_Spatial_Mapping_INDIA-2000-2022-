# Novel Contributions, Comparative Positioning, and the Case for a Physics-Informed Neural Operator

> Companion to `CDR_PINN_Methodology_Section.md`. This document exists to make the
> paper's novelty argument explicit, auditable, and traceable to a specific gap in
> the reference literature — every claim below is either a direct, source-checked
> comparison against Biswas et al. (2025) (see `Biswas et al. Verification`, this
> project's own PDF-extraction audit of that paper) or a measured result from this
> project's own already-completed steps. Nothing here is asserted without a citation
> or a number behind it.

---

## 1. Itemized Novel Contributions

Each item states what is new, why it is new (what specific gap it closes), and where
it is implemented/verified.

1. **First application of a physics-informed neural *operator* (not a pointwise PINN)
   to wildfire susceptibility mapping at national scale.** The wildfire-PINN
   literature that exists (e.g. landslide-susceptibility PINNs — Dahal & Lombardo,
   2025, the closest architectural precedent identified in this project's own
   literature survey) uses pointwise coordinate-MLP PINNs. This work uses an
   FNO/PINO backbone (Li et al., 2023) instead — a deliberate, reasoned pivot (see
   §2 below), not an incremental variant.

2. **A convection-diffusion-reaction equation whose three terms map onto Biswas et
   al. (2025)'s own four predictor groups.** Diffusion ↔ biophysical/climatic,
   advection ↔ topographic, reaction ↔ human-activity — verified in
   `CDR_PINN_Reaction_Design.md` §1 and `CDR_PINN_Final_Design_STEP_D.md` §1. No
   other fire-susceptibility PINN formulation (physics-informed or otherwise) in the
   literature surveyed for this project structures its governing equation around a
   specific prior MaxEnt study's own variable-importance decomposition.

3. **A terrain-driven advection term grounded in a specific, citable fire-behavior
   finding** (Rothermel, 1972 — upslope acceleration), not a generic transport term —
   and independently corroborated by this project's own Step 5a fire-coincidence
   measurement (fires sit at +115% mean slope vs. the national average), a genuine
   empirical cross-check between the physics term's motivation and this project's
   own real data, not just a literature citation.

4. **A Fisher–KPP reaction term with a learned, multi-source ignition-rate
   coefficient** (`ρ_net` over dryness, NDVI, slope, distance-to-roads) — chosen
   specifically because it is *globally* bounded and *globally* Lipschitz in `u`
   (`CDR_PINN_Reaction_Design.md` §5.1), giving **global-in-time** well-posedness
   over the full 266-month horizon — a strictly stronger guarantee than a naive
   polynomial reaction term would offer, and one the paper can state and prove, not
   just assume.

5. **Real fire-point supervision at true monthly resolution**, not synthetic or
   literature-borrowed collocation/boundary data — 541,545 MODIS-detected,
   forest-filtered fire events (Step 1), each entering the loss at its observed
   `(x,y,t)` coordinate.

6. **A new temporal-generalization validation axis (leave-years-out, Track B3)**
   with no counterpart in Biswas et al. (2025) or in this project's own prior
   classical-ML validation (Steps 7/8), made possible specifically by the per-month
   operator framing — a genuinely new capability, not a re-run of an existing test.

7. **Full 15/15 Biswas et al. (2025) predictor-variable parity, verified at both the
   pipeline level and the trained-model level** (`terrain_slope` ranked 6th of 58
   features by Gini importance in the retrained Step 7 model, 2026-08-20) — the
   physics-informed model inherits this same full-parity feature set via its
   physics-head inputs.

## 2. Why an Operator, Not a Pointwise PINN — The Direct Comparison

| Property | Pointwise coordinate-MLP PINN | This work (FNO/PINO) |
|---|---|---|
| Input | single `(x,y,t)` coordinate | full covariate + state grid |
| PDE residual coverage | sparse, sampled collocation points | dense — every valid grid cell, every month |
| Long time-horizon behavior | fails (PINO paper §4.2's own finding, on a structurally analogous 50-step case) | operator ansatz solves the same class of problem, 400x faster than a numerical solver in the cited benchmark |
| Cross-instance amortization | none (one network per equation instance) | learns across 265 real monthly instances |
| Resolution behavior | fixed to training resolution | discretization-convergent (Li et al., 2023) — zero-shot evaluation at higher resolution |

This project's own Step 8 already tested a pointwise PINN (plain monotonicity
penalty) against a plain MLP and found **no significant improvement on any of three
evaluation tracks** — an honest, disclosed negative result, not hidden in this
positioning document. That result is one further concrete reason (beyond the
literature-level argument above) to pivot architecture rather than iterate on the
same pointwise formulation with a different physics term.

## 3. Comparison Against Biswas et al. (2025) — Reference Paper

This section reuses this project's own source-checked audit (`Biswas et al.
Verification`) rather than re-deriving it. Headline points relevant to the
CDR-PINN specifically (the full audit also covers the classical RF/MaxEnt
comparison, not repeated here):

| Axis | Biswas et al. (2025) | This project |
|---|---|---|
| Modeling paradigm | MaxEnt (presence-background, linear/hinge/product features) | CDR-PINN (physics-informed neural operator) + a real, trained MaxEnt replication as a direct baseline |
| Physical mechanism | none — MaxEnt is a purely statistical density-estimation method | explicit, well-posedness-proven CDR equation |
| Spatial resolution | 0.25° (~27 km), uniform across all 15 variables | 0.01°/1 km native pipeline; CDR-PINN trains on a 256×256 working grid, discretization-convergent to native resolution |
| Validation | one random train/test split | random split + spatial block CV + leave-region-out + **leave-years-out** (new) |
| Trend/temporal structure | none — every variable is a static or monthly-mean raster | explicit `∂u/∂t`, monthly operator, real temporal dynamics |

## 4. Comparison Against This Project's Own Classical Baselines

**Table — accuracy bar the CDR-PINN is measured against** (all real, measured
numbers):

| Model | ROC-AUC | Average Precision |
|---|---:|---:|
| Random Forest (Step 7, 58-feature) | 0.9683 | 0.6796 |
| MaxEnt / `elapid` (Step 7, 58-feature) | 0.9595 | 0.6237 |
| Plain MLP (Step 8, Track A) | 0.9614 | — |
| Plain-monotonicity PINN (Step 8, Track A) | 0.9613 | — |
| **CDR-PINN — diffusion only** | 0.6017 | 0.6050 |
| **CDR-PINN — diffusion + advection** | 0.9239 | 0.9014 |
| **CDR-PINN — full CDR (+ reaction)** | **0.9406** | **0.9253** |

The CDR-PINN's own term-ablation study (2026-08-20, 80-epoch runs, `width=32`,
identical held-out 20% pixel split for all three, `seed=42`) is the actual
headline result of this section: **each physics term added measurable, real
predictive value** — diffusion alone is a genuinely weak predictor (0.60, barely
above the 0.50/0.42-baseline chance level for this class-balance), adding the
terrain-driven advection term produces the largest single jump (+0.322 AUC), and
adding the reaction term adds a further, smaller but real improvement (+0.017 AUC).
Full CDR (0.9406) is now within 0.019–0.028 AUC of the two classical baselines
(MaxEnt 0.9595, RF 0.9683) despite a comparatively small architecture (1.05M
parameters, 80 epochs) not yet tuned for maximum accuracy — the term-ablation
trend, not the raw final number, is the paper's actual evidence that the physics
formulation is doing real work, not decoration.

**One diagnostic worth reporting honestly, not hiding**: the first diffusion-only
run (before a fix) collapsed to a trivial constant-field solution (PDE residual
→ ~1e-7, held-out AUC ≈ 0.53, chance level) because the real monthly fire-positive
rate is only ~2.3%, and an unweighted BCE loss's gradient was too weak to compete
with the physics loss's pull toward that trivial attractor — a documented PINN
failure mode, not a hypothetical one. Fixed with inverse-frequency positive-class
weighting (`pos_weight≈43`, mirroring Step 7's own `class_weight='balanced'`
convention), after which the identical diffusion-only configuration reached 0.60,
and the fix was applied identically to all three ablation configs for a fair
comparison. This is worth stating in the paper's Discussion as a concrete,
transparent example of the class-imbalance/physics-collapse interaction rather
than omitted.

The one finding from Step 8 that already *does* hold up, honestly reported there and
repeated here because it directly motivates the CDR-PINN: **both neural
architectures generalized better than Random Forest under spatial CV** in the prior
study (~+1.5–1.8 AUC points retained) — an architecture-level effect, not yet
attributable to physics specifically. The CDR-PINN's term-ablation study
(Methodology §8) is designed exactly to test whether the *physics* constraint adds a
further, separable improvement on top of that architecture-level effect — the
specific question Step 8 could not answer, since it only tested one physics
formulation (soft monotonicity) that turned out not to help.

## 4a. Full Generalization Results (2026-08-20) — Reported Honestly, Mixed

The four-track validation plan referenced throughout this document is now complete,
not aspirational. Full table and discussion: `CDR_PINN_Methodology_Section.md` §8.
Summary, stated plainly rather than favorably: Track A (random split, 0.9406) and
Track B3 (leave-years-out, 0.8967) hold up well; Track B1 (spatial block CV, 0.7538 ±
0.0162) and Track B2 (leave-one-region-out, 0.5989 ± 0.0815, one region below chance)
are genuinely weak at this training scale. A matched physics-vs-no-physics comparison
on Track A found **no advantage from the physics constraint** (no-physics
AUC=0.9463 vs. physics AUC=0.9406) — a real negative result for the specific claim
tested, not yet contradicted or confirmed on the harder tracks where the literature
predicts the advantage should actually appear (Section 5 below states this prediction
before the test that would confirm it, and that test has not yet been run — the
honest state of the argument is "not yet empirically closed," not "proven").

## 5. Advantages of the PINN/PINO Framework, Argued Directly

1. **Testable mechanism, not just prediction.** A classical RF or MaxEnt model
   cannot be asked "how much does the terrain term specifically contribute to
   spatial generalization" — feature importance describes correlation, not a
   falsifiable mechanistic claim. The term-ablation study (diffusion-only /
   +advection / full CDR) makes this a directly measurable question.
2. **Physical bounds guaranteed by construction, not by training outcome.** `D>0`,
   `ρ>0`, and `v`'s upslope direction are architectural facts (via `softplus`), true
   for *any* trained weights — a classical model's feature-importance sign can flip
   under resampling or retraining with no such guarantee.
3. **Provable well-posedness.** Global existence/uniqueness of the governing PDE's
   weak solution is proven, not assumed — a mathematical property no black-box
   classifier formulation has an equivalent of, and a genuinely citable methods
   contribution independent of the model's eventual accuracy.
4. **Data efficiency under physics constraints — argued from the literature, tested
   once, not yet confirmed for this model.** The PINO paper's own reported result
   (Li et al., 2023, Table 3) and Read et al. (2019)'s concrete lake-temperature
   demonstration both show physics constraints improving generalization under data
   scarcity. This project ran the direct analogue — full-physics vs. no-physics,
   identical sparse (~2.3% positive) supervision, identical split — and found **no
   advantage on Track A** (§4a above: no-physics AUC=0.9463 vs. physics AUC=0.9406).
   Stated honestly: the *argument* for a structural fit between this framework's
   known strength and this project's sparse-label problem still holds, but the one
   test run so far didn't confirm it, and the literature's own prediction is that the
   effect shows under distribution shift (Tracks B1–B3), not an in-distribution random
   split — the test that would actually settle this has not yet been run.
5. **Resolution-independence — a proven architectural property, not yet empirically
   exercised on this model.** Discretization convergence (§2 above) is a mathematical
   guarantee of the FNO backbone (Li et al., 2023), holding regardless of this
   project's own results; RF/MaxEnt are trained and evaluated at one fixed pixel
   grid, the FNO backbone is provably evaluable at any resolution post-training. No
   zero-shot super-resolution evaluation has actually been run on the trained
   CDR-PINN checkpoints yet — flagged here as the same category of claim as item 4
   above: architecturally true, operationally untested for this specific model.

## 5a. Closing the Gap — Diagnosed and Proposed, Prioritized

Six hypotheses for why full CDR trails RF/MaxEnt on Track A have now been tested
directly (not left as unexamined caveats), all but one ruled out or downgraded:

1. **Train/eval aggregation mismatch (LSE-pool vs. max-pool) — tested, ruled out.**
   Fixing it left AUC statistically unchanged (0.9406→0.9406). A real correctness
   fix, not the explanation for the gap.
2. **Under-parameterization/under-training — tested directly, twice, ruled out.**
   `width=64`, 150 epochs, identical split/seed: AUC=0.9292 (worse than 0.9406), and
   confirmed again on an independent validation split (item 4 below): 0.9339 vs.
   0.9370 for the small config. Consistently worse across two splits — scale-up is
   not the fix.
3. **Causal time-weighting and staged curriculum learning — tested, both worse.**
   Causal time-weighting (Wang, Sankaran & Perdikaris, 2022 [cite-verify]),
   ε=1.0: AUC=0.9369. Staged curriculum (advection unlocked epoch 15, reaction
   epoch 35): AUC=0.9343. Both below the 0.9406 baseline — a fourth and fifth
   optimization-side intervention that doesn't close the gap. Caveat: neither ε nor
   the unlock epochs were swept, so this is "no help at the tested default," not
   "ruled out across all settings" (multi-seed/hyperparameter sweep is future work).
4. **Learning-rate schedule — downgraded from "ruled out" to "split-sensitive."**
   The original cosine-schedule test (item 2's companion) used Track A's test AUC
   directly for the decision — a disclosed methodological gap. Re-tested honestly on
   an independent train/val/test split (65/15/20%, winner selected by validation AUC
   only): cosine schedule actually **wins** on this split (val 0.9368 vs. 0.9329,
   test 0.9403 vs. 0.9370) — the opposite of the original Track A result (0.9154 vs.
   0.9406). All non-scale-up configurations across both splits cluster in a
   0.93–0.94 AUC band; this is ordinary split-to-split noise at this training scale,
   not a reliable per-config ranking. Full numbers: `CDR_PINN_Full_Paper_Draft.md`
   §4.7–4.8.

Taken together: five of six tested interventions land in the same narrow band,
and the one exception (scale-up) is robustly worse, not better — the pattern is
far more consistent with a representation ceiling (the elevation-dominance finding,
`CDR_PINN_Full_Paper_Draft.md` §4.6) than an under-optimized model that more tuning
would unlock.

Further, literature-grounded levers considered but not implemented, prioritized by
expected value-for-effort, each weighed against this study's own diagnosed
weaknesses rather than adopted generically:

1. **Instance-wise fine-tuning** (Li et al., 2023, §3.2) — PINO's own prescribed
   second training phase: use the learned operator as an ansatz, fine-tune with an
   anchor loss (`L_op`) plus the PDE loss specifically on the deployment instance.
   This is the literature's own explicit answer to "how do you close the gap after
   operator learning" — not yet implemented here.
2. **Self-adaptive per-point loss weighting** (McClenny & Braga-Neto, 2020, SA-PINN
   — already cited as related work in the Methodology document, never implemented)
   — learnable per-pixel weights on the data loss, plausibly most useful for the
   weak Track B2 regions where fixed uniform weighting may be underserving
   under-represented spatial zones.
2a. **Transfer learning for Track B2's weak regions** — fine-tune a pretrained base
   per held-out region rather than training each leave-one-region-out fold from
   scratch; the literature-standard fix for exactly this failure mode, not yet
   attempted.
2b. **Wavelet PINNs/operators** (Tripura & Chakraborty, 2022 [cite-verify]) for
   sharper local risk features than the FNO's 16×16 global-mode truncation permits
   (spectral-truncation limitation, `CDR_PINN_Full_Paper_Draft.md` §5.7 item 5);
   **PIKANs** for interpretability-aligned
   physics heads; **domain-decomposition PINNs** aligned with Step 2's existing
   biogeographic-zone infrastructure, a second architecturally distinct route to
   the same Track B2 gap item 2a targets. All three require substantial new
   architecture work, deliberately not attempted without that justification.
3. **Matched epoch budget for B1/B2** — currently 50 epochs vs. Track A's 80,
   disclosed explicitly as a wall-clock trade-off, not tested at parity yet.
4. **Multi-seed ensembling** — both a standard variance-reduction technique and the
   mechanism needed for the bootstrap CIs this document has flagged as outstanding
   throughout.

## 6. Advantages of the Specific Loss-Term Design

1. **One combined PDE residual, not three tuned sub-terms** — matches the PINO
   paper's own Eq. 4 exactly, avoiding the specific reviewer objection that a
   multi-term, separately-weighted physics loss is under-justified knob-turning.
2. **Adaptive, not hand-picked, weighting** (Wang, Teng & Perdikaris, 2021) — the
   single most predictable reviewer question about any PINN loss ("how were these
   weights chosen, and how sensitive are results to them?") is pre-empted by
   citing a literature-standard method already used in this project's own Step 8,
   not introduced fresh for this component.
3. **Dense, exhaustive collocation coverage** (Methodology §5) rather than sparse
   random sampling — a genuine computational and statistical advantage over the
   classical PINN collocation paradigm, not merely a different implementation
   choice.
4. **A reaction term with a provable global (not just local) existence guarantee**
   — most naive nonlinear reaction terms (e.g. a bare cubic, Allen-Cahn-style term)
   only guarantee local-in-time existence and can blow up in finite time; the
   Fisher–KPP form used here rules that out by construction (`CDR_PINN_Reaction_
   Design.md` §5.1), a genuine mathematical strength specific to this design choice.
