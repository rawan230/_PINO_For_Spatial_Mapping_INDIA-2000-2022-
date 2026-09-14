# Old / Superseded Files

Kept for history — not current. Do not cite from these while writing the methodology
section; use `Design_and_Paper/CDR_PINN_Methodology_Section.md` and the project root's
`Complete_Methodology_Section.md` instead.

| File | Why it's here |
|---|---|
| `_partial_check.json` | A leftover raw notebook-JSON dump from an early Step 7/8 planning pass (title: "Step 7 — Physics-Informed Fire-Risk Model (PINN vs. baseline ladder)"), unrelated to the current CDR-PINN implementation. Not referenced by any script or doc. |
| `CDR_PINN_Architecture_Diagram.jpg` | An early, hand-drafted architecture diagram from the design phase (2026-08-21). **Factually stale**: it labels the spectral-conv boundary handling "Fourier continuation (zero-pad)," but the actual, verified implementation uses a Neumann whole-sample-symmetric extension instead — zero-padding was tried and rejected (>99% higher edge error). Never wired into any methodology doc. Superseded by `CDR_PINN_Data/cdr_pinn_architecture_diagram.png`, generated directly from the real model code and embedded in the methodology docs as of 2026-09-02. |
| `cdr_pinn_full_cdr_PRE_STANDARD_PROTOCOL_backup.pt` | Explicitly self-labeled backup of the pre-standard-protocol checkpoint (the old ad-hoc 80-epoch/no-validation run, AUC 0.9406). Superseded by `CDR_PINN_Data/cdr_pinn_full_cdr_standard_protocol.pt` (validated 65/15/20 split, test AUC 0.9398). Not referenced by any script. |
| `cdr_pinn_checkpoint_verify.pt` | A one-off diagnostic checkpoint (2026-08-20 metric-fix verification, §8.1 of the methodology). Not referenced by any current script. |
| `LATEX_Manuscript_2026-08-21/` + `.zip` | A LaTeX/IEEE-TGRS compile of the CDR-PINN manuscript (`main.tex`, `references.bib`, compiled PDF), dated 2026-08-21. Found misplaced inside the `NDVI_DATA_INDIA_` repo (Step 2) — relocated here, its rightful home. Kept as an old snapshot rather than deleted since it predates substantial later paper-draft work (standard-protocol adoption, the 22yr/20yr ablation, the B1/B2/B3 train-AUC diagnostic, the architecture diagram, the novelty section) — the current working draft is `Design_and_Paper/CDR_PINN_Full_Paper_Draft.md`. |

Moved 2026-09-14.
