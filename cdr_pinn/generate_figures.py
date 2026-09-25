"""
CDR-PINO figures, drawn from RESULT FILES only (audit 2026-09-25 rewrite). No number is
hard-coded; nothing is trained.

Inputs (CDR_PINN_Data/unified/, written by run_unified_protocol.py / term_magnitudes.py,
exact reference filenames): track_{A,B1,B2,B3,B3orig}.json, pred_<tag>.npz,
term_magnitudes_<tag>.json, and optionally ckpt_A_full_s42.pt (map only).

Figures (to CDR_PINN_Data/unified/figures/):
  1. cdr_ablation_by_track.png      test ROC-AUC per physics config and track (mean +/- SD over seeds)
  2. cdr_roc_pr_trackA.png          ROC / PR of A_full_s42 vs A_nophys_s42 (+ same-cells RF if present)
  3. cdr_term_magnitudes.png        mean |term| of the CDR residual per checkpoint (log scale)
  4. cdr_susceptibility_map.png     only with --map: CPU rollout of ckpt_A_full_s42.pt, LSE-pooled
  5. legacy permutation-importance / response-curve plots, only if the historical v1 JSONs
     exist in CDR_PINN_Data/ -- titled as HISTORICAL (v1 checkpoint, pre-audit protocol).

Usage:  python generate_figures.py [--unified_dir DIR] [--map]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score  # noqa: E402

CDR_DIR = os.path.dirname(os.path.abspath(__file__))
STEP8_DIR = os.path.dirname(CDR_DIR)
sys.path.insert(0, CDR_DIR)
DATA_DIR = os.path.join(STEP8_DIR, "CDR_PINN_Data")
UNIFIED_DIR = os.path.join(DATA_DIR, "unified")
from analyze_unified import BRIDGE_PRED_DIR  # noqa: E402

CONFIG_ORDER = ["nophys", "diff", "diffadv", "full"]
CONFIG_LABEL = {"nophys": "No physics", "diff": "Diffusion", "diffadv": "Diff + adv", "full": "Full CDR"}
TRACKS = ["A", "B1", "B2", "B3", "B3orig"]


def load_track(U, track):
    f = os.path.join(U, f"track_{track}.json")
    return json.load(open(f))["results"] if os.path.exists(f) else []


def seed_means(results, cfg):
    """B1/B2: mean over folds/regions within a seed, then over seeds."""
    by_seed = {}
    for r in results:
        if r["config"] == cfg:
            by_seed.setdefault(r["seed"], []).append(r["test_auc"])
    v = [np.mean(x) for x in by_seed.values()]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)) if v else None


def fig_ablation(U, out):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.2
    any_bar = False
    for j, cfg in enumerate(CONFIG_ORDER):
        xs, ms, sds = [], [], []
        for i, tr in enumerate(TRACKS):
            sm = seed_means(load_track(U, tr), cfg)
            if sm:
                xs.append(i + (j - 1.5) * width); ms.append(sm[0]); sds.append(sm[1])
        if xs:
            any_bar = True
            bars = ax.bar(xs, ms, width, yerr=sds, capsize=3, label=CONFIG_LABEL[cfg])
            for bar, m, sd in zip(bars, ms, sds):
                ax.text(bar.get_x() + bar.get_width() / 2, m + sd + 0.008, f"{m:.3f}", ha="center", va="bottom",
                        fontsize=6.5, rotation=90)
    if not any_bar:
        plt.close(fig); print("skip ablation: no track_*.json"); return
    ax.set_xticks(range(len(TRACKS))); ax.set_xticklabels(TRACKS)
    ax.set_ylabel("Test ROC-AUC (mean ± SD over seeds)"); ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_title("CDR-PINO physics ablation, unified protocol (B3orig = historical leaky terminal label)")
    ax.legend(ncol=4, fontsize=8, loc="lower left"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(out, "cdr_ablation_by_track.png"), dpi=150); plt.close(fig)
    print("Saved cdr_ablation_by_track.png")


def fig_roc_pr(U, out, seed=42):
    curves = []
    for cfg in ("full", "nophys"):
        f = os.path.join(U, f"pred_A_{cfg}_s{seed}.npz")
        if os.path.exists(f):
            z = np.load(f); curves.append((f"CDR-PINO {CONFIG_LABEL[cfg]}", z["y"].astype(int), z["p"].astype(float)))
    fb = os.path.join(BRIDGE_PRED_DIR, "bridge12_A__RF_cdr7.npz")
    if curves and os.path.exists(fb):
        z = np.load(fb)
        if np.array_equal(z["y"].astype(int), curves[0][1]):
            curves.append(("RF, same cells & covariates", z["y"].astype(int), z["p"].astype(float)))
    if not curves:
        print("skip ROC/PR: no Track A predictions"); return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, y, p in curves:
        fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
        a1.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.4f})")
        a2.plot(rc, pr, label=f"{name} (AP={average_precision_score(y, p):.4f})")
    prev = curves[0][1].mean()
    a1.plot([0, 1], [0, 1], color="gray", ls="--", label="Chance")
    a2.axhline(prev, color="gray", ls="--", label=f"No skill (prevalence={prev:.3f})")
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate"); a1.set_title(f"ROC, Track A test cells (seed {seed})")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision"); a2.set_title(f"Precision-recall, Track A test cells (seed {seed})")
    for a in (a1, a2):
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "cdr_roc_pr_trackA.png"), dpi=150); plt.close(fig)
    print("Saved cdr_roc_pr_trackA.png")


def fig_terms(U, out):
    files = sorted(glob.glob(os.path.join(U, "term_magnitudes_*.json")))
    if not files:
        print("skip term magnitudes: no term_magnitudes_*.json"); return
    tags, vals = [], []
    for f in files:
        o = json.load(open(f)); s = o["summary"]
        tags.append(o.get("tag", os.path.basename(f)[16:-5]))
        vals.append([s["dudt"]["mean"], s["diff"]["mean"], s["adv"]["mean"], s["react"]["mean"], s["resid"]["mean"]])
    vals = np.array(vals)
    names = ["|∂u/∂t|", "|D∇²u|", "|v·∇u|", "|R|", "RMS residual"]
    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(tags)), 4.8))
    w = 0.16
    for k, nm in enumerate(names):
        ax.bar(np.arange(len(tags)) + (k - 2) * w, vals[:, k], w, label=nm)
    ax.set_yscale("log"); ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean over valid cells × 265 transitions")
    ax.set_title("Magnitude of each term of the implemented CDR residual (trained checkpoints)")
    ax.legend(fontsize=8, ncol=5); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(out, "cdr_term_magnitudes.png"), dpi=150); plt.close(fig)
    print("Saved cdr_term_magnitudes.png")


def fig_map(U, out, tag="A_full_s42"):
    import torch
    from model import CDRPINN
    from losses import lse_pool
    from preprocessing import load_tensors, covariate_stack, TARGET_H, TARGET_W
    from build_monthly_stacks import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX
    from run_unified_protocol import WIDTH, MODES, N_LAYERS
    ck = os.path.join(U, f"ckpt_{tag}.pt")
    if not os.path.exists(ck):
        print(f"skip map: {ck} not found"); return
    tensors, _, ndvi_f1, T = load_tensors("cpu")
    m = CDRPINN(n_static_channels=7, width=WIDTH, modes_h=MODES, modes_w=MODES, n_layers=N_LAYERS)
    state = torch.load(ck, map_location="cpu", weights_only=False)
    m.load_state_dict(state["model_state"]); m.eval()
    with torch.no_grad():
        u = torch.zeros(1, 1, TARGET_H, TARGET_W); sc = []
        for ti in range(T - 1):
            u = m(u, covariate_stack(tensors, ti)); sc.append(torch.sigmoid(u[0, 0]))
        pooled = lse_pool(torch.stack(sc, 0), dim=0, tau=5.0).numpy()
    pooled = np.where(~np.isnan(ndvi_f1), pooled, np.nan)
    auc = state.get("result", {}).get("test_auc")
    fig, ax = plt.subplots(figsize=(9, 10))
    im = ax.imshow(pooled, cmap="YlOrRd", extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], origin="upper", vmin=0, vmax=1)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"CDR-PINO fire-susceptibility map ({tag}" + (f", test AUC={auc:.4f})" if auc is not None else ")"))
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("LSE-pooled predicted probability")
    fig.tight_layout(); fig.savefig(os.path.join(out, "cdr_susceptibility_map.png"), dpi=150); plt.close(fig)
    print("Saved cdr_susceptibility_map.png")


def fig_legacy(out):
    """Historical v1 analyses (permutation importance, response curves) of the pre-audit
    checkpoint -- plotted only if their JSONs exist, and labelled as historical."""
    fp = os.path.join(DATA_DIR, "cdr_pinn_permutation_importance.json")
    if os.path.exists(fp):
        perm = json.load(open(fp))
        names = [k for k in perm if k != "baseline"]
        drops = np.array([perm[k]["auc_drop"] for k in names]); o = np.argsort(drops)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([names[i] for i in o], drops[o], color="#1f77b4")
        ax.set_xlabel("AUC drop when covariate is spatially permuted")
        ax.set_title(f"HISTORICAL (v1 checkpoint): permutation importance, baseline AUC={perm['baseline']['auc']:.4f}")
        ax.grid(alpha=0.3, axis="x"); fig.tight_layout()
        fig.savefig(os.path.join(out, "historical_v1_permutation_importance.png"), dpi=150); plt.close(fig)
        print("Saved historical_v1_permutation_importance.png")
    fr = os.path.join(DATA_DIR, "cdr_pinn_response_curves.json")
    if os.path.exists(fr):
        resp = json.load(open(fr)); covs = list(resp)
        nc = 3; nr = int(np.ceil(len(covs) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(15, 4 * nr), squeeze=False)
        for ax, cov in zip(axes.flat, covs):
            xs, ys = resp[cov]["sweep_values"], resp[cov]["predicted_probability"]
            ax.plot(xs, ys, marker="o", ms=3, color="#2ca02c")
            ax.set_title(f"{cov} (Δ={max(ys) - min(ys):.4f})"); ax.set_xlabel(cov); ax.set_ylabel("Predicted probability")
            ax.grid(alpha=0.3)
        for ax in list(axes.flat)[len(covs):]:
            ax.axis("off")
        fig.suptitle("HISTORICAL (v1 checkpoint): response curves"); fig.tight_layout()
        fig.savefig(os.path.join(out, "historical_v1_response_curves.png"), dpi=150); plt.close(fig)
        print("Saved historical_v1_response_curves.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified_dir", default=UNIFIED_DIR)
    ap.add_argument("--out_dir", default=None, help="default: <unified_dir>/figures")
    ap.add_argument("--map", action="store_true", help="also draw the susceptibility map (CPU rollout, ~1 min)")
    args = ap.parse_args()
    out = args.out_dir or os.path.join(args.unified_dir, "figures")
    os.makedirs(out, exist_ok=True)
    fig_ablation(args.unified_dir, out)
    fig_roc_pr(args.unified_dir, out)
    fig_terms(args.unified_dir, out)
    if args.map:
        fig_map(args.unified_dir, out)
    fig_legacy(out)


if __name__ == "__main__":
    main()
