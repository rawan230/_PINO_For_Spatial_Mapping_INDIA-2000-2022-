"""
Compile the unified CDR-PINO runs (run_unified_protocol.py) into the tables the paper
quotes -- port of the audit's results/code/analyze_cdr.py.

  1. Seed-level results and a per-track / per-config summary (B1/B2: mean over folds or
     regions within a seed, then mean +/- SD across seeds).
  2. Paired physics-config vs 'nophys' tests on IDENTICAL test cells, per seed:
     DeLong for Track A and B3/B3orig (cell-months pooled); 2-degree block bootstrap for
     B1/B2 (folds / regions pooled within a seed).
  3. Forest-population re-scoring (test cells with forest_frac > 0, the audit's primary
     evaluation population) for A/B1/B2, and forest cell-months for B3.
  4. CDR-PINO vs the classical "bridge" models (RF / MaxEnt / LogReg trained on
     CDR-PINO's own 12 km cells, covariates, partitions and label) on the same cells,
     plus B3 vs the persistence null models -- only if those prediction files exist in
     BRIDGE_PRED_DIR (written by the classical-model code as
     bridge12_<track>__<model>[_f<k>|_r<k>].npz with arrays y, p).
  5. TERM_MAGNITUDES_summary.csv collected from every term_magnitudes_<tag>.json
     (written by term_magnitudes.py) found in the unified dir.
  6. A train/validation AUC history figure for Track A (seed 42), from the run JSONs.

Nothing is trained or re-inferred: every number comes from the files written by
run_unified_protocol.py (track_*.json, partitions.npz, pred_<tag>.npz) and
term_magnitudes.py, so the tables can be regenerated from copied result files alone.

CPU only, ~2-5 min. Outputs to <unified dir>/analysis/.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

CDR_DIR = os.path.dirname(os.path.abspath(__file__))
STEP8_DIR = os.path.dirname(CDR_DIR)
PROJECT_ROOT = os.path.dirname(STEP8_DIR)
sys.path.insert(0, CDR_DIR)
from eval_utils import delong_paired, block_boot_diff, metrics  # noqa: E402
from preprocessing import DATA_PATH  # noqa: E402

UNIFIED_DIR = os.path.join(STEP8_DIR, "CDR_PINN_Data", "unified")
# Classical same-cells ("bridge") predictions, produced by the Step 7 / audit classical code
BRIDGE_PRED_DIR = os.path.join(PROJECT_ROOT, "Integrated_Analysis", "Model_Outputs", "bridge_predictions")
CLASSICAL_TAGS = ("RF_cdr7", "MaxEnt_cdr7", "LogReg_cdr7", "RF_v2agg", "MaxEnt_v2agg", "LogReg_v2agg")
B3_NULL_TAGS = ("null_climatological_frequency", "null_seasonal_frequency", "RF_monthly_cdr7", "RF_monthly_cdr7_month")
N_BOOT = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified_dir", default=UNIFIED_DIR)
    ap.add_argument("--bridge_dir", default=BRIDGE_PRED_DIR)
    ap.add_argument("--out_dir", default=None, help="default: <unified_dir>/analysis")
    args = ap.parse_args()
    U, PB = args.unified_dir, args.bridge_dir
    OUTD = args.out_dir or os.path.join(U, "analysis")
    os.makedirs(OUTD, exist_ok=True)

    part = np.load(os.path.join(U, "partitions.npz"))
    valid = part["valid"]; block = part["block_id"]
    d = np.load(DATA_PATH)
    forest12 = (np.nan_to_num(d["forest_frac"]) > 0)

    rows = []
    for f in sorted(glob.glob(os.path.join(U, "track_*.json"))):
        for r in json.load(open(f))["results"]:
            rows.append({k: v for k, v in r.items() if k != "history"})
    if not rows:
        print(f"No track_*.json in {U} -- run run_unified_protocol.py first."); return
    R = pd.DataFrame(rows)
    R["track"] = R.tag.str.split("_").str[0]
    R.loc[R.tag.str.startswith("B3orig"), "track"] = "B3orig"
    for c in ("fold", "region"):
        if c not in R.columns:
            R[c] = np.nan
    R.to_csv(os.path.join(OUTD, "SEED_LEVEL_RESULTS_cdr.csv"), index=False)

    def pred(tag):
        z = np.load(os.path.join(U, f"pred_{tag}.npz")); return z["y"].astype(int), z["p"].astype(float)

    fold_of = {int(b): int(k) for b, k in part["b1_fold_of_block"]}
    fold_grid = np.vectorize(lambda b: fold_of.get(int(b), -1))(block)

    def test_mask(track, extra):
        if track == "A":
            return valid & part["trackA_test"]
        if track == "B1":
            return valid & (fold_grid == extra)
        if track == "B2":
            return part["b2_region"] == extra
        return None

    def extra_of(row):
        ex = row["fold"] if row["track"] == "B1" else (row["region"] if row["track"] == "B2" else None)
        return None if ex is None or pd.isna(ex) else int(ex)

    # ---------------- 1 + 2: summary and paired physics-vs-nophys ----------------
    summ, paired = [], []
    for track in ("A", "B1", "B2", "B3", "B3orig"):
        sub = R[R.track == track]
        for cfg in sub.config.unique():
            s = sub[sub.config == cfg]
            per_seed = s.groupby("seed").agg(auc=("test_auc", "mean"), ap=("test_ap", "mean"))
            summ.append(dict(track=track, config=cfg, n_runs=len(s), seeds=sorted(s.seed.unique().tolist()),
                             auc_mean=float(per_seed.auc.mean()),
                             auc_sd_across_seeds=float(per_seed.auc.std(ddof=1)) if len(per_seed) > 1 else np.nan,
                             auc_min=float(s.test_auc.min()), auc_max=float(s.test_auc.max()),
                             auc_sd_across_folds_seed42=float(s[s.seed == 42].test_auc.std(ddof=0)) if track in ("B1", "B2") else np.nan,
                             ap_mean=float(per_seed.ap.mean()), prevalence=float(s.test_prevalence.mean()),
                             train_time_mean_sec=float(s.train_time_sec.mean()), best_epoch_mean=float(s.best_epoch.mean()),
                             c_adv_mean=float(s.c_adv.mean())))
        for cfg in [c for c in sub.config.unique() if c != "nophys"]:
            for seed in sorted(sub.seed.unique()):
                a = sub[(sub.config == cfg) & (sub.seed == seed)].sort_values("tag")
                b = sub[(sub.config == "nophys") & (sub.seed == seed)].sort_values("tag")
                if a.empty or b.empty or len(a) != len(b):
                    continue
                ys, p1s, p2s, bls = [], [], [], []
                for (_, ra), (_, rb) in zip(a.iterrows(), b.iterrows()):
                    y1, p1 = pred(ra.tag); y2, p2 = pred(rb.tag)
                    assert np.array_equal(y1, y2), f"{ra.tag} vs {rb.tag}: different test cells"
                    ys.append(y1); p1s.append(p1); p2s.append(p2)
                    m = test_mask(track, extra_of(ra)) if track in ("A", "B1", "B2") else None
                    bls.append(block[m] if m is not None else np.zeros(len(y1), int))
                y, p1, p2, bl = map(np.concatenate, (ys, p1s, p2s, bls))
                row = dict(track=track, config=cfg, vs="nophys", seed=int(seed), n=len(y), prevalence=float(y.mean()))
                if track == "A" or track.startswith("B3"):
                    row.update({f"delong_{k}": v for k, v in delong_paired(y, p1, p2).items()})
                else:
                    row.update({f"blockboot_{k}": v for k, v in block_boot_diff(y, p1, p2, bl, n=N_BOOT).items()})
                paired.append(row)
    S = pd.DataFrame(summ); P = pd.DataFrame(paired)
    S.to_csv(os.path.join(OUTD, "SUMMARY_cdr_unified.csv"), index=False)
    P.to_csv(os.path.join(OUTD, "PAIRED_physics_vs_nophys.csv"), index=False)
    S[S.track.isin(["A", "B1", "B2", "B3"])].to_csv(os.path.join(OUTD, "ABLATION_RESULTS.csv"), index=False)

    # ---------------- 3: forest-population re-scoring ----------------
    fr = []
    n_valid = int(valid.sum())
    forest_valid = forest12[valid]
    for _, r in R.iterrows():
        y, p = pred(r.tag)
        if r.track in ("A", "B1", "B2"):
            fm = forest12[test_mask(r.track, extra_of(r))]
        else:  # B3 / B3orig: predictions are (test months x valid cells), row-major
            if len(y) % n_valid:
                continue
            fm = np.tile(forest_valid, len(y) // n_valid)
        mt = metrics(y[fm], p[fm])
        fr.append(dict(tag=r.tag, track=r.track, config=r.config, seed=r.seed, forest_auc=mt.get("roc_auc"),
                       forest_ap=mt.get("ap"), forest_n=mt["n"], forest_prev=mt["prevalence"], all_auc=r.test_auc))
    F = pd.DataFrame(fr)
    F.to_csv(os.path.join(OUTD, "FOREST_POPULATION_cdr.csv"), index=False)
    if len(F):
        F.groupby(["track", "config"]).agg(forest_auc_mean=("forest_auc", "mean"), forest_auc_sd=("forest_auc", "std"),
                                           all_auc_mean=("all_auc", "mean"), n_runs=("tag", "count")
                                           ).reset_index().to_csv(os.path.join(OUTD, "FOREST_POPULATION_summary.csv"), index=False)

    # ---------------- 4: CDR-PINO vs classical bridge models on the same cells ----------------
    cmp = []
    if not os.path.isdir(PB):
        print(f"NOTE: bridge prediction folder not found ({PB}); skipping same-cells classical comparison")
    else:
        # Track A: DeLong per seed
        for btag in CLASSICAL_TAGS:
            fb = os.path.join(PB, f"bridge12_A__{btag}.npz")
            if not os.path.exists(fb):
                continue
            zb = np.load(fb)
            for seed in (42, 43, 44):
                for cfg in ("full", "nophys"):
                    tg = f"A_{cfg}_s{seed}"
                    if not os.path.exists(os.path.join(U, f"pred_{tg}.npz")):
                        continue
                    y, p = pred(tg)
                    if not np.array_equal(y, zb["y"].astype(int)):
                        print(f"WARNING: {tg} vs {btag}: populations differ, skipped"); continue
                    cmp.append(dict(track="A", cdr=tg, classical=btag,
                                    **{f"delong_{k}": v for k, v in delong_paired(y, p, zb["p"]).items()}))
        # B1 / B2: folds or regions pooled within a seed, 2-degree block bootstrap
        for track, key, n_parts in (("B1", "f", 3), ("B2", "r", 6)):
            for btag in CLASSICAL_TAGS:
                fbs = [os.path.join(PB, f"bridge12_{track}__{btag}_{key}{k}.npz") for k in range(n_parts)]
                if not all(os.path.exists(f) for f in fbs):
                    continue
                for seed in (42, 43, 44):
                    for cfg in ("full", "nophys"):
                        tgs = [f"{track}_{key}{k}_{cfg}_s{seed}" for k in range(n_parts)]
                        if not all(os.path.exists(os.path.join(U, f"pred_{t}.npz")) for t in tgs):
                            continue
                        ys, p1s, p2s, bls, ok = [], [], [], [], True
                        for k, (tg, fb) in enumerate(zip(tgs, fbs)):
                            y, p = pred(tg); zb = np.load(fb)
                            if not np.array_equal(y, zb["y"].astype(int)):
                                ok = False; break
                            ys.append(y); p1s.append(p); p2s.append(zb["p"].astype(float))
                            bls.append(block[test_mask(track, k)])
                        if not ok:
                            print(f"WARNING: {track} {cfg} s{seed} vs {btag}: populations differ, skipped"); continue
                        y, p1, p2, bl = map(np.concatenate, (ys, p1s, p2s, bls))
                        cmp.append(dict(track=track, cdr=f"{track}_{cfg}_s{seed}", classical=btag,
                                        **{f"blockboot_{k}": v for k, v in block_boot_diff(y, p1, p2, bl, n=N_BOOT).items()}))
        # B3: vs persistence nulls and monthly RF, identical cell-months
        for btag in B3_NULL_TAGS:
            fb = os.path.join(PB, f"bridge12_B3__{btag}.npz")
            if not os.path.exists(fb):
                continue
            zb = np.load(fb)
            for tg in R[R.track.isin(["B3", "B3orig"])].tag:
                y, p = pred(tg)
                if len(y) != len(zb["y"]) or not np.array_equal(y, zb["y"].astype(int)):
                    continue
                cmp.append(dict(track="B3", cdr=tg, classical=btag,
                                **{f"delong_{k}": v for k, v in delong_paired(y, p, zb["p"]).items()}))
    pd.DataFrame(cmp).to_csv(os.path.join(OUTD, "PAIRED_cdr_vs_classical_same_cells.csv"), index=False)

    # ---------------- 5: physics-term magnitudes (from term_magnitudes.py output) ----------------
    tm_rows = []
    for f in sorted(glob.glob(os.path.join(U, "term_magnitudes_*.json"))):
        o = json.load(open(f)); s_ = o["summary"]
        tm_rows.append(dict(tag=o.get("tag", os.path.basename(f)[16:-5]), median_abs_dudt=s_["dudt"]["median"],
                            mean_abs_dudt=s_["dudt"]["mean"], mean_abs_diffusion=s_["diff"]["mean"],
                            mean_abs_advection=s_["adv"]["mean"], mean_abs_reaction=s_["react"]["mean"],
                            rms_residual=s_["resid"]["mean"],
                            residual_over_mean_abs_dudt=o["rms_residual_over_mean_abs_dudt"],
                            share_diffusion=o["share_of_rhs_magnitude"]["diffusion"],
                            share_advection=o["share_of_rhs_magnitude"]["advection"],
                            share_reaction=o["share_of_rhs_magnitude"]["reaction"], c_adv=o["c_adv"]))
    if tm_rows:
        pd.DataFrame(tm_rows).to_csv(os.path.join(OUTD, "TERM_MAGNITUDES_summary.csv"), index=False)

    # ---------------- 6: Track A training-history figure ----------------
    fa = os.path.join(U, "track_A.json")
    if os.path.exists(fa):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            res = [r for r in json.load(open(fa))["results"] if r["seed"] == 42]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            for r in res:
                h = r.get("history", [])
                ep = [x["epoch"] for x in h]
                ax1.plot(ep, [x["val_auc"] for x in h], marker="o", label=f"{r['config']} (val)")
                ax1.plot(ep, [x["train_auc"] for x in h], linestyle="--", alpha=0.6, label=f"{r['config']} (train)")
                ax2.plot(ep, [x["val_loss"] for x in h], marker="o", label=r["config"])
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("ROC-AUC"); ax1.set_title("Track A, seed 42: train / validation AUC")
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation BCE"); ax2.set_title("Track A, seed 42: validation loss")
            for ax in (ax1, ax2):
                ax.grid(alpha=0.3); ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(OUTD, "trackA_training_history_s42.png"), dpi=150)
            plt.close(fig)
        except Exception as e:  # figure is optional
            print(f"WARNING: history figure not written: {e}")

    print(S.to_string())
    if len(P):
        diff_cols = [c for c in P.columns if c.endswith("_diff")]
        print(P.groupby(["track", "config"])[diff_cols].mean().to_string())
    print(f"\nSaved tables to {OUTD}")


if __name__ == "__main__":
    main()
