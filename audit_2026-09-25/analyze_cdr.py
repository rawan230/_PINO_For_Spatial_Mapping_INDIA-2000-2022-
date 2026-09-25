"""
Compile the unified CDR-PINO runs into seed-level / summary tables, the physics-term ablation,
the generalisation tracks, and paired statistical comparisons:
  * physics configs vs 'nophys' on identical test cells (DeLong for Track A; block bootstrap
    over 2 deg blocks for B1/B2; pooled cell-month DeLong for B3), per seed
  * CDR-PINO (full) vs the classical bridge models on the same 12 km cells (bridge12_*)
  * B3 vs the persistence null models
Outputs to results/cdr_pino/ and results/ablation/, results/generalization/.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from audit_common import RES  # noqa: E402
from eval_utils import delong_paired, block_boot_diff, metrics  # noqa: E402

U = os.path.join(RES, "cdr_pino", "unified")
PB = os.path.join(RES, "baseline", "predictions")
os.makedirs(os.path.join(RES, "ablation"), exist_ok=True)
os.makedirs(os.path.join(RES, "generalization"), exist_ok=True)
part = np.load(os.path.join(U, "partitions.npz"))
valid = part["valid"]; block = part["block_id"]
d = np.load(os.path.join(RES, "..", "Physics_Informed_FireRisk_Model", "CDR_PINN_Data", "cdr_pinn_monthly_stacks.npz"))
forest12 = (np.nan_to_num(d["forest_frac"]) > 0)

rows = []
for f in sorted(glob.glob(os.path.join(U, "track_*.json"))):
    for r in json.load(open(f))["results"]:
        rows.append({k: v for k, v in r.items() if k not in ("history",)})
R = pd.DataFrame(rows)
R["track"] = R.tag.str.split("_").str[0]
R.loc[R.tag.str.startswith("B3orig"), "track"] = "B3orig"
R.to_csv(os.path.join(RES, "cdr_pino", "SEED_LEVEL_RESULTS_cdr.csv"), index=False)


def pred(tag):
    z = np.load(os.path.join(U, f"pred_{tag}.npz")); return z["y"].astype(int), z["p"].astype(float)


def test_mask(track, extra):
    if track == "A":
        return valid & part["trackA_test"]
    if track == "B1":
        fold_of = {int(b): int(k) for b, k in part["b1_fold_of_block"]}
        fold = np.vectorize(lambda b: fold_of.get(int(b), -1))(block)
        return valid & (fold == extra)
    if track == "B2":
        return part["b2_region"] == extra
    return None


summ, paired = [], []
for track in ("A", "B1", "B2", "B3", "B3orig"):
    sub = R[R.track == track]
    for cfg in sub.config.unique():
        s = sub[sub.config == cfg]
        per_seed = s.groupby("seed").agg(auc=("test_auc", "mean"), ap=("test_ap", "mean"))  # B1/B2: mean over folds/regions
        summ.append(dict(track=track, config=cfg, n_runs=len(s), seeds=sorted(s.seed.unique().tolist()),
                         auc_mean=float(per_seed.auc.mean()), auc_sd_across_seeds=float(per_seed.auc.std(ddof=1)) if len(per_seed) > 1 else np.nan,
                         auc_min=float(s.test_auc.min()), auc_max=float(s.test_auc.max()),
                         auc_sd_across_folds_seed42=float(s[s.seed == 42].test_auc.std(ddof=0)) if track in ("B1", "B2") else np.nan,
                         ap_mean=float(per_seed.ap.mean()), prevalence=float(s.test_prevalence.mean()),
                         train_time_mean_sec=float(s.train_time_sec.mean()), best_epoch_mean=float(s.best_epoch.mean()),
                         c_adv_mean=float(s.c_adv.mean())))
    # paired: each physics config vs nophys, per seed and fold, on identical cells
    for cfg in [c for c in sub.config.unique() if c != "nophys"]:
        for seed in sorted(sub.seed.unique()):
            a = sub[(sub.config == cfg) & (sub.seed == seed)]; b = sub[(sub.config == "nophys") & (sub.seed == seed)]
            if a.empty or b.empty:
                continue
            ys, p1s, p2s, bls = [], [], [], []
            for (_, ra), (_, rb) in zip(a.sort_values("tag").iterrows(), b.sort_values("tag").iterrows()):
                y1, p1 = pred(ra.tag); y2, p2 = pred(rb.tag)
                assert np.array_equal(y1, y2)
                ys.append(y1); p1s.append(p1); p2s.append(p2)
                ex = ra["fold"] if track == "B1" else (ra["region"] if track == "B2" else None)
                m = test_mask(track, None if ex is None or pd.isna(ex) else int(ex)) if track in ("A", "B1", "B2") else None
                bls.append(block[m] if m is not None else np.zeros(len(y1), int))
            y, p1, p2, bl = map(np.concatenate, (ys, p1s, p2s, bls))
            row = dict(track=track, config=cfg, vs="nophys", seed=int(seed), n=len(y), prevalence=float(y.mean()))
            if track == "A" or track.startswith("B3"):
                row.update({f"delong_{k}": v for k, v in delong_paired(y, p1, p2).items()})
            else:
                row.update({f"blockboot_{k}": v for k, v in block_boot_diff(y, p1, p2, bl, n=300).items()})
            paired.append(row)
S = pd.DataFrame(summ); P = pd.DataFrame(paired)
S.to_csv(os.path.join(RES, "cdr_pino", "SUMMARY_cdr_unified.csv"), index=False)
P.to_csv(os.path.join(RES, "ablation", "PAIRED_physics_vs_nophys.csv"), index=False)
S[S.track.isin(["A", "B1", "B2", "B3"])].to_csv(os.path.join(RES, "ablation", "ABLATION_RESULTS.csv"), index=False)

# forest-only population re-scoring of CDR predictions (spatial tracks; test cells with forest_frac>0)
fr = []
for _, r in R[R.track.isin(["A", "B1", "B2"])].iterrows():
    y, p = pred(r.tag)
    ex = r["fold"] if r.track == "B1" else (r["region"] if r.track == "B2" else None)
    m = test_mask(r.track, None if ex is None or pd.isna(ex) else int(ex))
    fm = forest12[m]
    mt = metrics(y[fm], p[fm])
    fr.append(dict(tag=r.tag, track=r.track, config=r.config, seed=r.seed, forest_auc=mt.get("roc_auc"), forest_ap=mt.get("ap"),
                   forest_n=mt["n"], forest_prev=mt["prevalence"]))
pd.DataFrame(fr).to_csv(os.path.join(RES, "cdr_pino", "FOREST_POPULATION_cdr.csv"), index=False)

# CDR-PINO full vs bridge classical models on the same cells (Track A, seed-wise; B3 vs nulls)
cmp = []
for track, bexp in (("A", "bridge12_A"),):
    for btag in ("RF_cdr7", "MaxEnt_cdr7", "LogReg_cdr7", "RF_v2agg", "MaxEnt_v2agg", "LogReg_v2agg"):
        fb = os.path.join(PB, f"{bexp}__{btag}.npz")
        if not os.path.exists(fb):
            continue
        zb = np.load(fb)
        for seed in (42, 43, 44):
            for cfg in ("full", "nophys"):
                tg = f"A_{cfg}_s{seed}"
                if not os.path.exists(os.path.join(U, f"pred_{tg}.npz")):
                    continue
                y, p = pred(tg)
                assert np.array_equal(y, zb["y"].astype(int)), "populations differ"
                cmp.append(dict(track=track, cdr=tg, classical=btag, **{f"delong_{k}": v for k, v in delong_paired(y, p, zb["p"]).items()}))
for btag in ("null_climatological_frequency", "null_seasonal_frequency", "RF_monthly_cdr7", "RF_monthly_cdr7_month"):
    fb = os.path.join(PB, f"bridge12_B3__{btag}.npz")
    if not os.path.exists(fb):
        continue
    zb = np.load(fb)
    for tg in [t for t in R[R.track.isin(["B3", "B3orig"])].tag]:
        y, p = pred(tg)
        if len(y) != len(zb["y"]) or not np.array_equal(y, zb["y"].astype(int)):
            continue
        cmp.append(dict(track="B3", cdr=tg, classical=btag, **{f"delong_{k}": v for k, v in delong_paired(y, p, zb["p"]).items()}))
pd.DataFrame(cmp).to_csv(os.path.join(RES, "generalization", "PAIRED_cdr_vs_classical_same_cells.csv"), index=False)
print(S.to_string())
print(P.groupby(["track", "config"]).agg(**{c: (c, "mean") for c in P.columns if c.endswith("diff")}).to_string() if len(P) else "no paired")
