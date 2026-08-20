"""
Biswas-et-al.-style permutation importance for CDR-PINN (per-covariate, not
per-flat-feature, since inputs route through three physics heads rather than one
feature vector -- CDR_PINN_Study_Clarifications_QA.md Q5). Loads the already-trained
full_cdr checkpoint (no retraining needed -- inference only), shuffles each
covariate spatially (a single consistent spatial permutation applied to that
covariate at every month it appears, preserving temporal structure while destroying
spatial correspondence), and reports the resulting held-out AUC drop -- directly
comparable to Biswas et al.'s own Table 3 permutation-importance methodology.
"""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from model import CDRPINN
from losses import lse_pool
from train import build_masks
from build_monthly_stacks import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, TARGET_H, TARGET_W

DATA_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_monthly_stacks.npz"
CKPT_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_full_cdr.pt"
OUT_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_permutation_importance.json"
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

d = np.load(DATA_PATH)
H, W = TARGET_H, TARGET_W
n_months = len(d["months"])


def t(x):
    return torch.tensor(np.nan_to_num(x, nan=0.0), dtype=torch.float32, device=device)


ndvi_f1_np = d["ndvi_f1"]
valid_np, boundary_np, train_np, test_np = build_masks(ndvi_f1_np)
fire_ever_binary_np = (d["fire_ever_frac"] > 0).astype(np.float32)

base = dict(
    ndvi_f1=t(ndvi_f1_np), ndvi_anomaly=t(d["ndvi_anomaly"]), forest_frac=t(d["forest_frac"]),
    dryness=t(d["dryness_proxy"]), slope=t(d["slope"]), dist_roads=t(d["dist_roads"]),
    elevation=t(d["elevation"]), grad_e_x=t(d["grad_e_x"]), grad_e_y=t(d["grad_e_y"]),
)

ckpt = torch.load(CKPT_PATH, map_location=device)
model = CDRPINN(n_static_channels=7, width=32, modes_h=16, modes_w=16, n_layers=4).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded checkpoint (history had {len(ckpt.get('history', []))} epochs)")


def covariate_stack(cov, ti):
    return torch.stack([
        cov["ndvi_f1"], cov["ndvi_anomaly"][ti] if cov["ndvi_anomaly"].dim() == 3 else cov["ndvi_anomaly"],
        cov["forest_frac"], cov["dryness"][ti] if cov["dryness"].dim() == 3 else cov["dryness"],
        cov["slope"], cov["dist_roads"], cov["elevation"],
    ], dim=0).unsqueeze(0)


def evaluate(cov):
    with torch.no_grad():
        u = torch.zeros(1, 1, H, W, device=device)
        scores = []
        for ti in range(n_months - 1):
            u_next = model(u, covariate_stack(cov, ti))
            scores.append(torch.sigmoid(u_next.squeeze(0).squeeze(0)))
            u = u_next
        score_stack = torch.stack(scores, dim=0)
        pooled = lse_pool(score_stack, dim=0, tau=5.0).cpu().numpy()

    eval_mask = valid_np & test_np
    y_true = fire_ever_binary_np[eval_mask]
    y_score = pooled[eval_mask]
    ok = ~np.isnan(y_score) & ~np.isnan(y_true)
    y_true, y_score = y_true[ok], y_score[ok]
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return auc, ap


print("Computing baseline (unpermuted) AUC...")
baseline_auc, baseline_ap = evaluate(base)
print(f"Baseline: AUC={baseline_auc:.4f}, AP={baseline_ap:.4f}")

rng = np.random.RandomState(SEED)
valid_rows, valid_cols = np.where(valid_np)
n_valid = len(valid_rows)
perm_order = rng.permutation(n_valid)  # one consistent spatial permutation for all covariates


def spatial_permute_2d(field_np):
    out = field_np.copy()
    vals = field_np[valid_rows, valid_cols]
    out[valid_rows, valid_cols] = vals[perm_order]
    return out


def spatial_permute_3d(field_np):
    # (T,H,W) -- permute the SAME spatial locations consistently at every timestep
    out = field_np.copy()
    for ti in range(field_np.shape[0]):
        vals = field_np[ti][valid_rows, valid_cols]
        out[ti][valid_rows, valid_cols] = vals[perm_order]
    return out


STATIC_COVARIATES = ["ndvi_f1", "forest_frac", "slope", "dist_roads", "elevation"]
DYNAMIC_COVARIATES = ["ndvi_anomaly", "dryness"]  # (T,H,W) in the npz

results = {"baseline": {"auc": baseline_auc, "ap": baseline_ap}}

for name in STATIC_COVARIATES:
    cov = dict(base)
    permuted_np = spatial_permute_2d({"ndvi_f1": ndvi_f1_np, "forest_frac": d["forest_frac"],
                                       "slope": d["slope"], "dist_roads": d["dist_roads"],
                                       "elevation": d["elevation"]}[name])
    cov[name] = t(permuted_np)
    auc, ap = evaluate(cov)
    drop = baseline_auc - auc
    results[name] = {"auc": auc, "ap": ap, "auc_drop": drop, "pct_of_baseline": 100 * drop / baseline_auc}
    print(f"[{name}] permuted AUC={auc:.4f} (drop={drop:+.4f}, {100*drop/baseline_auc:.2f}% of baseline)")

npz_key_map = {"ndvi_anomaly": "ndvi_anomaly", "dryness": "dryness_proxy"}
for name in DYNAMIC_COVARIATES:
    cov = dict(base)
    permuted_np = spatial_permute_3d(d[npz_key_map[name]])
    cov[name] = t(permuted_np)
    auc, ap = evaluate(cov)
    drop = baseline_auc - auc
    results[name] = {"auc": auc, "ap": ap, "auc_drop": drop, "pct_of_baseline": 100 * drop / baseline_auc}
    print(f"[{name}] permuted AUC={auc:.4f} (drop={drop:+.4f}, {100*drop/baseline_auc:.2f}% of baseline)")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT_PATH}")

ranked = sorted([(k, v["auc_drop"]) for k, v in results.items() if k != "baseline"], key=lambda x: -x[1])
print("\nRanked by AUC drop (largest = most important):")
for name, drop in ranked:
    print(f"  {name}: {drop:+.4f}")
