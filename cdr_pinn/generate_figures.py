"""
Generates the missing CDR-PINN figures a completeness audit flagged (Step8 audit,
items 1 and 5): a full-country susceptibility probability map, ROC/PR curves,
a permutation-importance bar chart, and response-curve line plots. Inference-only
against the current canonical checkpoint (cdr_pinn_full_cdr.pt, the standard-
protocol model) -- no retraining.
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from model import CDRPINN
from losses import lse_pool
from preprocessing import (
    build_masks_3way, load_tensors, covariate_stack, SEED, TARGET_H, TARGET_W,
)
from build_monthly_stacks import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX

CKPT_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_full_cdr.pt"
DATA_DIR = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

tensors, fire_ever_binary_np, ndvi_f1_np, n_months = load_tensors(device)
valid_np, boundary_np, train_np, val_np, test_np = build_masks_3way(ndvi_f1_np)
H, W = TARGET_H, TARGET_W

ckpt = torch.load(CKPT_PATH, map_location=device)
model = CDRPINN(n_static_channels=7, width=32, modes_h=16, modes_w=16, n_layers=4).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("Checkpoint loaded (canonical standard-protocol model).")

# ---- Full-grid inference (every valid pixel, not just test) ----
with torch.no_grad():
    u = torch.zeros(1, 1, H, W, device=device)
    scores = []
    for ti in range(n_months - 1):
        u_next = model(u, covariate_stack(tensors, ti))
        scores.append(torch.sigmoid(u_next.squeeze(0).squeeze(0)))
        u = u_next
    pooled = lse_pool(torch.stack(scores, dim=0), dim=0, tau=5.0).cpu().numpy()

pooled_masked = np.where(valid_np, pooled, np.nan)

# ==== FIGURE 1: Full-country susceptibility probability map ====
fig, ax = plt.subplots(figsize=(9, 10))
extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
im = ax.imshow(pooled_masked, cmap="YlOrRd", extent=extent, origin="upper", vmin=0, vmax=1)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("CDR-PINN Fire-Susceptibility Probability Map — India\n(standard-protocol checkpoint, test AUC=0.9398)")
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("Predicted fire-ever probability (LSE-pooled)")
fig.tight_layout()
fig.savefig(f"{DATA_DIR}/cdr_pinn_susceptibility_map.png", dpi=150)
print(f"Saved: {DATA_DIR}/cdr_pinn_susceptibility_map.png")
plt.close(fig)

# ==== FIGURE 2: ROC / PR curves (test set) ====
m = valid_np & test_np
y_true = fire_ever_binary_np[m]
y_score = pooled[m]
ok = ~np.isnan(y_score) & ~np.isnan(y_true)
y_true, y_score = y_true[ok], y_score[ok]
auc = roc_auc_score(y_true, y_score)
ap = average_precision_score(y_true, y_score)
fpr, tpr, _ = roc_curve(y_true, y_score)
prec, rec, _ = precision_recall_curve(y_true, y_score)
no_skill = y_true.mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(fpr, tpr, color="#1f77b4", label=f"CDR-PINN (AUC={auc:.4f})")
ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance")
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve (test set)")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(rec, prec, color="#d62728", label=f"CDR-PINN (AP={ap:.4f})")
ax2.axhline(no_skill, color="gray", linestyle="--", label=f"No-skill (prevalence={no_skill:.3f})")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve (test set)")
ax2.legend(); ax2.grid(alpha=0.3)

fig.suptitle("CDR-PINN Held-Out Test Performance (standard-protocol checkpoint)")
fig.tight_layout()
fig.savefig(f"{DATA_DIR}/cdr_pinn_roc_pr_curves.png", dpi=150)
print(f"Saved: {DATA_DIR}/cdr_pinn_roc_pr_curves.png  (AUC={auc:.4f}, AP={ap:.4f})")
plt.close(fig)

# ==== FIGURE 3: Permutation importance bar chart ====
with open(f"{DATA_DIR}/cdr_pinn_permutation_importance.json") as f:
    perm = json.load(f)
names = [k for k in perm.keys() if k != "baseline"]
drops = [perm[k]["auc_drop"] for k in names]
order = np.argsort(drops)[::-1]
names_sorted = [names[i] for i in order]
drops_sorted = [drops[i] for i in order]

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#d62728" if d == max(drops_sorted) else "#1f77b4" for d in drops_sorted]
ax.barh(names_sorted[::-1], drops_sorted[::-1], color=colors[::-1])
ax.set_xlabel("AUC drop when covariate is spatially permuted")
ax.set_title(f"CDR-PINN Permutation Importance (baseline AUC={perm['baseline']['auc']:.4f})")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(f"{DATA_DIR}/cdr_pinn_permutation_importance.png", dpi=150)
print(f"Saved: {DATA_DIR}/cdr_pinn_permutation_importance.png")
plt.close(fig)

# ==== FIGURE 4: Response curves (line plots, one panel per covariate) ====
with open(f"{DATA_DIR}/cdr_pinn_response_curves.json") as f:
    resp = json.load(f)
covs = list(resp.keys())
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, cov in zip(axes.flat, covs):
    xs = resp[cov]["sweep_values"]
    ys = resp[cov]["predicted_probability"]
    ax.plot(xs, ys, color="#2ca02c", marker="o", markersize=3)
    delta = max(ys) - min(ys)
    ax.set_title(f"{cov} (Δ={delta:.4f})")
    ax.set_xlabel(cov); ax.set_ylabel("Predicted probability")
    ax.grid(alpha=0.3)
fig.suptitle("CDR-PINN Response Curves (Biswas et al. Figs. 8/9 analogue)")
fig.tight_layout()
fig.savefig(f"{DATA_DIR}/cdr_pinn_response_curves.png", dpi=150)
print(f"Saved: {DATA_DIR}/cdr_pinn_response_curves.png")
plt.close(fig)

print("\nAll 4 figures generated successfully.")
