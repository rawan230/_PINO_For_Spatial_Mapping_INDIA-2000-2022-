"""Train-vs-validation AUC overfitting/underfitting diagnostic for Tracks B1/B2/B3
and the 15-run Jackknife sweep -- extends the same diagnostic already built for
the 22yr/20yr ablation (train_standard_protocol.py / train_20yr_subset.py) to
every other CDR-PINN training run in this study, per the project's standing ML
rigor requirement. Reads train_auc_history/val_auc_history already saved by the
refactored run_validation_tracks.py and jackknife_test.py -- no retraining here."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data"

# ------------------------------------------------------------------------- #
# Figure 1: B1 / B2 / B3 train-vs-val AUC trajectories
# ------------------------------------------------------------------------- #
b1 = json.load(open(f"{DATA_DIR}/cdr_pinn_validation_tracks_b1.json"))["B1"]
b2 = json.load(open(f"{DATA_DIR}/cdr_pinn_validation_tracks_b2.json"))["B2"]
b3 = json.load(open(f"{DATA_DIR}/cdr_pinn_validation_tracks_b3.json"))["B3"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
track_data = [
    ("Track B1: spatial block CV\n(2\u00b0\u00d72\u00b0 blocks, 3 folds)", b1, "#2166ac"),
    ("Track B2: leave-one-region-out\n(KMeans, 6 regions)", b2, "#c9782c"),
    ("Track B3: leave-years-out\n(temporal generalization, 1 run)", b3, "#2e8b57"),
]

for ax, (title, runs, color) in zip(axes, track_data):
    for i, r in enumerate(runs):
        th = np.array(r["train_auc_history"])
        vh = np.array(r["val_auc_history"])
        label_t = "Train AUC" if i == 0 else None
        label_v = "Val AUC" if i == 0 else None
        ax.plot(th[:, 0], th[:, 1], "-", color=color, alpha=0.85, linewidth=1.6, label=label_t)
        ax.plot(vh[:, 0], vh[:, 1], "--", color=color, alpha=0.55, linewidth=1.6, label=label_v)
        ax.annotate(r["tag"].split("_")[1], (th[-1, 0], th[-1, 1]), fontsize=6.5, color=color,
                    xytext=(3, 2), textcoords="offset points")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5)
axes[0].set_ylabel("ROC-AUC")

fig.suptitle("CDR-PINN Train-vs-Validation AUC: Spatial/Temporal Generalization Tracks (B1/B2/B3)",
              fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out1 = f"{DATA_DIR}/cdr_pinn_tracks_train_val_auc.png"
fig.savefig(out1, dpi=150, facecolor="white")
print(f"Saved: {out1}")

# Print final train/val gap per fold for the text summary
print("\n--- Final train/val AUC gap per fold (last checkpoint) ---")
for title, runs, _ in track_data:
    for r in runs:
        t_final = r["train_auc_history"][-1][1]
        v_final = r["val_auc_history"][-1][1]
        print(f"  {r['tag']}: train={t_final:.4f} val={v_final:.4f} gap={t_final - v_final:+.4f}")

# ------------------------------------------------------------------------- #
# Figure 2: Jackknife train/val AUC gap summary (15 runs)
# ------------------------------------------------------------------------- #
jk = json.load(open(f"{DATA_DIR}/cdr_pinn_jackknife_results.json"))
COVARIATES = ["ndvi_f1", "ndvi_anomaly", "forest_frac", "dryness", "slope", "dist_roads", "elevation"]

labels, train_finals, val_finals, gaps = [], [], [], []
labels.append("all"); r = jk["all"]
train_finals.append(r["train_auc_history"][-1][1]); val_finals.append(r["val_auc_history"][-1][1])
for cov in COVARIATES:
    for mode in ["without", "only"]:
        key = f"{mode}_{cov}"
        r = jk[key]
        labels.append(key)
        train_finals.append(r["train_auc_history"][-1][1])
        val_finals.append(r["val_auc_history"][-1][1])
gaps = [t - v for t, v in zip(train_finals, val_finals)]

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
x = np.arange(len(labels))
ax1.plot(x, train_finals, "o-", color="#4472a8", label="Final train AUC", markersize=5)
ax1.plot(x, val_finals, "o--", color="#c97b2e", label="Final val AUC", markersize=5)
ax1.set_ylabel("ROC-AUC")
ax1.set_title("Jackknife sweep (15 runs, 40-epoch budget): final train vs. validation AUC", fontsize=11.5)
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(alpha=0.25)

colors = ["#7a7a7a" if lab == "all" else ("#4472a8" if lab.startswith("without") else "#c97b2e") for lab in labels]
ax2.bar(x, gaps, color=colors)
ax2.axhline(0, color="#333333", linewidth=0.8)
ax2.set_ylabel("Train \u2212 Val AUC gap")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
ax2.grid(alpha=0.25, axis="y")

from matplotlib.patches import Patch
legend_elems = [Patch(facecolor="#7a7a7a", label="all (baseline)"),
                Patch(facecolor="#4472a8", label="without-X"),
                Patch(facecolor="#c97b2e", label="only-X")]
ax2.legend(handles=legend_elems, loc="upper right", fontsize=8.5)

fig2.tight_layout()
out2 = f"{DATA_DIR}/cdr_pinn_jackknife_train_val_auc.png"
fig2.savefig(out2, dpi=150, facecolor="white")
print(f"Saved: {out2}")
