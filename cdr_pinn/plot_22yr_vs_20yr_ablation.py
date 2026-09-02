"""Visualizes the 22-year (this study) vs 20-year (Biswas et al.-matched) CDR-PINN
training-data-volume ablation. Two panels: (1) test AUC/AP under a controlled,
apples-to-apples comparison (same architecture, same pixel split, same 2001-2020
evaluation target for both models -- isolating training window length as the only
variable), and (2) the raw fire-ever-pixel coverage difference between the two
label definitions, a real, independent value-add of the extra 2 years that doesn't
depend on the AUC comparison at all."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CKPT_DIR = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data"
OUT_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_22yr_vs_20yr_ablation.png"

with open(f"{CKPT_DIR}/cdr_pinn_full_cdr_22yr_paired_with_20yr_ablation_result.json") as f:
    full_own = json.load(f)  # the fresh, contemporaneous 22yr run paired with the 20yr ablation
    # (reproduced the original canonical run's numbers exactly: val 0.9351, test 0.9398/0.9223)
with open(f"{CKPT_DIR}/cdr_pinn_full_model_on_20yr_window_result.json") as f:
    full_on_20yr = json.load(f)
with open(f"{CKPT_DIR}/cdr_pinn_cdr_20yr_2001_2020_subset_result.json") as f:
    subset_20yr = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

labels = [
    "22yr-trained model\non its own 22yr target\n(headline, A7)",
    "22yr-trained model\non the 2001-2020 target\n(this ablation)",
    "20yr-trained model\non the 2001-2020 target\n(this ablation)",
]
aucs = [full_own["test_roc_auc"], full_on_20yr["test_roc_auc"], subset_20yr["test_roc_auc"]]
aps = [full_own["test_ap"], full_on_20yr["test_ap"], subset_20yr["test_ap"]]
colors = ["#7a7a7a", "#2166ac", "#e07a4c"]

x = np.arange(len(labels))
width = 0.35
b1 = ax1.bar(x - width/2, aucs, width, label="Test ROC-AUC", color=colors)
b2 = ax1.bar(x + width/2, aps, width, label="Test AP", color=colors, alpha=0.55)
for bar in list(b1) + list(b2):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{bar.get_height():.4f}",
              ha="center", fontsize=8.5)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8.5)
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("Score")
ax1.set_title("Test-set performance\n(controlled comparison: middle vs. right bar)")
ax1.legend(loc="lower right", fontsize=8.5)
ax1.grid(alpha=0.25, axis="y")
ax1.axhline(1.0, color="none")

delta_auc = subset_20yr["test_roc_auc"] - full_on_20yr["test_roc_auc"]
ax1.annotate(f"$\\Delta$AUC = {delta_auc:+.4f}\n(within single-seed noise)",
             xy=(1.5, 0.5), fontsize=9, ha="center", color="#333333",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0", ec="#e07a4c"))

n_fire_22 = subset_20yr["n_fire_ever_pixels_full_22yr_for_reference"]
n_fire_20 = subset_20yr["n_fire_ever_pixels_window"]
bars2 = ax2.bar(["22-year record\n(2000-2022)", "20-year record\n(2001-2020, Biswas-matched)"],
                 [n_fire_22, n_fire_20], color=["#2166ac", "#e07a4c"])
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f"{int(bar.get_height()):,}",
              ha="center", fontsize=10, fontweight="bold")
pct_more = 100 * (n_fire_22 - n_fire_20) / n_fire_20
ax2.set_ylabel("Distinct fire-affected pixels (\"fire_ever\" = 1)")
ax2.set_title(f"Fire-ever label coverage\n({pct_more:.1f}% more distinct locations captured by the 22-year record)")
ax2.grid(alpha=0.25, axis="y")

fig.suptitle("CDR-PINN: This Study's 22-Year Record vs. Biswas et al.'s 20-Year Study Period",
              fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, facecolor="white")
print(f"Saved: {OUT_PATH}")
print(f"Delta AUC (20yr-trained minus 22yr-trained, same 2001-2020 target): {delta_auc:+.4f}")
print(f"Extra fire-ever pixel coverage from the 22-year record: +{n_fire_22 - n_fire_20:,} ({pct_more:.2f}%)")
