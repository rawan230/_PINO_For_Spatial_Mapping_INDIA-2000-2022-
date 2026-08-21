"""Regenerates the standard-protocol loss-curve plot from the saved result JSON, with
train loss (composite, includes PDE/BC/IC terms, scale ~O(10)) and validation loss
(pure BCE, scale ~O(0.1)) on separate twin y-axes -- plotting them on one shared axis
made the validation curve unreadable (a flat line pinned near zero relative to the much
larger composite scale). No retraining needed; this is a pure visualization fix."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_full_cdr_standard_protocol_result.json"
PLOT_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_full_cdr_standard_protocol_loss_curve.png"

with open(RESULT_PATH) as f:
    r = json.load(f)

train_loss_history = r["train_loss_history"]
val_loss_history = r["val_loss_history"]
val_auc_history = r["val_auc_history"]
val_epochs = r["val_epochs"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

l1, = ax1.plot(range(1, len(train_loss_history) + 1), train_loss_history,
                label="Train loss (composite: data+PDE+BC+IC)", color="#1f77b4")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train loss (composite)", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")

ax1b = ax1.twinx()
l2, = ax1b.plot(val_epochs, val_loss_history, label="Validation loss (BCE)", color="#d62728", marker="o")
ax1b.set_ylabel("Validation loss (BCE)", color="#d62728")
ax1b.tick_params(axis="y", labelcolor="#d62728")

ax1.set_title("Loss (optimized quantity) -- separate scales")
ax1.grid(alpha=0.3)
ax1.legend(handles=[l1, l2], loc="upper right")

ax2.plot(val_epochs, val_auc_history, label="Validation ROC-AUC", color="#2ca02c", marker="o")
best_ep = val_epochs[val_auc_history.index(max(val_auc_history))]
ax2.axvline(best_ep, color="gray", linestyle="--", alpha=0.6, label=f"Selected checkpoint (epoch {best_ep})")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("ROC-AUC")
ax2.set_title("Validation AUC (selection criterion)")
ax2.legend()
ax2.grid(alpha=0.3)

fig.suptitle(f"CDR-PINN training diagnostics: {r['config']}  "
             f"(final test AUC={r['test_roc_auc']:.4f})")
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=150)
print(f"Saved: {PLOT_PATH}")
