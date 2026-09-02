"""
The STANDARD training protocol for CDR-PINN -- supersedes the ad-hoc train.py (no
validation set, no regularization search, no adaptive LR, no loss-curve diagnostics)
and the earlier train_standard_protocol.py draft (fixed cosine schedule, no
regularization, no early stopping, no plots). Standing requirements this satisfies
(feedback_ml_rigor_standing_requirements, 2026-08-21):

1. Real epoch-vs-error diagnostic plot (train + validation loss), saved to disk --
   not just terminal print statements.
2. Regularization selected from a validated search (hp_search_weight_decay.py), not a
   default accepted untested -- AdamW weight decay is the standard FNO/PINO knob;
   spectral mode truncation (16x16, unchanged) is the architecture's other, already-
   fixed capacity control.
3. Adaptive learning rate: ReduceLROnPlateau, which responds to observed validation-
   loss plateaus during training itself -- more genuinely "adaptive" than a fixed
   schedule shape (e.g. cosine) decided once by an earlier one-off comparison.
4. Early stopping on validation loss (patience-based) -- directly acts on
   overfitting/underfitting rather than just diagnosing it after the fact.
5. Genuine 65/15/20 train/val/test split (preprocessing.py); validation drives every
   decision (weight decay, LR reduction, stopping point); test is touched exactly once.
6. Uses the corrected forest_frac input (forest_frac_baseline, 2001) via
   preprocessing.py -- forest_frac_recent was dropped from Step 6's parquet 2026-08-21
   as a data-leakage fix.

Architecture note (standing requirement 6/7 -- explicit CNN-terminology description,
previously absent from every doc in this project): the FNO/PINO backbone has no spatial
kernel or stride in the CNN sense -- each spectral block truncates to a fixed 16x16
Fourier-mode window (a global spectral low-pass filter, not a local sliding window) and
mixes channels via a POINTWISE 1x1 convolution (kernel_size=1, stride=1, padding=0 --
literally a per-pixel linear layer applied identically at every spatial location, the
"local" residual path in every FNO block). Channel width is 32 throughout the 4 spectral
layers; input channel count is 7 (the static+time-varying covariate stack). There is NO
pooling layer anywhere in this architecture -- FNO operates at full grid resolution
end-to-end by design, since resolution-independence (Li et al., 2023) depends on never
downsampling the spatial grid; the closest functional analogue to pooling is the 16x16
Fourier-mode truncation itself, which discards high-spatial-frequency information
globally (a spectral operation) rather than locally pooling neighboring pixels.
"""
import json
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score

from model import CDRPINN
from losses import data_loss_monthly, data_loss_terminal, pde_loss, bc_loss, ic_loss, AdaptiveLossBalancer, lse_pool
from preprocessing import (
    build_masks_3way, load_tensors, covariate_stack, physics_covariates,
    grid_metadata, compute_pos_weight, SEED, TARGET_H, TARGET_W,
)

CKPT_DIR = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data"
WINDOW = 24
N_EPOCHS = 80
WIDTH = 32
N_LAYERS = 4
MODES = 16
LR = 1e-3
CONFIG_NAME = "full_cdr_standard_protocol"
VAL_EVERY = 5           # validation rollout is expensive (full 266-month sequential pass); check periodically, not every epoch
EARLY_STOP_PATIENCE = 4  # in units of VAL_EVERY checks (=20 epochs) with no val_loss improvement


def load_weight_decay():
    try:
        with open(f"{CKPT_DIR}/cdr_pinn_weight_decay_search.json") as f:
            wd = json.load(f)["winner_weight_decay"]
        print(f"Using validated weight_decay={wd:.0e} from hp_search_weight_decay.py")
        return wd
    except FileNotFoundError:
        print("WARNING: no weight_decay search result found, defaulting to 0.0 (run hp_search_weight_decay.py first for a validated value)")
        return 0.0


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Standard protocol: {CONFIG_NAME} === Device: {device}")

    weight_decay = load_weight_decay()

    tensors, fire_ever_binary_np, ndvi_f1_np, n_months = load_tensors(device)
    valid_np, boundary_np, train_np, val_np, test_np = build_masks_3way(ndvi_f1_np)
    print(f"Split sizes (valid pixels): train={train_np.sum()}, val={val_np.sum()}, test={test_np.sum()}")

    H, W = TARGET_H, TARGET_W
    valid_mask = torch.tensor(valid_np, dtype=torch.bool, device=device)
    boundary_mask = torch.tensor(boundary_np, dtype=torch.bool, device=device)
    train_data_mask = torch.tensor(valid_np & train_np, dtype=torch.bool, device=device)

    monthly_pos_rate, pos_weight = compute_pos_weight(tensors, train_data_mask)
    print(f"Monthly fire-positive rate (train pixels): {monthly_pos_rate:.4%} -> pos_weight={pos_weight:.2f}")

    lat_rad, dlat_rad, dlon_rad = grid_metadata(device)

    model = CDRPINN(n_static_channels=7, width=WIDTH, modes_h=MODES, modes_w=MODES, n_layers=N_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)  # patience in units of VAL_EVERY checks
    balancer = AdaptiveLossBalancer(["data", "pde", "bc", "ic"], update_every=5)

    def compute_rollout():
        """The expensive part (a full sequential 265-month forward pass) -- computed
        ONCE per validation check and reused to score against every mask (train/val/
        test), rather than re-rolling-out per mask."""
        model.eval()
        with torch.no_grad():
            u = torch.zeros(1, 1, H, W, device=device)
            scores = []
            for ti in range(n_months - 1):
                u_next = model(u, covariate_stack(tensors, ti))
                scores.append(torch.sigmoid(u_next.squeeze(0).squeeze(0)))
                u = u_next
            pooled = lse_pool(torch.stack(scores, dim=0), dim=0, tau=5.0)
        model.train()
        return pooled

    def score_from_pooled(pooled, mask_np):
        m_np = valid_np & mask_np
        y_true = fire_ever_binary_np[m_np]
        y_score = pooled.cpu().numpy()[m_np]
        ok = ~np.isnan(y_score) & ~np.isnan(y_true)
        y_true, y_score = y_true[ok], y_score[ok]
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        m = torch.tensor(m_np, dtype=torch.bool, device=device)
        y_true_t = torch.tensor(fire_ever_binary_np, dtype=torch.float32, device=device)
        p = pooled.clamp(1e-6, 1 - 1e-6)
        bce = -(y_true_t * torch.log(p) + (1 - y_true_t) * torch.log(1 - p))
        loss = bce[m].mean().item()
        return loss, auc, ap

    def rollout_and_score(mask_np):
        """Convenience wrapper for one-off scoring (e.g. the final test evaluation)."""
        return score_from_pooled(compute_rollout(), mask_np)

    train_loss_history, val_loss_history, val_auc_history, train_auc_history, val_epochs = [], [], [], [], []
    # Checkpoint selection and early stopping are driven by VALIDATION AUC, not loss --
    # a first pass using val_loss found the two diverge for this model (loss plateaus
    # while AUC keeps climbing, a known effect under heavy class-imbalance reweighting):
    # stopping on loss would have thrown away a still-improving model. AUC is also the
    # metric this whole paper reports and compares across models, so selecting on it
    # directly is the more defensible choice, not just the empirically better one.
    # ReduceLROnPlateau still monitors loss (a reasonable, standard split of concerns --
    # LR scheduling and checkpoint selection do not need the same target metric).
    best_val_auc, best_state, epochs_without_improvement = -1.0, None, 0
    t_start = time.time()

    for epoch in range(N_EPOCHS):
        u = torch.zeros(1, 1, H, W, device=device)
        epoch_total = 0.0
        n_windows = 0
        for start in range(0, n_months - 1, WINDOW):
            end = min(start + WINDOW, n_months - 1)
            window_ic = ic_loss(u) if start == 0 else torch.zeros((), device=device)
            traj = [u.detach()]
            window_pde, window_bc, window_data = 0.0, 0.0, 0.0
            for ti in range(start, end):
                u_next = model(u, covariate_stack(tensors, ti))
                residual, _ = model.pde_residual(
                    u, u_next, dt_months=1.0, covariates=physics_covariates(tensors, ti),
                    lat_rad_1d=lat_rad, dlon_rad=dlon_rad, dlat_rad=dlat_rad,
                    use_diffusion=True, use_advection=True, use_reaction=True)
                window_pde = window_pde + pde_loss(residual.squeeze(0), valid_mask)
                mid = 0.5 * (u.squeeze(0).squeeze(0) + u_next.squeeze(0).squeeze(0))
                window_bc = window_bc + bc_loss(mid, lat_rad, dlon_rad, dlat_rad, boundary_mask)
                window_data = window_data + data_loss_monthly(
                    u_next.squeeze(0).squeeze(0), tensors["fire_indicator"][ti + 1], train_data_mask,
                    pos_weight=pos_weight)
                traj.append(u_next)
                u = u_next
            n_steps = end - start
            window_pde, window_bc, window_data = window_pde / n_steps, window_bc / n_steps, window_data / n_steps
            if end >= n_months - 1:
                traj_stack = torch.cat(traj, dim=0).squeeze(1)
                window_terminal = data_loss_terminal(traj_stack, tensors["fire_ever_frac"], train_data_mask, tau=5.0)
                window_data = 0.5 * window_data + 0.5 * window_terminal
            losses = {"data": window_data, "pde": window_pde, "bc": window_bc, "ic": window_ic}
            total, _ = balancer.combine(losses, list(model.parameters()))
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            u = u.detach()
            epoch_total += float(total.detach())
            n_windows += 1

        epoch_train_loss = epoch_total / n_windows
        train_loss_history.append(epoch_train_loss)

        if torch.isnan(total):
            print("STOPPING: loss went NaN.")
            break

        if (epoch + 1) % VAL_EVERY == 0 or epoch == N_EPOCHS - 1:
            pooled = compute_rollout()  # one rollout, scored against both train and val masks below
            val_loss, val_auc, val_ap = score_from_pooled(pooled, val_np)
            _, train_auc, _ = score_from_pooled(pooled, train_np)
            val_loss_history.append(val_loss)
            val_auc_history.append(val_auc)
            train_auc_history.append(train_auc)
            val_epochs.append(epoch + 1)
            scheduler.step(val_loss)  # LR scheduling still monitors loss -- a separate concern from checkpoint selection
            print(f"[{CONFIG_NAME} | Epoch {epoch+1}/{N_EPOCHS}] train_loss={epoch_train_loss:.4e} "
                  f"train_auc={train_auc:.4f} val_loss={val_loss:.4e} val_auc={val_auc:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} (elapsed {time.time()-t_start:.1f}s)")

            if val_auc > best_val_auc + 1e-4:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                    print(f"EARLY STOPPING at epoch {epoch+1}: no val_auc improvement for "
                          f"{EARLY_STOP_PATIENCE * VAL_EVERY} epochs (best={best_val_auc:.4f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Restored best-validation-AUC checkpoint for final evaluation.")

    train_time = time.time() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0

    val_loss_final, val_auc, val_ap = rollout_and_score(val_np)
    test_loss_final, test_auc, test_ap = rollout_and_score(test_np)
    print(f"[{CONFIG_NAME}] FINAL VALIDATION: ROC-AUC={val_auc:.4f}, AP={val_ap:.4f}")
    print(f"[{CONFIG_NAME}] FINAL TEST (untouched by any decision): ROC-AUC={test_auc:.4f}, AP={test_ap:.4f}")

    # Epoch-vs-error diagnostic plot (standing requirement 1) -- two panels: loss
    # (what's optimized) and AUC (what's actually selected on and reported), since
    # this run found the two diverge for this model and both are worth showing.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train loss (composite, per-epoch)", color="#1f77b4")
        ax1.plot(val_epochs, val_loss_history, label="Validation loss (BCE)", color="#d62728", marker="o")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Loss (optimized quantity)")
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(val_epochs, train_auc_history, label="Train ROC-AUC", color="#1f77b4", marker="s")
        ax2.plot(val_epochs, val_auc_history, label="Validation ROC-AUC", color="#2ca02c", marker="o")
        best_ep = val_epochs[val_auc_history.index(max(val_auc_history))]
        ax2.axvline(best_ep, color="gray", linestyle="--", alpha=0.6, label=f"Selected checkpoint (epoch {best_ep})")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("ROC-AUC")
        ax2.set_title("Train vs. Validation AUC (overfitting/underfitting diagnostic)")
        ax2.legend(); ax2.grid(alpha=0.3)

        fig.suptitle(f"CDR-PINN training diagnostics: {CONFIG_NAME}")
        fig.tight_layout()
        plot_path = f"{CKPT_DIR}/cdr_pinn_{CONFIG_NAME}_loss_curve.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved loss-curve plot: {plot_path}")
    except Exception as e:
        print(f"WARNING: could not save loss-curve plot: {e}")

    result = {
        "config": CONFIG_NAME,
        "protocol": "3-way split (65/15/20), AdamW+validated weight_decay, ReduceLROnPlateau (loss-monitored), early stopping (AUC-monitored), LSE-pool eval",
        "weight_decay": weight_decay, "n_params": n_params, "n_epochs_run": len(train_loss_history),
        "n_epochs_budget": N_EPOCHS, "train_time_sec": train_time, "peak_gpu_mem_mb": peak_mem,
        "val_roc_auc": float(val_auc), "val_ap": float(val_ap),
        "test_roc_auc": float(test_auc), "test_ap": float(test_ap),
        "train_loss_history": train_loss_history, "val_loss_history": val_loss_history,
        "val_auc_history": val_auc_history, "train_auc_history": train_auc_history, "val_epochs": val_epochs,
    }
    ckpt_path = f"{CKPT_DIR}/cdr_pinn_{CONFIG_NAME}.pt"
    torch.save({"model_state": model.state_dict(), "result": result}, ckpt_path)
    result_path = f"{CKPT_DIR}/cdr_pinn_{CONFIG_NAME}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {ckpt_path}, {result_path}")


if __name__ == "__main__":
    main()
