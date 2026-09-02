"""
Controlled ablation: does this study's full 22-year record (Nov 2000-Dec 2022, 266
months) improve CDR-PINN over a 20-year record matching Biswas et al.'s exact 2001-2020
study period (240 months, month indices 2:242 of the full stack)? Everything except the
temporal training window is held identical to train_standard_protocol.py: same
architecture (WIDTH=32, N_LAYERS=4, MODES=16), same seed=42 65/15/20 pixel split (so the
train/val/test PIXEL partition is byte-identical to the full-period run -- only the
temporal extent of supervision differs), same AdamW + validated weight_decay, same
ReduceLROnPlateau (loss-monitored) + early stopping (val-AUC-monitored, patience=4),
same LSE-pool (tau=5.0) terminal aggregation.

The terminal label is recomputed for this window specifically -- fire_ever_2001_2020,
"did this pixel burn within 2001-2020" -- rather than reusing the full-period fire_ever
label (which would leak information from the 2 years this run never sees). This is the
one deliberate content difference from train_standard_protocol.py; every other choice is
copied exactly so a test-AUC delta between the two runs is attributable to training data
volume, not a confound.

u_0=0 is reset at the window's own first month (Jan 2001), not carried over from any
2000 state -- the CDR PDE's own IC compatibility argument (Methodology Section 9.3)
applies identically at any t=0 origin.
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
    grid_metadata, SEED, TARGET_H, TARGET_W, DATA_PATH,
)

CKPT_DIR = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data"
WINDOW = 24
N_EPOCHS = 80
WIDTH = 32
N_LAYERS = 4
MODES = 16
LR = 1e-3
CONFIG_NAME = "cdr_20yr_2001_2020_subset"
VAL_EVERY = 5
EARLY_STOP_PATIENCE = 4

WINDOW_START_IDX = 2    # 2001-01 in the full 266-month array
WINDOW_END_IDX = 242    # exclusive -- 2020-12 is index 241, so slice [2:242] = 240 months


def load_weight_decay():
    try:
        with open(f"{CKPT_DIR}/cdr_pinn_weight_decay_search.json") as f:
            wd = json.load(f)["winner_weight_decay"]
        print(f"Using the SAME validated weight_decay={wd:.0e} as the full-period run (hp_search_weight_decay.py) "
              f"-- not re-searched, to keep this a controlled single-variable (data volume) ablation.")
        return wd
    except FileNotFoundError:
        print("WARNING: no weight_decay search result found, defaulting to 0.0")
        return 0.0


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== 20-year subset ablation: {CONFIG_NAME} === Device: {device}")

    weight_decay = load_weight_decay()

    tensors, fire_ever_binary_full_np, ndvi_f1_np, n_months_full = load_tensors(device)
    d = np.load(DATA_PATH)
    months = d["months"]
    print(f"Full stack: {n_months_full} months ({months[0]} .. {months[-1]})")
    window_months = months[WINDOW_START_IDX:WINDOW_END_IDX]
    n_months = WINDOW_END_IDX - WINDOW_START_IDX
    print(f"Restricted window: {n_months} months ({window_months[0]} .. {window_months[-1]}) "
          f"-- matches Biswas et al.'s exact 2001-2020 study period")
    assert n_months == 240, f"expected exactly 240 months, got {n_months}"

    # Recompute the terminal label for THIS window only -- "did this pixel burn within
    # 2001-2020", not the full-period fire_ever (which would leak 2000/2021-2022 fires
    # this run never trains on).
    fire_indicator_window = tensors["fire_indicator"][WINDOW_START_IDX:WINDOW_END_IDX]
    fire_ever_frac_window = fire_indicator_window.max(dim=0).values  # (H, W), in {0,1}
    fire_ever_binary_np = fire_ever_frac_window.cpu().numpy()
    n_fire_full = int((fire_ever_binary_full_np > 0).sum())
    n_fire_window = int((fire_ever_binary_np > 0).sum())
    print(f"Fire-ever pixels: full 22yr record = {n_fire_full:,}, 2001-2020 window = {n_fire_window:,} "
          f"({100*(n_fire_full - n_fire_window)/max(n_fire_full,1):.2f}% fewer fire-positive pixels "
          f"when the 2 extra years are excluded)")

    valid_np, boundary_np, train_np, val_np, test_np = build_masks_3way(ndvi_f1_np)
    print(f"Split sizes (IDENTICAL seed=42 pixel partition as the full-period run): "
          f"train={train_np.sum()}, val={val_np.sum()}, test={test_np.sum()}")

    H, W = TARGET_H, TARGET_W
    valid_mask = torch.tensor(valid_np, dtype=torch.bool, device=device)
    boundary_mask = torch.tensor(boundary_np, dtype=torch.bool, device=device)
    train_data_mask = torch.tensor(valid_np & train_np, dtype=torch.bool, device=device)

    # pos_weight measured from THIS window's train pixels only (matches
    # compute_pos_weight's own convention of measuring from what's actually trained on)
    monthly_pos_rate = fire_indicator_window[:, train_data_mask].mean().item()
    pos_weight = (1.0 - monthly_pos_rate) / max(monthly_pos_rate, 1e-6)
    print(f"Monthly fire-positive rate (train pixels, 2001-2020 window): {monthly_pos_rate:.4%} -> pos_weight={pos_weight:.2f}")

    lat_rad, dlat_rad, dlon_rad = grid_metadata(device)

    def cov_stack(ti):
        # ti is a window-local index (0..239); shift by WINDOW_START_IDX for the
        # time-varying channels (ndvi_anomaly, dryness), which are indexed into the
        # FULL tensors dict since preprocessing.py loads the whole archive.
        full_ti = WINDOW_START_IDX + ti
        return torch.stack([
            tensors["ndvi_f1"], tensors["ndvi_anomaly"][full_ti], tensors["forest_frac"],
            tensors["dryness"][full_ti], tensors["slope"], tensors["dist_roads"], tensors["elevation"],
        ], dim=0).unsqueeze(0)

    def phys_cov(ti):
        full_ti = WINDOW_START_IDX + ti
        return {
            "ndvi_f1": tensors["ndvi_f1"].unsqueeze(0), "forest_frac": tensors["forest_frac"].unsqueeze(0),
            "ndvi_anomaly": tensors["ndvi_anomaly"][full_ti].unsqueeze(0),
            "grad_e_x": tensors["grad_e_x"].unsqueeze(0), "grad_e_y": tensors["grad_e_y"].unsqueeze(0),
            "dryness": tensors["dryness"][full_ti].unsqueeze(0), "slope": tensors["slope"].unsqueeze(0),
            "dist_roads": tensors["dist_roads"].unsqueeze(0),
        }

    model = CDRPINN(n_static_channels=7, width=WIDTH, modes_h=MODES, modes_w=MODES, n_layers=N_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    balancer = AdaptiveLossBalancer(["data", "pde", "bc", "ic"], update_every=5)

    def rollout_and_score(mask_np):
        model.eval()
        with torch.no_grad():
            u = torch.zeros(1, 1, H, W, device=device)
            scores = []
            for ti in range(n_months - 1):
                u_next = model(u, cov_stack(ti))
                scores.append(torch.sigmoid(u_next.squeeze(0).squeeze(0)))
                u = u_next
            pooled = lse_pool(torch.stack(scores, dim=0), dim=0, tau=5.0)
        model.train()
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

    train_loss_history, val_loss_history, val_auc_history, val_epochs = [], [], [], []
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
                u_next = model(u, cov_stack(ti))
                residual, _ = model.pde_residual(
                    u, u_next, dt_months=1.0, covariates=phys_cov(ti),
                    lat_rad_1d=lat_rad, dlon_rad=dlon_rad, dlat_rad=dlat_rad,
                    use_diffusion=True, use_advection=True, use_reaction=True)
                window_pde = window_pde + pde_loss(residual.squeeze(0), valid_mask)
                mid = 0.5 * (u.squeeze(0).squeeze(0) + u_next.squeeze(0).squeeze(0))
                window_bc = window_bc + bc_loss(mid, lat_rad, dlon_rad, dlat_rad, boundary_mask)
                window_data = window_data + data_loss_monthly(
                    u_next.squeeze(0).squeeze(0), fire_indicator_window[ti + 1], train_data_mask,
                    pos_weight=pos_weight)
                traj.append(u_next)
                u = u_next
            n_steps = end - start
            window_pde, window_bc, window_data = window_pde / n_steps, window_bc / n_steps, window_data / n_steps
            if end >= n_months - 1:
                traj_stack = torch.cat(traj, dim=0).squeeze(1)
                window_terminal = data_loss_terminal(traj_stack, fire_ever_frac_window, train_data_mask, tau=5.0)
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
            val_loss, val_auc, val_ap = rollout_and_score(val_np)
            val_loss_history.append(val_loss)
            val_auc_history.append(val_auc)
            val_epochs.append(epoch + 1)
            scheduler.step(val_loss)
            print(f"[{CONFIG_NAME} | Epoch {epoch+1}/{N_EPOCHS}] train_loss={epoch_train_loss:.4e} "
                  f"val_loss={val_loss:.4e} val_auc={val_auc:.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
                  f"(elapsed {time.time()-t_start:.1f}s)")

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

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train loss (composite, per-epoch)", color="#1f77b4")
        ax1.plot(val_epochs, val_loss_history, label="Validation loss (BCE)", color="#d62728", marker="o")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Loss (optimized quantity) -- 2001-2020 subset")
        ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(val_epochs, val_auc_history, label="Validation ROC-AUC", color="#2ca02c", marker="o")
        best_ep = val_epochs[val_auc_history.index(max(val_auc_history))]
        ax2.axvline(best_ep, color="gray", linestyle="--", alpha=0.6, label=f"Selected checkpoint (epoch {best_ep})")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("ROC-AUC")
        ax2.set_title("Validation AUC (selection criterion) -- 2001-2020 subset")
        ax2.legend(); ax2.grid(alpha=0.3)

        fig.suptitle(f"CDR-PINN training diagnostics: {CONFIG_NAME} (20-year ablation)")
        fig.tight_layout()
        plot_path = f"{CKPT_DIR}/cdr_pinn_{CONFIG_NAME}_loss_curve.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved loss-curve plot: {plot_path}")
    except Exception as e:
        print(f"WARNING: could not save loss-curve plot: {e}")

    result = {
        "config": CONFIG_NAME,
        "protocol": "IDENTICAL to full_cdr_standard_protocol except training window restricted to 2001-2020 "
                    "(240 months, matching Biswas et al. exactly) and terminal label recomputed for that window",
        "window_start_month": str(window_months[0]), "window_end_month": str(window_months[-1]),
        "n_months_trained": n_months, "n_fire_ever_pixels_window": n_fire_window,
        "n_fire_ever_pixels_full_22yr_for_reference": n_fire_full,
        "weight_decay": weight_decay, "n_params": n_params, "n_epochs_run": len(train_loss_history),
        "n_epochs_budget": N_EPOCHS, "train_time_sec": train_time, "peak_gpu_mem_mb": peak_mem,
        "val_roc_auc": float(val_auc), "val_ap": float(val_ap),
        "test_roc_auc": float(test_auc), "test_ap": float(test_ap),
        "train_loss_history": train_loss_history, "val_loss_history": val_loss_history,
        "val_auc_history": val_auc_history, "val_epochs": val_epochs,
    }
    ckpt_path = f"{CKPT_DIR}/cdr_pinn_{CONFIG_NAME}.pt"
    torch.save({"model_state": model.state_dict(), "result": result}, ckpt_path)
    result_path = f"{CKPT_DIR}/cdr_pinn_{CONFIG_NAME}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {ckpt_path}, {result_path}")


if __name__ == "__main__":
    main()
