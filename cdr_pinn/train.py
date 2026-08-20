"""
CDR-PINN training + term-ablation study + held-out validation.

Autoregressive per-month rollout (u_0=0, the design's own IC), truncated BPTT in
WINDOW-month chunks. Three ablation configurations share this exact script,
differing only in which terms contribute to the optimized PDE residual
(model.CDRPINN.pde_residual's use_diffusion/use_advection/use_reaction flags):
'diffusion_only', 'diffusion_advection', 'full_cdr'.

Validation: an 80/20 random split of valid in-India pixels (seed=42, matching
this project's own established convention -- Step 7's own split). DATA losses
(monthly + terminal) are computed on TRAIN pixels only; the PDE-residual and
boundary losses are computed on ALL valid pixels regardless of split, since they
reference no label at all -- a genuine PINN-specific property (physics
constraints don't leak test-set supervision) worth exploiting, not an oversight.
After training, ROC-AUC/AP are computed on the held-out TEST pixels only,
against the real (binarized) fire_ever ground truth -- directly comparable to
Step 7's own RF/MaxEnt headline metric.
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score

from model import CDRPINN
from losses import data_loss_monthly, data_loss_terminal, pde_loss, bc_loss, ic_loss, AdaptiveLossBalancer
from build_monthly_stacks import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, TARGET_H, TARGET_W

DATA_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_monthly_stacks.npz"
CKPT_DIR = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data"

WINDOW = 24
N_EPOCHS = 80
WIDTH = 32
N_LAYERS = 4
MODES = 16
LR = 1e-3
SEED = 42

CONFIGS = {
    "diffusion_only":     dict(use_diffusion=True, use_advection=False, use_reaction=False),
    "diffusion_advection": dict(use_diffusion=True, use_advection=True,  use_reaction=False),
    "full_cdr":            dict(use_diffusion=True, use_advection=True,  use_reaction=True),
}


def build_masks(ndvi_f1, seed=SEED, test_frac=0.2):
    valid = ~np.isnan(ndvi_f1)
    up = np.roll(valid, 1, axis=0); down = np.roll(valid, -1, axis=0)
    left = np.roll(valid, 1, axis=1); right = np.roll(valid, -1, axis=1)
    boundary = valid & (~up | ~down | ~left | ~right)

    rng = np.random.RandomState(seed)
    valid_idx = np.argwhere(valid)
    n_test = int(len(valid_idx) * test_frac)
    perm = rng.permutation(len(valid_idx))
    test_idx = valid_idx[perm[:n_test]]
    train_idx = valid_idx[perm[n_test:]]

    train_mask = np.zeros_like(valid)
    train_mask[train_idx[:, 0], train_idx[:, 1]] = True
    test_mask = np.zeros_like(valid)
    test_mask[test_idx[:, 0], test_idx[:, 1]] = True
    return valid, boundary, train_mask, test_mask


def run(config_name):
    term_flags = CONFIGS[config_name]
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Config: {config_name} ({term_flags}) === Device: {device}")

    d = np.load(DATA_PATH)
    H, W = TARGET_H, TARGET_W
    n_months = len(d["months"])

    def t(x):
        return torch.tensor(np.nan_to_num(x, nan=0.0), dtype=torch.float32, device=device)

    ndvi_f1 = d["ndvi_f1"]
    valid_np, boundary_np, train_np, test_np = build_masks(ndvi_f1)
    valid_mask = torch.tensor(valid_np, dtype=torch.bool, device=device)
    boundary_mask = torch.tensor(boundary_np, dtype=torch.bool, device=device)
    train_data_mask = torch.tensor(valid_np & train_np, dtype=torch.bool, device=device)
    test_mask = torch.tensor(valid_np & test_np, dtype=torch.bool, device=device)
    print(f"Valid pixels: {valid_np.sum()} | train: {train_data_mask.sum().item()} | "
          f"test (held out from ALL data losses): {test_mask.sum().item()}")

    ndvi_f1_t = t(ndvi_f1)
    ndvi_anomaly_t = t(d["ndvi_anomaly"])
    forest_frac_t = t(d["forest_frac"])
    dryness_t = t(d["dryness_proxy"])
    slope_t = t(d["slope"])
    dist_roads_t = t(d["dist_roads"])
    elevation_t = t(d["elevation"])
    grad_e_x_t = t(d["grad_e_x"])
    grad_e_y_t = t(d["grad_e_y"])
    fire_indicator_t = t(d["fire_indicator"])
    fire_ever_frac_t = t(d["fire_ever_frac"])
    fire_ever_binary_np = (d["fire_ever_frac"] > 0).astype(np.float32)

    # inverse-frequency positive-class weight for the monthly BCE (measured directly
    # from train pixels, not the global rate, so it reflects exactly what this run
    # trains on) -- see losses.py's data_loss_monthly docstring for why this matters
    monthly_pos_rate = fire_indicator_t[:, train_data_mask].mean().item()
    pos_weight = (1.0 - monthly_pos_rate) / max(monthly_pos_rate, 1e-6)
    print(f"Monthly fire-positive rate (train pixels): {monthly_pos_rate:.4%} -> pos_weight={pos_weight:.2f}")

    lat_deg = np.linspace(LAT_MAX, LAT_MIN, H)
    lat_rad = torch.tensor(np.radians(lat_deg), dtype=torch.float32, device=device)
    dlat_rad = float(np.radians(abs(lat_deg[1] - lat_deg[0])))
    dlon_rad = float(np.radians((LON_MAX - LON_MIN) / W))

    def covariate_stack(ti):
        return torch.stack([
            ndvi_f1_t, ndvi_anomaly_t[ti], forest_frac_t, dryness_t[ti],
            slope_t, dist_roads_t, elevation_t,
        ], dim=0).unsqueeze(0)

    def physics_covariates(ti):
        return {
            "ndvi_f1": ndvi_f1_t.unsqueeze(0), "forest_frac": forest_frac_t.unsqueeze(0),
            "ndvi_anomaly": ndvi_anomaly_t[ti].unsqueeze(0),
            "grad_e_x": grad_e_x_t.unsqueeze(0), "grad_e_y": grad_e_y_t.unsqueeze(0),
            "dryness": dryness_t[ti].unsqueeze(0), "slope": slope_t.unsqueeze(0),
            "dist_roads": dist_roads_t.unsqueeze(0),
        }

    model = CDRPINN(n_static_channels=7, width=WIDTH, modes_h=MODES, modes_w=MODES, n_layers=N_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=LR)
    balancer = AdaptiveLossBalancer(["data", "pde", "bc", "ic"], update_every=5)

    history = []
    t_start = time.time()

    for epoch in range(N_EPOCHS):
        u = torch.zeros(1, 1, H, W, device=device)
        epoch_loss_sum = {"data": 0.0, "pde": 0.0, "bc": 0.0, "ic": 0.0, "total": 0.0}
        n_windows = 0
        last_window_traj = None

        for start in range(0, n_months - 1, WINDOW):
            end = min(start + WINDOW, n_months - 1)
            window_data, window_pde, window_bc = 0.0, 0.0, 0.0
            window_ic = ic_loss(u) if start == 0 else torch.zeros((), device=device)
            traj = [u.detach()]

            for ti in range(start, end):
                cov_stack = covariate_stack(ti)
                u_next = model(u, cov_stack)
                residual, _ = model.pde_residual(
                    u, u_next, dt_months=1.0, covariates=physics_covariates(ti),
                    lat_rad_1d=lat_rad, dlon_rad=dlon_rad, dlat_rad=dlat_rad,
                    **term_flags,
                )
                window_pde = window_pde + pde_loss(residual.squeeze(0), valid_mask)  # physics: full grid, no label leak
                mid = 0.5 * (u.squeeze(0).squeeze(0) + u_next.squeeze(0).squeeze(0))
                window_bc = window_bc + bc_loss(mid, lat_rad, dlon_rad, dlat_rad, boundary_mask)
                window_data = window_data + data_loss_monthly(
                    u_next.squeeze(0).squeeze(0), fire_indicator_t[ti + 1], train_data_mask,
                    pos_weight=pos_weight)  # TRAIN pixels only, class-imbalance corrected
                traj.append(u_next)
                u = u_next

            n_steps = end - start
            window_data, window_pde, window_bc = window_data / n_steps, window_pde / n_steps, window_bc / n_steps

            is_last_window = (end >= n_months - 1)
            if is_last_window:
                traj_stack = torch.cat(traj, dim=0).squeeze(1)
                window_terminal = data_loss_terminal(traj_stack, fire_ever_frac_t, train_data_mask, tau=5.0)
                window_data = 0.5 * window_data + 0.5 * window_terminal
                last_window_traj = traj_stack.detach()

            losses = {"data": window_data, "pde": window_pde, "bc": window_bc, "ic": window_ic}
            total, weights = balancer.combine(losses, list(model.parameters()))

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            u = u.detach()

            for k in ("data", "pde", "bc", "ic"):
                epoch_loss_sum[k] += float(losses[k].detach())
            epoch_loss_sum["total"] += float(total.detach())
            n_windows += 1

        avg = {k: v / n_windows for k, v in epoch_loss_sum.items()}
        history.append(avg)
        if (epoch + 1) % 10 == 0 or epoch == N_EPOCHS - 1:
            elapsed = time.time() - t_start
            print(f"[{config_name} | Epoch {epoch+1}/{N_EPOCHS}] total={avg['total']:.4e} "
                  f"data={avg['data']:.4e} pde={avg['pde']:.4e} bc={avg['bc']:.4e} (elapsed {elapsed:.1f}s)")

        if any(np.isnan(v) for v in avg.values()):
            print(f"STOPPING [{config_name}]: loss went NaN.")
            break

    train_time = time.time() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0

    # ---- Held-out evaluation: full forward rollout (no grad), LSE-pooled terminal
    #      score, ROC-AUC/AP on TEST pixels only, against real binary fire_ever ----
    model.eval()
    with torch.no_grad():
        u = torch.zeros(1, 1, H, W, device=device)
        max_score = torch.full((H, W), -1e9, device=device)
        for ti in range(n_months - 1):
            u_next = model(u, covariate_stack(ti))
            s = torch.sigmoid(u_next.squeeze(0).squeeze(0))
            max_score = torch.maximum(max_score, s)
            u = u_next
        pred_score = max_score.cpu().numpy()

    y_true = fire_ever_binary_np[test_np & valid_np]
    y_score = pred_score[test_np & valid_np]
    valid_eval = ~np.isnan(y_score) & ~np.isnan(y_true)
    y_true, y_score = y_true[valid_eval], y_score[valid_eval]

    if len(np.unique(y_true)) < 2:
        auc, ap = float("nan"), float("nan")
        print(f"WARNING [{config_name}]: held-out test set has only one class present, AUC undefined.")
    else:
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

    print(f"[{config_name}] HELD-OUT TEST: n={len(y_true)}, positive_frac={y_true.mean():.4f}, "
          f"ROC-AUC={auc:.4f}, AP={ap:.4f}")

    result = {
        "config": config_name, "term_flags": term_flags, "n_params": n_params,
        "n_epochs": N_EPOCHS, "train_time_sec": train_time, "peak_gpu_mem_mb": peak_mem,
        "final_total_loss": history[-1]["total"], "final_data_loss": history[-1]["data"],
        "final_pde_loss": history[-1]["pde"], "final_bc_loss": history[-1]["bc"],
        "held_out_n": int(len(y_true)), "held_out_positive_frac": float(y_true.mean()),
        "held_out_roc_auc": float(auc), "held_out_ap": float(ap),
    }

    ckpt_path = f"{CKPT_DIR}/cdr_pinn_{config_name}.pt"
    torch.save({"model_state": model.state_dict(), "history": history, "result": result}, ckpt_path)
    result_path = f"{CKPT_DIR}/cdr_pinn_{config_name}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {ckpt_path}, {result_path}\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    configs_to_run = list(CONFIGS.keys()) if args.config == "all" else [args.config]
    all_results = []
    for cfg in configs_to_run:
        all_results.append(run(cfg))

    if len(all_results) > 1:
        print("\n=== ABLATION SUMMARY ===")
        for r in all_results:
            print(f"  {r['config']:20s} ROC-AUC={r['held_out_roc_auc']:.4f}  AP={r['held_out_ap']:.4f}  "
                  f"train_time={r['train_time_sec']:.1f}s")
        with open(f"{CKPT_DIR}/cdr_pinn_ablation_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)
