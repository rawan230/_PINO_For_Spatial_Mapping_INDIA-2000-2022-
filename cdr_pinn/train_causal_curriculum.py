"""
Two PINN-literature techniques not yet tried, tested directly against the
established full_cdr Track-A baseline (AUC=0.9406), rather than assumed to help:

1. Causal time-weighting (Wang, Sankaran & Perdikaris, 2022, "Respecting causality
   is all you need for training physics-informed neural networks", Computer Methods
   in Applied Mechanics and Engineering, 421:116813 [cite-verify]): standard PINN
   training treats every timestep's residual loss equally, which can let the model
   fit a LATER month's residual well while violating an EARLIER one -- physically
   backwards for a time-marching system. Fix: weight month i's residual loss by
   w_i = exp(-eps * cumulative residual loss from months < i, within the current
   window), so the optimizer must reduce earlier-time error before later-time error
   is allowed to contribute much gradient.

2. Staged curriculum learning: rather than training with all three CDR terms
   active from epoch 1 (as every previous run in this study did), progressively
   unlock advection then reaction partway through training -- distinct from the
   term-ABLATION study (which trains three separate models to measure each term's
   final contribution); this trains ONE model that curriculum-learns the full
   equation.

Run with --mode causal, --mode curriculum, or --mode both (default) to test each
independently against the same Track-A-equivalent split/seed.
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score

from model import CDRPINN
from losses import data_loss_monthly, data_loss_terminal, pde_loss, bc_loss, ic_loss, AdaptiveLossBalancer, lse_pool
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
CAUSAL_EPS = 1.0          # causality-strictness; larger = stricter temporal ordering enforced
CURRICULUM_ADVECTION_EPOCH = 15   # advection term switches on after this epoch
CURRICULUM_REACTION_EPOCH = 35    # reaction term switches on after this epoch


def build_masks(ndvi_f1, seed=SEED, test_frac=0.2):
    valid = ~np.isnan(ndvi_f1)
    up = np.roll(valid, 1, axis=0); down = np.roll(valid, -1, axis=0)
    left = np.roll(valid, 1, axis=1); right = np.roll(valid, -1, axis=1)
    boundary = valid & (~up | ~down | ~left | ~right)
    rng = np.random.RandomState(seed)
    valid_idx = np.argwhere(valid)
    n_test = int(len(valid_idx) * test_frac)
    perm = rng.permutation(len(valid_idx))
    test_idx, train_idx = valid_idx[perm[:n_test]], valid_idx[perm[n_test:]]
    train_mask = np.zeros_like(valid); train_mask[train_idx[:, 0], train_idx[:, 1]] = True
    test_mask = np.zeros_like(valid); test_mask[test_idx[:, 0], test_idx[:, 1]] = True
    return valid, boundary, train_mask, test_mask


def run(mode):
    """mode: 'causal', 'curriculum', or 'baseline' (re-verification, no new technique)."""
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== mode={mode} === Device: {device}")

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
    fire_ever_binary_np = (d["fire_ever_frac"] > 0).astype(np.float32)

    tens = dict(
        ndvi_f1=t(ndvi_f1), ndvi_anomaly=t(d["ndvi_anomaly"]), forest_frac=t(d["forest_frac"]),
        dryness=t(d["dryness_proxy"]), slope=t(d["slope"]), dist_roads=t(d["dist_roads"]),
        elevation=t(d["elevation"]), grad_e_x=t(d["grad_e_x"]), grad_e_y=t(d["grad_e_y"]),
        fire_indicator=t(d["fire_indicator"]), fire_ever_frac=t(d["fire_ever_frac"]),
    )

    lat_deg = np.linspace(LAT_MAX, LAT_MIN, H)
    lat_rad = torch.tensor(np.radians(lat_deg), dtype=torch.float32, device=device)
    dlat_rad = float(np.radians(abs(lat_deg[1] - lat_deg[0])))
    dlon_rad = float(np.radians((LON_MAX - LON_MIN) / W))

    monthly_pos_rate = tens["fire_indicator"][:, train_data_mask].mean().item()
    pos_weight = (1.0 - monthly_pos_rate) / max(monthly_pos_rate, 1e-6)

    def covariate_stack(ti):
        return torch.stack([
            tens["ndvi_f1"], tens["ndvi_anomaly"][ti], tens["forest_frac"], tens["dryness"][ti],
            tens["slope"], tens["dist_roads"], tens["elevation"],
        ], dim=0).unsqueeze(0)

    def physics_covariates(ti):
        return {
            "ndvi_f1": tens["ndvi_f1"].unsqueeze(0), "forest_frac": tens["forest_frac"].unsqueeze(0),
            "ndvi_anomaly": tens["ndvi_anomaly"][ti].unsqueeze(0),
            "grad_e_x": tens["grad_e_x"].unsqueeze(0), "grad_e_y": tens["grad_e_y"].unsqueeze(0),
            "dryness": tens["dryness"][ti].unsqueeze(0), "slope": tens["slope"].unsqueeze(0),
            "dist_roads": tens["dist_roads"].unsqueeze(0),
        }

    model = CDRPINN(n_static_channels=7, width=WIDTH, modes_h=MODES, modes_w=MODES, n_layers=N_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    balancer = AdaptiveLossBalancer(["data", "pde", "bc", "ic"], update_every=5)

    t_start = time.time()
    history = []
    for epoch in range(N_EPOCHS):
        if mode == "curriculum":
            use_advection = epoch >= CURRICULUM_ADVECTION_EPOCH
            use_reaction = epoch >= CURRICULUM_REACTION_EPOCH
        else:
            use_advection, use_reaction = True, True
        term_flags = dict(use_diffusion=True, use_advection=use_advection, use_reaction=use_reaction)

        u = torch.zeros(1, 1, H, W, device=device)
        epoch_loss_sum = {"data": 0.0, "pde": 0.0, "bc": 0.0, "ic": 0.0, "total": 0.0}
        n_windows = 0

        for start in range(0, n_months - 1, WINDOW):
            end = min(start + WINDOW, n_months - 1)
            window_ic = ic_loss(u) if start == 0 else torch.zeros((), device=device)
            traj = [u.detach()]

            if mode == "causal":
                cumulative_residual = 0.0  # running, detached -- drives the causal weight only
                window_pde_weighted, window_bc, window_data = 0.0, 0.0, 0.0
                for ti in range(start, end):
                    cov_stack = covariate_stack(ti)
                    u_next = model(u, cov_stack)
                    residual, _ = model.pde_residual(
                        u, u_next, dt_months=1.0, covariates=physics_covariates(ti),
                        lat_rad_1d=lat_rad, dlon_rad=dlon_rad, dlat_rad=dlat_rad, **term_flags)
                    step_pde_loss = pde_loss(residual.squeeze(0), valid_mask)
                    causal_weight = float(np.exp(-CAUSAL_EPS * cumulative_residual))
                    window_pde_weighted = window_pde_weighted + causal_weight * step_pde_loss
                    cumulative_residual += step_pde_loss.detach().item()

                    mid = 0.5 * (u.squeeze(0).squeeze(0) + u_next.squeeze(0).squeeze(0))
                    window_bc = window_bc + bc_loss(mid, lat_rad, dlon_rad, dlat_rad, boundary_mask)
                    window_data = window_data + data_loss_monthly(
                        u_next.squeeze(0).squeeze(0), tens["fire_indicator"][ti + 1], train_data_mask,
                        pos_weight=pos_weight)
                    traj.append(u_next)
                    u = u_next
                n_steps = end - start
                window_pde = window_pde_weighted / n_steps
                window_bc = window_bc / n_steps
                window_data = window_data / n_steps
            else:
                window_pde, window_bc, window_data = 0.0, 0.0, 0.0
                for ti in range(start, end):
                    cov_stack = covariate_stack(ti)
                    u_next = model(u, cov_stack)
                    residual, _ = model.pde_residual(
                        u, u_next, dt_months=1.0, covariates=physics_covariates(ti),
                        lat_rad_1d=lat_rad, dlon_rad=dlon_rad, dlat_rad=dlat_rad, **term_flags)
                    window_pde = window_pde + pde_loss(residual.squeeze(0), valid_mask)
                    mid = 0.5 * (u.squeeze(0).squeeze(0) + u_next.squeeze(0).squeeze(0))
                    window_bc = window_bc + bc_loss(mid, lat_rad, dlon_rad, dlat_rad, boundary_mask)
                    window_data = window_data + data_loss_monthly(
                        u_next.squeeze(0).squeeze(0), tens["fire_indicator"][ti + 1], train_data_mask,
                        pos_weight=pos_weight)
                    traj.append(u_next)
                    u = u_next
                n_steps = end - start
                window_pde, window_bc, window_data = window_pde / n_steps, window_bc / n_steps, window_data / n_steps

            is_last_window = (end >= n_months - 1)
            if is_last_window:
                traj_stack = torch.cat(traj, dim=0).squeeze(1)
                window_terminal = data_loss_terminal(traj_stack, tens["fire_ever_frac"], train_data_mask, tau=5.0)
                window_data = 0.5 * window_data + 0.5 * window_terminal

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
            print(f"[{mode} | epoch {epoch+1}/{N_EPOCHS}] total={avg['total']:.4e} data={avg['data']:.4e} "
                  f"pde={avg['pde']:.4e} adv={use_advection} react={use_reaction} "
                  f"(elapsed {time.time()-t_start:.1f}s)")
        if any(np.isnan(v) for v in avg.values()):
            print(f"STOPPING [{mode}]: NaN.")
            break

    train_time = time.time() - t_start

    model.eval()
    with torch.no_grad():
        u = torch.zeros(1, 1, H, W, device=device)
        scores = []
        for ti in range(n_months - 1):
            u_next = model(u, covariate_stack(ti))
            scores.append(torch.sigmoid(u_next.squeeze(0).squeeze(0)))
            u = u_next
        pooled = lse_pool(torch.stack(scores, dim=0), dim=0, tau=5.0).cpu().numpy()

    eval_mask = valid_np & test_np
    y_true = fire_ever_binary_np[eval_mask]
    y_score = pooled[eval_mask]
    ok = ~np.isnan(y_score) & ~np.isnan(y_true)
    y_true, y_score = y_true[ok], y_score[ok]
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    print(f"[{mode}] HELD-OUT TEST: AUC={auc:.4f}, AP={ap:.4f} (train {train_time:.1f}s)")

    result = {"mode": mode, "auc": auc, "ap": ap, "train_time_sec": train_time}
    with open(f"{CKPT_DIR}/cdr_pinn_{mode}_result.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["causal", "curriculum", "both"], default="both")
    args = parser.parse_args()
    modes = ["causal", "curriculum"] if args.mode == "both" else [args.mode]
    results = [run(m) for m in modes]
    print("\n=== SUMMARY (baseline full_cdr AUC=0.9406) ===")
    for r in results:
        print(f"  {r['mode']}: AUC={r['auc']:.4f}, AP={r['ap']:.4f}")
