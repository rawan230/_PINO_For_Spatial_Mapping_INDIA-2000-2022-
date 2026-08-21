"""
Validated hyperparameter search: weight decay (L2 regularization via AdamW), the
literature-standard regularization knob for FNO/PINO training (spectral mode
truncation is FNO's other, architectural capacity control -- already fixed at 16x16
modes, not searched here). Selected by VALIDATION AUC on the standard 65/15/20 split
(preprocessing.py) -- test pixels are never touched by this script.

Reduced epoch budget (40, half of the standard 80) for the search itself -- relative
comparison across candidates, not each candidate's own maximal convergence, same
budget-reduction logic already used for this project's Jackknife test.
"""
import json
import time
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from model import CDRPINN
from losses import data_loss_monthly, data_loss_terminal, pde_loss, bc_loss, ic_loss, AdaptiveLossBalancer, lse_pool
from preprocessing import (
    build_masks_3way, load_tensors, covariate_stack, physics_covariates,
    grid_metadata, compute_pos_weight, SEED, TARGET_H, TARGET_W,
)

CKPT_DIR = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data"
WINDOW = 24
SEARCH_EPOCHS = 40
WIDTH = 32
N_LAYERS = 4
MODES = 16
LR = 1e-3
CANDIDATES = [0.0, 1e-5, 1e-4]


def run_one(weight_decay, tensors, fire_ever_binary_np, valid_np, train_np, val_np,
            boundary_np, n_months, device):
    torch.manual_seed(SEED)
    H, W = TARGET_H, TARGET_W
    valid_mask = torch.tensor(valid_np, dtype=torch.bool, device=device)
    boundary_mask = torch.tensor(boundary_np, dtype=torch.bool, device=device)
    train_data_mask = torch.tensor(valid_np & train_np, dtype=torch.bool, device=device)
    _, pos_weight = compute_pos_weight(tensors, train_data_mask)
    lat_rad, dlat_rad, dlon_rad = grid_metadata(device)

    model = CDRPINN(n_static_channels=7, width=WIDTH, modes_h=MODES, modes_w=MODES, n_layers=N_LAYERS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    balancer = AdaptiveLossBalancer(["data", "pde", "bc", "ic"], update_every=5)

    for epoch in range(SEARCH_EPOCHS):
        u = torch.zeros(1, 1, H, W, device=device)
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
        if torch.isnan(total):
            return {"weight_decay": weight_decay, "val_auc": float("nan"), "nan": True}

    model.eval()
    with torch.no_grad():
        u = torch.zeros(1, 1, H, W, device=device)
        scores = []
        for ti in range(n_months - 1):
            u_next = model(u, covariate_stack(tensors, ti))
            scores.append(torch.sigmoid(u_next.squeeze(0).squeeze(0)))
            u = u_next
        pooled = lse_pool(torch.stack(scores, dim=0), dim=0, tau=5.0).cpu().numpy()

    m = valid_np & val_np
    y_true = fire_ever_binary_np[m]
    y_score = pooled[m]
    ok = ~np.isnan(y_score) & ~np.isnan(y_true)
    val_auc = roc_auc_score(y_true[ok], y_score[ok])
    return {"weight_decay": weight_decay, "val_auc": float(val_auc), "nan": False}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, searching weight_decay in {CANDIDATES} at {SEARCH_EPOCHS} epochs each")
    tensors, fire_ever_binary_np, ndvi_f1_np, n_months = load_tensors(device)
    valid_np, boundary_np, train_np, val_np, test_np = build_masks_3way(ndvi_f1_np)

    results = []
    t0 = time.time()
    for wd in CANDIDATES:
        r = run_one(wd, tensors, fire_ever_binary_np, valid_np, train_np, val_np, boundary_np, n_months, device)
        results.append(r)
        print(f"  weight_decay={wd:.0e} -> val_auc={r['val_auc']:.4f}  ({time.time()-t0:.0f}s elapsed)")

    valid_results = [r for r in results if not r["nan"]]
    winner = max(valid_results, key=lambda r: r["val_auc"])
    print(f"\nWinner (by validation AUC): weight_decay={winner['weight_decay']:.0e}, val_auc={winner['val_auc']:.4f}")

    out = {"candidates": results, "winner_weight_decay": winner["weight_decay"], "winner_val_auc": winner["val_auc"]}
    with open(f"{CKPT_DIR}/cdr_pinn_weight_decay_search.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {CKPT_DIR}/cdr_pinn_weight_decay_search.json")


if __name__ == "__main__":
    main()
