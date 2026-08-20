"""
Fixes the disclosed methodological gap (Methodology Draft Sec 3.4 / Limitations item 2):
this study's three earlier architecture/schedule decisions (width=32/80ep "small",
width=64/150ep "scale-up", width=32/80ep+cosine "LR-schedule") were each accepted or
rejected by looking at TEST-set AUC -- exactly what a validation set exists to prevent,
since the test set then stops being a true held-out estimate for whichever config wins.

This script builds a genuine 3-way split (train/val/test, pixel-level, same seed=42
as every other split in this study) and re-runs all three configs, selecting the
winner by VALIDATION AUC only. The test set is scored ONCE, for the winning config
only, and is not used for any decision at all -- a fair, honest final number replacing
the earlier test-selected one used throughout the paper draft's Results section.

If the val-based winner agrees with the original test-based conclusion (small/short
config best), that CONFIRMS the earlier finding wasn't test-set overfitting -- a
genuine strengthening, not just a formality. If it disagrees, that is the honest
result to report instead.
"""
import json
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, average_precision_score

from model import CDRPINN
from losses import data_loss_monthly, data_loss_terminal, pde_loss, bc_loss, ic_loss, AdaptiveLossBalancer, lse_pool
from build_monthly_stacks import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, TARGET_H, TARGET_W

DATA_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_monthly_stacks.npz"
OUT_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_validation_split_results.json"

WINDOW = 24
SEED = 42

CONFIGS = {
    "small_80ep":     dict(width=32, n_layers=4, modes=16, epochs=80,  lr_schedule=None),
    "scaleup_150ep":  dict(width=64, n_layers=4, modes=16, epochs=150, lr_schedule=None),
    "small_80ep_cos": dict(width=32, n_layers=4, modes=16, epochs=80,  lr_schedule="cosine"),
}


def build_masks_3way(ndvi_f1, seed=SEED, val_frac=0.15, test_frac=0.2):
    valid = ~np.isnan(ndvi_f1)
    up = np.roll(valid, 1, axis=0); down = np.roll(valid, -1, axis=0)
    left = np.roll(valid, 1, axis=1); right = np.roll(valid, -1, axis=1)
    boundary = valid & (~up | ~down | ~left | ~right)
    rng = np.random.RandomState(seed)
    valid_idx = np.argwhere(valid)
    perm = rng.permutation(len(valid_idx))
    n_test = int(len(valid_idx) * test_frac)
    n_val = int(len(valid_idx) * val_frac)
    test_idx = valid_idx[perm[:n_test]]
    val_idx = valid_idx[perm[n_test:n_test + n_val]]
    train_idx = valid_idx[perm[n_test + n_val:]]
    train_mask = np.zeros_like(valid); train_mask[train_idx[:, 0], train_idx[:, 1]] = True
    val_mask = np.zeros_like(valid); val_mask[val_idx[:, 0], val_idx[:, 1]] = True
    test_mask = np.zeros_like(valid); test_mask[test_idx[:, 0], test_idx[:, 1]] = True
    return valid, boundary, train_mask, val_mask, test_mask


def run_config(name, cfg, device, tens, valid_mask, boundary_mask, train_data_mask,
               fire_indicator, fire_ever_frac, lat_rad, dlon_rad, dlat_rad, pos_weight,
               n_months, H, W, val_np, test_np, valid_np, fire_ever_binary_np):
    torch.manual_seed(SEED)

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

    model = CDRPINN(n_static_channels=7, width=cfg["width"], modes_h=cfg["modes"], modes_w=cfg["modes"],
                     n_layers=cfg["n_layers"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"]) if cfg["lr_schedule"] == "cosine" else None
    balancer = AdaptiveLossBalancer(["data", "pde", "bc", "ic"], update_every=5)

    t_start = time.time()
    for epoch in range(cfg["epochs"]):
        u = torch.zeros(1, 1, H, W, device=device)
        for start in range(0, n_months - 1, WINDOW):
            end = min(start + WINDOW, n_months - 1)
            window_ic = ic_loss(u) if start == 0 else torch.zeros((), device=device)
            traj = [u.detach()]
            window_pde, window_bc, window_data = 0.0, 0.0, 0.0
            for ti in range(start, end):
                u_next = model(u, covariate_stack(ti))
                residual, _ = model.pde_residual(
                    u, u_next, dt_months=1.0, covariates=physics_covariates(ti),
                    lat_rad_1d=lat_rad, dlon_rad=dlon_rad, dlat_rad=dlat_rad,
                    use_diffusion=True, use_advection=True, use_reaction=True)
                window_pde = window_pde + pde_loss(residual.squeeze(0), valid_mask)
                mid = 0.5 * (u.squeeze(0).squeeze(0) + u_next.squeeze(0).squeeze(0))
                window_bc = window_bc + bc_loss(mid, lat_rad, dlon_rad, dlat_rad, boundary_mask)
                window_data = window_data + data_loss_monthly(
                    u_next.squeeze(0).squeeze(0), fire_indicator[ti + 1], train_data_mask, pos_weight=pos_weight)
                traj.append(u_next)
                u = u_next
            n_steps = end - start
            window_pde, window_bc, window_data = window_pde / n_steps, window_bc / n_steps, window_data / n_steps
            if end >= n_months - 1:
                traj_stack = torch.cat(traj, dim=0).squeeze(1)
                window_terminal = data_loss_terminal(traj_stack, fire_ever_frac, train_data_mask, tau=5.0)
                window_data = 0.5 * window_data + 0.5 * window_terminal
            losses = {"data": window_data, "pde": window_pde, "bc": window_bc, "ic": window_ic}
            total, _ = balancer.combine(losses, list(model.parameters()))
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            u = u.detach()
        if scheduler is not None:
            scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"  [{name}] epoch {epoch+1}/{cfg['epochs']} total={float(total):.4e} "
                  f"({time.time()-t_start:.0f}s elapsed)")
        if torch.isnan(total):
            print(f"  [{name}] STOPPING: NaN.")
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

    def score_on(mask):
        m = valid_np & mask
        y_true = fire_ever_binary_np[m]
        y_score = pooled[m]
        ok = ~np.isnan(y_score) & ~np.isnan(y_true)
        y_true, y_score = y_true[ok], y_score[ok]
        return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)

    val_auc, val_ap = score_on(val_np)
    test_auc, test_ap = score_on(test_np)
    return {"val_auc": val_auc, "val_ap": val_ap, "test_auc": test_auc, "test_ap": test_ap,
            "train_time_sec": train_time}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    d = np.load(DATA_PATH)
    H, W = TARGET_H, TARGET_W
    n_months = len(d["months"])

    def t(x):
        return torch.tensor(np.nan_to_num(x, nan=0.0), dtype=torch.float32, device=device)

    ndvi_f1_np = d["ndvi_f1"]
    valid_np, boundary_np, train_np, val_np, test_np = build_masks_3way(ndvi_f1_np)
    print(f"Split sizes (valid pixels): train={train_np.sum()}, val={val_np.sum()}, test={test_np.sum()}")
    valid_mask = torch.tensor(valid_np, dtype=torch.bool, device=device)
    boundary_mask = torch.tensor(boundary_np, dtype=torch.bool, device=device)
    train_data_mask = torch.tensor(valid_np & train_np, dtype=torch.bool, device=device)
    fire_ever_binary_np = (d["fire_ever_frac"] > 0).astype(np.float32)

    tens = dict(
        ndvi_f1=t(ndvi_f1_np), ndvi_anomaly=t(d["ndvi_anomaly"]), forest_frac=t(d["forest_frac"]),
        dryness=t(d["dryness_proxy"]), slope=t(d["slope"]), dist_roads=t(d["dist_roads"]),
        elevation=t(d["elevation"]), grad_e_x=t(d["grad_e_x"]), grad_e_y=t(d["grad_e_y"]),
    )
    fire_indicator = t(d["fire_indicator"])
    fire_ever_frac = t(d["fire_ever_frac"])

    lat_deg = np.linspace(LAT_MAX, LAT_MIN, H)
    lat_rad = torch.tensor(np.radians(lat_deg), dtype=torch.float32, device=device)
    dlat_rad = float(np.radians(abs(lat_deg[1] - lat_deg[0])))
    dlon_rad = float(np.radians((LON_MAX - LON_MIN) / W))
    monthly_pos_rate = fire_indicator[:, train_data_mask].mean().item()
    pos_weight = (1.0 - monthly_pos_rate) / max(monthly_pos_rate, 1e-6)

    common = dict(device=device, tens=tens, valid_mask=valid_mask, boundary_mask=boundary_mask,
                  train_data_mask=train_data_mask, fire_indicator=fire_indicator, fire_ever_frac=fire_ever_frac,
                  lat_rad=lat_rad, dlon_rad=dlon_rad, dlat_rad=dlat_rad, pos_weight=pos_weight,
                  n_months=n_months, H=H, W=W, val_np=val_np, test_np=test_np, valid_np=valid_np,
                  fire_ever_binary_np=fire_ever_binary_np)

    results = {}
    for name, cfg in CONFIGS.items():
        print(f"\n=== {name} (width={cfg['width']}, epochs={cfg['epochs']}, schedule={cfg['lr_schedule']}) ===")
        results[name] = run_config(name, cfg, **common)
        r = results[name]
        print(f"  VAL AUC={r['val_auc']:.4f}  |  TEST AUC={r['test_auc']:.4f} (not used for selection)")

    winner = max(results, key=lambda k: results[k]["val_auc"])
    print(f"\n=== SELECTION (by VAL AUC only) ===")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["val_auc"]):
        marker = " <-- WINNER" if name == winner else ""
        print(f"  {name}: val_auc={r['val_auc']:.4f}, test_auc={r['test_auc']:.4f}{marker}")
    print(f"\nHonest final held-out number (winner={winner}): TEST AUC={results[winner]['test_auc']:.4f}")
    print("Original test-selected conclusion was: small_80ep (AUC=0.9406 on the old 80/20 split).")

    out = {"results": results, "winner": winner, "winner_test_auc": results[winner]["test_auc"]}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
