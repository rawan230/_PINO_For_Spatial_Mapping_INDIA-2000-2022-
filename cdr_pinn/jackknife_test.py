"""
Biswas et al. (2025) Fig. 10-style Jackknife variable-importance test -- the one
variable-understanding analysis from the reference paper NOT YET reproduced in this
study (permutation importance and response curves already were, both inference-only
against a single trained checkpoint). Jackknife is fundamentally different: it needs
per-variable RETRAINING, not just inference-time perturbation of a fixed model.

For each of the 7 covariates, train two models (constant-mean-field masking keeps
architecture/input shape unchanged -- masked covariates carry zero spatial/temporal
information, same trick as the response-curve sweep's constant fields):
  - "without X"  : all 7 covariates normal EXCEPT X held at its domain-mean constant
  - "only X"     : X normal, all OTHER 6 held at their domain-mean constants
Plus "all" (every covariate normal) retrained at the SAME reduced epoch budget as the
without/only runs, for a fair apples-to-apples comparison array (the existing
full_cdr.pt checkpoint used 80 epochs; this script's own "all" run uses --epochs to
match the jackknife runs exactly).

Reduced epoch budget (default 40, half of the original 80-epoch full_cdr run) is a
deliberate, documented compute/time tradeoff for a 15-run comparison sweep, consistent
with standard MaxEnt Jackknife practice (relative comparison across variables, not
each run's own maximal convergence). Same architecture/seed/split as every other
CDR-PINN experiment in this study for direct comparability.
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
from train import build_masks

DATA_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_monthly_stacks.npz"
OUT_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_jackknife_results.json"

WINDOW = 24
WIDTH = 32
N_LAYERS = 4
MODES = 16
LR = 1e-3
SEED = 42

COVARIATES = ["ndvi_f1", "ndvi_anomaly", "forest_frac", "dryness", "slope", "dist_roads", "elevation"]


def train_one(covariate_mode, held_out_or_only, epochs, device, tens, means, valid_mask, boundary_mask,
              train_data_mask, fire_indicator, fire_ever_frac, lat_rad, dlon_rad, dlat_rad, pos_weight,
              n_months, H, W, test_np, valid_np, fire_ever_binary_np):
    """covariate_mode: 'all' | 'without' | 'only'. held_out_or_only: covariate name (ignored if mode=='all')."""
    torch.manual_seed(SEED)

    def cov_value(name, ti):
        base = tens[name] if name not in ("ndvi_anomaly", "dryness") else tens[name][ti]
        if covariate_mode == "all":
            return base
        if covariate_mode == "without":
            return means[name] if name == held_out_or_only else base
        if covariate_mode == "only":
            return base if name == held_out_or_only else means[name]
        raise ValueError(covariate_mode)

    def covariate_stack(ti):
        return torch.stack([
            cov_value("ndvi_f1", ti), cov_value("ndvi_anomaly", ti), cov_value("forest_frac", ti),
            cov_value("dryness", ti), cov_value("slope", ti), cov_value("dist_roads", ti),
            cov_value("elevation", ti),
        ], dim=0).unsqueeze(0)

    def physics_covariates(ti):
        return {
            "ndvi_f1": cov_value("ndvi_f1", ti).unsqueeze(0), "forest_frac": cov_value("forest_frac", ti).unsqueeze(0),
            "ndvi_anomaly": cov_value("ndvi_anomaly", ti).unsqueeze(0),
            "grad_e_x": tens["grad_e_x"].unsqueeze(0), "grad_e_y": tens["grad_e_y"].unsqueeze(0),
            "dryness": cov_value("dryness", ti).unsqueeze(0), "slope": cov_value("slope", ti).unsqueeze(0),
            "dist_roads": cov_value("dist_roads", ti).unsqueeze(0),
        }

    model = CDRPINN(n_static_channels=7, width=WIDTH, modes_h=MODES, modes_w=MODES, n_layers=N_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    balancer = AdaptiveLossBalancer(["data", "pde", "bc", "ic"], update_every=5)

    t_start = time.time()
    for epoch in range(epochs):
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
        if torch.isnan(total):
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
    return {"auc": auc, "ap": ap, "train_time_sec": train_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, epochs/run: {args.epochs}, 15 runs total (7 without + 7 only + 1 all)")

    d = np.load(DATA_PATH)
    H, W = TARGET_H, TARGET_W
    n_months = len(d["months"])

    def t(x):
        return torch.tensor(np.nan_to_num(x, nan=0.0), dtype=torch.float32, device=device)

    ndvi_f1_np = d["ndvi_f1"]
    valid_np, boundary_np, train_np, test_np = build_masks(ndvi_f1_np)
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

    means = {}
    for name in COVARIATES:
        arr = tens[name][0] if name in ("ndvi_anomaly", "dryness") else tens[name]
        m = arr[valid_mask].mean()
        means[name] = torch.full((H, W), float(m), dtype=torch.float32, device=device)

    lat_deg = np.linspace(LAT_MAX, LAT_MIN, H)
    lat_rad = torch.tensor(np.radians(lat_deg), dtype=torch.float32, device=device)
    dlat_rad = float(np.radians(abs(lat_deg[1] - lat_deg[0])))
    dlon_rad = float(np.radians((LON_MAX - LON_MIN) / W))
    monthly_pos_rate = fire_indicator[:, train_data_mask].mean().item()
    pos_weight = (1.0 - monthly_pos_rate) / max(monthly_pos_rate, 1e-6)

    common = dict(device=device, tens=tens, means=means, valid_mask=valid_mask, boundary_mask=boundary_mask,
                  train_data_mask=train_data_mask, fire_indicator=fire_indicator, fire_ever_frac=fire_ever_frac,
                  lat_rad=lat_rad, dlon_rad=dlon_rad, dlat_rad=dlat_rad, pos_weight=pos_weight,
                  n_months=n_months, H=H, W=W, test_np=test_np, valid_np=valid_np,
                  fire_ever_binary_np=fire_ever_binary_np)

    results = {}
    t0 = time.time()
    print(f"\n[1/15] all variables (baseline @ {args.epochs} epochs)")
    results["all"] = train_one("all", None, args.epochs, **common)
    print(f"  AUC={results['all']['auc']:.4f}  ({time.time()-t0:.0f}s elapsed)")

    run_i = 1
    for cov in COVARIATES:
        run_i += 1
        print(f"[{run_i}/15] without {cov}")
        results[f"without_{cov}"] = train_one("without", cov, args.epochs, **common)
        print(f"  AUC={results[f'without_{cov}']['auc']:.4f}  ({time.time()-t0:.0f}s elapsed)")

        run_i += 1
        print(f"[{run_i}/15] only {cov}")
        results[f"only_{cov}"] = train_one("only", cov, args.epochs, **common)
        print(f"  AUC={results[f'only_{cov}']['auc']:.4f}  ({time.time()-t0:.0f}s elapsed)")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}  (total {time.time()-t0:.0f}s)")

    all_auc = results["all"]["auc"]
    print(f"\n=== JACKKNIFE SUMMARY (all-variables AUC={all_auc:.4f} @ {args.epochs} epochs) ===")
    print(f"{'covariate':<14} {'without-X AUC':>14} {'drop-when-removed':>19} {'only-X AUC':>12} {'gain-alone':>11}")
    for cov in COVARIATES:
        w = results[f"without_{cov}"]["auc"]
        o = results[f"only_{cov}"]["auc"]
        print(f"{cov:<14} {w:>14.4f} {all_auc - w:>19.4f} {o:>12.4f} {o - 0.5:>11.4f}")


if __name__ == "__main__":
    main()
