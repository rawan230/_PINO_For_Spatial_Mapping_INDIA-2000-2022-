"""
Phase 23-25 numerical physics audit: on a trained CDR-PINO checkpoint, roll out all 265
monthly transitions and measure, over valid in-India cells, the magnitude of every term of
the implemented residual  r = du/dt - D lap(u) + v.grad(u) - rho s(1-s),  s = sigmoid(u),
plus the learned coefficient fields. Answers: are the physics terms numerically active,
and does the trained state satisfy the PDE?  Units as implemented: x,y in km (R=6371 km),
t in months (dt=1), u dimensionless (logit), D -> km^2/month, v -> km/month (c_adv carries
km^2 m^-1 month^-1 because grad(E) is in m/km).
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = r"D:\FOREST FIRE MAPPING(INDIA)"
sys.path.insert(0, os.path.join(ROOT, "Physics_Informed_FireRisk_Model", "cdr_pinn"))
from model import CDRPINN  # noqa: E402
from preprocessing import load_tensors, covariate_stack, physics_covariates, grid_metadata  # noqa: E402

ck_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "cdr_pino", "reproduction", "cdr_pinn_full_cdr_standard_protocol.pt")
tag = sys.argv[2] if len(sys.argv) > 2 else "trackA_full_repro"
dev = "cpu"
torch.set_num_threads(8)
tensors, fe, ndvi_f1, T = load_tensors(dev)
valid = torch.tensor(~np.isnan(ndvi_f1))
up = np.roll(~np.isnan(ndvi_f1), 1, 0); v_ = ~np.isnan(ndvi_f1)
bnd = v_ & (~np.roll(v_, 1, 0) | ~np.roll(v_, -1, 0) | ~np.roll(v_, 1, 1) | ~np.roll(v_, -1, 1))
lat, dlat, dlon = grid_metadata(dev)
m = CDRPINN(n_static_channels=7, width=32, modes_h=16, modes_w=16, n_layers=4)
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
m.load_state_dict(ck["model_state"]); m.eval()
stats = {k: [] for k in ("dudt", "diff", "adv", "react", "resid", "D_mean", "D_max", "rho_mean", "rho_max", "vmag_mean", "u_mean", "u_std", "s_mean")}
with torch.no_grad():
    u = torch.zeros(1, 1, *valid.shape)
    for ti in range(T - 1):
        un = m(u, covariate_stack(tensors, ti))
        r, parts = m.pde_residual(u, un, 1.0, physics_covariates(tensors, ti), lat, dlon, dlat)
        du = (un - u)[0, 0]
        lap = parts["laplacian"][0]; D = parts["D"][0]
        vx, vy = parts["vx"][0], parts["vy"][0]
        # advection term recomputed exactly as in pde_residual (gradient at the midpoint)
        adv = (r[0] - du + D * lap + parts["reaction"][0])  # r = du - D lap + adv - R  -> adv = r - du + D lap + R
        M = valid
        A = lambda x: float(x[M].abs().mean())
        stats["dudt"].append(A(du)); stats["diff"].append(A(D * lap)); stats["adv"].append(A(adv))
        stats["react"].append(A(parts["reaction"][0])); stats["resid"].append(float(r[0][M].pow(2).mean().sqrt()))
        stats["D_mean"].append(float(D[M].mean())); stats["D_max"].append(float(D[M].max()))
        rho = parts["reaction"][0] / (torch.sigmoid(0.5 * (u + un))[0, 0] * (1 - torch.sigmoid(0.5 * (u + un))[0, 0])).clamp_min(1e-12)
        stats["rho_mean"].append(float(rho[M].mean())); stats["rho_max"].append(float(rho[M].max()))
        stats["vmag_mean"].append(float(torch.sqrt(vx ** 2 + vy ** 2)[M].mean()))
        stats["u_mean"].append(float(un[0, 0][M].mean())); stats["u_std"].append(float(un[0, 0][M].std()))
        stats["s_mean"].append(float(torch.sigmoid(un[0, 0])[M].mean()))
        u = un
summ = {k: dict(mean=float(np.mean(v)), median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v))) for k, v in stats.items()}
tot = summ["diff"]["mean"] + summ["adv"]["mean"] + summ["react"]["mean"]
out = dict(checkpoint=ck_path, tag=tag, n_valid=int(valid.sum()), n_boundary_ring=int(bnd.sum()), T_transitions=T - 1,
           c_adv=float(torch.nn.functional.softplus(m.v_head.c_raw)), w_ndvi_anom=float(torch.nn.functional.softplus(m.D_head.w_raw)),
           grad_e_abs_mean_m_per_km=float(tensors["grad_e_x"].abs()[valid].mean()), summary=summ,
           share_of_rhs_magnitude=dict(diffusion=summ["diff"]["mean"] / tot, advection=summ["adv"]["mean"] / tot, reaction=summ["react"]["mean"] / tot),
           rms_residual_over_mean_abs_dudt=summ["resid"]["mean"] / max(summ["dudt"]["mean"], 1e-30),
           per_month=stats)
os.makedirs(os.path.join(ROOT, "results", "cdr_pino"), exist_ok=True)
with open(os.path.join(ROOT, "results", "cdr_pino", f"term_magnitudes_{tag}.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_month"}, indent=1))
