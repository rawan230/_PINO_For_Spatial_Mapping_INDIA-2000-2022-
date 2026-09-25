"""
Numerical physics audit of trained CDR-PINO checkpoints (port of the audit's
results/code/cdr_term_magnitudes.py).

For a checkpoint, roll out all T-1 = 265 monthly transitions and measure, over the
valid in-India cells, the magnitude of every term of the FULL implemented residual
    r = du/dt - D lap(u) + v.grad(u) - rho s(1-s),   s = sigmoid(u), u = logit
(all three terms are evaluated regardless of which configuration the checkpoint was
trained with), plus the learned coefficient fields. Answers two questions: are the
physics terms numerically active, and does the trained state satisfy the PDE?

Units as implemented: x, y in km (R = 6371 km), t in months (dt = 1), u dimensionless
(logit); D -> km^2/month, v -> km/month (c_adv carries km^2 m^-1 month^-1 because
grad(E) is in m/km).

Runs on CPU (inference only, ~1-3 min per checkpoint).

Usage:
  python term_magnitudes.py                                   # A_{nophys,diff,diffadv,full}_s42
  python term_magnitudes.py --ckpt path\\to\\ckpt.pt --tag mytag
  python term_magnitudes.py --tags A_full_s42,A_full_s43
Output: <unified dir>/term_magnitudes_<tag>.json and TERM_MAGNITUDES_summary.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

CDR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CDR_DIR)
from model import CDRPINN  # noqa: E402
from preprocessing import load_tensors, covariate_stack, physics_covariates, grid_metadata  # noqa: E402
from run_unified_protocol import OUT as UNIFIED_DIR, WIDTH, MODES, N_LAYERS  # noqa: E402

DEFAULT_TAGS = "A_nophys_s42,A_diff_s42,A_diffadv_s42,A_full_s42"


def term_magnitudes(ck_path, tag, tensors, ndvi_f1, T, out_dir):
    dev = "cpu"
    v_np = ~np.isnan(ndvi_f1)
    valid = torch.tensor(v_np)
    bnd = v_np & (~np.roll(v_np, 1, 0) | ~np.roll(v_np, -1, 0) | ~np.roll(v_np, 1, 1) | ~np.roll(v_np, -1, 1))
    lat, dlat, dlon = grid_metadata(dev)
    m = CDRPINN(n_static_channels=7, width=WIDTH, modes_h=MODES, modes_w=MODES, n_layers=N_LAYERS)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state"]); m.eval()
    stats = {k: [] for k in ("dudt", "diff", "adv", "react", "resid", "D_mean", "D_max", "rho_mean", "rho_max",
                             "vmag_mean", "u_mean", "u_std", "s_mean")}
    M = valid

    def A(x):
        return float(x[M].abs().mean())

    with torch.no_grad():
        u = torch.zeros(1, 1, *valid.shape)
        for ti in range(T - 1):
            un = m(u, covariate_stack(tensors, ti))
            r, parts = m.pde_residual(u, un, 1.0, physics_covariates(tensors, ti), lat, dlon, dlat)
            du = (un - u)[0, 0]
            lap = parts["laplacian"][0]; D = parts["D"][0]
            vx, vy = parts["vx"][0], parts["vy"][0]
            # advection term recovered exactly as assembled in pde_residual:
            # r = du - D lap + adv - R  ->  adv = r - du + D lap + R
            adv = (r[0] - du + D * lap + parts["reaction"][0])
            stats["dudt"].append(A(du)); stats["diff"].append(A(D * lap)); stats["adv"].append(A(adv))
            stats["react"].append(A(parts["reaction"][0])); stats["resid"].append(float(r[0][M].pow(2).mean().sqrt()))
            stats["D_mean"].append(float(D[M].mean())); stats["D_max"].append(float(D[M].max()))
            s_mid = torch.sigmoid(0.5 * (u + un))[0, 0]
            rho = parts["reaction"][0] / (s_mid * (1 - s_mid)).clamp_min(1e-12)
            stats["rho_mean"].append(float(rho[M].mean())); stats["rho_max"].append(float(rho[M].max()))
            stats["vmag_mean"].append(float(torch.sqrt(vx ** 2 + vy ** 2)[M].mean()))
            stats["u_mean"].append(float(un[0, 0][M].mean())); stats["u_std"].append(float(un[0, 0][M].std()))
            stats["s_mean"].append(float(torch.sigmoid(un[0, 0])[M].mean()))
            u = un
    summ = {k: dict(mean=float(np.mean(v)), median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)))
            for k, v in stats.items()}
    tot = summ["diff"]["mean"] + summ["adv"]["mean"] + summ["react"]["mean"]
    out = dict(checkpoint=ck_path, tag=tag, n_valid=int(valid.sum()), n_boundary_ring=int(bnd.sum()), T_transitions=T - 1,
               c_adv=float(torch.nn.functional.softplus(m.v_head.c_raw)),
               w_ndvi_anom=float(torch.nn.functional.softplus(m.D_head.w_raw)),
               grad_e_abs_mean_m_per_km=float(tensors["grad_e_x"].abs()[valid].mean()), summary=summ,
               share_of_rhs_magnitude=dict(diffusion=summ["diff"]["mean"] / tot, advection=summ["adv"]["mean"] / tot,
                                           reaction=summ["react"]["mean"] / tot),
               rms_residual_over_mean_abs_dudt=summ["resid"]["mean"] / max(summ["dudt"]["mean"], 1e-30),
               per_month=stats)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"term_magnitudes_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_month"}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="single checkpoint path (overrides --tags)")
    ap.add_argument("--tag", default=None, help="tag for --ckpt output")
    ap.add_argument("--tags", default=DEFAULT_TAGS, help="comma list of unified run tags (ckpt_<tag>.pt)")
    ap.add_argument("--unified_dir", default=UNIFIED_DIR)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tensors, _, ndvi_f1, T = load_tensors("cpu")
    if args.ckpt:
        jobs = [(args.ckpt, args.tag or os.path.splitext(os.path.basename(args.ckpt))[0])]
    else:
        jobs = [(os.path.join(args.unified_dir, f"ckpt_{t}.pt"), t) for t in args.tags.split(",")]
    rows = []
    for ck, tag in jobs:
        if not os.path.exists(ck):
            print(f"skip {tag}: {ck} not found"); continue
        o = term_magnitudes(ck, tag, tensors, ndvi_f1, T, args.unified_dir)
        s = o["summary"]
        rows.append(dict(tag=tag, median_abs_dudt=s["dudt"]["median"], mean_abs_diffusion=s["diff"]["mean"],
                         mean_abs_advection=s["adv"]["mean"], mean_abs_reaction=s["react"]["mean"],
                         rms_residual=s["resid"]["mean"], residual_over_mean_abs_dudt=o["rms_residual_over_mean_abs_dudt"],
                         share_diffusion=o["share_of_rhs_magnitude"]["diffusion"],
                         share_advection=o["share_of_rhs_magnitude"]["advection"],
                         share_reaction=o["share_of_rhs_magnitude"]["reaction"], c_adv=o["c_adv"]))
    if rows:
        import pandas as pd
        p = os.path.join(args.unified_dir, "TERM_MAGNITUDES_summary.csv")
        pd.DataFrame(rows).to_csv(p, index=False)
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
