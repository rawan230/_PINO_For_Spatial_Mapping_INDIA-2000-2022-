"""
AUDIT 2026-09-24 -- unified CDR-PINO experiment runner.

Why this exists (audit findings it addresses):
  * Historical Track A (train_standard_protocol.py) and Tracks B1/B2/B3
    (run_validation_tracks.py) used DIFFERENT protocols: A = AdamW + validated
    weight decay + ReduceLROnPlateau + 80-epoch budget; B = Adam + cosine, 50-epoch
    budget (B1/B2). They are separately-trained models, not one checkpoint.
    -> Here ONE protocol (Track A's validated one) is applied to every track.
  * Historical B3 supervised the terminal (LSE-pooled) data loss with the POOLED
    2000-2022 fire_ever label, which includes the held-out test years' fires
    (run_validation_tracks.py:271) -- label leakage.
    -> Here, for B3 the terminal label is rebuilt from TRAINING MONTHS ONLY.
  * Historical B1/B2 carved validation pixels at random from the training region,
    so early stopping monitored an in-distribution score (~0.94) while the test was
    out-of-distribution (~0.75). -> Here validation for spatial tracks is carved
    as whole 2deg blocks from the training region (spatially blocked validation).
  * Ablation, B1, B2, B3 were single-seed. -> 3 seeds (42, 43, 44); fold/region/
    held-out-year assignment is fixed across seeds and configs (seed=42), so seeds
    vary only model initialisation / training noise.
  * No per-cell predictions were saved, so no paired test was possible.
    -> Every run saves its test-cell (or test-cell-month) predictions.

Nothing about the model, losses, covariates, grid or data is changed: model.py,
losses.py, spectral_ops.py and the monthly-stack npz are used as-is.

Physics configurations (identical budget/protocol):
  nophys   : data + IC losses only (PDE and BC removed)
  diff     : + PDE residual with diffusion term only (+ BC)
  diffadv  : + diffusion + advection
  full     : + diffusion + advection + reaction   (full CDR)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = r"D:\FOREST FIRE MAPPING(INDIA)"
CDR_DIR = os.path.join(ROOT, "Physics_Informed_FireRisk_Model", "cdr_pinn")
sys.path.insert(0, CDR_DIR)

from model import CDRPINN  # noqa: E402
from losses import (data_loss_monthly, data_loss_terminal, pde_loss, bc_loss,  # noqa: E402
                    ic_loss, AdaptiveLossBalancer, lse_pool)
from build_monthly_stacks import LON_MIN, LAT_MIN, TARGET_H, TARGET_W  # noqa: E402
from preprocessing import load_tensors, covariate_stack, physics_covariates, grid_metadata  # noqa: E402

DATA_PATH = os.path.join(ROOT, "Physics_Informed_FireRisk_Model", "CDR_PINN_Data", "cdr_pinn_monthly_stacks.npz")
OUT = os.path.join(ROOT, "results", "cdr_pino", "unified")
os.makedirs(OUT, exist_ok=True)

# Track A's validated protocol, applied to every track
WINDOW, N_EPOCHS, WIDTH, N_LAYERS, MODES, LR = 24, 80, 32, 4, 16, 1e-3
WEIGHT_DECAY = 0.0          # validated winner, cdr_pinn_weight_decay_search.json
VAL_EVERY, PATIENCE = 5, 4  # early stopping on validation AUC, 20 epochs without improvement
SPLIT_SEED = 42             # fold / region / year / pixel assignment -- fixed
CONFIGS = {
    "nophys":  dict(use_physics=False, flags=dict(use_diffusion=False, use_advection=False, use_reaction=False)),
    "diff":    dict(use_physics=True,  flags=dict(use_diffusion=True,  use_advection=False, use_reaction=False)),
    "diffadv": dict(use_physics=True,  flags=dict(use_diffusion=True,  use_advection=True,  use_reaction=False)),
    "full":    dict(use_physics=True,  flags=dict(use_diffusion=True,  use_advection=True,  use_reaction=True)),
}


def git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(n), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Data / splits
# --------------------------------------------------------------------------- #
class Ctx:
    def __init__(self, device):
        self.device = device
        self.tensors, self.fire_ever_np, ndvi_f1_np, self.n_months = load_tensors(device)
        self.H, self.W = TARGET_H, TARGET_W
        self.valid = ~np.isnan(ndvi_f1_np)
        up = np.roll(self.valid, 1, 0); dn = np.roll(self.valid, -1, 0)
        lf = np.roll(self.valid, 1, 1); rt = np.roll(self.valid, -1, 1)
        self.boundary = self.valid & (~up | ~dn | ~lf | ~rt)
        self.lat_rad, self.dlat_rad, self.dlon_rad = grid_metadata(device)
        d = np.load(DATA_PATH)
        self.months = d["months"]
        self.years = np.array([int(m[:4]) for m in self.months])
        self.fire_ind_np = d["fire_indicator"]              # (T,H,W) binary per month
        self.forest_frac_np = d["forest_frac"]
        from build_monthly_stacks import LON_MAX, LAT_MAX
        lat_deg = np.linspace(LAT_MAX, LAT_MIN, self.H)
        lon_deg = np.linspace(LON_MIN, LON_MAX, self.W)
        self.lon_grid, self.lat_grid = np.meshgrid(lon_deg, lat_deg)
        self.block_id = (np.floor((self.lon_grid - LON_MIN) / 2.0).astype(int) * 1000
                         + np.floor((self.lat_grid - LAT_MIN) / 2.0).astype(int))


def random_pixel_split(ctx):
    """Track A: identical to preprocessing.build_masks_3way (65/15/20, seed 42)."""
    from preprocessing import build_masks_3way
    d = np.load(DATA_PATH)
    _, _, tr, va, te = build_masks_3way(d["ndvi_f1"])
    return tr, va, te


def carve_block_validation(ctx, train_mask, frac=0.1875, seed=SPLIT_SEED):
    """Spatially blocked validation: whole 2deg blocks from the training region."""
    blocks = np.unique(ctx.block_id[train_mask])
    rng = np.random.RandomState(seed + 1000)
    n_val = max(1, int(round(len(blocks) * frac)))
    val_blocks = rng.choice(blocks, size=n_val, replace=False)
    val = train_mask & np.isin(ctx.block_id, val_blocks)
    return train_mask & ~val, val


def b1_folds(ctx, n_folds=3):
    """IDENTICAL fold assignment to run_validation_tracks.run_track_b1."""
    valid_blocks = np.unique(ctx.block_id[ctx.valid])
    perm = np.random.RandomState(SPLIT_SEED).permutation(valid_blocks)
    return [set(f.tolist()) for f in np.array_split(perm, n_folds)]


def b2_regions(ctx, n_regions=6):
    """IDENTICAL region assignment to run_validation_tracks.run_track_b2."""
    coords = np.stack([ctx.lon_grid[ctx.valid], ctx.lat_grid[ctx.valid]], axis=1)
    km = KMeans(n_clusters=n_regions, random_state=SPLIT_SEED, n_init=10).fit(coords)
    region = np.full(ctx.lon_grid.shape, -1, dtype=int)
    region[ctx.valid] = km.labels_
    return region, km


def b3_years(ctx, test_frac=0.2):
    """IDENTICAL held-out years to run_validation_tracks.run_track_b3."""
    years = ctx.years[:-1]
    uy = np.unique(years)
    rng = np.random.RandomState(SPLIT_SEED)
    test_years = set(rng.choice(uy, size=max(1, int(len(uy) * test_frac)), replace=False).tolist())
    train_years = sorted(set(uy.tolist()) - test_years)
    rng_v = np.random.RandomState(SPLIT_SEED + 1000)
    n_val = max(1, int(round(len(train_years) * 0.1875)))
    val_years = set(rng_v.choice(train_years, size=n_val, replace=False).tolist())
    fit = np.array([y not in test_years and y not in val_years for y in years])
    val = np.array([y in val_years for y in years])
    test = np.array([y in test_years for y in years])
    return fit, val, test, sorted(test_years), sorted(val_years)


# --------------------------------------------------------------------------- #
# Training (one protocol for every track)
# --------------------------------------------------------------------------- #
def train_eval(ctx, cfg_name, seed, fit_pix, val_pix, test_pix,
               fit_months=None, val_months=None, test_months=None,
               terminal_label_from_fit_months=False, tag=""):
    dev = ctx.device
    cfg = CONFIGS[cfg_name]
    torch.manual_seed(seed)
    np.random.seed(seed)
    H, W, T = ctx.H, ctx.W, ctx.n_months
    t_ = ctx.tensors
    temporal = fit_months is not None
    if not temporal:
        fit_months = val_months = test_months = np.ones(T - 1, dtype=bool)

    valid_t = torch.tensor(ctx.valid, dtype=torch.bool, device=dev)
    bnd_t = torch.tensor(ctx.boundary, dtype=torch.bool, device=dev)
    fit_t = torch.tensor(ctx.valid & fit_pix, dtype=torch.bool, device=dev)

    # terminal label: pooled over ALL months (historical) or over FIT months only (leak-free B3)
    if terminal_label_from_fit_months:
        idx = np.where(fit_months)[0] + 1        # prediction index ti -> label month ti+1
        term_np = (ctx.fire_ind_np[idx].max(axis=0) > 0).astype(np.float32)
        term_label = torch.tensor(term_np, device=dev)
    else:
        term_label = t_["fire_ever_frac"]

    fi = t_["fire_indicator"]
    pos_rate = fi[1:][torch.tensor(fit_months, device=dev)][:, fit_t].mean().item()
    pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)

    model = CDRPINN(n_static_channels=7, width=WIDTH, modes_h=MODES, modes_w=MODES, n_layers=N_LAYERS).to(dev)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    names = ["data", "pde", "bc", "ic"] if cfg["use_physics"] else ["data", "ic"]
    bal = AdaptiveLossBalancer(names, update_every=5)

    def rollout():
        model.eval()
        with torch.no_grad():
            u = torch.zeros(1, 1, H, W, device=dev)
            out = []
            for ti in range(T - 1):
                u = model(u, covariate_stack(t_, ti))
                out.append(torch.sigmoid(u[0, 0]))
            s = torch.stack(out, 0)
        model.train()
        return s  # (T-1,H,W) probabilities

    def score(s, pix, months):
        """Spatial tracks: LSE-pooled score vs fire_ever (historical scoring).
        Temporal track: per-cell-month score vs monthly indicator over `months`."""
        m = ctx.valid & pix
        if not temporal:
            pooled = lse_pool(s, dim=0, tau=5.0).cpu().numpy()
            y, p = ctx.fire_ever_np[m], pooled[m]
        else:
            sm = s[torch.tensor(months, device=dev)].cpu().numpy()
            lab = ctx.fire_ind_np[1:][months]
            y, p = lab[:, m].ravel(), sm[:, m].ravel()
        if len(np.unique(y)) < 2:
            return float("nan"), float("nan"), y, p
        return roc_auc_score(y, p), average_precision_score(y, p), y, p

    best_auc, best_state, best_ep, bad = -1.0, None, 0, 0
    hist = []
    t0 = time.time()
    for ep in range(N_EPOCHS):
        u = torch.zeros(1, 1, H, W, device=dev)
        tot_ep, nw = 0.0, 0
        for start in range(0, T - 1, WINDOW):
            end = min(start + WINDOW, T - 1)
            L_ic = ic_loss(u) if start == 0 else torch.zeros((), device=dev)
            traj, flags = [u.detach()], [True]
            L_pde = L_bc = L_data = 0.0
            nd = 0
            for ti in range(start, end):
                un = model(u, covariate_stack(t_, ti))
                if cfg["use_physics"]:
                    r, _ = model.pde_residual(u, un, dt_months=1.0, covariates=physics_covariates(t_, ti),
                                              lat_rad_1d=ctx.lat_rad, dlon_rad=ctx.dlon_rad,
                                              dlat_rad=ctx.dlat_rad, **cfg["flags"])
                    L_pde = L_pde + pde_loss(r.squeeze(0), valid_t)
                    mid = 0.5 * (u[0, 0] + un[0, 0])
                    L_bc = L_bc + bc_loss(mid, ctx.lat_rad, ctx.dlon_rad, ctx.dlat_rad, bnd_t)
                if fit_months[ti]:
                    L_data = L_data + data_loss_monthly(un[0, 0], fi[ti + 1], fit_t, pos_weight=pos_weight)
                    nd += 1
                traj.append(un); flags.append(bool(fit_months[ti]))
                u = un
            n = end - start
            L_data = L_data / nd if nd else torch.zeros((), device=dev)
            if end >= T - 1:
                ts = torch.cat(traj, 0).squeeze(1)
                sel = np.where(np.array(flags))[0]
                if len(sel) > 1:
                    L_term = data_loss_terminal(ts[sel], term_label, fit_t, tau=5.0)
                    L_data = 0.5 * L_data + 0.5 * L_term
            losses = {"data": L_data, "ic": L_ic}
            if cfg["use_physics"]:
                losses.update(pde=L_pde / n, bc=L_bc / n)
            total, _ = bal.combine(losses, list(model.parameters()))
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            u = u.detach()
            tot_ep += float(total.detach()); nw += 1
        if not np.isfinite(tot_ep):
            print(f"[{tag}] NaN loss at epoch {ep+1}, stopping"); break
        if (ep + 1) % VAL_EVERY == 0 or ep == N_EPOCHS - 1:
            s = rollout()
            va_auc, va_ap, yv, pv = score(s, val_pix, val_months)
            tr_auc, _, _, _ = score(s, fit_pix, fit_months)
            pv_c = np.clip(pv, 1e-6, 1 - 1e-6)
            va_loss = float(-(yv * np.log(pv_c) + (1 - yv) * np.log(1 - pv_c)).mean())
            sched.step(va_loss)
            hist.append(dict(epoch=ep + 1, train_loss=tot_ep / nw, train_auc=tr_auc, val_auc=va_auc,
                             val_loss=va_loss, lr=opt.param_groups[0]["lr"]))
            print(f"  [{tag}] ep{ep+1} train_auc={tr_auc:.4f} val_auc={va_auc:.4f} val_loss={va_loss:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if va_auc > best_auc + 1e-4:
                best_auc, best_ep, bad = va_auc, ep + 1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    s = rollout()
    te_auc, te_ap, yt, pt = score(s, test_pix, test_months)
    va_auc, va_ap, _, _ = score(s, val_pix, val_months)
    peak = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else 0.0
    torch.cuda.reset_peak_memory_stats() if dev == "cuda" else None
    # physical coefficient diagnostics at the selected checkpoint
    with torch.no_grad():
        c_adv = float(torch.nn.functional.softplus(model.v_head.c_raw))
        w_d = float(torch.nn.functional.softplus(model.D_head.w_raw))
    res = dict(tag=tag, config=cfg_name, seed=seed, test_auc=float(te_auc), test_ap=float(te_ap),
               val_auc=float(va_auc), val_ap=float(va_ap), best_epoch=best_ep, epochs_run=ep + 1,
               train_time_sec=train_time, peak_gpu_mem_mb=peak, pos_weight=pos_weight,
               n_test=int(len(yt)), test_prevalence=float(np.mean(yt)),
               c_adv=c_adv, w_ndvi_anom=w_d, history=hist)
    # save predictions + state for paired tests / checkpoint manifest
    np.savez_compressed(os.path.join(OUT, f"pred_{tag}.npz"), y=yt.astype(np.int8), p=pt.astype(np.float32))
    ck = os.path.join(OUT, f"ckpt_{tag}.pt")
    torch.save({"model_state": model.state_dict(), "result": {k: v for k, v in res.items() if k != "history"}}, ck)
    res["checkpoint"] = ck
    res["checkpoint_sha256"] = sha256(ck)
    print(f"[{tag}] TEST AUC={te_auc:.4f} AP={te_ap:.4f} (val {va_auc:.4f}, best ep {best_ep}, {train_time:.0f}s)", flush=True)
    return res


def save(name, payload):
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(payload, f, indent=1, default=float)


def main():
    global N_EPOCHS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="A,B3,B3orig,B1,B2")
    ap.add_argument("--configs", default="nophys,diff,diffadv,full")
    ap.add_argument("--b2_configs", default="nophys,full")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    N_EPOCHS, OUT = args.epochs, args.out
    os.makedirs(OUT, exist_ok=True)
    dev = args.device
    ctx = Ctx(dev)
    seeds = [int(s) for s in args.seeds.split(",")]
    configs = args.configs.split(",")
    meta = dict(audit_date="2026-09-24", code_commit=git_commit(os.path.join(ROOT, "Physics_Informed_FireRisk_Model")),
                data=DATA_PATH, data_sha256=sha256(DATA_PATH), torch=torch.__version__, python=sys.version.split()[0],
                gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                protocol=dict(window=WINDOW, epochs=N_EPOCHS, width=WIDTH, layers=N_LAYERS, modes=MODES, lr=LR,
                              weight_decay=WEIGHT_DECAY, optimizer="AdamW", scheduler="ReduceLROnPlateau(val BCE, factor .5, patience 2 checks)",
                              early_stop="val AUC, check every 5 ep, patience 4 checks", split_seed=SPLIT_SEED))
    save("meta", meta)
    # export the exact partitions so the classical models can reuse them
    folds = b1_folds(ctx)
    region, km = b2_regions(ctx)
    fit_m, val_m, test_m, ty, vy = b3_years(ctx)
    tr, va, te = random_pixel_split(ctx)
    np.savez_compressed(os.path.join(OUT, "partitions.npz"), valid=ctx.valid, block_id=ctx.block_id,
                        b1_fold_of_block=np.array([[b, k] for k, f in enumerate(folds) for b in f]),
                        b2_region=region, b2_centroids=km.cluster_centers_, lon_grid=ctx.lon_grid,
                        lat_grid=ctx.lat_grid, trackA_train=tr, trackA_val=va, trackA_test=te,
                        b3_test_years=np.array(ty), b3_val_years=np.array(vy))

    for track in args.tracks.split(","):
        results = []
        if track == "A":
            tr, va, te = random_pixel_split(ctx)
            for c in configs:
                for s in seeds:
                    results.append(train_eval(ctx, c, s, tr, va, te, tag=f"A_{c}_s{s}"))
                    save("track_A", dict(meta=meta, results=results))
        elif track in ("B3", "B3orig"):
            fit_m, val_m, test_m, ty, vy = b3_years(ctx)
            allp = np.ones_like(ctx.valid)
            leakfree = track == "B3"
            cfgs = configs if leakfree else ["nophys", "full"]
            for c in cfgs:
                for s in seeds:
                    r = train_eval(ctx, c, s, allp, allp, allp, fit_m, val_m, test_m,
                                   terminal_label_from_fit_months=leakfree, tag=f"{track}_{c}_s{s}")
                    r.update(test_years=ty, val_years=vy)
                    results.append(r)
                    save(f"track_{track}", dict(meta=meta, results=results))
        elif track == "B1":
            folds = b1_folds(ctx)
            for c in configs:
                for s in seeds:
                    for k, fb in enumerate(folds):
                        te = ctx.valid & np.isin(ctx.block_id, list(fb))
                        tr_all = ctx.valid & ~te
                        tr, va = carve_block_validation(ctx, tr_all)
                        r = train_eval(ctx, c, s, tr, va, te, tag=f"B1_f{k}_{c}_s{s}")
                        r.update(fold=k, n_test_blocks=len(fb))
                        results.append(r)
                        save("track_B1", dict(meta=meta, results=results))
        elif track == "B2":
            region, km = b2_regions(ctx)
            for c in args.b2_configs.split(","):
                for s in seeds:
                    for rid in range(6):
                        te = region == rid
                        tr_all = ctx.valid & ~te
                        tr, va = carve_block_validation(ctx, tr_all)
                        r = train_eval(ctx, c, s, tr, va, te, tag=f"B2_r{rid}_{c}_s{s}")
                        r.update(region=rid, centroid=km.cluster_centers_[rid].tolist())
                        results.append(r)
                        save("track_B2", dict(meta=meta, results=results))


if __name__ == "__main__":
    main()
