"""
Tracks B1 (spatial block CV), B2 (leave-one-region-out), B3 (leave-years-out,
new), and a physics-vs-no-physics data-efficiency comparison, all against the
identical full_cdr configuration and data as Track A (train.py).

Shares model.py/losses.py; reimplements the training loop here (rather than
importing train.run()) because each track needs a different train/test
*masking* strategy (spatial pixel masks for B1/B2, temporal month masks for
B3, and a physics-on/off toggle for the data-efficiency test) that doesn't
cleanly fit train.py's single random-pixel-split assumption.
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score

from model import CDRPINN
from losses import data_loss_monthly, data_loss_terminal, pde_loss, bc_loss, ic_loss, AdaptiveLossBalancer
from build_monthly_stacks import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, TARGET_H, TARGET_W

DATA_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_monthly_stacks.npz"
CKPT_DIR = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data"

WINDOW = 24
WIDTH = 32
N_LAYERS = 4
MODES = 16
LR = 1e-3
SEED = 42
FULL_CDR = dict(use_diffusion=True, use_advection=True, use_reaction=True)


def load_data(device):
    d = np.load(DATA_PATH)
    H, W = TARGET_H, TARGET_W
    n_months = len(d["months"])

    def t(x):
        return torch.tensor(np.nan_to_num(x, nan=0.0), dtype=torch.float32, device=device)

    ndvi_f1 = d["ndvi_f1"]
    valid_np = ~np.isnan(ndvi_f1)
    up = np.roll(valid_np, 1, axis=0); down = np.roll(valid_np, -1, axis=0)
    left = np.roll(valid_np, 1, axis=1); right = np.roll(valid_np, -1, axis=1)
    boundary_np = valid_np & (~up | ~down | ~left | ~right)

    tensors = dict(
        ndvi_f1=t(ndvi_f1), ndvi_anomaly=t(d["ndvi_anomaly"]), forest_frac=t(d["forest_frac"]),
        dryness=t(d["dryness_proxy"]), slope=t(d["slope"]), dist_roads=t(d["dist_roads"]),
        elevation=t(d["elevation"]), grad_e_x=t(d["grad_e_x"]), grad_e_y=t(d["grad_e_y"]),
        fire_indicator=t(d["fire_indicator"]), fire_ever_frac=t(d["fire_ever_frac"]),
    )
    fire_ever_binary_np = (d["fire_ever_frac"] > 0).astype(np.float32)
    months = d["months"]
    years_np = np.array([int(m[:4]) for m in months])

    lat_deg = np.linspace(LAT_MAX, LAT_MIN, H)
    lon_deg = np.linspace(LON_MIN, LON_MAX, W)
    lat_rad = torch.tensor(np.radians(lat_deg), dtype=torch.float32, device=device)
    dlat_rad = float(np.radians(abs(lat_deg[1] - lat_deg[0])))
    dlon_rad = float(np.radians((LON_MAX - LON_MIN) / W))
    lon_grid, lat_grid = np.meshgrid(lon_deg, lat_deg)  # (H,W) each

    return dict(
        H=H, W=W, n_months=n_months, valid_np=valid_np, boundary_np=boundary_np,
        tensors=tensors, fire_ever_binary_np=fire_ever_binary_np, years_np=years_np,
        lat_rad=lat_rad, dlat_rad=dlat_rad, dlon_rad=dlon_rad, lon_grid=lon_grid, lat_grid=lat_grid,
    )


def train_and_eval(ctx, device, train_pixel_mask, test_pixel_mask, tag,
                    epochs=80, term_flags=FULL_CDR, use_physics=True,
                    train_month_mask=None, eval_month_mask=None, seed=SEED,
                    val_frac=0.1875, val_every=5, patience=4):
    """train_pixel_mask/test_pixel_mask: (H,W) bool -- which pixels' data-loss
    supervision is used for training vs. held out for evaluation (spatial split).
    train_month_mask/eval_month_mask: (n_months-1,) bool, optional -- which
    MONTHS' data-loss supervision is used for training vs. evaluation (temporal
    split, Track B3). If None, all months are used for both (spatial-only split).
    use_physics=False zeroes out the PDE+BC loss entirely (data-efficiency test).
    seed: model-init/training-noise seed -- fold/region/year assignment stays
    fixed across seeds (set with the module-level SEED constant in each track
    function) so multi-seed runs vary only the model's own stochasticity, not
    which pixels/years land in which split.

    Genuine validation-set-driven early stopping (previously missing here --
    every B1/B2/B3/data-efficiency/multiseed run trained for a fixed epoch
    budget with no validation monitoring at all, unlike train_standard_protocol.py's
    Track A run, which already does this). val_frac=0.1875 of the TRAIN portion
    is carved out as validation (never touches test_pixel_mask/test years), giving
    ~65/15/20 overall on an 80/20 track split -- matching the standard protocol's
    ratios. For a spatial split (B1/B2/data-eff/multiseed A), validation pixels are
    carved from train_pixel_mask. For a temporal split (B3, train_month_mask given
    and different from eval_month_mask), validation TRAIN YEARS are carved instead
    (pixel-level val makes no sense when the held-out dimension is time)."""
    torch.manual_seed(seed)
    H, W, n_months = ctx["H"], ctx["W"], ctx["n_months"]
    t_ = ctx["tensors"]
    valid_mask = torch.tensor(ctx["valid_np"], dtype=torch.bool, device=device)
    boundary_mask = torch.tensor(ctx["boundary_np"], dtype=torch.bool, device=device)
    lat_rad, dlon_rad, dlat_rad = ctx["lat_rad"], ctx["dlon_rad"], ctx["dlat_rad"]

    is_temporal_split = (train_month_mask is not None) and (eval_month_mask is not None) \
        and not np.array_equal(train_month_mask, eval_month_mask)

    val_rng = np.random.RandomState(seed + 1000)
    if is_temporal_split:
        # Carve validation YEARS out of the train years only (test years untouched).
        train_years_present = sorted(set(ctx["years_np"][:-1][train_month_mask].tolist()))
        n_val_years = max(1, int(round(len(train_years_present) * val_frac)))
        val_years = set(val_rng.choice(train_years_present, size=n_val_years, replace=False).tolist())
        fit_month_mask = train_month_mask & np.array([y not in val_years for y in ctx["years_np"][:-1]])
        val_month_mask = train_month_mask & np.array([y in val_years for y in ctx["years_np"][:-1]])
        val_pixel_mask_for_eval = ctx["valid_np"]  # all pixels, val split is temporal here
        train_data_mask = torch.tensor(ctx["valid_np"] & train_pixel_mask, dtype=torch.bool, device=device)
    else:
        # Carve validation PIXELS out of the train pixel mask only (test pixels untouched).
        train_idx = np.argwhere(ctx["valid_np"] & train_pixel_mask)
        perm = val_rng.permutation(len(train_idx))
        n_val = int(len(train_idx) * val_frac)
        val_idx, fit_idx = train_idx[perm[:n_val]], train_idx[perm[n_val:]]
        fit_pixel_mask = np.zeros_like(ctx["valid_np"]); fit_pixel_mask[fit_idx[:, 0], fit_idx[:, 1]] = True
        val_pixel_mask_for_eval = np.zeros_like(ctx["valid_np"]); val_pixel_mask_for_eval[val_idx[:, 0], val_idx[:, 1]] = True
        train_data_mask = torch.tensor(fit_pixel_mask, dtype=torch.bool, device=device)
        fit_month_mask = train_month_mask if train_month_mask is not None else np.ones(n_months - 1, dtype=bool)
        val_month_mask = fit_month_mask  # same months, held-out pixels

    train_month_mask = fit_month_mask
    if eval_month_mask is None:
        eval_month_mask = np.ones(n_months - 1, dtype=bool)

    monthly_pos_rate = t_["fire_indicator"][:-1][train_month_mask][:, train_data_mask].mean().item()
    pos_weight = (1.0 - monthly_pos_rate) / max(monthly_pos_rate, 1e-6)

    def covariate_stack(ti):
        return torch.stack([
            t_["ndvi_f1"], t_["ndvi_anomaly"][ti], t_["forest_frac"], t_["dryness"][ti],
            t_["slope"], t_["dist_roads"], t_["elevation"],
        ], dim=0).unsqueeze(0)

    def physics_covariates(ti):
        return {
            "ndvi_f1": t_["ndvi_f1"].unsqueeze(0), "forest_frac": t_["forest_frac"].unsqueeze(0),
            "ndvi_anomaly": t_["ndvi_anomaly"][ti].unsqueeze(0),
            "grad_e_x": t_["grad_e_x"].unsqueeze(0), "grad_e_y": t_["grad_e_y"].unsqueeze(0),
            "dryness": t_["dryness"][ti].unsqueeze(0), "slope": t_["slope"].unsqueeze(0),
            "dist_roads": t_["dist_roads"].unsqueeze(0),
        }

    from losses import lse_pool as _lse_pool

    def compute_rollout(model_, temporal_per_month=False):
        """One expensive sequential rollout pass over all months. Returns either
        a (n_months-1, H, W) per-month sigmoid-score array (temporal_per_month=True,
        needed because Track B3's train/val/test split is on MONTHS, so every month
        must be scoreable independently) or a single LSE-pooled (H,W) array
        (spatial-split tracks, where train/val/test differ only by PIXEL). Scoring
        this once and slicing it against multiple pixel/month masks (see
        score_from_rollout below) is what lets a train-AUC be computed at no extra
        rollout cost beyond the val-AUC that was already being computed here."""
        model_.eval()
        with torch.no_grad():
            u_ = torch.zeros(1, 1, H, W, device=device)
            if temporal_per_month:
                all_scores = []
                for ti in range(n_months - 1):
                    u_next_ = model_(u_, covariate_stack(ti))
                    all_scores.append(torch.sigmoid(u_next_.squeeze(0).squeeze(0)).cpu().numpy())
                    u_ = u_next_
                result = np.stack(all_scores, axis=0)
            else:
                scores = []
                for ti in range(n_months - 1):
                    u_next_ = model_(u_, covariate_stack(ti))
                    scores.append(torch.sigmoid(u_next_.squeeze(0).squeeze(0)))
                    u_ = u_next_
                score_stack = torch.stack(scores, dim=0)
                result = _lse_pool(score_stack, dim=0, tau=5.0).cpu().numpy()
        model_.train()
        return result

    def score_from_rollout(rollout, eval_pixel_mask_np, month_mask_np, temporal_per_month=False):
        if temporal_per_month:
            y_true_list, y_score_list = [], []
            for ti in range(n_months - 1):
                if month_mask_np[ti]:
                    y_score_list.append(rollout[ti][eval_pixel_mask_np])
                    y_true_list.append(t_["fire_indicator"][ti + 1].cpu().numpy()[eval_pixel_mask_np])
            y_true, y_score = np.concatenate(y_true_list), np.concatenate(y_score_list)
        else:
            y_true = ctx["fire_ever_binary_np"][eval_pixel_mask_np]
            y_score = rollout[eval_pixel_mask_np]
        valid_ = ~np.isnan(y_score) & ~np.isnan(y_true)
        y_true, y_score = y_true[valid_], y_score[valid_]
        if len(np.unique(y_true)) < 2:
            return float("nan"), float("nan")
        return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)

    def rollout_auc(model_, eval_pixel_mask_np, month_mask_np, temporal_per_month=False):
        """Single-shot convenience wrapper (final test eval only -- one rollout, one score)."""
        rollout = compute_rollout(model_, temporal_per_month)
        return score_from_rollout(rollout, eval_pixel_mask_np, month_mask_np, temporal_per_month)

    model = CDRPINN(n_static_channels=7, width=WIDTH, modes_h=MODES, modes_w=MODES, n_layers=N_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Cosine LR decay -- added specifically to test the hypothesis that the width=64
    # scale-up regression (AUC 0.9406->0.9292) was an LR-schedule/overfitting
    # interaction rather than a physics-formulation problem: the fixed lr=1e-3 used
    # for every prior run was never tuned for the larger model's different
    # optimization landscape. eta_min=lr/100 rather than 0, so late epochs keep a
    # small but nonzero step rather than fully freezing.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR / 100)
    loss_names = ["data", "ic"] if not use_physics else ["data", "pde", "bc", "ic"]
    balancer = AdaptiveLossBalancer(loss_names, update_every=5)

    best_val_auc, best_epoch, best_state, epochs_since_improve = -1.0, 0, None, 0
    train_data_mask_np = train_data_mask.cpu().numpy()
    train_auc_history, val_auc_history = [], []

    t_start = time.time()
    for epoch in range(epochs):
        u = torch.zeros(1, 1, H, W, device=device)
        for start in range(0, n_months - 1, WINDOW):
            end = min(start + WINDOW, n_months - 1)
            window_data = 0.0
            window_pde, window_bc = 0.0, 0.0
            window_ic = ic_loss(u) if start == 0 else torch.zeros((), device=device)
            traj = [u.detach()]
            n_data_terms = 0

            for ti in range(start, end):
                cov_stack = covariate_stack(ti)
                u_next = model(u, cov_stack)

                if use_physics:
                    residual, _ = model.pde_residual(
                        u, u_next, dt_months=1.0, covariates=physics_covariates(ti),
                        lat_rad_1d=lat_rad, dlon_rad=dlon_rad, dlat_rad=dlat_rad, **term_flags)
                    window_pde = window_pde + pde_loss(residual.squeeze(0), valid_mask)
                    mid = 0.5 * (u.squeeze(0).squeeze(0) + u_next.squeeze(0).squeeze(0))
                    window_bc = window_bc + bc_loss(mid, lat_rad, dlon_rad, dlat_rad, boundary_mask)

                if train_month_mask[ti]:
                    window_data = window_data + data_loss_monthly(
                        u_next.squeeze(0).squeeze(0), t_["fire_indicator"][ti + 1], train_data_mask,
                        pos_weight=pos_weight)
                    n_data_terms += 1
                traj.append(u_next)
                u = u_next

            n_steps = end - start
            if use_physics:
                window_pde, window_bc = window_pde / n_steps, window_bc / n_steps
            window_data = window_data / max(n_data_terms, 1) if n_data_terms > 0 else torch.zeros((), device=device)

            is_last_window = (end >= n_months - 1)
            if is_last_window:
                traj_stack = torch.cat(traj, dim=0).squeeze(1)
                # traj_stack has length (end-start+1): traj[0]=u at window start (always
                # "in", it's just the carried-over state, not a supervised month), then
                # traj[1:] correspond to ti=start..end-1 in this WINDOW's own local
                # indexing -- must NOT be indexed with a mask built at full-sequence (266)
                # length (that was the bug: an out-of-bounds CUDA gather).
                window_train_flags = [True] + [bool(train_month_mask[ti]) for ti in range(start, end)]
                idx = np.where(np.array(window_train_flags))[0]
                if len(idx) > 1:
                    window_terminal = data_loss_terminal(traj_stack[idx], t_["fire_ever_frac"], train_data_mask, tau=5.0)
                    window_data = 0.5 * window_data + 0.5 * window_terminal

            if use_physics:
                losses = {"data": window_data, "pde": window_pde, "bc": window_bc, "ic": window_ic}
            else:
                losses = {"data": window_data, "ic": window_ic}
            total, weights = balancer.combine(losses, list(model.parameters()))

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            u = u.detach()

        scheduler.step()

        # ---- Validation-based checkpoint selection + early stopping (this is the
        # fix: every track previously trained for a fixed epoch budget with zero
        # validation monitoring, unlike train_standard_protocol.py's Track A). ----
        if (epoch + 1) % val_every == 0 or epoch == epochs - 1:
            rollout = compute_rollout(model, temporal_per_month=is_temporal_split)
            val_auc, val_ap = score_from_rollout(rollout, val_pixel_mask_for_eval, val_month_mask,
                                                  temporal_per_month=is_temporal_split)
            train_auc, _ = score_from_rollout(rollout, train_data_mask_np, train_month_mask,
                                               temporal_per_month=is_temporal_split)
            train_auc_history.append([epoch + 1, train_auc])
            val_auc_history.append([epoch + 1, val_auc])
            improved = (not np.isnan(val_auc)) and (val_auc > best_val_auc)
            print(f"  [{tag}] epoch {epoch+1}/{epochs}  train_AUC={train_auc:.4f}  val_AUC={val_auc:.4f}"
                  + ("  <- best" if improved else ""))
            if improved:
                best_val_auc, best_epoch = val_auc, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"  [{tag}] early stop at epoch {epoch+1} (no val_AUC improvement in {patience} checks)")
                break

    train_time = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Final test evaluation, on the true held-out test_pixel_mask/test years
    # (never touched by training or by the validation carve-out above), using the
    # best-val-AUC checkpoint -- same LSE-pooled scoring path as validation. ----
    eval_pixel_mask = ctx["valid_np"] & test_pixel_mask
    auc, ap = rollout_auc(model, eval_pixel_mask, eval_month_mask, temporal_per_month=is_temporal_split)

    result = {"tag": tag, "roc_auc": float(auc), "ap": float(ap), "train_time_sec": train_time,
              "epochs_run": epoch + 1, "best_epoch": best_epoch, "best_val_auc": float(best_val_auc),
              "train_auc_history": train_auc_history, "val_auc_history": val_auc_history}
    print(f"[{tag}] AUC={result['roc_auc']:.4f} AP={result['ap']:.4f}  "
          f"(best val_AUC={best_val_auc:.4f} @ epoch {best_epoch}, stopped @ {epoch+1}/{epochs}, {train_time:.1f}s)")
    return result


def run_track_b1(ctx, device, n_folds=3, epochs=50, use_physics=True, seed=SEED):
    tag_suffix = "physics" if use_physics else "nophysics"
    print(f"\n=== Track B1: {n_folds}-fold spatial block CV (2deg x 2deg blocks) [{tag_suffix}, seed={seed}] ===")
    block_deg = 2.0
    block_lon = np.floor((ctx["lon_grid"] - LON_MIN) / block_deg).astype(int)
    block_lat = np.floor((ctx["lat_grid"] - LAT_MIN) / block_deg).astype(int)
    block_id = block_lon * 1000 + block_lat
    valid_blocks = np.unique(block_id[ctx["valid_np"]])
    rng = np.random.RandomState(SEED)  # fold assignment stays fixed across seeds/physics-toggle for a fair comparison
    perm = rng.permutation(valid_blocks)
    folds = np.array_split(perm, n_folds)

    results = []
    for k in range(n_folds):
        test_blocks = set(folds[k].tolist())
        test_mask = np.isin(block_id, list(test_blocks))
        train_mask = ctx["valid_np"] & ~test_mask
        r = train_and_eval(ctx, device, train_mask, test_mask, tag=f"B1_fold{k}_{tag_suffix}_seed{seed}",
                            epochs=epochs, use_physics=use_physics, seed=seed)
        results.append(r)
    aucs = [r["roc_auc"] for r in results if not np.isnan(r["roc_auc"])]
    print(f"Track B1 [{tag_suffix}, seed={seed}] mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    return results


def run_track_b2(ctx, device, n_regions=6, epochs=50, use_physics=True, seed=SEED):
    tag_suffix = "physics" if use_physics else "nophysics"
    print(f"\n=== Track B2: leave-one-region-out ({n_regions} KMeans regions) [{tag_suffix}, seed={seed}] ===")
    coords = np.stack([ctx["lon_grid"][ctx["valid_np"]], ctx["lat_grid"][ctx["valid_np"]]], axis=1)
    km = KMeans(n_clusters=n_regions, random_state=SEED, n_init=10).fit(coords)  # region assignment fixed across seeds/physics-toggle
    region_grid = np.full(ctx["lon_grid"].shape, -1, dtype=int)
    region_grid[ctx["valid_np"]] = km.labels_

    results = []
    for r_id in range(n_regions):
        test_mask = region_grid == r_id
        train_mask = ctx["valid_np"] & ~test_mask
        n_test = test_mask.sum()
        if n_test < 20:
            print(f"  region {r_id}: only {n_test} pixels, skipping (too small for a meaningful AUC)")
            continue
        r = train_and_eval(ctx, device, train_mask, test_mask, tag=f"B2_region{r_id}_{tag_suffix}_seed{seed}",
                            epochs=epochs, use_physics=use_physics, seed=seed)
        results.append(r)
    aucs = [r["roc_auc"] for r in results if not np.isnan(r["roc_auc"])]
    print(f"Track B2 [{tag_suffix}, seed={seed}] mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    return results


def run_track_b3(ctx, device, test_frac=0.2, epochs=80, use_physics=True, seed=SEED):
    tag_suffix = "physics" if use_physics else "nophysics"
    print(f"\n=== Track B3: leave-years-out (temporal generalization) [{tag_suffix}, seed={seed}] ===")
    years = ctx["years_np"][:-1]  # aligned to the (n_months-1) prediction indices
    unique_years = np.unique(years)
    rng = np.random.RandomState(SEED)  # held-out years stay fixed across seeds/physics-toggle for a fair comparison
    n_test_years = max(1, int(len(unique_years) * test_frac))
    test_years = set(rng.choice(unique_years, size=n_test_years, replace=False).tolist())
    train_month_mask = np.array([y not in test_years for y in years])
    eval_month_mask = ~train_month_mask
    print(f"  Test years: {sorted(test_years)} ({eval_month_mask.sum()} held-out months)")

    all_pixels = ctx["valid_np"]
    r = train_and_eval(ctx, device, all_pixels, all_pixels, tag=f"B3_leave_years_out_{tag_suffix}_seed{seed}",
                        epochs=epochs, use_physics=use_physics, seed=seed,
                        train_month_mask=train_month_mask, eval_month_mask=eval_month_mask)
    return [r]


def run_data_efficiency_test(ctx, device, epochs=80):
    print("\n=== Data-efficiency test: full-physics vs. no-physics, identical sparse supervision ===")
    rng = np.random.RandomState(SEED)
    valid_idx = np.argwhere(ctx["valid_np"])
    n_test = int(len(valid_idx) * 0.2)
    perm = rng.permutation(len(valid_idx))
    test_idx, train_idx = valid_idx[perm[:n_test]], valid_idx[perm[n_test:]]
    train_mask = np.zeros_like(ctx["valid_np"]); train_mask[train_idx[:, 0], train_idx[:, 1]] = True
    test_mask = np.zeros_like(ctx["valid_np"]); test_mask[test_idx[:, 0], test_idx[:, 1]] = True

    r_physics = train_and_eval(ctx, device, train_mask, test_mask, tag="dataeff_full_physics",
                                epochs=epochs, use_physics=True)
    r_nophysics = train_and_eval(ctx, device, train_mask, test_mask, tag="dataeff_no_physics",
                                  epochs=epochs, use_physics=False)
    print(f"Physics AUC={r_physics['roc_auc']:.4f} vs. No-physics AUC={r_nophysics['roc_auc']:.4f} "
          f"(same architecture, same data, same split)")
    return [r_physics, r_nophysics]


def run_physics_vs_nophysics_all_tracks(ctx, device):
    """The item repeatedly flagged throughout this study as the single most
    important unresolved experiment: physics-vs-no-physics was previously only
    tested on Track A (random split). This runs it on B1/B2/B3 too -- the harder
    tracks, where the literature predicts a physics-informed advantage should
    actually appear under distribution shift, if it exists at all."""
    results = {}
    for track_name, fn in [("B1", run_track_b1), ("B2", run_track_b2), ("B3", run_track_b3)]:
        print(f"\n{'='*70}\nTrack {track_name}: physics vs. no-physics\n{'='*70}")
        r_physics = fn(ctx, device, use_physics=True)
        r_nophysics = fn(ctx, device, use_physics=False)
        aucs_p = [r["roc_auc"] for r in r_physics if not np.isnan(r["roc_auc"])]
        aucs_np = [r["roc_auc"] for r in r_nophysics if not np.isnan(r["roc_auc"])]
        mean_p, mean_np = float(np.mean(aucs_p)), float(np.mean(aucs_np))
        print(f"\n>>> Track {track_name} SUMMARY: physics={mean_p:.4f}, no-physics={mean_np:.4f}, "
              f"delta={mean_p - mean_np:+.4f} ({'physics helps' if mean_p > mean_np else 'no physics advantage'})")
        results[track_name] = {
            "physics": r_physics, "no_physics": r_nophysics,
            "physics_mean_auc": mean_p, "no_physics_mean_auc": mean_np,
            "delta": mean_p - mean_np,
        }
    return results


def run_multiseed_track_a(ctx, device, seeds=(42, 43, 44), epochs=80):
    """Multi-seed robustness for the study's single most-cited number (Track A /
    full CDR random-split AUC) -- item 4 of the CDR-PINN completeness audit.
    Full multi-seed x all-tracks x physics-toggle would be ~180 runs, well beyond
    this pass's scope; this specifically targets the headline number, the one
    every other comparison in the paper is anchored to."""
    print(f"\n{'='*70}\nMulti-seed Track A (full CDR, random split): seeds={seeds}\n{'='*70}")
    rng = np.random.RandomState(SEED)  # split itself fixed across seeds -- only model init/training noise varies
    valid_idx = np.argwhere(ctx["valid_np"])
    n_test = int(len(valid_idx) * 0.2)
    perm = rng.permutation(len(valid_idx))
    test_idx, train_idx = valid_idx[perm[:n_test]], valid_idx[perm[n_test:]]
    train_mask = np.zeros_like(ctx["valid_np"]); train_mask[train_idx[:, 0], train_idx[:, 1]] = True
    test_mask = np.zeros_like(ctx["valid_np"]); test_mask[test_idx[:, 0], test_idx[:, 1]] = True

    results = []
    for s in seeds:
        r = train_and_eval(ctx, device, train_mask, test_mask, tag=f"trackA_multiseed_seed{s}",
                            epochs=epochs, seed=s)
        results.append(r)
        print(f"  seed={s}: AUC={r['roc_auc']:.4f}")
    aucs = [r["roc_auc"] for r in results if not np.isnan(r["roc_auc"])]
    mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs))
    print(f">>> Multi-seed Track A: {mean_auc:.4f} +/- {std_auc:.4f} (n={len(aucs)} seeds)")
    return {"per_seed": results, "mean_auc": mean_auc, "std_auc": std_auc, "seeds": list(seeds)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["b1", "b2", "b3", "dataeff", "physics_vs_nophysics_all", "multiseed_a", "all"], default="all")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    ctx = load_data(device)
    print(f"Valid pixels: {ctx['valid_np'].sum()}, months: {ctx['n_months']}")

    all_results = {}
    if args.track in ("b1", "all"):
        all_results["B1"] = run_track_b1(ctx, device)
    if args.track in ("b2", "all"):
        all_results["B2"] = run_track_b2(ctx, device)
    if args.track in ("b3", "all"):
        all_results["B3"] = run_track_b3(ctx, device)
    if args.track in ("dataeff", "all"):
        all_results["dataeff"] = run_data_efficiency_test(ctx, device)
    if args.track == "physics_vs_nophysics_all":
        all_results["physics_vs_nophysics_all_tracks"] = run_physics_vs_nophysics_all_tracks(ctx, device)
    if args.track == "multiseed_a":
        all_results["multiseed_track_a"] = run_multiseed_track_a(ctx, device)

    out_path = f"{CKPT_DIR}/cdr_pinn_validation_tracks_{args.track}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")
