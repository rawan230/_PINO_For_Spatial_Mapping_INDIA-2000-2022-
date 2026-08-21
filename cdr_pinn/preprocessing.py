"""
CDR-PINN preprocessing -- the single, standard source of truth for how raw monthly-
stack data becomes model-ready tensors and splits. Factored out of train_standard_
protocol.py so every training/evaluation/diagnostic script builds its splits and
covariate tensors identically, rather than each script re-implementing (and risking
silent drift in) its own copy.

Standard protocol (adopted 2026-08-21, replacing the old ad-hoc 80/20-split-only
convention used throughout this project's earlier scripts):
  - Genuine 65/15/20% train/validation/test split of valid in-India pixels, seed=42.
  - Validation pixels are used for any architecture/hyperparameter decision (e.g. the
    LR-schedule choice adopted in train_standard_protocol.py); test pixels are touched
    exactly once, at the very end, for the final reported number.
  - forest_frac is loaded from the post-2026-08-21 corrected field (built from
    forest_frac_baseline / 2001 land cover via add_missing_static_fields.py) -- NOT the
    original forest_frac_recent (2020), which was dropped from Step 6's parquet as a
    data-leakage fix (forest_frac_recent/current overlapped the fire label's own
    2000-2022 window, a real reverse-causality risk from post-fire LULC
    reclassification). Any script importing this module automatically gets the
    corrected feature -- there's no separate code path for the old one.
"""
import numpy as np
import torch

from build_monthly_stacks import LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, TARGET_H, TARGET_W

DATA_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_monthly_stacks.npz"
SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.20

COVARIATE_NAMES = ["ndvi_f1", "ndvi_anomaly", "forest_frac", "dryness", "slope", "dist_roads", "elevation"]


def build_masks_3way(ndvi_f1, seed=SEED, val_frac=VAL_FRAC, test_frac=TEST_FRAC):
    """Pixel-level 65/15/20 train/val/test split over valid (non-NaN) in-India pixels,
    plus a boundary mask (pixels touching a NaN neighbour, for the Neumann BC loss)."""
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


def load_tensors(device):
    """Loads the monthly-stack npz and returns every tensor a training/eval script
    needs, keyed by name -- one place to add/rename a covariate, not N places."""
    d = np.load(DATA_PATH)

    def t(x):
        return torch.tensor(np.nan_to_num(x, nan=0.0), dtype=torch.float32, device=device)

    tensors = {
        "ndvi_f1": t(d["ndvi_f1"]),
        "ndvi_anomaly": t(d["ndvi_anomaly"]),
        "forest_frac": t(d["forest_frac"]),  # corrected 2026-08-21 -- see module docstring
        "dryness": t(d["dryness_proxy"]),
        "slope": t(d["slope"]),
        "dist_roads": t(d["dist_roads"]),
        "elevation": t(d["elevation"]),
        "grad_e_x": t(d["grad_e_x"]),
        "grad_e_y": t(d["grad_e_y"]),
        "fire_indicator": t(d["fire_indicator"]),
        "fire_ever_frac": t(d["fire_ever_frac"]),
    }
    fire_ever_binary_np = (d["fire_ever_frac"] > 0).astype(np.float32)
    ndvi_f1_np = d["ndvi_f1"]
    n_months = len(d["months"])
    return tensors, fire_ever_binary_np, ndvi_f1_np, n_months


def covariate_stack(tensors, ti):
    """Static + time-varying covariates stacked for one month's forward pass,
    in the fixed channel order every CDRPINN checkpoint expects."""
    return torch.stack([
        tensors["ndvi_f1"], tensors["ndvi_anomaly"][ti], tensors["forest_frac"],
        tensors["dryness"][ti], tensors["slope"], tensors["dist_roads"], tensors["elevation"],
    ], dim=0).unsqueeze(0)


def physics_covariates(tensors, ti):
    """Covariates the physics heads (D/v/rho) read from -- a superset of
    covariate_stack's channels since grad_e_x/grad_e_y feed the advection head only."""
    return {
        "ndvi_f1": tensors["ndvi_f1"].unsqueeze(0), "forest_frac": tensors["forest_frac"].unsqueeze(0),
        "ndvi_anomaly": tensors["ndvi_anomaly"][ti].unsqueeze(0),
        "grad_e_x": tensors["grad_e_x"].unsqueeze(0), "grad_e_y": tensors["grad_e_y"].unsqueeze(0),
        "dryness": tensors["dryness"][ti].unsqueeze(0), "slope": tensors["slope"].unsqueeze(0),
        "dist_roads": tensors["dist_roads"].unsqueeze(0),
    }


def grid_metadata(device):
    """Lat/lon spacing in radians, needed by the spherical differential operators."""
    lat_deg = np.linspace(LAT_MAX, LAT_MIN, TARGET_H)
    lat_rad = torch.tensor(np.radians(lat_deg), dtype=torch.float32, device=device)
    dlat_rad = float(np.radians(abs(lat_deg[1] - lat_deg[0])))
    dlon_rad = float(np.radians((LON_MAX - LON_MIN) / TARGET_W))
    return lat_rad, dlat_rad, dlon_rad


def compute_pos_weight(tensors, train_data_mask):
    """Inverse-frequency positive-class weight, measured directly from TRAIN pixels
    only (not global rate), so it reflects exactly what a given run trains on."""
    monthly_pos_rate = tensors["fire_indicator"][:, train_data_mask].mean().item()
    pos_weight = (1.0 - monthly_pos_rate) / max(monthly_pos_rate, 1e-6)
    return monthly_pos_rate, pos_weight


if __name__ == "__main__":
    # Self-check: confirm the split is reproducible and the corrected forest_frac loads.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensors, fire_ever_binary_np, ndvi_f1_np, n_months = load_tensors(device)
    valid, boundary, train_m, val_m, test_m = build_masks_3way(ndvi_f1_np)
    print(f"Valid={valid.sum()}  train={train_m.sum()}  val={val_m.sum()}  test={test_m.sum()}  "
          f"(fractions: {train_m.sum()/valid.sum():.3f}/{val_m.sum()/valid.sum():.3f}/{test_m.sum()/valid.sum():.3f})")
    ff = tensors["forest_frac"].cpu().numpy()
    print(f"forest_frac range=[{np.nanmin(ff):.4f}, {np.nanmax(ff):.4f}], nonzero={np.sum(ff > 0)}  "
          f"(should match add_missing_static_fields.py's post-fix report: nonzero=14342)")
