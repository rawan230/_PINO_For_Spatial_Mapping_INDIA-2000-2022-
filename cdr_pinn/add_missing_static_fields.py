"""Adds forest_frac (from the Step 6 parquet, same rasterisation as fire_ever_frac)
and grad_e_x/grad_e_y (spherical gradient of the resampled elevation field,
via the verified spectral_ops) to the existing monthly-stack npz, without
re-running the expensive NDVI/FLDAS extraction.

Must be run after EVERY build_monthly_stacks.py run (the builder does not write
these three keys).

Audit 2026-09-25: forest_frac stays the 2001 (pre-fire-period) land-cover fraction.
The parquet is read through build_monthly_stacks.parquet_column_to_target, i.e. the
true native NDVI transform + floor-on-edges indexing. The column is looked up by name
in FOREST_FRAC_COLUMNS order, so a regenerated Step 6 parquet that keeps the
historical name (forest_frac_baseline) works unchanged.
"""
import argparse

import numpy as np
import torch
import pyarrow.parquet as pq

from build_monthly_stacks import (
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, TARGET_H, TARGET_W,
    PARQUET_PATH, DEFAULT_OUT_PATH, parquet_column_to_target,
)
from spectral_ops import spherical_gradient, neumann_periodic_extend, neumann_periodic_crop

IN_PATH = DEFAULT_OUT_PATH  # r"...\CDR_PINN_Data\cdr_pinn_monthly_stacks.npz"
# 2001 land-cover forest fraction; first name present in the parquet wins
FOREST_FRAC_COLUMNS = ("forest_frac_baseline", "forest_frac_2001", "v2_forest_frac_2001")


def build_forest_frac_grid():
    # forest_frac_recent (2020) and forest_frac_current (2022) were dropped from Step 6's
    # parquet 2026-08-21 -- both fell inside the 2000-2022 pooled fire label window, a real
    # reverse-causality leakage risk (post-fire LULC reclassification literature). Only
    # the 2001 baseline survives as the sole forest-fraction feature.
    names = pq.read_schema(PARQUET_PATH).names
    col = next((c for c in FOREST_FRAC_COLUMNS if c in names), None)
    if col is None:
        raise KeyError(f"none of {FOREST_FRAC_COLUMNS} found in {PARQUET_PATH}")
    print(f"  using parquet column '{col}'")
    return parquet_column_to_target(col)


def main(path=IN_PATH):
    print(f"Loading {path} ...")
    d = dict(np.load(path))

    print("Building forest_frac (from Step 6/7 parquet, resampled to target grid)...")
    forest_frac = build_forest_frac_grid()
    d["forest_frac"] = forest_frac

    print("Computing elevation gradient (verified spherical gradient operator)...")
    elevation = d["elevation"]
    elev_filled = np.nan_to_num(elevation, nan=float(np.nanmean(elevation)))
    lat_deg = np.linspace(LAT_MAX, LAT_MIN, TARGET_H)  # row 0 = top = max lat, matches from_bounds/reproject convention
    lat_rad = torch.tensor(np.radians(lat_deg), dtype=torch.float32)
    dlat_rad = float(np.radians(abs(lat_deg[1] - lat_deg[0])))
    dlon_rad = float(np.radians((LON_MAX - LON_MIN) / TARGET_W))

    e = torch.tensor(elev_filled, dtype=torch.float32)
    e_ext = neumann_periodic_extend(neumann_periodic_extend(e, axis=-1), axis=-2)
    n_h = e_ext.shape[-2]
    lat_ext = torch.cat([lat_rad, lat_rad.flip(0)[1:-1]])[:n_h]
    gx_ext, gy_ext = spherical_gradient(e_ext, lat_ext, dlon_rad, dlat_rad)
    H, W = e.shape
    gx = neumann_periodic_crop(neumann_periodic_crop(gx_ext, -2, H), -1, W).numpy()
    gy = neumann_periodic_crop(neumann_periodic_crop(gy_ext, -2, H), -1, W).numpy()
    d["grad_e_x"] = gx.astype(np.float32)
    d["grad_e_y"] = gy.astype(np.float32)

    print(f"Saving extended stack back to {path} ...")
    np.savez_compressed(path, **d)
    print("Done. New keys:", [k for k in d.keys() if k in ("forest_frac", "grad_e_x", "grad_e_y")])
    print(f"forest_frac range: [{np.nanmin(forest_frac):.4f}, {np.nanmax(forest_frac):.4f}], "
          f"nonzero cells: {(forest_frac > 0).sum()}")
    print(f"grad_e_x range: [{gx.min():.4e}, {gx.max():.4e}], grad_e_y range: [{gy.min():.4e}, {gy.max():.4e}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=IN_PATH, help="stack npz to extend in place")
    main(ap.parse_args().path)
