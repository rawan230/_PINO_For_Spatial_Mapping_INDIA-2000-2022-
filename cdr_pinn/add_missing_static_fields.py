"""Adds forest_frac (from the Step 6/7 parquet, same pattern as fire_ever_grid)
and grad_e_x/grad_e_y (spherical gradient of the resampled elevation field,
via the verified spectral_ops) to the existing monthly-stack npz, without
re-running the expensive NDVI/FLDAS extraction."""
import numpy as np
import torch
from rasterio.transform import from_bounds
import pyarrow.parquet as pq

from build_monthly_stacks import (
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, TARGET_H, TARGET_W, TARGET_CRS,
    PARQUET_PATH, resample_to_target,
)
from spectral_ops import spherical_gradient, neumann_periodic_extend, neumann_periodic_crop

IN_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data\cdr_pinn_monthly_stacks.npz"
OUT_PATH = IN_PATH  # overwrite with the extended set


def build_forest_frac_grid():
    table = pq.read_table(PARQUET_PATH, columns=["lon", "lat", "forest_frac_recent"])
    df = table.to_pandas()
    native_h, native_w = 3641, 3504
    native_transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, native_w, native_h)
    inv = ~native_transform
    cols, rows = inv * (df["lon"].values, df["lat"].values)
    rows = np.clip(rows.astype(int), 0, native_h - 1)
    cols = np.clip(cols.astype(int), 0, native_w - 1)
    grid = np.full((native_h, native_w), np.nan, dtype=np.float32)
    grid[rows, cols] = df["forest_frac_recent"].values.astype(np.float32)
    return resample_to_target(grid, native_transform, TARGET_CRS)


def main():
    print(f"Loading {IN_PATH} ...")
    d = dict(np.load(IN_PATH))

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

    print(f"Saving extended stack back to {OUT_PATH} ...")
    np.savez_compressed(OUT_PATH, **d)
    print("Done. New keys:", [k for k in d.keys() if k in ("forest_frac", "grad_e_x", "grad_e_y")])
    print(f"forest_frac range: [{np.nanmin(forest_frac):.4f}, {np.nanmax(forest_frac):.4f}], "
          f"nonzero cells: {(forest_frac > 0).sum()}")
    print(f"grad_e_x range: [{gx.min():.4e}, {gx.max():.4e}], grad_e_y range: [{gy.min():.4e}, {gy.max():.4e}]")


if __name__ == "__main__":
    main()
