"""
Builds the per-month (a_t, u_t) training data the CDR-PINN needs, from raw
source files -- none of this exists pre-built anywhere in the project (confirmed
by direct recon of NDVI_DATA_INDIA_ and the FLDAS folder: both notebooks compute
monthly stacks transiently in RAM and only persist time-mean summaries).

Target grid: 256x256, same real India bounds used throughout the CDR-PINN design
docs (lon [68.20,97.40], lat [6.75,37.09]) -- downsampled via Resampling.average
(this project's established convention for continuous-field aggregation, e.g.
Step 4/6's FLDAS and land-cover resampling), not naive/incorrect block-decimation.

Run with --smoke-test first (3 sample months) before the full 266-month build.
"""
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import xarray as xr

# ---------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------- #
TARGET_H, TARGET_W = 256, 256
LON_MIN, LON_MAX = 68.20, 97.40
LAT_MIN, LAT_MAX = 6.75, 37.09
TARGET_TRANSFORM = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, TARGET_W, TARGET_H)
TARGET_CRS = "EPSG:4326"

STUDY_MONTHS = pd.date_range("2000-11-01", "2022-12-01", freq="MS")  # 266 months
BASELINE_START, BASELINE_END = "2001-01-01", "2020-12-31"

NDVI_DIR = r"D:\FOREST FIRE MAPPING(INDIA)\NDVI_DATA_INDIA_\NDVI TIF File_INDIA"
FLDAS_DIR = r"D:\FOREST FIRE MAPPING(INDIA)\FLDAS Noah Land Surface Model L4 Global Monthly 0.1 x 0.1 degree (MERRA-2 and CHIRPS) (FLDAS_NOAH01_C_GL_M)"
FIRE_CSV = r"D:\FOREST FIRE MAPPING(INDIA)\Forest fire Extraction in INDIA(2000-2022)\Forest_Fire_Outputs\all_forest_fires_2000_2022.csv"
PARQUET_PATH = r"D:\FOREST FIRE MAPPING(INDIA)\Integrated_Analysis\Integrated_Outputs\Integrated_FireRisk_Pixels.parquet"
TERRAIN_DIR = r"D:\FOREST FIRE MAPPING(INDIA)\Terrain_Elevation_Slope_Aspect_Analysis\Terrain_Outputs"
ACCESS_DIR = r"D:\FOREST FIRE MAPPING(INDIA)\Distance_Roads_Railways_Waterways_Analysis\Accessibility_Outputs"

OUT_DIR = r"D:\FOREST FIRE MAPPING(INDIA)\Physics_Informed_FireRisk_Model\CDR_PINN_Data"


def doy_to_date(doy_str):
    year = int(doy_str[:4])
    day = int(doy_str[4:7])
    return pd.Timestamp(year, 1, 1) + pd.Timedelta(days=day - 1)


def resample_to_target(src_array, src_transform, src_crs, resampling=Resampling.average, src_nodata=np.nan):
    dst = np.full((TARGET_H, TARGET_W), np.nan, dtype=np.float32)
    reproject(
        source=src_array.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=TARGET_TRANSFORM,
        dst_crs=TARGET_CRS,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst


# ---------------------------------------------------------------------- #
# NDVI
# ---------------------------------------------------------------------- #

def index_ndvi_files():
    ndvi_files = glob.glob(os.path.join(NDVI_DIR, "MOD13A3.061__1_km_monthly_NDVI_doy*_aid0001.tif"))
    qa_files = glob.glob(os.path.join(NDVI_DIR, "MOD13A3.061__1_km_monthly_pixel_reliability_doy*_aid0001.tif"))

    def date_from_path(p):
        m = re.search(r"doy(\d{7})", os.path.basename(p))
        return doy_to_date(m.group(1))

    ndvi_by_date = {date_from_path(p).replace(day=1): p for p in ndvi_files}
    qa_by_date = {date_from_path(p).replace(day=1): p for p in qa_files}
    return ndvi_by_date, qa_by_date


def process_ndvi_month(ndvi_path, qa_path):
    with rasterio.open(ndvi_path) as src:
        raw = src.read(1).astype(np.float32)
        transform, crs = src.transform, src.crs
        nodata = src.nodata
    with rasterio.open(qa_path) as src:
        qa = src.read(1).astype(np.float32)

    ndvi = raw * 0.0001
    ndvi[raw == (nodata if nodata is not None else -3000)] = np.nan
    ndvi[(ndvi < -0.2) | (ndvi > 1.0)] = np.nan
    ndvi[~np.isin(qa, [0, 1])] = np.nan  # keep Good(0)/Marginal(1) only

    return resample_to_target(ndvi, transform, crs)


# ---------------------------------------------------------------------- #
# FLDAS
# ---------------------------------------------------------------------- #

def index_fldas_files():
    files = glob.glob(os.path.join(FLDAS_DIR, "FLDAS_NOAH01_C_GL_M.A??????.001.nc"))
    files = [f for f in files if "(1)" not in f and "(2)" not in f]
    out = {}
    for f in files:
        m = re.search(r"\.A(\d{6})\.001\.nc$", os.path.basename(f))
        if m:
            ym = m.group(1)
            out[pd.Timestamp(int(ym[:4]), int(ym[4:6]), 1)] = f
    return out


def process_fldas_month(nc_path):
    ds = xr.open_dataset(nc_path)
    ds = ds.sel(Y=slice(5, 38.5), X=slice(67, 98.5))
    lat = ds["Y"].values
    lon = ds["X"].values
    # north-up transform for reproject: rasterio expects row0 = max lat
    flip = lat[0] < lat[-1]
    if flip:
        ds = ds.isel(Y=slice(None, None, -1))
        lat = ds["Y"].values
    src_transform = from_bounds(lon.min(), lat.min(), lon.max(), lat.max(), len(lon), len(lat))

    tair_k = ds["Tair_f_tavg"].isel(time=0).values.astype(np.float32)
    wind = ds["Wind_f_tavg"].isel(time=0).values.astype(np.float32)
    qair = ds["Qair_f_tavg"].isel(time=0).values.astype(np.float32)
    psurf_pa = ds["Psurf_f_tavg"].isel(time=0).values.astype(np.float32)
    rainf = ds["Rainf_f_tavg"].isel(time=0).values.astype(np.float32)
    lwnet = ds["Lwnet_tavg"].isel(time=0).values.astype(np.float32)
    soilm_surf = ds["SoilMoi00_10cm_tavg"].isel(time=0).values.astype(np.float32)
    ds.close()

    tc = tair_k - 273.15
    es_hpa = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
    p_hpa = psurf_pa / 100.0
    e_hpa = qair * p_hpa / (0.622 + 0.378 * qair)
    rh = np.clip(100.0 * e_hpa / es_hpa, 0, 100).astype(np.float32)
    precip_mm_month = rainf * 86400.0 * 30.0  # kg/m2/s -> mm/month (30-day approximation, consistent w/ project convention)

    variables = {
        "tair_k": tair_k, "wind": wind, "rh": rh, "precip_mm": precip_mm_month,
        "lwnet": lwnet, "soilm_surf": soilm_surf,
    }
    return {k: resample_to_target(v, src_transform, "EPSG:4326") for k, v in variables.items()}


# ---------------------------------------------------------------------- #
# Static layers (terrain/accessibility) + terminal aggregate label
# ---------------------------------------------------------------------- #

def process_static_grid(tif_path):
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
        transform, crs, nodata = src.transform, src.crs, src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return resample_to_target(arr, transform, crs)


def build_fire_ever_grid():
    """Resample the already-validated Step 6/7 fire_ever label (native NDVI
    resolution) down to the target grid via averaging -> a fractional
    burned-pixel-proportion soft label, reusing real ground truth rather than
    recomputing a coarser one from scratch."""
    import pyarrow.parquet as pq
    table = pq.read_table(PARQUET_PATH, columns=["lon", "lat", "fire_ever"])
    df = table.to_pandas()
    # build a native-resolution grid matching F1_NDVI_QA_mean.tif's transform, then resample
    native_h, native_w = 3641, 3504
    native_transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, native_w, native_h)
    inv = ~native_transform
    cols, rows = inv * (df["lon"].values, df["lat"].values)
    rows = np.clip(rows.astype(int), 0, native_h - 1)
    cols = np.clip(cols.astype(int), 0, native_w - 1)
    grid = np.full((native_h, native_w), np.nan, dtype=np.float32)
    grid[rows, cols] = df["fire_ever"].values.astype(np.float32)
    return resample_to_target(grid, native_transform, TARGET_CRS)


def build_monthly_fire_indicator(month_ts):
    """Bin raw (already forest-filtered, per Step 1) fire points for one
    calendar month directly into the target grid."""
    global _fire_df
    sub = _fire_df[(_fire_df["year"] == month_ts.year) & (_fire_df["month"] == month_ts.month)]
    grid = np.zeros((TARGET_H, TARGET_W), dtype=np.float32)
    if len(sub) == 0:
        return grid
    inv = ~TARGET_TRANSFORM
    cols, rows = inv * (sub["longitude"].values, sub["latitude"].values)
    rows = np.clip(rows.astype(int), 0, TARGET_H - 1)
    cols = np.clip(cols.astype(int), 0, TARGET_W - 1)
    np.add.at(grid, (rows, cols), 1.0)
    return (grid > 0).astype(np.float32)


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #

def main(smoke_test):
    global _fire_df
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Indexing source files...")
    ndvi_by_date, qa_by_date = index_ndvi_files()
    fldas_by_date = index_fldas_files()
    _fire_df = pd.read_csv(FIRE_CSV, usecols=["longitude", "latitude", "year", "month"])
    print(f"  NDVI months found: {len(ndvi_by_date)}, FLDAS months found: {len(fldas_by_date)}")

    months = list(STUDY_MONTHS)
    if smoke_test:
        months = [pd.Timestamp(2001, 1, 1), pd.Timestamp(2010, 8, 1), pd.Timestamp(2022, 12, 1)]
        print(f"[SMOKE TEST MODE] processing only {len(months)} sample months: {[m.date() for m in months]}")

    n = len(months)
    ndvi_stack = np.full((n, TARGET_H, TARGET_W), np.nan, dtype=np.float32)
    fldas_stacks = {k: np.full((n, TARGET_H, TARGET_W), np.nan, dtype=np.float32)
                    for k in ["tair_k", "wind", "rh", "precip_mm", "lwnet", "soilm_surf"]}
    fire_indicator = np.zeros((n, TARGET_H, TARGET_W), dtype=np.float32)
    month_index = pd.DataFrame({"date": months, "year": [m.year for m in months], "month": [m.month for m in months]})

    missing_ndvi, missing_fldas = [], []
    for i, m in enumerate(months):
        if m in ndvi_by_date and m in qa_by_date:
            ndvi_stack[i] = process_ndvi_month(ndvi_by_date[m], qa_by_date[m])
        else:
            missing_ndvi.append(m)

        if m in fldas_by_date:
            fvars = process_fldas_month(fldas_by_date[m])
            for k, v in fvars.items():
                fldas_stacks[k][i] = v
        else:
            missing_fldas.append(m)

        fire_indicator[i] = build_monthly_fire_indicator(m)

        if (i + 1) % max(1, n // 10) == 0 or i == n - 1:
            print(f"  processed {i+1}/{n} months ({m.date()})")

    if missing_ndvi:
        print(f"WARNING: {len(missing_ndvi)} months missing NDVI source file: {[m.date() for m in missing_ndvi][:5]}...")
    if missing_fldas:
        print(f"WARNING: {len(missing_fldas)} months missing FLDAS source file: {[m.date() for m in missing_fldas][:5]}...")

    # climatology + anomaly (baseline 2001-2020, per project convention)
    baseline_mask = (month_index["date"] >= BASELINE_START) & (month_index["date"] <= BASELINE_END)
    ndvi_clim = np.full((12, TARGET_H, TARGET_W), np.nan, dtype=np.float32)
    for mo in range(1, 13):
        sel = baseline_mask & (month_index["month"] == mo)
        if sel.any():
            ndvi_clim[mo - 1] = np.nanmean(ndvi_stack[sel.values], axis=0)
    ndvi_anomaly = ndvi_stack - ndvi_clim[month_index["month"].values - 1]

    fldas_anomaly = {}
    for k, stack in fldas_stacks.items():
        clim = np.full((12, TARGET_H, TARGET_W), np.nan, dtype=np.float32)
        for mo in range(1, 13):
            sel = baseline_mask & (month_index["month"] == mo)
            if sel.any():
                clim[mo - 1] = np.nanmean(stack[sel.values], axis=0)
        fldas_anomaly[k] = stack - clim[month_index["month"].values - 1]

    # dryness proxy: standardized combination of FLDAS anomalies, physically-fixed
    # signs per the Tetens/VPD qualitative direction already cited in the diffusion
    # design doc (warmer + lower-humidity + drier-soil + lower-precip anomalies all
    # push dryness UP) -- z-scored per-variable over the whole stack before combining
    # so no single variable's units dominate the sum.
    def zscore(a):
        return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-8)

    dryness_proxy = (zscore(fldas_anomaly["tair_k"])
                      - zscore(fldas_anomaly["rh"])
                      - zscore(fldas_anomaly["soilm_surf"])
                      - zscore(fldas_anomaly["precip_mm"])) / 4.0

    print("Building static layers (elevation, slope, dist_roads, dist_railways, dist_waterways)...")
    elevation = process_static_grid(os.path.join(TERRAIN_DIR, "T1_Elevation_native_1km.tif"))
    slope = process_static_grid(os.path.join(TERRAIN_DIR, "T2_Slope_native_1km.tif"))
    dist_roads = process_static_grid(os.path.join(ACCESS_DIR, "D1_Distance_to_Roads_native_1km.tif"))
    dist_railways = process_static_grid(os.path.join(ACCESS_DIR, "D2_Distance_to_Railways_native_1km.tif"))
    dist_waterways = process_static_grid(os.path.join(ACCESS_DIR, "D3_Distance_to_Waterways_native_1km.tif"))

    print("Building fire_ever terminal aggregate label from Step 6/7's own validated parquet...")
    fire_ever_frac = build_fire_ever_grid()

    ndvi_f1 = np.nanmean(ndvi_stack, axis=0)  # whole-period mean, matches F1's own definition

    out_path = os.path.join(OUT_DIR, "cdr_pinn_monthly_stacks_smoketest.npz" if smoke_test
                             else "cdr_pinn_monthly_stacks.npz")
    np.savez_compressed(
        out_path,
        months=np.array([m.isoformat() for m in months]),
        ndvi_stack=ndvi_stack, ndvi_anomaly=ndvi_anomaly, ndvi_f1=ndvi_f1,
        fire_indicator=fire_indicator, fire_ever_frac=fire_ever_frac,
        dryness_proxy=dryness_proxy,
        elevation=elevation, slope=slope,
        dist_roads=dist_roads, dist_railways=dist_railways, dist_waterways=dist_waterways,
        **{f"fldas_{k}": v for k, v in fldas_stacks.items()},
        **{f"fldas_{k}_anomaly": v for k, v in fldas_anomaly.items()},
    )
    print(f"\nSaved: {out_path}")
    print(f"  ndvi_stack shape: {ndvi_stack.shape}, NaN fraction: {np.isnan(ndvi_stack).mean():.3f}")
    print(f"  fire_indicator: {fire_indicator.shape}, months with any fire: {(fire_indicator.sum(axis=(1,2))>0).sum()}/{n}")
    print(f"  fire_ever_frac: nonzero cells: {(fire_ever_frac>0).sum()}, max frac: {np.nanmax(fire_ever_frac):.4f}")
    print(f"  dryness_proxy range: [{np.nanmin(dryness_proxy):.3f}, {np.nanmax(dryness_proxy):.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test)
