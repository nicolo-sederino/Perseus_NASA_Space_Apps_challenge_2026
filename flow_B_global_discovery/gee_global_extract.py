#!/usr/bin/env python3
"""
NASA Space Apps Challenge 2026 - Team Perseus
Flow B (Global Discovery): Google Earth Engine (GEE) global abiotic-terrain extractor

AI-Assistance Disclosure (NASA Space Apps Submission Guide):
This script was developed with AI code-generation assistance and reviewed/validated by Team Perseus.

What it does
------------
1. Builds a continuous, approximately equal-area global lat/lon grid (default 0.5 deg).
2. Removes every cell closer than --buffer-km (default 50 km) to ANY Flow A site (Haversine buffer),
   so Flow B can only ever surface new, uncatalogued locations.
3. Stage 1 (cheap, 1 km): drops water, forests, wetlands, croplands, urban cells and vegetated cells
   (MODIS peak NDVI), server-side in GEE.
4. Stage 2 (30 m - 11 km): extracts measured satellite / reanalysis layers for the survivors AND for the
   Flow A anchor sites (anchors are only used to calibrate/validate, they are tagged is_known_site=True):
     * NASA NASADEM 30 m (+ Copernicus GLO-30 fallback at high latitude): elevation, multi-scale slope,
       slope statistics in a ~1 km window, detrended RMS roughness, safe-landing / ultra-flat fractions
     * NASA/USGS Landsat 8/9 Collection 2 Level-2 surface reflectance (cloud-masked median composite)
     * NASA MODIS Aqua MYD11A2 day/night land-surface temperature (real diurnal Delta-T, T_min, T_max)
     * ECMWF ERA5-Land monthly: 10 m wind, surface pressure, downward solar radiation, surface soil moisture
     * OpenLandMap soil texture (sand/clay) and bulk density (250 m)
5. Derives the Flow B feature vector in Python (all formulas visible below) and writes
   flow_B_global_discovery/cache/global_abiotic_grid.csv  (+ extraction_metadata.json).

Honest limitations (also written to the metadata file)
------------------------------------------------------
* True 1 m / 5 m slopes do not exist globally, so Slope_1m / Slope_5m are NOT produced (not fabricated).
  The discovery engine automatically restricts the feature space to columns that exist in BOTH the
  Flow A catalogs and this grid.
* Mineral "wt%" columns (FeO, TiO2, salts, water ice) are spectral-index proxies from Landsat, not
  measured abundances.
* UV_Flux_Index is a proxy derived from ERA5-Land downward shortwave radiation.
* Geotechnical columns come from soil-texture pedotransfer relations (see GeotechParams), not field tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from perseus_common import (
    find_col,
    load_flow_a_anchors,
    min_distance_to_sites_km,
    read_earth_catalog,
)

try:
    import ee
except ImportError:  # pragma: no cover
    ee = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("PerseusGEE")

# MODIS MCD12Q1 IGBP classes
LC_NAMES = {
    1: "Evergreen Needleleaf Forest", 2: "Evergreen Broadleaf Forest", 3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest", 5: "Mixed Forest", 6: "Closed Shrubland", 7: "Open Shrubland",
    8: "Woody Savanna", 9: "Savanna", 10: "Grassland", 11: "Permanent Wetland", 12: "Cropland",
    13: "Urban", 14: "Cropland/Natural Mosaic", 15: "Snow and Ice", 16: "Barren", 17: "Water",
}
LC_EXCLUDE = [1, 2, 3, 4, 5, 11, 12, 13, 14, 17]


@dataclass
class GeotechParams:
    """Tunable pedotransfer constants (engineering conventions, NOT measurements). Edit or expose via CLI."""
    phi_sand_deg: float = 36.0       # typical friction angle, dense sand
    phi_silt_deg: float = 30.0
    phi_clay_deg: float = 22.0
    density_phi_gain_deg_per_1000: float = 4.0   # extra friction per +1000 kg/m3 above 1500 kg/m3
    cohesion_clay_kpa_per_pct: float = 0.20
    cohesion_silt_kpa_per_pct: float = 0.02
    cohesion_max_kpa: float = 30.0
    gravity_m_s2: float = 9.81       # set 1.62 (Moon) / 3.71 (Mars) to express bearing capacity in that gravity
    footing_width_m: float = 1.0     # strip footing width B for the Terzaghi/Vesic bearing capacity


@dataclass
class DerivationParams:
    ti_scale: float = 6800.0         # Apparent Thermal Inertia -> catalog Thermal_Inertia scale (auto-calibrated on anchors)
    uv_scale_w_m2: float = 25.0      # UV_Flux_Index = ssrd_W_m2 / uv_scale (proxy)
    eolian_w_wind: float = 0.45
    eolian_w_dry: float = 0.30
    eolian_w_erodible: float = 0.25
    eolian_wind_lo: float = 3.0      # m/s, peak monthly-mean wind giving 0 deflation potential
    eolian_wind_hi: float = 10.0     # m/s, giving the maximum
    eolian_dry_ref_m3m3: float = 0.15


# --------------------------------------------------------------------------------------------------
# Grid construction
# --------------------------------------------------------------------------------------------------
def build_grid(step_deg: float, lat_min: float, lat_max: float,
               bbox: Optional[Tuple[float, float, float, float]] = None) -> pd.DataFrame:
    """Approximately equal-area grid: the number of longitudes per row shrinks with cos(lat)."""
    frames = []
    lats = np.arange(lat_min + step_deg / 2.0, lat_max, step_deg)
    for lat in lats:
        n_lon = max(1, int(round(360.0 * math.cos(math.radians(lat)) / step_deg)))
        lons = -180.0 + (np.arange(n_lon) + 0.5) * (360.0 / n_lon)
        frames.append(pd.DataFrame({"latitude": np.round(lat, 4), "longitude": np.round(lons, 4)}))
    grid = pd.concat(frames, ignore_index=True)
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        grid = grid[
            (grid.longitude >= min_lon) & (grid.longitude <= max_lon)
            & (grid.latitude >= min_lat) & (grid.latitude <= max_lat)
        ].reset_index(drop=True)
    return grid


# --------------------------------------------------------------------------------------------------
# Earth Engine layers
# --------------------------------------------------------------------------------------------------
def stage1_image(y0: int, y1: int) -> "ee.Image":
    """Light-weight 1 km mask image: peak NDVI + IGBP land cover class."""
    ndvi_max = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterDate(f"{y0}-01-01", f"{y1 + 1}-01-01")
        .select("NDVI").max().multiply(0.0001).rename("ndvi_max")
    )
    lc = (
        ee.ImageCollection("MODIS/061/MCD12Q1")
        .sort("system:time_start", False).first()
        .select("LC_Type1").rename("lc")
    )
    return ndvi_max.addBands(lc)


def stage2_image(y0: int, y1: int, safe_slope_deg: float, flat_slope_deg: float, win_px: int) -> "ee.Image":
    """Full measurement stack. Band names are the raw columns consumed by derive_features()."""
    start, end = f"{y0}-01-01", f"{y1 + 1}-01-01"

    # ---- NASA NASADEM 30 m, Copernicus GLO-30 where NASADEM has no data (|lat| > ~60 deg)
    nasadem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
    glo = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM")
    glo_mosaic = glo.mosaic().setDefaultProjection(glo.first().projection()).rename("elevation")
    dem = nasadem.unmask(glo_mosaic).rename("elevation").setDefaultProjection(nasadem.projection())

    slope = ee.Terrain.slope(dem).rename("slope")
    kernel = ee.Kernel.circle(radius=win_px, units="pixels")  # ~1 km diameter for win_px = 16 (30 m pixels)
    mean_slope = slope.reduceNeighborhood(ee.Reducer.mean(), kernel).rename("mean_slope")
    slope_p75 = slope.reduceNeighborhood(ee.Reducer.percentile([75]), kernel).rename("slope_p75")
    safe_frac = slope.lte(safe_slope_deg).reduceNeighborhood(ee.Reducer.mean(), kernel).rename("safe_frac")
    flat_frac = slope.lte(flat_slope_deg).reduceNeighborhood(ee.Reducer.mean(), kernel).rename("flat_frac")
    # Detrended RMS height: residual to the 3x3 local mean (removes the planar trend), RMS over the window
    resid = dem.subtract(dem.focal_mean(1, "square", "pixels"))
    rms_rough = resid.pow(2).reduceNeighborhood(ee.Reducer.mean(), kernel).sqrt().rename("rms_rough")
    slope_90 = ee.Terrain.slope(dem.focal_mean(1, "square", "pixels")).rename("slope_90")
    slope_270 = ee.Terrain.slope(dem.focal_mean(4, "square", "pixels")).rename("slope_270")

    # ---- NASA/USGS Landsat 8/9 C2 L2 surface reflectance, cloud-masked median composite
    def prep(img):
        qa = img.select("QA_PIXEL")
        clear = qa.bitwiseAnd(30).eq(0)  # bits 1-4: dilated cloud, cirrus, cloud, cloud shadow (snow is kept)
        refl = (
            img.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
                       ["blue", "green", "red", "nir", "swir1", "swir2"])
            .multiply(0.0000275).add(-0.2)
        )
        return refl.updateMask(clear)

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(start, end).map(prep)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterDate(start, end).map(prep)
    landsat = l8.merge(l9)
    refl_median = landsat.median()
    n_obs = landsat.select("blue").count().rename("n_obs")

    # ---- NASA MODIS Aqua land surface temperature (13:30 day / 01:30 night overpasses)
    lst = ee.ImageCollection("MODIS/061/MYD11A2").filterDate(start, end)
    stats = ee.Reducer.mean().combine(ee.Reducer.percentile([5, 95]), sharedInputs=True)
    day = lst.select("LST_Day_1km").reduce(stats).multiply(0.02).rename(["day_mean", "day_p5", "day_p95"])
    night = lst.select("LST_Night_1km").reduce(stats).multiply(0.02).rename(["night_mean", "night_p5", "night_p95"])
    clear_frac = lst.select("LST_Day_1km").count().divide(ee.Image.constant(lst.size())).rename("clear_frac")

    # ---- ECMWF ERA5-Land monthly aggregates (~11 km)
    era = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate(start, end)
    wind = era.map(
        lambda im: im.select("u_component_of_wind_10m").hypot(im.select("v_component_of_wind_10m")).rename("wind")
    )
    wind_mean = wind.mean().rename("wind_mean")
    wind_max = wind.max().rename("wind_max_monthly")
    sp_pa = era.select("surface_pressure").mean().rename("sp_pa")
    ssrd = (
        era.select("surface_solar_radiation_downwards_sum").mean().divide(30.4375 * 86400.0).rename("ssrd_wm2")
    )
    swvl1 = era.select("volumetric_soil_water_layer_1").mean().rename("swvl1")

    # ---- OpenLandMap soil (250 m): texture in % (kg/kg) and bulk density (stored as 10 x kg/m3)
    sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select("b0").rename("sand_pct")
    clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select("b0").rename("clay_pct")
    bd = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").multiply(10)
    bd_015 = bd.select(["b0", "b10"]).reduce(ee.Reducer.mean()).rename("bd_0_15")
    bd_060 = bd.select(["b0", "b10", "b30", "b60"]).reduce(ee.Reducer.mean()).rename("bd_0_60")

    return ee.Image.cat([
        dem, slope, mean_slope, slope_p75, safe_frac, flat_frac, rms_rough, slope_90, slope_270,
        refl_median, n_obs, day, night, clear_frac,
        wind_mean, wind_max, sp_pa, ssrd, swvl1,
        sand, clay, bd_015, bd_060,
    ])


# --------------------------------------------------------------------------------------------------
# Batched, parallel, self-splitting extraction
# --------------------------------------------------------------------------------------------------
def init_earth_engine(project: str) -> None:
    if ee is None:
        sys.exit("earthengine-api is not installed. Run: pip install earthengine-api")
    try:
        ee.Initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception as exc:
        sys.exit(
            f"Earth Engine initialisation failed: {exc}\n"
            "Run `earthengine authenticate` once, make sure the Earth Engine API is enabled and the project "
            f"'{project}' is registered for Earth Engine (https://code.earthengine.google.com/register)."
        )


def _reduce_points(batch: pd.DataFrame, image: "ee.Image", scale: float, tile_scale: int) -> pd.DataFrame:
    feats = [
        ee.Feature(ee.Geometry.Point([float(lo), float(la)]), {"pid": int(p)})
        for p, la, lo in zip(batch["pid"], batch["latitude"], batch["longitude"])
    ]
    fc = ee.FeatureCollection(feats)
    out = image.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=scale, tileScale=tile_scale).getInfo()
    return pd.DataFrame([f["properties"] for f in out["features"]])


def _extract_with_split(batch: pd.DataFrame, image: "ee.Image", scale: float, tile_scale: int,
                        min_size: int = 50, depth: int = 0) -> pd.DataFrame:
    try:
        return _reduce_points(batch, image, scale, tile_scale)
    except Exception as exc:  # rate limit, memory, timeout ...
        if len(batch) <= min_size or depth > 6:
            logger.warning(f"Giving up on a batch of {len(batch)} points: {str(exc)[:160]}")
            return pd.DataFrame()
        logger.info(f"Batch of {len(batch)} failed ({str(exc)[:90]}); splitting and retrying")
        time.sleep(2.0 + depth)
        mid = len(batch) // 2
        a = _extract_with_split(batch.iloc[:mid], image, scale, tile_scale, min_size, depth + 1)
        b = _extract_with_split(batch.iloc[mid:], image, scale, tile_scale, min_size, depth + 1)
        return pd.concat([a, b], ignore_index=True)


def run_stage(points: pd.DataFrame, image: "ee.Image", scale: float, label: str,
              batch_size: int, workers: int, tile_scale: int) -> pd.DataFrame:
    """Spatially sorts the points (compact batches => fewer Landsat/DEM tiles per request) and runs them in parallel."""
    pts = points.copy()
    pts["_band"] = np.floor((pts["latitude"] + 90.0) / 5.0)
    pts = pts.sort_values(["_band", "longitude"]).drop(columns="_band").reset_index(drop=True)
    batches = [pts.iloc[i:i + batch_size] for i in range(0, len(pts), batch_size)]
    logger.info(f"[{label}] {len(pts)} points in {len(batches)} batches ({workers} parallel workers)")
    t0 = time.time()
    results: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_extract_with_split, b, image, scale, tile_scale) for b in batches]
        for k, fut in enumerate(as_completed(futs), start=1):
            res = fut.result()
            if not res.empty:
                results.append(res)
            if k % 5 == 0 or k == len(futs):
                logger.info(f"[{label}] {k}/{len(futs)} batches done in {time.time() - t0:.0f}s")
    if not results:
        return pd.DataFrame(columns=["pid"])
    return pd.concat(results, ignore_index=True)


# --------------------------------------------------------------------------------------------------
# Feature derivation (pure Python, formulas visible and editable)
# --------------------------------------------------------------------------------------------------
def derive_features(raw: pd.DataFrame, dp: DerivationParams, gp: GeotechParams) -> pd.DataFrame:
    d = raw.copy()
    eps = 1e-6
    for c in ["blue", "green", "red", "nir", "swir1", "swir2"]:
        d[c] = d[c].clip(0.0, 1.5)

    d["NDVI"] = (d["nir"] - d["red"]) / (d["nir"] + d["red"] + eps)
    d["NDWI"] = (d["green"] - d["nir"]) / (d["green"] + d["nir"] + eps)
    d["NDSI"] = (d["green"] - d["swir1"]) / (d["green"] + d["swir1"] + eps)

    # ---- Morphometry (measured from NASADEM / GLO-30)
    d["Altitude_m"] = d["elevation"]
    d["Mean_Slope_deg"] = d["mean_slope"]
    d["Traverse_Slope_deg"] = d["slope_p75"]
    d["RMS_Roughness_m"] = d["rms_rough"]
    d["Safe_Landing_Frac"] = d["safe_frac"]
    d["Ultra_Flat_Frac"] = d["flat_frac"]
    d["Slope_30m_deg"] = d["slope"]
    d["Slope_90m_deg"] = d["slope_90"]
    d["Slope_270m_deg"] = d["slope_270"]

    # ---- Albedo + spectral mineral proxies (Landsat)
    d["Albedo"] = (
        0.356 * d["blue"] + 0.130 * d["red"] + 0.373 * d["nir"] + 0.085 * d["swir1"] + 0.072 * d["swir2"] - 0.0018
    ).clip(0.02, 0.95)
    d["Ferric_Oxide_Idx"] = (d["red"] / (d["blue"] + 0.01)).clip(0.3, 6.0)
    vis = (d["blue"] + d["green"] + d["red"]) / 3.0
    swir_sal = (d["swir1"] - d["swir2"]) / (d["swir1"] + d["swir2"] + eps)
    evap = vis * (1.0 + swir_sal.clip(lower=0.0) * 2.5) * (1.0 - d["NDSI"].clip(lower=0.0))
    d["Salts_Evaporites_wt_pct"] = (evap * 65.0).clip(1.0, 45.0)
    dark = ((0.32 - d["Albedo"]) / 0.26).clip(0.0, 1.0)
    mafic_ratio = ((d["swir1"] / (d["nir"] + 0.02)).clip(0.4, 2.0)) / 2.0
    d["Mafic_Basalt_Idx"] = ((0.70 * dark + 0.30 * mafic_ratio) * 100.0).clip(5.0, 98.0)
    d["FeO_wt_pct"] = (3.5 + (d["Mafic_Basalt_Idx"] / 100.0) * 13.0).clip(2.5, 18.0)
    d["TiO2_wt_pct"] = (0.25 + ((d["Mafic_Basalt_Idx"] / 100.0) ** 1.4) * 5.5).clip(0.2, 8.0)
    d["Water_Ice_wt_pct"] = ((d["NDSI"] - 0.15).clip(lower=0.0) * 95.0).clip(0.0, 85.0)

    # ---- Thermophysics (measured MODIS Aqua LST)
    d["T_max_K"] = d["day_p95"]
    d["T_min_K"] = d["night_p5"]
    d["T_mean_K"] = 0.5 * (d["day_mean"] + d["night_mean"])
    d["Diurnal_Delta_T_K"] = (d["day_mean"] - d["night_mean"]).clip(lower=0.5)
    d["ATI"] = (1.0 - d["Albedo"]) / d["Diurnal_Delta_T_K"]          # Apparent Thermal Inertia
    d["Thermal_Inertia"] = (dp.ti_scale * d["ATI"]).clip(10.0, 1000.0)
    d["Sun_Visibility_pct"] = (d["clear_frac"] * 100.0).clip(0.0, 100.0)

    # ---- Atmosphere / radiation / wind (ERA5-Land)
    d["Surface_Pressure_mbar"] = d["sp_pa"] / 100.0
    d["UV_Flux_Index"] = (d["ssrd_wm2"] / dp.uv_scale_w_m2).clip(0.0, 20.0)
    d["Wind_Mean_mps"] = d["wind_mean"]
    d["Wind_Peak_Monthly_mps"] = d["wind_max_monthly"]

    # ---- Soil mechanics (OpenLandMap texture + ERA5-Land moisture)
    sand = d["sand_pct"].clip(0, 100)
    clay = d["clay_pct"].clip(0, 100)
    silt = (100.0 - sand - clay).clip(lower=0.0)
    tot = (sand + silt + clay).replace(0.0, np.nan)
    rho = d["bd_0_60"]
    d["Density_0_15cm_kg_m3"] = d["bd_0_15"]
    d["Density_0_60cm_kg_m3"] = d["bd_0_60"]
    phi = (sand * gp.phi_sand_deg + silt * gp.phi_silt_deg + clay * gp.phi_clay_deg) / tot
    phi = (phi + gp.density_phi_gain_deg_per_1000 * (rho - 1500.0) / 1000.0).clip(20.0, 45.0)
    theta = d["swvl1"].clip(0.0, 0.5)
    moisture_factor = 0.3 + 0.7 * (theta / 0.25).clip(0.0, 1.0)
    coh = ((gp.cohesion_clay_kpa_per_pct * clay + gp.cohesion_silt_kpa_per_pct * silt) * moisture_factor).clip(
        0.0, gp.cohesion_max_kpa
    )
    phi_r = np.radians(phi)
    nq = np.exp(np.pi * np.tan(phi_r)) * np.tan(np.radians(45.0) + phi_r / 2.0) ** 2
    nc = (nq - 1.0) / np.tan(phi_r)
    ng = 2.0 * (nq + 1.0) * np.tan(phi_r)
    gamma_kn_m3 = rho * gp.gravity_m_s2 / 1000.0
    d["Friction_Angle_deg"] = phi
    d["Soil_Cohesion_kPa"] = coh
    d["Cohesion_kPa"] = coh
    d["Bearing_Capacity_kPa"] = coh * nc + 0.5 * gamma_kn_m3 * gp.footing_width_m * ng
    d["Soil_Data_Available"] = d["sand_pct"].notna().astype(int)

    # ---- Aeolian deflation potential (ERA5-Land wind peak, soil dryness, texture erodibility)
    wind_term = ((d["wind_max_monthly"] - dp.eolian_wind_lo) / (dp.eolian_wind_hi - dp.eolian_wind_lo)).clip(0, 1)
    dry_term = 1.0 - (theta / dp.eolian_dry_ref_m3m3).clip(0.0, 1.0)
    erodible = (0.6 * sand / 100.0 + 0.4 * (1.0 - clay / 100.0)).clip(0.0, 1.0)
    d["Eolian_Deflation_Idx"] = 100.0 * (
        dp.eolian_w_wind * wind_term + dp.eolian_w_dry * dry_term + dp.eolian_w_erodible * erodible
    )
    return d


def calibrate_thermal_inertia(df: pd.DataFrame, anchors: pd.DataFrame, data_dir: Path, dp: DerivationParams) -> float:
    """
    Fits ONE scale factor k so that k * ATI matches the catalog Thermal_Inertia at the Flow A anchors
    (median ratio, robust to outliers). Returns the scale actually used.
    """
    ratios: List[float] = []
    for fname in ("Earth_Lunar_Analogs_Catalog.csv", "Earth_Mars_Analogs_Catalog.csv"):
        path = data_dir / fname
        if not path.exists():
            continue
        cat = read_earth_catalog(path)
        ti_col = find_col(cat, ["Thermal_Inertia"])
        if ti_col is None:
            continue
        for _, row in cat.iterrows():
            m = df[
                df["is_known_site"]
                & ((df["latitude"] - float(row["Latitude"])).abs() < 0.08)
                & ((df["longitude"] - float(row["Longitude"])).abs() < 0.08)
            ]
            if m.empty:
                continue
            ati = float(m.iloc[0]["ATI"])
            ti = pd.to_numeric(row[ti_col], errors="coerce")
            if np.isfinite(ati) and ati > 0 and np.isfinite(ti) and ti > 0:
                ratios.append(float(ti) / ati)
    if len(ratios) >= 5:
        k = float(np.median(ratios))
        logger.info(f"Thermal-inertia scale calibrated on {len(ratios)} Flow A anchors: k = {k:.1f}")
        return k
    logger.warning(f"Only {len(ratios)} anchors with catalog Thermal_Inertia: keeping default ti_scale={dp.ti_scale}")
    return dp.ti_scale


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Team Perseus - Flow B global abiotic extractor (Google Earth Engine)")
    ap.add_argument("--project", required=True, help="Google Cloud project id registered for Earth Engine")
    ap.add_argument("--mode", choices=["smoke", "fast", "standard", "full"], default="standard",
                    help="grid step: fast=1.0, standard=0.5, full=0.25 degrees (smoke = 2.0 + --bbox recommended)")
    ap.add_argument("--step-deg", type=float, default=None, help="override grid step (degrees)")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
                    default=None, help="restrict to a bounding box (smoke tests / regional zoom)")
    ap.add_argument("--lat-min", type=float, default=-85.0)
    ap.add_argument("--lat-max", type=float, default=83.0)
    ap.add_argument("--buffer-km", type=float, default=50.0, help="exclusion radius around every Flow A site")
    ap.add_argument("--years", type=int, nargs=2, default=[2021, 2024], metavar=("START", "END"))
    ap.add_argument("--stage1-ndvi-max", type=float, default=0.20, help="coarse MODIS peak-NDVI cut (stage 1)")
    ap.add_argument("--ndvi-max", type=float, default=0.15, help="strict Landsat NDVI cut for discovery cells")
    ap.add_argument("--safe-slope-deg", type=float, default=5.0)
    ap.add_argument("--flat-slope-deg", type=float, default=2.0)
    ap.add_argument("--window-px", type=int, default=16, help="neighbourhood radius in 30 m pixels (16 ~ 1 km window)")
    ap.add_argument("--batch-size", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tile-scale", type=int, default=4)
    ap.add_argument("--gravity", type=float, default=9.81, help="gravity for bearing capacity (m/s2)")
    ap.add_argument("--footing-width", type=float, default=1.0)
    ap.add_argument("--ti-scale", type=float, default=None, help="force the ATI->Thermal_Inertia scale (skips calibration)")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    cache_dir = script_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    step = args.step_deg or {"smoke": 2.0, "fast": 1.0, "standard": 0.5, "full": 0.25}[args.mode]
    y0, y1 = args.years
    dp = DerivationParams()
    gp = GeotechParams(gravity_m_s2=args.gravity, footing_width_m=args.footing_width)
    if args.ti_scale:
        dp.ti_scale = args.ti_scale

    t0 = time.time()
    logger.info("=" * 78)
    logger.info("  NASA SPACE APPS 2026 - TEAM PERSEUS | FLOW B GEE GLOBAL EXTRACTOR")
    logger.info("=" * 78)
    init_earth_engine(args.project)

    anchors = load_flow_a_anchors(data_dir)
    grid = build_grid(step, args.lat_min, args.lat_max, tuple(args.bbox) if args.bbox else None)
    n_grid_raw = len(grid)
    dmin = min_distance_to_sites_km(grid.latitude.values, grid.longitude.values,
                                    anchors.latitude.values, anchors.longitude.values)
    grid = grid[dmin >= args.buffer_km].reset_index(drop=True)
    logger.info(f"Grid {step} deg: {n_grid_raw} cells, {n_grid_raw - len(grid)} removed by the "
                f"{args.buffer_km:.0f} km Flow A exclusion buffer -> {len(grid)} candidate cells")
    grid["pid"] = np.arange(len(grid))

    # ---------------- Stage 1: cheap global mask at 1 km
    s1 = run_stage(grid, stage1_image(y0, y1), 1000, "stage1", args.batch_size, args.workers, args.tile_scale)
    if s1.empty:
        sys.exit("Stage 1 returned no data. Check Earth Engine access / quotas.")
    s1 = s1.merge(grid, on="pid", how="inner")
    keep = (
        s1["lc"].notna() & ~s1["lc"].isin(LC_EXCLUDE) & s1["ndvi_max"].notna() & (s1["ndvi_max"] < args.stage1_ndvi_max)
    )
    cand = s1[keep].copy().reset_index(drop=True)
    cand["country"] = cand["lc"].astype(int).map(LC_NAMES).fillna("Unclassified")
    logger.info(f"Stage 1: {len(cand)}/{len(grid)} cells survive (land, non-forest, non-crop, peak NDVI < "
                f"{args.stage1_ndvi_max})")
    cand["site_name"] = [f"PERSEUS_{la:+07.2f}_{lo:+08.2f}" for la, lo in zip(cand.latitude, cand.longitude)]
    cand["catalog_role"] = "discovery_candidate"
    cand["is_known_site"] = False
    cand = cand[["site_name", "country", "latitude", "longitude", "catalog_role", "is_known_site"]]

    # ---------------- Stage 2: full measurement stack (candidates + Flow A anchors for calibration)
    anc = anchors.copy()
    anc["is_known_site"] = True
    if args.bbox:  # smoke test / regional zoom: only anchors inside the bbox
        min_lon, min_lat, max_lon, max_lat = args.bbox
        anc = anc[anc.longitude.between(min_lon, max_lon) & anc.latitude.between(min_lat, max_lat)]
    pool = pd.concat([cand, anc], ignore_index=True)
    pool["pid"] = np.arange(len(pool))
    img2 = stage2_image(y0, y1, args.safe_slope_deg, args.flat_slope_deg, args.window_px)
    s2 = run_stage(pool, img2, 30, "stage2", args.batch_size, args.workers, args.tile_scale)
    if s2.empty:
        sys.exit("Stage 2 returned no data.")
    raw = pool.merge(s2, on="pid", how="inner").drop(columns=["pid"])
    needed = ["elevation", "blue", "red", "nir", "day_mean", "night_mean"]
    raw = raw.dropna(subset=needed).reset_index(drop=True)
    logger.info(f"Stage 2: {len(raw)} cells with complete DEM + Landsat + MODIS LST measurements")

    # ---------------- Derive features, calibrate thermal inertia on the anchors, strict abiotic filter
    feats = derive_features(raw, dp, gp)
    if args.ti_scale is None:
        dp.ti_scale = calibrate_thermal_inertia(feats, anchors, data_dir, dp)
        feats = derive_features(raw, dp, gp)
    abiotic = feats["is_known_site"] | (feats["NDVI"] < args.ndvi_max)
    final = feats[abiotic].reset_index(drop=True)
    final = final.sort_values(["is_known_site", "latitude", "longitude"], ascending=[False, True, True])

    csv_path = cache_dir / "global_abiotic_grid.csv"
    final.round(5).to_csv(csv_path, index=False)

    meta: Dict[str, Any] = {
        "project": "NASA Space Apps Challenge 2026 - Team Perseus",
        "module": "Flow B Global Abiotic Grid Extraction (Google Earth Engine)",
        "ai_assisted_development": "Developed with AI code-generation assistance and validated by Team Perseus",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "grid_step_deg": step,
        "years": [y0, y1],
        "flow_a_exclusion_buffer_km": args.buffer_km,
        "grid_cells_total": n_grid_raw,
        "candidate_cells_after_stage1": int(len(cand)),
        "discovery_cells_kept": int((~final["is_known_site"]).sum()),
        "flow_a_calibration_anchors_kept": int(final["is_known_site"].sum()),
        "thermal_inertia_scale": dp.ti_scale,
        "geotech_params": asdict(gp),
        "derivation_params": asdict(dp),
        "datasets": {
            "elevation_slope": "NASA/NASADEM_HGT/001 (30 m) + COPERNICUS/DEM/GLO30 fallback",
            "surface_reflectance": "LANDSAT/LC08/C02/T1_L2 + LANDSAT/LC09/C02/T1_L2 (cloud-masked median)",
            "land_surface_temperature": "MODIS/061/MYD11A2 (day 13:30 / night 01:30)",
            "reanalysis": "ECMWF/ERA5_LAND/MONTHLY_AGGR (wind, pressure, solar radiation, soil moisture)",
            "soil": "OpenLandMap SOL sand / clay / bulk density (250 m, Antarctica not covered)",
            "coarse_mask": "MODIS/061/MOD13A3 peak NDVI + MODIS/061/MCD12Q1 land cover",
        },
        "limitations": [
            "Slope_1m / Slope_5m are not available globally and are intentionally not produced.",
            "FeO / TiO2 / salts / water-ice columns are Landsat spectral-index proxies, not measured abundances.",
            "UV_Flux_Index is a proxy from ERA5-Land downward shortwave radiation.",
            "Geotechnical columns are texture-based pedotransfer estimates (see geotech_params).",
            "Soil data is absent over Antarctica; those cells have Soil_Data_Available = 0.",
        ],
        "execution_time_seconds": round(time.time() - t0, 1),
    }
    (cache_dir / "extraction_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("=" * 78)
    logger.info(f"DONE in {meta['execution_time_seconds']} s | {meta['discovery_cells_kept']} discovery cells + "
                f"{meta['flow_a_calibration_anchors_kept']} calibration anchors -> {csv_path}")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
