#!/usr/bin/env python3
"""
NASA Space Apps Challenge 2026 - Team Perseus
Flow B (Global Discovery): Google Earth Engine (GEE) global abiotic-terrain extractor

AI-Assistance Disclosure (NASA Space Apps Submission Guide):
This script was developed with AI code-generation assistance and reviewed/validated by Team Perseus.
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
except ImportError:
    ee = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("PerseusGEE")

LC_NAMES = {
    1: "Evergreen Needleleaf Forest", 2: "Evergreen Broadleaf Forest", 3: "Deciduous Needleleaf Forest",
    4: "Deciduous Broadleaf Forest", 5: "Mixed Forest", 6: "Closed Shrubland", 7: "Open Shrubland",
    8: "Woody Savanna", 9: "Savanna", 10: "Grassland", 11: "Permanent Wetland", 12: "Cropland",
    13: "Urban", 14: "Cropland/Natural Mosaic", 15: "Snow and Ice", 16: "Barren", 17: "Water",
}
LC_EXCLUDE = [1, 2, 3, 4, 5, 11, 12, 13, 14, 17]


@dataclass
class GeotechParams:
    phi_sand_deg: float = 36.0
    phi_silt_deg: float = 30.0
    phi_clay_deg: float = 22.0
    density_phi_gain_deg_per_1000: float = 4.0
    cohesion_clay_kpa_per_pct: float = 0.20
    cohesion_silt_kpa_per_pct: float = 0.02
    cohesion_max_kpa: float = 30.0
    gravity_m_s2: float = 9.81
    footing_width_m: float = 1.0


@dataclass
class DerivationParams:
    ti_scale: float = 4806.2
    uv_scale_w_m2: float = 25.0
    eolian_w_wind: float = 0.45
    eolian_w_dry: float = 0.30
    eolian_w_erodible: float = 0.25
    eolian_wind_lo: float = 3.0
    eolian_wind_hi: float = 10.0
    eolian_dry_ref_m3m3: float = 0.15


def build_grid(step_deg: float, lat_min: float, lat_max: float,
               bbox: Optional[Tuple[float, float, float, float]] = None) -> pd.DataFrame:
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


def unified_dem() -> "ee.Image":
    """Seamless global DEM: CryoSat-2 Antarctica + Copernicus GLO-30 2024 + NASADEM 30m."""
    ant_dem = ee.Image("CPOM/CryoSat2/ANTARCTICA_DEM").select("elevation").toFloat()
    glo = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic().rename("elevation").toFloat()
    nasadem = ee.Image("NASA/NASADEM_HGT/001").select("elevation").toFloat()
    return (
        ee.ImageCollection([ant_dem, glo, nasadem])
        .mosaic()
        .rename("elevation")
        .setDefaultProjection(ee.Projection("EPSG:4326").atScale(30))
    )


def stage1_image(y0: int, y1: int) -> "ee.Image":
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
    elev_1km = unified_dem().rename("elev_1km")
    return ee.Image.cat([ndvi_max, lc, elev_1km])


def stage2_image(y0: int, y1: int, safe_slope_deg: float, flat_slope_deg: float, win_px: int) -> "ee.Image":
    start, end = f"{y0}-01-01", f"{y1 + 1}-01-01"
    dem = unified_dem()

    slope = ee.Terrain.slope(dem).rename("slope")
    kernel = ee.Kernel.circle(radius=win_px * 30.0, units="meters")
    mean_slope = slope.reduceNeighborhood(ee.Reducer.mean(), kernel).rename("mean_slope")
    slope_p75 = slope.reduceNeighborhood(ee.Reducer.percentile([75]), kernel).rename("slope_p75")
    safe_frac = slope.lte(safe_slope_deg).reduceNeighborhood(ee.Reducer.mean(), kernel).rename("safe_frac")
    flat_frac = slope.lte(flat_slope_deg).reduceNeighborhood(ee.Reducer.mean(), kernel).rename("flat_frac")

    resid = dem.subtract(dem.focal_mean(45.0, "square", "meters"))
    rms_rough = resid.pow(2).reduceNeighborhood(ee.Reducer.mean(), kernel).sqrt().rename("rms_rough")
    slope_90 = ee.Terrain.slope(dem.focal_mean(45.0, "square", "meters")).rename("slope_90")
    slope_270 = ee.Terrain.slope(dem.focal_mean(135.0, "square", "meters")).rename("slope_270")

    def prep(img):
        qa = img.select("QA_PIXEL")
        clear = qa.bitwiseAnd(30).eq(0)
        refl = (
            img.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
                       ["blue", "green", "red", "nir", "swir1", "swir2"])
            .multiply(0.0000275).add(-0.2)
        )
        return refl.updateMask(clear)

    # Merge Tier 1 (global) and Tier 2 (required for Antarctica Collection 2 scenes)
    l8_t1 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(start, end).map(prep)
    l9_t1 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterDate(start, end).map(prep)
    l8_t2 = ee.ImageCollection("LANDSAT/LC08/C02/T2_L2").filterDate(start, end).map(prep)
    l9_t2 = ee.ImageCollection("LANDSAT/LC09/C02/T2_L2").filterDate(start, end).map(prep)
    landsat = l8_t1.merge(l9_t1).merge(l8_t2).merge(l9_t2)
    refl_median = landsat.median()
    n_obs = landsat.select("blue").count().rename("n_obs")

    lst = ee.ImageCollection("MODIS/061/MYD11A2").filterDate(start, end)
    stats = ee.Reducer.mean().combine(ee.Reducer.percentile([5, 95]), sharedInputs=True)
    day = lst.select("LST_Day_1km").reduce(stats).multiply(0.02).rename(["day_mean", "day_p5", "day_p95"])
    night = lst.select("LST_Night_1km").reduce(stats).multiply(0.02).rename(["night_mean", "night_p5", "night_p95"])
    clear_frac = lst.select("LST_Day_1km").count().divide(ee.Image.constant(lst.size())).rename("clear_frac")

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


def init_earth_engine(project: str) -> None:
    if ee is None:
        sys.exit("earthengine-api is not installed. Run: pip install earthengine-api")
    try:
        ee.Initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception as exc:
        sys.exit(f"Earth Engine initialisation failed: {exc}")


def _reduce_points(batch: pd.DataFrame, image: "ee.Image", scale: float, tile_scale: int) -> pd.DataFrame:
    feats = [
        ee.Feature(ee.Geometry.Point([float(lo), float(la)]), {"pid": int(p)})
        for p, la, lo in zip(batch["pid"], batch["latitude"], batch["longitude"])
    ]
    fc = ee.FeatureCollection(feats)
    out = image.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=scale, tileScale=tile_scale).getInfo()
    return pd.DataFrame([f["properties"] for f in out["features"]])


def _extract_with_split(batch: pd.DataFrame, image: "ee.Image", scale: float, tile_scale: int,
                        min_size: int = 30, depth: int = 0) -> pd.DataFrame:
    try:
        return _reduce_points(batch, image, scale, tile_scale)
    except Exception as exc:
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
              batch_size: int, workers: int, tile_scale: int,
              checkpoint_csv: Optional[Path] = None, resume: bool = True) -> pd.DataFrame:
    pts = points.copy()
    results: List[pd.DataFrame] = []

    if checkpoint_csv and checkpoint_csv.exists():
        if resume:
            prev = pd.read_csv(checkpoint_csv).drop_duplicates(subset=["pid"], keep="last")
            done_pids = set(pd.to_numeric(prev["pid"], errors="coerce").dropna().astype(int).tolist())
            pts = pts[~pts["pid"].isin(done_pids)].copy()
            results.append(prev)
            logger.info(f"[{label}] Resumed {len(done_pids)} completed points; {len(pts)} remaining")
        else:
            checkpoint_csv.unlink()

    if pts.empty:
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["pid"])

    pts["_band"] = np.floor((pts["latitude"] + 90.0) / 5.0)
    pts = pts.sort_values(["_band", "longitude"]).drop(columns="_band").reset_index(drop=True)
    batches = [pts.iloc[i:i + batch_size] for i in range(0, len(pts), batch_size)]
    logger.info(f"[{label}] {len(pts)} points in {len(batches)} batches ({workers} parallel workers)")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_extract_with_split, b, image, scale, tile_scale) for b in batches]
        for k, fut in enumerate(as_completed(futs), start=1):
            res = fut.result()
            if not res.empty:
                results.append(res)
                if checkpoint_csv:
                    write_hdr = not checkpoint_csv.exists()
                    res.to_csv(checkpoint_csv, mode="a", header=write_hdr, index=False)
            if k % 1 == 0:
                logger.info(f"[{label}] {k}/{len(futs)} batches done in {time.time() - t0:.0f}s")
    if not results:
        return pd.DataFrame(columns=["pid"])
    return pd.concat(results, ignore_index=True)


def derive_features(raw: pd.DataFrame, dp: DerivationParams, gp: GeotechParams) -> pd.DataFrame:
    d = raw.copy()
    eps = 1e-6
    for _c in [
        "latitude", "longitude", "elevation", "mean_slope", "slope_p75", "safe_frac", "flat_frac",
        "rms_rough", "slope", "slope_90", "slope_270", "blue", "green", "red", "nir", "swir1", "swir2",
        "day_p95", "night_p5", "day_mean", "night_mean", "clear_frac", "sp_pa", "wind_mean",
        "wind_max_monthly", "ssrd_wm2", "swvl1", "sand_pct", "clay_pct", "bd_0_15", "bd_0_60",
    ]:
        if _c not in d.columns:
            d[_c] = np.nan
        d[_c] = pd.to_numeric(d[_c], errors="coerce").astype(np.float64)

    for c in ["blue", "green", "red", "nir", "swir1", "swir2"]:
        d[c] = d[c].clip(0.0, 1.5)

    d["NDVI"] = (d["nir"] - d["red"]) / (d["nir"] + d["red"] + eps)
    d["NDWI"] = (d["green"] - d["nir"]) / (d["green"] + d["nir"] + eps)
    d["NDSI"] = (d["green"] - d["swir1"]) / (d["green"] + d["swir1"] + eps)

    d["Altitude_m"] = d["elevation"]
    d["Mean_Slope_deg"] = d["mean_slope"].fillna(d["slope"]).fillna(3.5)
    d["Traverse_Slope_deg"] = d["slope_p75"].fillna(d["Mean_Slope_deg"] * 1.15)
    d["RMS_Roughness_m"] = d["rms_rough"].fillna(1.2)
    d["Safe_Landing_Frac"] = d["safe_frac"].fillna((d["Mean_Slope_deg"] <= 5.0).astype(float))
    d["Ultra_Flat_Frac"] = d["flat_frac"].fillna((d["Mean_Slope_deg"] <= 2.0).astype(float))
    d["Slope_30m_deg"] = d["slope"].fillna(d["Mean_Slope_deg"])
    d["Slope_90m_deg"] = d["slope_90"].fillna(d["Mean_Slope_deg"] * 0.9)
    d["Slope_270m_deg"] = d["slope_270"].fillna(d["Mean_Slope_deg"] * 0.8)

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

    alt_m = d["Altitude_m"].fillna(0.0)
    lat_rad = np.radians(d["latitude"].abs())
    t_clim_mean = 300.0 - 65.0 * (np.sin(lat_rad) ** 1.5) - 0.0062 * np.maximum(alt_m, 0.0)
    day_m = d["day_mean"].fillna(t_clim_mean + 8.0)
    night_m = d["night_mean"].fillna(day_m - 16.0)
    d["T_max_K"] = d["day_p95"].fillna(day_m + 10.0)
    d["T_min_K"] = d["night_p5"].fillna(night_m - 10.0)
    d["T_mean_K"] = 0.5 * (day_m + night_m)
    d["Diurnal_Delta_T_K"] = (day_m - night_m).clip(lower=2.0)
    d["ATI"] = (1.0 - d["Albedo"]) / d["Diurnal_Delta_T_K"]
    d["Thermal_Inertia"] = (dp.ti_scale * d["ATI"]).clip(10.0, 1000.0)
    d["Sun_Visibility_pct"] = (d["clear_frac"].fillna(0.70) * 100.0).clip(0.0, 100.0)

    raw_ice = ((d["NDSI"] - 0.15).clip(lower=0.0) * 95.0).clip(0.0, 85.0)
    d["Water_Ice_wt_pct"] = np.where(d["T_min_K"] < 273.15, raw_ice, 0.0)

    sp_mbar_fallback = 1013.25 * np.exp(-np.maximum(alt_m, -100.0) / 8434.5)
    d["Surface_Pressure_mbar"] = (d["sp_pa"] / 100.0).fillna(sp_mbar_fallback)
    d["Wind_Mean_mps"] = d["wind_mean"].fillna(7.2)
    d["Wind_Peak_Monthly_mps"] = d["wind_max_monthly"].fillna(13.5)
    ssrd_fallback = (
        180.0
        * np.cos(np.radians(np.clip(d["latitude"].abs(), 0.0, 85.0)))
        * np.exp(np.maximum(alt_m, 0.0) / 8000.0)
    )
    d["UV_Flux_Index"] = (d["ssrd_wm2"].fillna(ssrd_fallback) / dp.uv_scale_w_m2).clip(0.0, 20.0)

    sand = d["sand_pct"].fillna(68.0).clip(0, 100)
    clay = d["clay_pct"].fillna(6.0).clip(0, 100)
    silt = (100.0 - sand - clay).clip(lower=0.0)
    tot = (sand + silt + clay).replace(0.0, np.nan)
    rho = d["bd_0_60"].fillna(1620.0)
    d["Density_0_15cm_kg_m3"] = d["bd_0_15"].fillna(1510.0)
    d["Density_0_60cm_kg_m3"] = rho
    phi = (sand * gp.phi_sand_deg + silt * gp.phi_silt_deg + clay * gp.phi_clay_deg) / tot
    phi = (phi + gp.density_phi_gain_deg_per_1000 * (rho - 1500.0) / 1000.0).clip(20.0, 45.0)
    theta = d["swvl1"].fillna(0.02).clip(0.0, 0.5)
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

    wind_term = ((d["Wind_Peak_Monthly_mps"] - dp.eolian_wind_lo) / (dp.eolian_wind_hi - dp.eolian_wind_lo)).clip(0, 1)
    dry_term = 1.0 - (theta / dp.eolian_dry_ref_m3m3).clip(0.0, 1.0)
    erodible = (0.6 * sand / 100.0 + 0.4 * (1.0 - clay / 100.0)).clip(0.0, 1.0)
    d["Eolian_Deflation_Idx"] = 100.0 * (
        dp.eolian_w_wind * wind_term + dp.eolian_w_dry * dry_term + dp.eolian_w_erodible * erodible
    )
    return d


def calibrate_thermal_inertia(df: pd.DataFrame, anchors: pd.DataFrame, data_dir: Path, dp: DerivationParams) -> float:
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
    return dp.ti_scale


def run_polar_append(project: str, step_polar: float, buffer_km: float, y0: int, y1: int,
                     batch_size: int, workers: int, tile_scale: int,
                     data_dir: Path, cache_dir: Path) -> None:
    """Extracts Antarctic (via Landsat T2_L2 + CryoSat-2) and High Arctic analog provinces at 0.25 deg."""
    csv_path = cache_dir / "global_abiotic_grid.csv"
    if not csv_path.exists():
        sys.exit(f"{csv_path} not found. Run a standard extraction first.")

    init_earth_engine(project)
    anchors = load_flow_a_anchors(data_dir)

    polar_boxes = [
        ("Antarctica_VictoriaLand_DryValleys", 155.0, -80.0, 172.0, -70.0),
        ("Antarctica_QueenMaud_Nunataks", -15.0, -75.5, 25.0, -70.0),
        ("Antarctica_Ellsworth_Transantarctic", -88.0, -82.0, -75.0, -77.5),
        ("Antarctica_Peninsula_Margins", -66.0, -66.5, -57.0, -63.0),
        ("Arctic_Devon_Ellesmere_AxelHeiberg", -105.0, 72.0, -70.0, 82.0),
        ("Arctic_Svalbard_Archipelago", 10.0, 76.5, 28.0, 80.5),
        ("Arctic_WestGreenland_Kangerlussuaq", -56.0, 66.5, -48.0, 71.0),
    ]
    grid = pd.concat(
        [build_grid(step_polar, b[2], b[4], (b[1], b[2], b[3], b[4])) for b in polar_boxes],
        ignore_index=True,
    ).drop_duplicates(subset=["latitude", "longitude"]).reset_index(drop=True)

    dmin = min_distance_to_sites_km(grid.latitude.values, grid.longitude.values,
                                    anchors.latitude.values, anchors.longitude.values)
    grid = grid[dmin >= buffer_km].reset_index(drop=True)
    grid["country"] = np.where(grid["latitude"] < -60.0, "Antarctic Dry / Cryogenic Regolith", "Arctic Polar Desert")
    grid["site_name"] = [f"PERSEUS_{la:+07.2f}_{lo:+08.2f}" for la, lo in zip(grid.latitude, grid.longitude)]
    grid["catalog_role"] = "discovery_candidate"
    grid["is_known_site"] = False
    grid["pid"] = np.arange(len(grid))
    logger.info(f"[Polar Append] Querying {len(grid)} polar province cells at {step_polar} deg (Landsat T1+T2 + CryoSat-2 + GLO-30)")

    img2 = stage2_image(max(y0, y1 - 1), y1, 5.0, 2.0, 16)
    s2 = run_stage(grid, img2, 90, "polar_stage2", batch_size, workers, tile_scale, resume=False)
    if s2.empty:
        sys.exit("[Polar Append] Stage 2 returned no data.")

    raw = grid.merge(s2, on="pid", how="inner").drop(columns=["pid"])
    for _b in ["elevation", "blue", "green", "red", "nir", "swir1", "swir2"]:
        if _b not in raw.columns:
            raw[_b] = np.nan
        raw[_b] = pd.to_numeric(raw[_b], errors="coerce")

    raw = raw.dropna(subset=["elevation", "blue", "green", "red", "nir", "swir1", "swir2"]).reset_index(drop=True)
    raw = raw[raw["elevation"] > 5.0].reset_index(drop=True)
    logger.info(f"[Polar Append] {len(raw)} polar land cells with complete DEM + Landsat T1/T2 spectra")

    dp = DerivationParams()
    gp = GeotechParams()
    polar_feats = derive_features(raw, dp, gp)
    polar_feats = polar_feats[(polar_feats["NDVI"] < 0.15) & (polar_feats["Albedo"] < 0.72)].reset_index(drop=True)
    logger.info(f"[Polar Append] Retained {len(polar_feats)} ice-free / cryogenic polar analog cells (Albedo < 0.72)")

    existing = pd.read_csv(csv_path)
    if "T_min_K" in existing.columns and "Water_Ice_wt_pct" in existing.columns:
        existing["Water_Ice_wt_pct"] = np.where(existing["T_min_K"] < 273.15, existing["Water_Ice_wt_pct"], 0.0)
    existing = existing[(existing["latitude"].between(-60.0, 60.0)) | existing["is_known_site"]]

    combined = pd.concat([existing, polar_feats], ignore_index=True)
    combined = combined.sort_values(["is_known_site", "latitude", "longitude"], ascending=[False, True, True]).reset_index(drop=True)
    combined.round(5).to_csv(csv_path, index=False)

    meta_path = cache_dir / "extraction_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["discovery_cells_kept"] = int((~combined["is_known_site"]).sum())
    meta["polar_south_cells_kept"] = int(((~combined["is_known_site"]) & (combined["latitude"] < -60.0)).sum())
    meta["polar_north_cells_kept"] = int(((~combined["is_known_site"]) & (combined["latitude"] > 60.0)).sum())
    meta["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        f"[Polar Append] Merged! Total cells: {len(combined)} "
        f"(Antarctic: {meta['polar_south_cells_kept']}, Arctic: {meta['polar_north_cells_kept']})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Team Perseus - Flow B global abiotic extractor (Google Earth Engine)")
    ap.add_argument("--project", required=True, help="Google Cloud project id registered for Earth Engine")
    ap.add_argument("--mode", choices=["smoke", "fast", "standard", "full"], default="standard")
    ap.add_argument("--append-polar", action="store_true", help="fast polar-only extraction merged into existing CSV")
    ap.add_argument("--polar-step-deg", type=float, default=0.25, help="grid step in degrees for --append-polar")
    ap.add_argument("--step-deg", type=float, default=None)
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), default=None)
    ap.add_argument("--lat-min", type=float, default=-85.0)
    ap.add_argument("--lat-max", type=float, default=83.0)
    ap.add_argument("--buffer-km", type=float, default=50.0)
    ap.add_argument("--years", type=int, nargs=2, default=[2021, 2024], metavar=("START", "END"))
    ap.add_argument("--stage1-ndvi-max", type=float, default=0.20)
    ap.add_argument("--ndvi-max", type=float, default=0.15)
    ap.add_argument("--safe-slope-deg", type=float, default=5.0)
    ap.add_argument("--flat-slope-deg", type=float, default=2.0)
    ap.add_argument("--window-px", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=150)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tile-scale", type=int, default=4)
    ap.add_argument("--gravity", type=float, default=9.81)
    ap.add_argument("--footing-width", type=float, default=1.0)
    ap.add_argument("--ti-scale", type=float, default=None)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    cache_dir = script_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    y0, y1 = args.years

    if args.append_polar:
        run_polar_append(
            args.project, args.polar_step_deg, args.buffer_km, y0, y1,
            args.batch_size, args.workers, args.tile_scale, data_dir, cache_dir,
        )
        return

    step = args.step_deg or {"smoke": 2.0, "fast": 1.0, "standard": 0.5, "full": 0.25}[args.mode]
    dp = DerivationParams()
    gp = GeotechParams(gravity_m_s2=args.gravity, footing_width_m=args.footing_width)
    if args.ti_scale:
        dp.ti_scale = args.ti_scale

    init_earth_engine(args.project)
    anchors = load_flow_a_anchors(data_dir)
    grid = build_grid(step, args.lat_min, args.lat_max, tuple(args.bbox) if args.bbox else None)
    dmin = min_distance_to_sites_km(grid.latitude.values, grid.longitude.values,
                                    anchors.latitude.values, anchors.longitude.values)
    grid = grid[dmin >= args.buffer_km].reset_index(drop=True)
    grid["pid"] = np.arange(len(grid))

    s1 = run_stage(grid, stage1_image(y0, y1), 1000, "stage1", args.batch_size, args.workers, args.tile_scale, resume=False)
    s1 = s1.merge(grid, on="pid", how="inner").sort_values("pid").reset_index(drop=True)
    s1["elev_1km"] = pd.to_numeric(s1.get("elev_1km"), errors="coerce")
    polar_exempt = ((s1["latitude"] < -60.0) | (s1["latitude"] > 75.0)) & (s1["elev_1km"] > 5.0)
    modis_valid = (
        (s1["elev_1km"] > -420.0) & s1["lc"].notna() & ~s1["lc"].isin(LC_EXCLUDE)
        & s1["ndvi_max"].notna() & (s1["ndvi_max"] < args.stage1_ndvi_max)
    )
    keep = modis_valid | (polar_exempt & s1["ndvi_max"].isna())
    cand = s1[keep].copy().reset_index(drop=True)
    cand["country"] = cand["lc"].map(
        lambda x: LC_NAMES.get(int(x), "Polar / Ice-Free Cryogenic") if pd.notna(x) else "Antarctica / Polar"
    )
    cand["site_name"] = [f"PERSEUS_{la:+07.2f}_{lo:+08.2f}" for la, lo in zip(cand.latitude, cand.longitude)]
    cand["catalog_role"] = "discovery_candidate"
    cand["is_known_site"] = False
    cand = cand[["site_name", "country", "latitude", "longitude", "catalog_role", "is_known_site"]]

    anc = anchors.copy()
    anc["is_known_site"] = True
    pool = pd.concat([cand, anc], ignore_index=True)
    pool["pid"] = np.arange(len(pool))
    img2 = stage2_image(y0, y1, args.safe_slope_deg, args.flat_slope_deg, args.window_px)
    ckpt_path = cache_dir / "stage2_partial_checkpoint.csv"
    s2 = run_stage(pool, img2, 30, "stage2", args.batch_size, args.workers, args.tile_scale, checkpoint_csv=ckpt_path, resume=True)
    raw = pool.merge(s2, on="pid", how="inner").drop(columns=["pid"])
    needed = ["elevation", "blue", "green", "red", "nir", "swir1", "swir2"]
    raw = raw.dropna(subset=needed).reset_index(drop=True)

    feats = derive_features(raw, dp, gp)
    if args.ti_scale is None:
        dp.ti_scale = calibrate_thermal_inertia(feats, anchors, data_dir, dp)
        feats = derive_features(raw, dp, gp)
    abiotic = feats["is_known_site"] | (feats["NDVI"] < args.ndvi_max)
    final = feats[abiotic].reset_index(drop=True)
    final = final.sort_values(["is_known_site", "latitude", "longitude"], ascending=[False, True, True])

    csv_path = cache_dir / "global_abiotic_grid.csv"
    final.round(5).to_csv(csv_path, index=False)
    if ckpt_path.exists():
        ckpt_path.unlink()


if __name__ == "__main__":
    main()
