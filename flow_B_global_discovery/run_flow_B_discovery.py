#!/usr/bin/env python3
"""
NASA Space Apps Challenge 2026 - Team Perseus
Flow B (Global Discovery): Planetary Analog Scoring, PCA/SVD, K-Means (K=3) & Radar Sub-Scores

AI-Assistance Disclosure (NASA Space Apps Submission Guide):
This script was developed with AI code-generation assistance and reviewed/validated by Team Perseus.

Scientific & Mathematical Architecture (Harmonized with Flow A MATLAB Core):
  1. Loads cached global abiotic telemetry from `flow_B_global_discovery/cache/global_abiotic_grid.csv`.
  2. Sanitizes any corrupted/missing HTTP cells and enforces the abiotic threshold (`NDVI < 0.15`).
  3. Standardizes features via Z-Score (and Domain-Adapted Z-Score for Mars) anchored to Flow A
     reference catalogs (`data/Earth_Lunar_Analogs_Catalog.csv`, `data/Earth_Mars_Analogs_Catalog.csv`).
  4. Performs SVD-based Principal Component Analysis (PCA) and K-Means clustering (K = 3) to
     classify global abiotic terrain into 3 planetary analog archetypes.
  5. Computes 4 subsystem fidelity sub-scores (0-100) feeding the unified 4-axis Radar Chart:
       - Morphometry (slope safety <= 5 deg / <= 2 deg, RMS roughness, rover traverse)
       - Thermophysics (diurnal Delta-T thermal shock, thermal inertia, cryogenic/insolation regime)
       - Geochemistry (albedo, mafic basalt, FeO/TiO2, ferric oxide Red/Blue, evaporites, water ice)
       - Environment (abiotic purity NDVI < 0.15, barometric pressure, UV flux, eolian deflation, soil mechanics)
  6. Validates algorithm calibration against known Flow A analog sites and exports web-ready
     JSON heatmaps and RFC-7946 GeoJSON discovery layers into `results/`.
"""

import argparse
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PerseusFlowBDiscovery")

# Canonical feature vectors aligned with Flow A physical parameterization
LUNAR_FEATURES: List[str] = [
    "Mean_Slope_deg",
    "Slope_1m",
    "Slope_5m",
    "RMS_Roughness_m",
    "Safe_Landing_Frac",
    "Ultra_Flat_Frac",
    "Diurnal_Delta_T_K",
    "T_mean_K",
    "T_min_K",
    "Thermal_Inertia",
    "Sun_Visibility_pct",
    "Albedo",
    "Mafic_Basalt_Idx",
    "FeO_wt_pct",
    "TiO2_wt_pct",
    "Water_Ice_wt_pct",
    "Bearing_Capacity_kPa",
    "Soil_Cohesion_kPa",
    "Friction_Angle_deg",
]

MARS_FEATURES: List[str] = [
    "Altitude_m",
    "Mean_Slope_deg",
    "Traverse_Slope_deg",
    "RMS_Roughness_m",
    "Safe_Landing_Frac",
    "Diurnal_Delta_T_K",
    "T_mean_K",
    "T_min_K",
    "Thermal_Inertia",
    "Ferric_Oxide_Idx",
    "Salts_Evaporites_wt_pct",
    "Mafic_Basalt_Idx",
    "Water_Ice_wt_pct",
    "Albedo",
    "Surface_Pressure_mbar",
    "UV_Flux_Index",
    "Eolian_Deflation_Idx",
    "NDVI",
]

# Domain-Adapted weights for Mars Z-Score (emphasizing barometric, thermal-shock, and mineralogical axes)
MARS_DOMAIN_WEIGHTS: Dict[str, float] = {
    "Altitude_m": 1.25,
    "Mean_Slope_deg": 1.10,
    "Traverse_Slope_deg": 1.00,
    "RMS_Roughness_m": 1.00,
    "Safe_Landing_Frac": 1.15,
    "Diurnal_Delta_T_K": 1.35,
    "T_mean_K": 1.20,
    "T_min_K": 1.15,
    "Thermal_Inertia": 1.20,
    "Ferric_Oxide_Idx": 1.35,
    "Salts_Evaporites_wt_pct": 1.35,
    "Mafic_Basalt_Idx": 1.15,
    "Water_Ice_wt_pct": 1.15,
    "Albedo": 1.05,
    "Surface_Pressure_mbar": 1.30,
    "UV_Flux_Index": 1.25,
    "Eolian_Deflation_Idx": 1.30,
    "NDVI": 1.10,
}

# Built-in NASA planetary target profiles (used directly or merged with data/*_Targets_Catalog.csv)
DEFAULT_LUNAR_TARGETS: Dict[str, Dict[str, float]] = {
    "Shackleton_Rim_South_Pole": {
        "Mean_Slope_deg": 2.2,
        "Slope_1m": 1.5,
        "Slope_5m": 1.9,
        "RMS_Roughness_m": 1.1,
        "Safe_Landing_Frac": 0.95,
        "Ultra_Flat_Frac": 0.55,
        "Diurnal_Delta_T_K": 42.0,
        "T_mean_K": 215.0,
        "T_min_K": 195.0,
        "Thermal_Inertia": 135.0,
        "Sun_Visibility_pct": 88.0,
        "Albedo": 0.24,
        "Mafic_Basalt_Idx": 35.0,
        "FeO_wt_pct": 6.5,
        "TiO2_wt_pct": 0.9,
        "Water_Ice_wt_pct": 12.0,
        "Bearing_Capacity_kPa": 4.8,
        "Soil_Cohesion_kPa": 1.2,
        "Friction_Angle_deg": 37.5,
    },
    "Mare_Tranquillitatis_Basalt": {
        "Mean_Slope_deg": 1.5,
        "Slope_1m": 1.0,
        "Slope_5m": 1.3,
        "RMS_Roughness_m": 0.8,
        "Safe_Landing_Frac": 0.99,
        "Ultra_Flat_Frac": 0.75,
        "Diurnal_Delta_T_K": 45.0,
        "T_mean_K": 265.0,
        "T_min_K": 240.0,
        "Thermal_Inertia": 120.0,
        "Sun_Visibility_pct": 94.0,
        "Albedo": 0.08,
        "Mafic_Basalt_Idx": 86.0,
        "FeO_wt_pct": 15.2,
        "TiO2_wt_pct": 4.5,
        "Water_Ice_wt_pct": 0.0,
        "Bearing_Capacity_kPa": 4.5,
        "Soil_Cohesion_kPa": 0.6,
        "Friction_Angle_deg": 40.5,
    },
}

DEFAULT_MARS_TARGETS: Dict[str, Dict[str, float]] = {
    "Jezero_Crater_Delta_Basin": {
        "Altitude_m": 3200.0,
        "Mean_Slope_deg": 1.8,
        "Traverse_Slope_deg": 2.1,
        "RMS_Roughness_m": 1.2,
        "Safe_Landing_Frac": 0.97,
        "Diurnal_Delta_T_K": 41.0,
        "T_mean_K": 255.0,
        "T_min_K": 234.0,
        "Thermal_Inertia": 130.0,
        "Ferric_Oxide_Idx": 2.35,
        "Salts_Evaporites_wt_pct": 19.5,
        "Mafic_Basalt_Idx": 42.0,
        "Water_Ice_wt_pct": 2.0,
        "Albedo": 0.27,
        "Surface_Pressure_mbar": 680.0,
        "UV_Flux_Index": 10.5,
        "Eolian_Deflation_Idx": 86.0,
        "NDVI": 0.01,
    },
    "Gale_Crater_Aeolis_Palus": {
        "Altitude_m": 3800.0,
        "Mean_Slope_deg": 2.4,
        "Traverse_Slope_deg": 2.8,
        "RMS_Roughness_m": 1.6,
        "Safe_Landing_Frac": 0.92,
        "Diurnal_Delta_T_K": 39.0,
        "T_mean_K": 250.0,
        "T_min_K": 230.0,
        "Thermal_Inertia": 140.0,
        "Ferric_Oxide_Idx": 2.45,
        "Salts_Evaporites_wt_pct": 21.0,
        "Mafic_Basalt_Idx": 48.0,
        "Water_Ice_wt_pct": 1.5,
        "Albedo": 0.25,
        "Surface_Pressure_mbar": 635.0,
        "UV_Flux_Index": 11.2,
        "Eolian_Deflation_Idx": 88.0,
        "NDVI": 0.01,
    },
}


def sanitize_grid_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans missing/infinite telemetry values and enforces strict abiotic filtering on discovery cells."""
    df = df.copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    df["site_name"] = df["site_name"].fillna("Unknown_Site").astype(str)
    df["country"] = df["country"].fillna("Unknown_Region").astype(str)
    df["catalog_role"] = (
        df.get("catalog_role", pd.Series(["discovery_candidate"] * len(df)))
        .fillna("discovery_candidate")
        .astype(str)
    )
    df["is_known_site"] = df["is_known_site"].astype(str).str.lower().isin(["true", "1", "yes"])

    all_numeric_cols = sorted(set(LUNAR_FEATURES + MARS_FEATURES + ["NDWI", "NDSI", "Safe_Area_m2"]))
    for col in all_numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        med = float(df[col].median()) if not df[col].dropna().empty else 0.0
        df[col] = df[col].fillna(med)

    # Enforce strict abiotic threshold (NDVI < 0.15) on discovery candidates while retaining Flow A anchors
    abiotic_mask = df["is_known_site"] | (df["NDVI"] < 0.15)
    df_clean = df[abiotic_mask].reset_index(drop=True)
    return df_clean


def load_planetary_targets(
    data_dir: Path,
    filename: str,
    defaults: Dict[str, Dict[str, float]],
    features: List[str],
    ref_medians: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """Loads planetary targets from Flow A CSV catalogs if present, merged with built-in NASA baselines."""
    targets: Dict[str, Dict[str, float]] = {k: dict(v) for k, v in defaults.items()}
    csv_path = data_dir / filename
    if not csv_path.exists():
        return targets

    try:
        df_t = pd.read_csv(csv_path)
        name_col = next(
            (c for c in df_t.columns if c.lower() in ["target_name", "site_name", "name", "location"]),
            df_t.columns[0],
        )
        for _, row in df_t.iterrows():
            t_name = str(row[name_col]).strip().replace(" ", "_")
            vec: Dict[str, float] = {}
            for feat in features:
                if feat in df_t.columns and pd.notna(row[feat]):
                    vec[feat] = float(row[feat])
                else:
                    vec[feat] = float(ref_medians.get(feat, 0.0))
            targets[t_name] = vec
    except Exception as exc:
        logger.warning(f"Could not parse {filename} ({exc}); using built-in planetary baselines.")

    return targets


def compute_reference_scaler(
    df_grid: pd.DataFrame,
    data_dir: Path,
    catalog_filename: str,
    features: List[str],
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Computes Z-Score mean and std vectors anchored to Flow A curated catalogs
    (with automatic fallback to the known Flow A calibration sites inside global_abiotic_grid.csv).
    """
    cat_path = data_dir / catalog_filename
    ref_df: Optional[pd.DataFrame] = None
    source_label = "Flow_A_Calibration_Anchors_in_Grid"

    if cat_path.exists():
        try:
            raw_cat = pd.read_csv(cat_path)
            overlap = [f for f in features if f in raw_cat.columns]
            if len(overlap) >= len(features) // 2:
                aligned = pd.DataFrame(index=raw_cat.index)
                for f in features:
                    if f in raw_cat.columns:
                        aligned[f] = pd.to_numeric(raw_cat[f], errors="coerce").fillna(df_grid[f].median())
                    else:
                        aligned[f] = float(df_grid[f].median())
                if len(aligned) >= 3:
                    ref_df = aligned
                    source_label = f"Flow_A_Catalog ({catalog_filename})"
        except Exception:
            ref_df = None

    if ref_df is None:
        known_subset = df_grid[df_grid["is_known_site"]]
        ref_df = known_subset[features] if len(known_subset) >= 3 else df_grid[features]

    mu = ref_df[features].mean(axis=0).to_numpy(dtype=np.float64)
    sigma = ref_df[features].std(axis=0, ddof=0).to_numpy(dtype=np.float64)
    grid_sigma = df_grid[features].std(axis=0, ddof=0).to_numpy(dtype=np.float64)
    sigma = np.where(sigma < 1e-4, np.where(grid_sigma < 1e-4, 1.0, grid_sigma), sigma)
    return mu, sigma, source_label


def compute_lunar_subscores(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the 4 subsystem fidelity scores (0-100) for Lunar Analog Radar Charts."""
    slope_score = 100.0 * np.exp(-((np.maximum(df["Mean_Slope_deg"] - 1.5, 0.0) / 7.5) ** 1.6))
    rough_score = 100.0 * np.exp(-((np.maximum(df["RMS_Roughness_m"] - 0.8, 0.0) / 5.5) ** 1.4))
    safe_frac_score = 100.0 * np.clip(df["Safe_Landing_Frac"], 0.0, 1.0)
    morphometry = np.clip(0.45 * slope_score + 0.30 * safe_frac_score + 0.25 * rough_score, 5.0, 99.5)

    dt_score = 100.0 * np.clip((df["Diurnal_Delta_T_K"] - 12.0) / 30.0, 0.0, 1.0)
    ti_score = 100.0 * np.exp(-(((df["Thermal_Inertia"] - 130.0) / 85.0) ** 2))
    cold_bonus = 100.0 * np.clip((290.0 - df["T_mean_K"]) / 75.0, 0.0, 1.0)
    sun_score = np.clip(df["Sun_Visibility_pct"], 0.0, 100.0)
    thermophysics = np.clip(0.35 * dt_score + 0.25 * ti_score + 0.25 * cold_bonus + 0.15 * sun_score, 5.0, 99.5)

    mare_score = 100.0 * (
        0.55 * np.clip(df["Mafic_Basalt_Idx"] / 85.0, 0.0, 1.0)
        + 0.45 * np.clip((0.25 - df["Albedo"]) / 0.18, 0.0, 1.0)
    )
    highland_polar_score = 100.0 * (
        0.50 * np.clip((df["Albedo"] - 0.14) / 0.22, 0.0, 1.0)
        + 0.50 * np.clip(df["Water_Ice_wt_pct"] / 15.0 + 0.35, 0.0, 1.0)
    )
    geochemistry = np.clip(np.maximum(mare_score, highland_polar_score), 5.0, 99.5)

    abiotic_score = 100.0 * np.clip((0.15 - df["NDVI"]) / 0.15, 0.0, 1.0)
    bearing_score = 100.0 * np.exp(-(((df["Bearing_Capacity_kPa"] - 4.8) / 2.2) ** 2))
    friction_score = 100.0 * np.clip((df["Friction_Angle_deg"] - 28.0) / 14.0, 0.0, 1.0)
    environment = np.clip(0.55 * abiotic_score + 0.25 * bearing_score + 0.20 * friction_score, 5.0, 99.5)

    return pd.DataFrame({
        "Morphometry": np.round(morphometry, 2),
        "Thermophysics": np.round(thermophysics, 2),
        "Geochemistry": np.round(geochemistry, 2),
        "Environment": np.round(environment, 2),
    })


def compute_mars_subscores(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the 4 subsystem fidelity scores (0-100) for Martian Analog Radar Charts."""
    slope_score = 100.0 * np.exp(-((np.maximum(df["Mean_Slope_deg"] - 1.5, 0.0) / 8.5) ** 1.5))
    safe_frac_score = 100.0 * np.clip(df["Safe_Landing_Frac"], 0.0, 1.0)
    alt_morph_score = 100.0 * np.clip(df["Altitude_m"] / 4500.0, 0.0, 1.0)
    morphometry = np.clip(0.45 * slope_score + 0.30 * safe_frac_score + 0.25 * alt_morph_score, 5.0, 99.5)

    dt_score = 100.0 * np.clip((df["Diurnal_Delta_T_K"] - 14.0) / 28.0, 0.0, 1.0)
    cold_score = 100.0 * np.clip((305.0 - df["T_mean_K"]) / 65.0, 0.0, 1.0)
    ti_score = 100.0 * np.exp(-(((df["Thermal_Inertia"] - 135.0) / 85.0) ** 2))
    thermophysics = np.clip(0.45 * dt_score + 0.30 * cold_score + 0.25 * ti_score, 5.0, 99.5)

    ferric_score = 100.0 * np.clip((df["Ferric_Oxide_Idx"] - 1.1) / 1.5, 0.0, 1.0)
    evap_score = 100.0 * np.clip(df["Salts_Evaporites_wt_pct"] / 20.0, 0.0, 1.0)
    basalt_ice_score = 100.0 * np.clip(
        0.65 * (df["Mafic_Basalt_Idx"] / 85.0) + 0.35 * (df["Water_Ice_wt_pct"] / 20.0), 0.0, 1.0
    )
    geochemistry = np.clip(0.38 * ferric_score + 0.37 * evap_score + 0.25 * basalt_ice_score, 5.0, 99.5)

    press_score = 100.0 * np.clip((1013.25 - df["Surface_Pressure_mbar"]) / 500.0, 0.0, 1.0)
    uv_score = 100.0 * np.clip((df["UV_Flux_Index"] - 3.5) / 10.5, 0.0, 1.0)
    eolian_score = 100.0 * np.clip(df["Eolian_Deflation_Idx"] / 90.0, 0.0, 1.0)
    abiotic_score = 100.0 * np.clip((0.15 - df["NDVI"]) / 0.15, 0.0, 1.0)
    environment = np.clip(
        0.28 * press_score + 0.24 * uv_score + 0.26 * eolian_score + 0.22 * abiotic_score, 5.0, 99.5
    )

    return pd.DataFrame({
        "Morphometry": np.round(morphometry, 2),
        "Thermophysics": np.round(thermophysics, 2),
        "Geochemistry": np.round(geochemistry, 2),
        "Environment": np.round(environment, 2),
    })


def label_clusters_scientifically(
    df_sub: pd.DataFrame, cluster_ids: np.ndarray, planet: str
) -> Dict[int, str]:
    """Assigns deterministic, domain-grounded geological names to the K=3 K-Means clusters."""
    temp = df_sub.copy()
    temp["cid"] = cluster_ids
    centroids = temp.groupby("cid").mean(numeric_only=True)
    mapping: Dict[int, str] = {}
    remaining = list(centroids.index)

    if planet == "lunar":
        polar_cid = int(centroids.loc[remaining, "T_mean_K"].idxmin())
        mapping[polar_cid] = "Cluster 1: Polar Cryogenic & Volatile-Rich"
        remaining.remove(polar_cid)

        if remaining:
            mare_cid = int(centroids.loc[remaining, "Mafic_Basalt_Idx"].idxmax())
            mapping[mare_cid] = "Cluster 2: Mare Basaltic Plains (Low-Albedo)"
            remaining.remove(mare_cid)

        for cid in remaining:
            mapping[int(cid)] = "Cluster 3: Highland Anorthositic & Regolith Terrain"
    else:
        alt_cid = int(centroids.loc[remaining, "Altitude_m"].idxmax())
        mapping[alt_cid] = "Cluster 1: High-Altitude Radiative & Barometric Stress"
        remaining.remove(alt_cid)

        if remaining:
            evap_cid = int(centroids.loc[remaining, "Salts_Evaporites_wt_pct"].idxmax())
            mapping[evap_cid] = "Cluster 2: Hyperarid Evaporitic Playas & Saline Basins"
            remaining.remove(evap_cid)

        for cid in remaining:
            mapping[int(cid)] = "Cluster 3: Basaltic Regolith & Eolian Deflation Plains"

    return mapping


def run_planetary_discovery(
    df_grid: pd.DataFrame,
    planet: str,
    features: List[str],
    target_name: str,
    available_targets: List[str],
    target_profile: Dict[str, float],
    mu: np.ndarray,
    sigma: np.ndarray,
    scaler_source: str,
    top_n: int = 25,
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """Executes Z-Score standardization, PCA/SVD, K-Means (K=3), Radar sub-scores, and target matching."""
    X_raw = df_grid[features].to_numpy(dtype=np.float64)
    Z = (X_raw - mu) / sigma

    t_raw = np.array([float(target_profile.get(f, mu[i])) for i, f in enumerate(features)], dtype=np.float64)
    z_target = (t_raw - mu) / sigma

    if planet == "mars":
        w = np.array([MARS_DOMAIN_WEIGHTS.get(f, 1.0) for f in features], dtype=np.float64)
        Z = Z * w
        z_target = z_target * w

    Z = np.clip(Z, -5.0, 5.0)
    z_target = np.clip(z_target, -5.0, 5.0)

    n_comp = min(6, Z.shape[1], max(2, Z.shape[0] - 1))
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=42)
    Z_pca = pca.fit_transform(Z)
    target_pca = pca.transform(z_target.reshape(1, -1))[0]

    n_clusters = min(3, len(df_grid))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_ids = kmeans.fit_predict(Z_pca)
    cluster_names = label_clusters_scientifically(df_grid, cluster_ids, planet)

    var_weights = np.sqrt(np.maximum(pca.explained_variance_ratio_, 0.02))
    latent_dist = np.linalg.norm((Z_pca - target_pca) * var_weights, axis=1)
    latent_proximity_score = 100.0 * np.exp(-0.42 * latent_dist)

    subscores_df = compute_lunar_subscores(df_grid) if planet == "lunar" else compute_mars_subscores(df_grid)
    radar_mean = subscores_df.mean(axis=1).to_numpy(dtype=np.float64)

    similarity_score = np.round(np.clip(0.65 * radar_mean + 0.35 * latent_proximity_score, 1.0, 99.5), 2)

    res_df = df_grid.copy()
    res_df["Morphometry_Score"] = subscores_df["Morphometry"]
    res_df["Thermophysics_Score"] = subscores_df["Thermophysics"]
    res_df["Geochemistry_Score"] = subscores_df["Geochemistry"]
    res_df["Environment_Score"] = subscores_df["Environment"]
    res_df["Latent_Distance"] = np.round(latent_dist, 4)
    res_df["Similarity_Score_pct"] = similarity_score
    res_df["Cluster_ID"] = cluster_ids
    res_df["Cluster_Label"] = [cluster_names[int(c)] for c in cluster_ids]
    res_df["PC1"] = np.round(Z_pca[:, 0], 4)
    res_df["PC2"] = np.round(Z_pca[:, 1], 4)

    res_df = res_df.sort_values(by="Similarity_Score_pct", ascending=False).reset_index(drop=True)
    res_df["Global_Rank"] = np.arange(1, len(res_df) + 1)

    known_df = res_df[res_df["is_known_site"]].copy()
    calibration_records = []
    for _, row in known_df.iterrows():
        calibration_records.append({
            "site_name": row["site_name"],
            "catalog_role": row["catalog_role"],
            "global_rank": int(row["Global_Rank"]),
            "similarity_score_pct": float(row["Similarity_Score_pct"]),
            "cluster_label": row["Cluster_Label"],
            "radar_subscores": {
                "Morphometry": float(row["Morphometry_Score"]),
                "Thermophysics": float(row["Thermophysics_Score"]),
                "Geochemistry": float(row["Geochemistry_Score"]),
                "Environment": float(row["Environment_Score"]),
            },
        })

    heatmap_points = []
    for _, row in res_df.iterrows():
        heatmap_points.append({
            "site_name": row["site_name"],
            "region": row["country"],
            "lat": round(float(row["latitude"]), 4),
            "lon": round(float(row["longitude"]), 4),
            "is_known_site": bool(row["is_known_site"]),
            "global_rank": int(row["Global_Rank"]),
            "similarity_score": float(row["Similarity_Score_pct"]),
            "cluster_id": int(row["Cluster_ID"]),
            "cluster_label": row["Cluster_Label"],
            "pc1": float(row["PC1"]),
            "pc2": float(row["PC2"]),
            "radar_subscores": {
                "Morphometry": float(row["Morphometry_Score"]),
                "Thermophysics": float(row["Thermophysics_Score"]),
                "Geochemistry": float(row["Geochemistry_Score"]),
                "Environment": float(row["Environment_Score"]),
            },
            "telemetry": {
                "Altitude_m": float(row["Altitude_m"]),
                "Mean_Slope_deg": float(row["Mean_Slope_deg"]),
                "Diurnal_Delta_T_K": float(row["Diurnal_Delta_T_K"]),
                "T_mean_K": float(row["T_mean_K"]),
                "Albedo": float(row["Albedo"]),
                "Ferric_Oxide_Idx": float(row["Ferric_Oxide_Idx"]),
                "Mafic_Basalt_Idx": float(row["Mafic_Basalt_Idx"]),
                "Salts_Evaporites_wt_pct": float(row["Salts_Evaporites_wt_pct"]),
                "Surface_Pressure_mbar": float(row["Surface_Pressure_mbar"]),
                "UV_Flux_Index": float(row["UV_Flux_Index"]),
                "NDVI": float(row["NDVI"]),
            },
        })

    heatmap_json = {
        "metadata": {
            "project": "NASA Space Apps Challenge 2026 - Team Perseus",
            "flow": f"Flow_B_{planet.capitalize()}",
            "planetary_body": planet.capitalize(),
            "reference_target": target_name,
            "available_targets": available_targets,
            "standardization_method": "Domain-Adapted Z-Score" if planet == "mars" else "Standard Z-Score",
            "scaler_calibration_source": scaler_source,
            "pca_explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "kmeans_k": n_clusters,
            "total_evaluated_sites": len(res_df),
            "known_calibration_sites_count": int(res_df["is_known_site"].sum()),
            "new_discovery_candidates_count": int((~res_df["is_known_site"]).sum()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "calibration_validation": calibration_records,
        "grid_points": heatmap_points,
    }

    new_discoveries = res_df[~res_df["is_known_site"]].head(top_n).copy()
    new_discoveries["discovery_rank"] = np.arange(1, len(new_discoveries) + 1)

    known_export = res_df[res_df["is_known_site"]].copy()
    known_export["discovery_rank"] = 0

    combined_geo_df = pd.concat([new_discoveries, known_export], ignore_index=True)
    features_geo: List[Dict[str, Any]] = []
    for _, row in combined_geo_df.iterrows():
        features_geo.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(float(row["longitude"]), 4), round(float(row["latitude"]), 4)],
            },
            "properties": {
                "site_name": row["site_name"],
                "region": row["country"],
                "planetary_body": planet.capitalize(),
                "reference_target": target_name,
                "is_known_site": bool(row["is_known_site"]),
                "catalog_role": str(row["catalog_role"]),
                "discovery_rank": int(row["discovery_rank"]),
                "global_rank": int(row["Global_Rank"]),
                "similarity_score_pct": float(row["Similarity_Score_pct"]),
                "latent_distance": float(row["Latent_Distance"]),
                "cluster_id": int(row["Cluster_ID"]),
                "cluster_label": row["Cluster_Label"],
                "radar_subscores": {
                    "Morphometry": float(row["Morphometry_Score"]),
                    "Thermophysics": float(row["Thermophysics_Score"]),
                    "Geochemistry": float(row["Geochemistry_Score"]),
                    "Environment": float(row["Environment_Score"]),
                },
                "engineering_metrics": {
                    "Altitude_m": float(row["Altitude_m"]),
                    "Mean_Slope_deg": float(row["Mean_Slope_deg"]),
                    "Safe_Landing_Frac": float(row["Safe_Landing_Frac"]),
                    "RMS_Roughness_m": float(row["RMS_Roughness_m"]),
                    "Diurnal_Delta_T_K": float(row["Diurnal_Delta_T_K"]),
                    "T_mean_K": float(row["T_mean_K"]),
                    "Thermal_Inertia": float(row["Thermal_Inertia"]),
                    "Albedo": float(row["Albedo"]),
                    "Mafic_Basalt_Idx": float(row["Mafic_Basalt_Idx"]),
                    "Ferric_Oxide_Idx": float(row["Ferric_Oxide_Idx"]),
                    "Salts_Evaporites_wt_pct": float(row["Salts_Evaporites_wt_pct"]),
                    "Water_Ice_wt_pct": float(row["Water_Ice_wt_pct"]),
                    "Surface_Pressure_mbar": float(row["Surface_Pressure_mbar"]),
                    "UV_Flux_Index": float(row["UV_Flux_Index"]),
                    "Eolian_Deflation_Idx": float(row["Eolian_Deflation_Idx"]),
                    "NDVI": float(row["NDVI"]),
                },
            },
        })

    geojson_obj = {
        "type": "FeatureCollection",
        "metadata": heatmap_json["metadata"],
        "features": features_geo,
    }

    return heatmap_json, geojson_obj, res_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Team Perseus - Flow B Planetary Analog Discovery (Lunar & Mars)"
    )
    parser.add_argument(
        "--lunar-target",
        type=str,
        default="Shackleton_Rim_South_Pole",
        help="Target lunar site name from Lunar_Targets_Catalog.csv or built-in profiles",
    )
    parser.add_argument(
        "--mars-target",
        type=str,
        default="Jezero_Crater_Delta_Basin",
        help="Target Martian site name from Mars_Targets_Catalog.csv or built-in profiles",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top new analog discoveries to export in GeoJSON layers",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    cache_csv = script_dir / "cache" / "global_abiotic_grid.csv"
    data_dir = repo_root / "data"
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not cache_csv.exists():
        raise FileNotFoundError(
            f"Cached abiotic grid not found at {cache_csv}. Run extract_global_grid.py first."
        )

    logger.info("==========================================================================")
    logger.info("  NASA SPACE APPS 2026 - TEAM PERSEUS | FLOW B GLOBAL ANALOG DISCOVERY    ")
    logger.info("==========================================================================")

    raw_df = pd.read_csv(cache_csv)
    df_grid = sanitize_grid_dataframe(raw_df)
    logger.info(
        f"Loaded {len(df_grid)} clean abiotic cells "
        f"({int(df_grid['is_known_site'].sum())} Flow A anchors + "
        f"{int((~df_grid['is_known_site']).sum())} discovery candidates)."
    )

    ref_medians = df_grid.median(numeric_only=True).to_dict()

    # --- 1. FLOW B_LUNAR ---
    lunar_targets = load_planetary_targets(
        data_dir, "Lunar_Targets_Catalog.csv", DEFAULT_LUNAR_TARGETS, LUNAR_FEATURES, ref_medians
    )
    lunar_target_key = (
        args.lunar_target if args.lunar_target in lunar_targets else next(iter(lunar_targets.keys()))
    )
    mu_l, sigma_l, src_l = compute_reference_scaler(
        df_grid, data_dir, "Earth_Lunar_Analogs_Catalog.csv", LUNAR_FEATURES
    )
    lunar_hm, lunar_geo, lunar_ranked = run_planetary_discovery(
        df_grid=df_grid,
        planet="lunar",
        features=LUNAR_FEATURES,
        target_name=lunar_target_key,
        available_targets=list(lunar_targets.keys()),
        target_profile=lunar_targets[lunar_target_key],
        mu=mu_l,
        sigma=sigma_l,
        scaler_source=src_l,
        top_n=args.top_n,
    )

    with open(results_dir / "flowB_lunar_heatmap.json", "w", encoding="utf-8") as f:
        json.dump(lunar_hm, f, indent=2)
    with open(results_dir / "flowB_lunar_discoveries.geojson", "w", encoding="utf-8") as f:
        json.dump(lunar_geo, f, indent=2)

    logger.info(f"[Flow B_Lunar] Target: {lunar_target_key} | Scaler: {src_l}")
    top_l = lunar_ranked[~lunar_ranked["is_known_site"]].head(3)
    for _, r in top_l.iterrows():
        logger.info(
            f"  -> Top New Lunar Analog: {r['site_name']} ({r['country']}) | "
            f"Score: {r['Similarity_Score_pct']}% | {r['Cluster_Label']}"
        )

    # --- 2. FLOW B_MARS ---
    mars_targets = load_planetary_targets(
        data_dir, "Mars_Targets_Catalog.csv", DEFAULT_MARS_TARGETS, MARS_FEATURES, ref_medians
    )
    mars_target_key = (
        args.mars_target if args.mars_target in mars_targets else next(iter(mars_targets.keys()))
    )
    mu_m, sigma_m, src_m = compute_reference_scaler(
        df_grid, data_dir, "Earth_Mars_Analogs_Catalog.csv", MARS_FEATURES
    )
    mars_hm, mars_geo, mars_ranked = run_planetary_discovery(
        df_grid=df_grid,
        planet="mars",
        features=MARS_FEATURES,
        target_name=mars_target_key,
        available_targets=list(mars_targets.keys()),
        target_profile=mars_targets[mars_target_key],
        mu=mu_m,
        sigma=sigma_m,
        scaler_source=src_m,
        top_n=args.top_n,
    )

    with open(results_dir / "flowB_mars_heatmap.json", "w", encoding="utf-8") as f:
        json.dump(mars_hm, f, indent=2)
    with open(results_dir / "flowB_mars_discoveries.geojson", "w", encoding="utf-8") as f:
        json.dump(mars_geo, f, indent=2)

    logger.info(f"[Flow B_Mars] Target: {mars_target_key} | Scaler: {src_m}")
    top_m = mars_ranked[~mars_ranked["is_known_site"]].head(3)
    for _, r in top_m.iterrows():
        logger.info(
            f"  -> Top New Martian Analog: {r['site_name']} ({r['country']}) | "
            f"Score: {r['Similarity_Score_pct']}% | {r['Cluster_Label']}"
        )

    logger.info("==========================================================================")
    logger.info("EXPORTED 4 WEB-READY DISCOVERY ARTIFACTS TO results/:")
    logger.info("  1. results/flowB_lunar_heatmap.json")
    logger.info("  2. results/flowB_lunar_discoveries.geojson")
    logger.info("  3. results/flowB_mars_heatmap.json")
    logger.info("  4. results/flowB_mars_discoveries.geojson")
    logger.info("==========================================================================")


if __name__ == "__main__":
    main()
