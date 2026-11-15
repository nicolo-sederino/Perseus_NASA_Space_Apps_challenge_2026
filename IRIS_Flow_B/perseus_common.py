#!/usr/bin/env python3
"""
NASA Space Apps Challenge 2026 - Team Perseus
Flow B (Global Discovery): shared utilities.

AI-Assistance Disclosure (NASA Space Apps Submission Guide):
This module was developed with AI code-generation assistance and reviewed/validated by Team Perseus.

Provides:
  - Haversine great-circle distance helpers (used for the Flow A exclusion buffer)
  - Loader for the Flow A curated Earth analog catalogs (calibration space ONLY, never part of the search space)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("PerseusCommon")

EARTH_RADIUS_KM = 6371.0088

# Non-quantitative columns of the Flow A catalogs (same list as the MATLAB core, `meta_cols`)
META_COLS: List[str] = [
    "Site_Name", "Country", "Category", "Latitude", "Longitude",
    "Region_State", "Geological_Setting", "Best_Lunar_Match",
    "Primary_Use", "Analog_Category", "Operational_Status", "Dominant_Mineral",
]

FLOW_A_EARTH_CATALOGS: Dict[str, str] = {
    "lunar_known": "Earth_Lunar_Analogs_Catalog.csv",
    "mars_known": "Earth_Mars_Analogs_Catalog.csv",
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorised, broadcasting-friendly)."""
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def min_distance_to_sites_km(
    lat: np.ndarray, lon: np.ndarray, site_lat: np.ndarray, site_lon: np.ndarray
) -> np.ndarray:
    """For every (lat, lon) returns the distance in km to the nearest site. Chunked to bound memory."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if len(site_lat) == 0:
        return np.full(lat.shape, np.inf)
    out = np.empty(lat.shape, dtype=np.float64)
    chunk = 50_000
    for i in range(0, len(lat), chunk):
        d = haversine_km(
            lat[i:i + chunk, None], lon[i:i + chunk, None],
            np.asarray(site_lat)[None, :], np.asarray(site_lon)[None, :],
        )
        out[i:i + chunk] = d.min(axis=1)
    return out


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Case-insensitive column lookup."""
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def read_earth_catalog(path: Path) -> pd.DataFrame:
    """Reads a Flow A Earth analog catalog and normalises the 4 identity columns."""
    df = pd.read_csv(path)
    lat_c = find_col(df, ["Latitude", "Lat"])
    lon_c = find_col(df, ["Longitude", "Lon", "Long"])
    name_c = find_col(df, ["Site_Name", "Name", "Site"])
    if lat_c is None or lon_c is None or name_c is None:
        raise ValueError(f"{path.name}: needs Site_Name, Latitude and Longitude columns (found {list(df.columns)})")
    df = df.rename(columns={lat_c: "Latitude", lon_c: "Longitude", name_c: "Site_Name"})
    ctry_c = find_col(df, ["Country"])
    if ctry_c and ctry_c != "Country":
        df = df.rename(columns={ctry_c: "Country"})
    if "Country" not in df.columns:
        df["Country"] = "Unknown"
    return df


def load_flow_a_anchors(data_dir: Path) -> pd.DataFrame:
    """
    Returns one row per unique Flow A site (lunar + mars Earth analog catalogs merged when the
    same location appears in both).  These are the sites that must be EXCLUDED from the discovery
    search space (geographic buffer) while still defining the calibration space.
    """
    records: List[Dict] = []
    for role, fname in FLOW_A_EARTH_CATALOGS.items():
        path = data_dir / fname
        if not path.exists():
            logger.warning(f"Flow A catalog not found: {path}")
            continue
        df = read_earth_catalog(path)
        for _, row in df.iterrows():
            lat, lon = float(row["Latitude"]), float(row["Longitude"])
            name = str(row["Site_Name"]).strip()
            merged = False
            for rec in records:
                if abs(rec["latitude"] - lat) < 0.08 and abs(rec["longitude"] - lon) < 0.08:
                    rec["catalog_role"] = "both_known"
                    rec["site_name"] = f"{rec['site_name']} / {name}"
                    merged = True
                    break
            if not merged:
                records.append({
                    "site_name": name,
                    "country": str(row.get("Country", "Unknown")).strip(),
                    "latitude": lat,
                    "longitude": lon,
                    "catalog_role": role,
                })
    out = pd.DataFrame(records, columns=["site_name", "country", "latitude", "longitude", "catalog_role"])
    logger.info(f"Loaded {len(out)} unique Flow A anchor sites from {data_dir}")
    return out
