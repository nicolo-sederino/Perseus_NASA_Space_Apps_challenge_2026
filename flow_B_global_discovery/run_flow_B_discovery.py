#!/usr/bin/env python3
"""
NASA Space Apps Challenge 2026 - Team Perseus
Flow B (Global Discovery): Planetary analog discovery with Flow A calibration space and Flow A site exclusion

AI-Assistance Disclosure (NASA Space Apps Submission Guide):
This script was developed with AI code-generation assistance and reviewed/validated by Team Perseus.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from perseus_common import (
    META_COLS,
    find_col,
    load_flow_a_anchors,
    min_distance_to_sites_km,
    read_earth_catalog,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("PerseusFlowBDiscovery")

CATEGORIES = ["Morphometry", "Thermophysics", "Geochemistry", "Environment"]
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Morphometry": ["slope", "rough", "altitude", "elevation", "flat", "landing", "traverse", "boulder", "pit_", "hab_sep", "dte"],
    "Thermophysics": ["delta_t", "t_mean", "t_min", "t_max", "thermal", "sun_vis", "temperature", "insolation"],
    "Geochemistry": ["albedo", "mafic", "feo", "tio2", "ice", "ferric", "salt", "evapor", "mineral", "sio2", "mgo"],
}


@dataclass
class DiscoveryParams:
    buffer_km: float = 50.0
    max_slope_lander_deg: float = 5.0
    max_slope_rover_deg: float = 12.0
    min_safe_landing_frac: float = 0.50
    dt_min_k: Optional[float] = None
    dt_max_k: Optional[float] = None
    ndvi_max: float = 0.15
    hard_filter: str = "none"
    radar_weight: float = 0.25
    radar_tau: float = 1.5
    z_clip: Optional[float] = 4.5
    category_weights: Dict[str, float] = field(default_factory=lambda: {c: 1.0 for c in CATEGORIES})
    category_overrides: Dict[str, str] = field(default_factory=dict)
    exclude_features: List[str] = field(default_factory=list)
    min_features: int = 8
    max_missing_frac: float = 0.50
    kmeans_k: int = 3
    random_state: int = 42


def fnum(x: Any, nd: int = 4) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def categorize(feature: str, overrides: Dict[str, str]) -> str:
    if feature in overrides:
        return overrides[feature]
    low = feature.lower()
    for cat in ("Morphometry", "Thermophysics", "Geochemistry"):
        if any(k in low for k in CATEGORY_KEYWORDS[cat]):
            return cat
    return "Environment"


def load_params(path: Optional[str], cli: Dict[str, Any]) -> DiscoveryParams:
    p = DiscoveryParams()
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f.name for f in fields(DiscoveryParams)}
        for k, v in data.items():
            if k in valid:
                setattr(p, k, v)
            else:
                logger.warning(f"Unknown parameter in {path}: {k}")
    for k, v in cli.items():
        if v is not None:
            setattr(p, k, v)
    return p


def sanitize_grid(df: pd.DataFrame, p: DiscoveryParams) -> pd.DataFrame:
    df = df.copy()
    for c in ["latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    df["site_name"] = df.get("site_name", pd.Series(["Unknown"] * len(df))).fillna("Unknown").astype(str)
    df["country"] = df.get("country", pd.Series(["Unknown"] * len(df))).fillna("Unknown").astype(str)
    df["is_known_site"] = df["is_known_site"].astype(str).str.lower().isin(["true", "1", "yes"])
    for c in df.columns:
        if c not in ("site_name", "country", "catalog_role", "is_known_site"):
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if "NDVI" in df.columns:
        df = df[df["is_known_site"] | (df["NDVI"] < p.ndvi_max)].reset_index(drop=True)
    return df


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


@dataclass
class CalibrationSpace:
    planet: str
    features: List[str]
    method: str
    mu_e: np.ndarray
    sd_e: np.ndarray
    V: np.ndarray
    explained_pct: np.ndarray
    weights: np.ndarray
    Ze: np.ndarray
    Zt: np.ndarray
    Se: np.ndarray
    St: np.ndarray
    kmeans: KMeans
    earth_names: List[str]
    target_names: List[str]
    earth_coords: np.ndarray


def _zstats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1) if len(X) > 1 else np.ones(X.shape[1])
    sd = np.where(sd == 0, 1e-5, sd)
    return mu, sd


def build_calibration_space(
    planet: str, earth_df: pd.DataFrame, target_df: pd.DataFrame, grid_cols: List[str], p: DiscoveryParams
) -> CalibrationSpace:
    earth_cols = set(numeric_feature_columns(earth_df))
    target_cols = set(numeric_feature_columns(target_df))
    shared = [c for c in earth_df.columns if c in earth_cols and c in target_cols]
    in_grid = [c for c in shared if c in grid_cols and c not in p.exclude_features]
    dropped = [c for c in shared if c not in in_grid]
    if dropped:
        logger.info(f"[{planet}] catalog features without a global satellite equivalent (not used): {dropped}")
    if len(in_grid) < p.min_features:
        raise SystemExit(
            f"[{planet}] only {len(in_grid)} usable features ({in_grid}).\n"
            f"Catalog columns: {sorted(shared)}\nGrid columns:   {sorted(grid_cols)}"
        )
    feats = in_grid

    Xe = earth_df[feats].apply(pd.to_numeric, errors="coerce")
    Xt = target_df[feats].apply(pd.to_numeric, errors="coerce")
    med = pd.concat([Xe, Xt]).median()
    Xe = Xe.fillna(med).to_numpy(float)
    Xt = Xt.fillna(med).to_numpy(float)

    if planet == "lunar":
        mu, sd = _zstats(np.vstack([Xe, Xt]))
        mu_e, sd_e, mu_t, sd_t = mu, sd, mu, sd
        method = "Standard Z-Score (Earth analogs + Moon targets pooled, MATLAB parity)"
    else:
        mu_e, sd_e = _zstats(Xe)
        if len(Xt) >= 3:
            mu_t, sd_t = _zstats(Xt)
            method = "Domain-Adapted Z-Score (Earth stats for Earth, Mars stats for Mars targets, MATLAB parity)"
        else:
            mu_t, sd_t = mu_e, sd_e
            method = "Z-Score with Earth statistics"
    w = np.array([p.category_weights.get(categorize(f, p.category_overrides), 1.0) for f in feats], float)
    Ze = ((Xe - mu_e) / sd_e) * w
    Zt = ((Xt - mu_t) / sd_t) * w

    ref = np.vstack([Ze, Zt])
    _, S, Vt = np.linalg.svd(ref - ref.mean(axis=0), full_matrices=False)
    keep = S > 1e-9 * S.max()
    V = Vt.T[:, keep]
    eig = (S[keep] ** 2) / max(len(ref) - 1, 1)
    explained = 100.0 * eig / eig.sum()

    km = KMeans(n_clusters=min(p.kmeans_k, len(Ze)), random_state=p.random_state, n_init=20).fit(Ze)
    name_c = find_col(earth_df, ["Site_Name"]) or earth_df.columns[0]
    tname_c = find_col(target_df, ["Site_Name", "Target_Name", "Name"]) or target_df.columns[0]
    lat_c, lon_c = find_col(earth_df, ["Latitude"]), find_col(earth_df, ["Longitude"])
    return CalibrationSpace(
        planet=planet, features=feats, method=method, mu_e=mu_e, sd_e=sd_e, V=V, explained_pct=explained,
        weights=w, Ze=Ze, Zt=Zt, Se=Ze @ V, St=Zt @ V, kmeans=km,
        earth_names=earth_df[name_c].astype(str).tolist(),
        target_names=target_df[tname_c].astype(str).str.strip().str.replace(" ", "_").tolist(),
        earth_coords=earth_df[[lat_c, lon_c]].to_numpy(float) if lat_c and lon_c else np.zeros((len(Ze), 2)),
    )


def cluster_labels(space: CalibrationSpace) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for cid, cen in enumerate(space.kmeans.cluster_centers_):
        idx = np.argsort(-np.abs(cen))[:2]
        desc = ", ".join(f"{'high' if cen[i] > 0 else 'low'} {space.features[i].replace('_', ' ')}" for i in idx)
        out[cid] = f"Archetype {cid + 1}: {desc}"
    return out


def prepare_candidates(cand: pd.DataFrame, feats: List[str], p: DiscoveryParams) -> pd.DataFrame:
    miss = cand[feats].isna().mean(axis=1)
    cand = cand[miss <= p.max_missing_frac].copy()
    cand["Imputed_Feature_Count"] = cand[feats].isna().sum(axis=1).astype(int)
    cand[feats] = cand[feats].fillna(cand[feats].median())
    return cand.reset_index(drop=True)


def add_engineering_flags(df: pd.DataFrame, p: DiscoveryParams) -> pd.DataFrame:
    df = df.copy()
    ok_l = pd.Series(True, index=df.index)
    ok_r = pd.Series(True, index=df.index)
    if "Mean_Slope_deg" in df.columns:
        ok_l &= df["Mean_Slope_deg"] <= p.max_slope_lander_deg
    if "Safe_Landing_Frac" in df.columns:
        ok_l &= df["Safe_Landing_Frac"] >= p.min_safe_landing_frac
    if "Traverse_Slope_deg" in df.columns:
        ok_r &= df["Traverse_Slope_deg"] <= p.max_slope_rover_deg
    elif "Mean_Slope_deg" in df.columns:
        ok_r &= df["Mean_Slope_deg"] <= p.max_slope_rover_deg
    if "Diurnal_Delta_T_K" in df.columns:
        if p.dt_min_k is not None:
            ok_l &= df["Diurnal_Delta_T_K"] >= p.dt_min_k
            ok_r &= df["Diurnal_Delta_T_K"] >= p.dt_min_k
        if p.dt_max_k is not None:
            ok_l &= df["Diurnal_Delta_T_K"] <= p.dt_max_k
            ok_r &= df["Diurnal_Delta_T_K"] <= p.dt_max_k
    df["Lander_Suitable"] = ok_l
    df["Rover_Suitable"] = ok_r
    return df


def score_candidates(space: CalibrationSpace, cand: pd.DataFrame, t_idx: int, p: DiscoveryParams,
                     labels: Dict[int, str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    feats = space.features
    Zg = ((cand[feats].to_numpy(float) - space.mu_e) / space.sd_e) * space.weights
    if p.z_clip is not None:
        Zg = np.clip(Zg, -p.z_clip, p.z_clip)
    Sg = Zg @ space.V
    st = space.St[t_idx]
    zt = space.Zt[t_idx]

    d = np.linalg.norm(Sg - st, axis=1)
    cos = (Sg @ st) / (np.linalg.norm(Sg, axis=1) * np.linalg.norm(st) + 1e-12)
    d_cat = np.linalg.norm(space.Se - st, axis=1)
    d_best, d_med = float(d_cat.min()), float(np.median(d_cat))
    latent_score = 100.0 * np.exp(-math.log(2.0) * d / max(d_med, 1e-9))

    res = cand.copy()
    res["Latent_Distance"] = d
    res["Cosine_Similarity"] = cos
    res["Distance_vs_FlowA_Best"] = d / max(d_best, 1e-9)
    cats = [categorize(f, p.category_overrides) for f in feats]
    diffs = np.abs(Zg - zt)
    radar = {}
    for c in CATEGORIES:
        cols = [i for i, cc in enumerate(cats) if cc == c]
        radar[c] = 100.0 * np.exp(-diffs[:, cols].mean(axis=1) / p.radar_tau) if cols else np.full(len(res), 50.0)
        res[f"{c}_Score"] = radar[c]
    radar_mean = np.nanmean(np.vstack([radar[c] for c in CATEGORIES]), axis=0)
    res["Similarity_Score_pct"] = np.clip((1.0 - p.radar_weight) * latent_score + p.radar_weight * radar_mean, 0.0, 100.0)
    clu = space.kmeans.predict(Zg)
    res["Cluster_ID"] = clu
    res["Cluster_Label"] = [labels.get(int(c), f"Archetype {int(c) + 1}") for c in clu]
    res["PC1"] = Sg[:, 0]
    res["PC2"] = Sg[:, 1] if Sg.shape[1] > 1 else 0.0

    res = res.sort_values("Similarity_Score_pct", ascending=False).reset_index(drop=True)
    res["Global_Rank"] = np.arange(1, len(res) + 1)
    return res, {"d_best_flow_a": d_best, "d_median_flow_a": d_med}


def feature_parity_report(grid_anchors: pd.DataFrame, earth_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lat_c, lon_c = find_col(earth_df, ["Latitude"]), find_col(earth_df, ["Longitude"])
    if grid_anchors.empty or not lat_c or not lon_c:
        return pd.DataFrame()
    pairs = []
    for i, r in earth_df.iterrows():
        m = grid_anchors[
            ((grid_anchors["latitude"] - float(r[lat_c])).abs() < 0.08)
            & ((grid_anchors["longitude"] - float(r[lon_c])).abs() < 0.08)
        ]
        if not m.empty:
            pairs.append((i, m.index[0]))
    if len(pairs) < 3:
        return pd.DataFrame()
    shared = [c for c in numeric_feature_columns(earth_df) if c in grid_anchors.columns]
    for c in shared:
        a = pd.to_numeric(earth_df.loc[[i for i, _ in pairs], c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(grid_anchors.loc[[j for _, j in pairs], c], errors="coerce").to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3:
            continue
        sa, sb = pd.Series(a[ok]), pd.Series(b[ok])
        rows.append({
            "feature": c, "n_sites": int(ok.sum()),
            "pearson_r": fnum(sa.corr(sb), 3), "spearman_rho": fnum(sa.corr(sb, method="spearman"), 3),
            "mean_abs_error": fnum(np.mean(np.abs(a[ok] - b[ok])), 4),
            "bias_satellite_minus_catalog": fnum(np.mean(b[ok] - a[ok]), 4),
            "catalog_mean": fnum(a[ok].mean(), 4), "satellite_mean": fnum(b[ok].mean(), 4),
        })
    return pd.DataFrame(rows)


def export_planet(planet: str, space: CalibrationSpace, res: pd.DataFrame, target_name: str, available: List[str],
                  ref_stats: Dict[str, float], p: DiscoveryParams, anchors_scored: List[Dict[str, Any]],
                  top_n: int, max_map_points: int, results_dir: Path, exclusion_info: Dict[str, Any]) -> None:
    n_known_in_output = 0
    meta = {
        "project": "NASA Space Apps Challenge 2026 - Team Perseus",
        "flow": f"Flow_B_{planet.capitalize()}",
        "planetary_body": planet.capitalize(),
        "reference_target": target_name,
        "available_targets": available,
        "standardization_method": space.method,
        "calibration_features": space.features,
        "calibration_catalog_rows": int(len(space.Ze)),
        "pca_explained_variance_pct": [round(float(v), 2) for v in space.explained_pct],
        "kmeans_k": int(space.kmeans.n_clusters),
        "flow_a_exclusion_buffer_km": p.buffer_km,
        "flow_a_sites_excluded_from_search": exclusion_info["n_anchors"],
        "flow_a_sites_in_results": n_known_in_output,
        "total_evaluated_sites": int(len(res)),
        "new_discovery_candidates_count": int(len(res)),
        "ranking_metric": "Euclidean distance in PCA space to target blended with 4-axis Radar sub-scores",
        "similarity_score_definition": "100*exp(-ln2*d/d_median) blended with Radar sub-scores",
        "flow_a_reference_distance_best": round(ref_stats["d_best_flow_a"], 4),
        "flow_a_reference_distance_median": round(ref_stats["d_median_flow_a"], 4),
        "engineering_parameters": {k: v for k, v in asdict(p).items() if k not in ("category_weights", "category_overrides")},
        "category_weights": p.category_weights,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    def point(row: pd.Series) -> Dict[str, Any]:
        tele = {f: fnum(row[f], 3) for f in space.features}
        for extra in ("Altitude_m", "Mean_Slope_deg", "Diurnal_Delta_T_K"):
            if extra in row.index:
                tele[extra] = fnum(row[extra], 3)
        return {
            "site_name": row["site_name"], "region": row["country"],
            "lat": round(float(row["latitude"]), 4), "lon": round(float(row["longitude"]), 4),
            "is_known_site": False,
            "global_rank": int(row["Global_Rank"]),
            "similarity_score": fnum(row["Similarity_Score_pct"], 2),
            "latent_distance": fnum(row["Latent_Distance"], 4),
            "cosine_similarity": fnum(row["Cosine_Similarity"], 4),
            "distance_vs_flow_a_best": fnum(row["Distance_vs_FlowA_Best"], 3),
            "cluster_id": int(row["Cluster_ID"]), "cluster_label": row["Cluster_Label"],
            "pc1": fnum(row["PC1"], 4), "pc2": fnum(row["PC2"], 4),
            "radar_subscores": {c: fnum(row[f"{c}_Score"], 2) for c in CATEGORIES},
            "engineering": {
                "lander_suitable": bool(row["Lander_Suitable"]), "rover_suitable": bool(row["Rover_Suitable"]),
                "imputed_feature_count": int(row["Imputed_Feature_Count"]),
                "nearest_flow_a_site_km": fnum(row["Nearest_FlowA_km"], 1),
            },
            "telemetry": tele,
        }

    map_df = res.head(max_map_points)
    heat = {"metadata": meta, "calibration_validation": anchors_scored, "grid_points": [point(r) for _, r in map_df.iterrows()]}
    (results_dir / f"flowB_{planet}_heatmap.json").write_text(json.dumps(heat, indent=2), encoding="utf-8")

    feats_geo = []
    for _, r in res.head(top_n).iterrows():
        pt = point(r)
        feats_geo.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [pt["lon"], pt["lat"]]},
            "properties": {
                "site_name": pt["site_name"], "region": pt["region"], "planetary_body": planet.capitalize(),
                "reference_target": target_name, "is_known_site": False, "discovery_rank": pt["global_rank"],
                **{k: pt[k] for k in ("similarity_score", "latent_distance", "cosine_similarity",
                                      "distance_vs_flow_a_best", "cluster_id", "cluster_label",
                                      "radar_subscores", "engineering")},
                "engineering_metrics": pt["telemetry"],
            },
        })
    geo = {"type": "FeatureCollection", "metadata": meta, "features": feats_geo}
    (results_dir / f"flowB_{planet}_discoveries.geojson").write_text(json.dumps(geo, indent=2), encoding="utf-8")


def load_targets(data_dir: Path, fname: str) -> pd.DataFrame:
    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing Flow A target catalog: {path}")
    df = pd.read_csv(path)
    name_c = find_col(df, ["Site_Name", "Target_Name", "Name"]) or df.columns[0]
    df = df.rename(columns={name_c: "Site_Name"}) if name_c != "Site_Name" else df
    return df


def run_planet(planet: str, grid: pd.DataFrame, candidates: pd.DataFrame, anchors_in_grid: pd.DataFrame,
               data_dir: Path, results_dir: Path, target_arg: Optional[str], p: DiscoveryParams, top_n: int,
               max_map_points: int, all_targets: bool, exclusion_info: Dict[str, Any]) -> pd.DataFrame:
    earth_file = "Earth_Lunar_Analogs_Catalog.csv" if planet == "lunar" else "Earth_Mars_Analogs_Catalog.csv"
    tgt_file = "Lunar_Targets_Catalog.csv" if planet == "lunar" else "Mars_Targets_Catalog.csv"
    earth_df = read_earth_catalog(data_dir / earth_file)
    target_df = load_targets(data_dir, tgt_file)

    space = build_calibration_space(planet, earth_df, target_df, list(grid.columns), p)
    labels = cluster_labels(space)
    logger.info(f"[{planet}] calibration space: {len(space.features)} features | {len(earth_df)} Earth analogs + "
                f"{len(target_df)} targets | PCA comps {len(space.explained_pct)} | {space.method}")

    cand = prepare_candidates(candidates, space.features, p)
    cand = add_engineering_flags(cand, p)
    if p.hard_filter == "lander":
        cand = cand[cand["Lander_Suitable"]].reset_index(drop=True)
    elif p.hard_filter == "rover":
        cand = cand[cand["Rover_Suitable"]].reset_index(drop=True)
    logger.info(f"[{planet}] {len(cand)} discovery cells enter scoring (hard_filter={p.hard_filter})")
    if cand.empty:
        raise SystemExit(f"[{planet}] no candidate cells left after filtering; relax the thresholds.")

    t_names = space.target_names
    chosen = target_arg.replace(" ", "_") if target_arg else t_names[0]
    if chosen not in t_names:
        logger.warning(f"[{planet}] target '{chosen}' not found, using '{t_names[0]}'. Available: {t_names}")
        chosen = t_names[0]
    t_idx = t_names.index(chosen)

    res, ref_stats = score_candidates(space, cand, t_idx, p, labels)

    anchors_scored: List[Dict[str, Any]] = []
    if not anchors_in_grid.empty:
        a = prepare_candidates(anchors_in_grid.copy(), space.features, DiscoveryParams(max_missing_frac=1.0))
        a = add_engineering_flags(a, p)
        a_res, _ = score_candidates(space, a, t_idx, p, labels)
        for _, r in a_res.iterrows():
            anchors_scored.append({
                "site_name": r["site_name"], "similarity_score_pct": fnum(r["Similarity_Score_pct"], 2),
                "distance_vs_flow_a_best": fnum(r["Distance_vs_FlowA_Best"], 3), "cluster_label": r["Cluster_Label"],
            })
    export_planet(planet, space, res, chosen, t_names, ref_stats, p, anchors_scored, top_n, max_map_points,
                  results_dir, exclusion_info)

    parity = feature_parity_report(anchors_in_grid.reset_index(drop=True), earth_df)
    if not parity.empty:
        parity.to_csv(results_dir / f"flowB_{planet}_feature_parity_report.csv", index=False)
        logger.info(f"[{planet}] satellite-vs-catalog parity (median Spearman rho over features): "
                    f"{parity['spearman_rho'].median():.2f}")

    top = res.head(3)
    for _, r in top.iterrows():
        logger.info(f"[{planet}] top new analog for {chosen}: {r['site_name']} ({r['country']}) | "
                    f"score {r['Similarity_Score_pct']:.1f}% | d/d_best_FlowA = {r['Distance_vs_FlowA_Best']:.2f}")

    if all_targets:
        compact: Dict[str, Any] = {}
        for ti, tn in enumerate(t_names):
            rr, _ = score_candidates(space, cand, ti, p, labels)
            compact[tn] = [
                {"site_name": r["site_name"], "lat": round(float(r["latitude"]), 4), "lon": round(float(r["longitude"]), 4),
                 "similarity_score": fnum(r["Similarity_Score_pct"], 2), "distance_vs_flow_a_best": fnum(r["Distance_vs_FlowA_Best"], 3)}
                for _, r in rr.head(top_n).iterrows()
            ]
        (results_dir / f"flowB_{planet}_all_targets_top{top_n}.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Team Perseus - Flow B Planetary Analog Discovery (Lunar & Mars)")
    ap.add_argument("--lunar-target", default=None)
    ap.add_argument("--mars-target", default=None)
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--max-map-points", type=int, default=3000)
    ap.add_argument("--all-targets", action="store_true")
    ap.add_argument("--params", default=None)
    ap.add_argument("--write-default-params", default=None, metavar="FILE")
    ap.add_argument("--grid", default=None)
    ap.add_argument("--buffer-km", type=float, default=None)
    ap.add_argument("--max-slope-lander", type=float, default=None, dest="max_slope_lander_deg")
    ap.add_argument("--max-slope-rover", type=float, default=None, dest="max_slope_rover_deg")
    ap.add_argument("--min-safe-frac", type=float, default=None, dest="min_safe_landing_frac")
    ap.add_argument("--dt-min", type=float, default=None, dest="dt_min_k")
    ap.add_argument("--dt-max", type=float, default=None, dest="dt_max_k")
    ap.add_argument("--hard-filter", choices=["none", "lander", "rover"], default=None)
    ap.add_argument("--radar-weight", type=float, default=None)
    ap.add_argument("--z-clip", type=float, default=None)
    ap.add_argument("--min-features", type=int, default=None)
    args = ap.parse_args()

    if args.write_default_params:
        Path(args.write_default_params).write_text(json.dumps(asdict(DiscoveryParams()), indent=2), encoding="utf-8")
        print(f"Default parameters written to {args.write_default_params}")
        return

    cli = {k: getattr(args, k) for k in ("buffer_km", "max_slope_lander_deg", "max_slope_rover_deg",
                                         "min_safe_landing_frac", "dt_min_k", "dt_max_k", "hard_filter",
                                         "radar_weight", "z_clip", "min_features")}
    p = load_params(args.params, cli)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_dir = repo_root / "data"
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    grid_csv = Path(args.grid) if args.grid else script_dir / "cache" / "global_abiotic_grid.csv"
    if not grid_csv.exists():
        raise FileNotFoundError(f"{grid_csv} not found. Run gee_global_extract.py first.")

    logger.info("=" * 78)
    logger.info("  NASA SPACE APPS 2026 - TEAM PERSEUS | FLOW B GLOBAL ANALOG DISCOVERY")
    logger.info("=" * 78)

    grid = sanitize_grid(pd.read_csv(grid_csv), p)
    anchors = load_flow_a_anchors(data_dir)

    grid["Nearest_FlowA_km"] = min_distance_to_sites_km(
        grid.latitude.values, grid.longitude.values, anchors.latitude.values, anchors.longitude.values
    )
    in_grid_anchor = grid["is_known_site"]
    anchors_in_grid = grid[in_grid_anchor].copy()
    pool = grid[~in_grid_anchor]
    candidates = pool[pool["Nearest_FlowA_km"] >= p.buffer_km].copy().reset_index(drop=True)
    n_excl = int(len(pool) - len(candidates))
    assert (candidates["Nearest_FlowA_km"] >= p.buffer_km).all()
    exclusion_info = {
        "buffer_km": p.buffer_km, "n_anchors": int(len(anchors)), "cells_removed_by_buffer": n_excl,
        "candidates_remaining": int(len(candidates)),
        "min_distance_to_any_flow_a_site_km_in_search_space": fnum(candidates["Nearest_FlowA_km"].min(), 2),
        "flow_a_sites": anchors[["site_name", "latitude", "longitude"]].to_dict("records"),
    }
    (results_dir / "flowB_exclusion_report.json").write_text(json.dumps(exclusion_info, indent=2), encoding="utf-8")
    logger.info(f"Flow A exclusion: {len(anchors)} sites, {p.buffer_km:.0f} km buffer -> {n_excl} cells removed, "
                f"{len(candidates)} discovery cells remain (min distance to Flow A = "
                f"{exclusion_info['min_distance_to_any_flow_a_site_km_in_search_space']} km)")

    for planet, tgt in (("lunar", args.lunar_target), ("mars", args.mars_target)):
        run_planet(planet, grid, candidates, anchors_in_grid, data_dir, results_dir, tgt, p, args.top_n,
                   args.max_map_points, args.all_targets, exclusion_info)

    logger.info("=" * 78)
    logger.info("EXPORTED to results/: flowB_{lunar,mars}_heatmap.json, flowB_{lunar,mars}_discoveries.geojson,")
    logger.info("  flowB_exclusion_report.json, flowB_*_feature_parity_report.csv")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()
