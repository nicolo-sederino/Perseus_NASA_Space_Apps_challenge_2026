# Flow B: Global Planetary Analog Discovery Engine (`flow_B_global_discovery`)

**NASA Space Apps Challenge 2026 — Team Perseus**  
*Challenge: Identify Earth Locations that Analog the Permanent Moon Base Locations and Mars*

> **AI-Assistance Disclosure (NASA Space Apps Submission Guide):**  
> The Python modules in this directory were developed with AI code-generation assistance and subsequently reviewed, calibrated, and scientifically validated by Team Perseus.

---

## 1. Overview & Scientific Purpose

While **Flow A** (`core_matlab/analog_site_matcher.m`) evaluates and ranks **31 curated, literature-established terrestrial analog sites**, **Flow B** is a cloud-native, planetary-scale discovery pipeline built on **Google Earth Engine (GEE)** to uncover **previously uncatalogued Earth analog locations** for both the **Moon** (e.g., *Shackleton Rim / Artemis South Pole*) and **Mars** (e.g., *Olympus Mons*, *Jezero Crater*, *Gale Crater*).

### Core Design Principles
1. **Strict Zero-Overlap Discovery (Haversine Spatial Exclusion):**  
   The 31 curated Flow A analog sites are used **exclusively as an invisible statistical calibration space** to define feature standardization ($\mu, \sigma$), Singular Value Decomposition / Principal Component Analysis (SVD/PCA) axes, and K-Means geological archetypes. Prior to candidate ranking, every terrestrial grid cell within a configurable great-circle exclusion buffer (default: **$50\text{ km}$**) of any Flow A site is purged via Haversine filtering, guaranteeing that **100% of Flow B outputs are genuinely new candidate locations**.
2. **Mathematical Parity with Flow A (MATLAB Core):**  
   Feature selection is dynamically intersected across the Flow A Earth analog catalogs, planetary target catalogs, and satellite telemetry grid. Lunar scoring applies pooled standard Z-score normalization, whereas Martian scoring applies **Domain-Adapted Z-Score normalization** (standardizing Earth cells with Earth statistics and Martian targets with Martian statistics) to bridge cross-planetary atmospheric and thermal regime shifts.
3. **Server-Side Cloud Telemetry & Polar Coverage:**  
   All multi-sensor remote sensing reductions execute server-side inside Google Earth Engine datacenters across an equal-area global grid ($-85^\circ$ to $+83^\circ$ latitude), combining 30 m digital elevation models, cloud-masked Landsat 8/9 surface reflectance, MODIS Aqua day/night thermal infrared, ERA5-Land climate reanalysis, and OpenLandMap soil profiles—with explicit polar exemptions and barometric/radiative fallbacks for Antarctica and the High Arctic.

---

## 2. Module Architecture

| File | Role | Description |
| :--- | :--- | :--- |
| `gee_global_extract.py` | **Cloud Telemetry Extractor** | Builds an equal-area global grid, applies the $50\text{ km}$ Flow A Haversine exclusion buffer, runs a two-stage server-side Google Earth Engine reduction, derives physical/geotechnical features, auto-calibrates thermal inertia, and outputs `cache/global_abiotic_grid.csv`. |
| `run_flow_B_discovery.py` | **Statistical & PCA Discovery Engine** | Constructs the Lunar and Martian calibration spaces, projects global discovery cells into PCA latent space, assigns K-Means ($K=3$) geological archetypes, evaluates Lander/Rover engineering flags, computes 4-axis Radar sub-scores, and exports JSON/GeoJSON/CSV deliverables to `../results/`. |
| `perseus_common.py` | **Shared Geospatial Utilities** | Implements chunked, vectorized Haversine great-circle distance calculations ($R_{\oplus} = 6371.0088\text{ km}$), case-insensitive CSV schema normalization, and Flow A anchor deduplication. |
| `generate_local_viewer.py` | **Interactive Map Generator** | Compiles a standalone, zero-dependency Leaflet web map (`../results/perseus_local_viewer.html`) featuring Esri True-Color Satellite & Dark Tactical basemaps, Lunar/Mars layer toggles, interactive $50\text{ km}$ Flow A exclusion zone overlays, and telemetry popups. |
| `requirements.txt` | **Python Dependencies** | Pins `earthengine-api`, `numpy`, `pandas`, and `scikit-learn`. |

---

## 3. Data Pipeline & Physical Methodology

### Stage 1: Equal-Area Grid, Exclusion Buffer & Coarse Abiotic Mask ($1\text{ km}$)
* **Equal-Area Sampling:** Grid spacing in longitude scales inversely with $\cos(\text{latitude})$ to prevent polar oversampling (`--mode smoke`: $2.0^\circ$, `fast`: $1.0^\circ$, `standard`: $0.5^\circ$, `full`: $0.25^\circ$, yielding $\sim 656,348$ initial cells globally at $0.25^\circ$).
* **Flow A Exclusion Buffer:** Cells within `--buffer-km` (default $50.0\text{ km}$) of any Flow A site are removed before satellite querying.
* **Abiotic & Land-Cover Filter:** Uses `MODIS/061/MCD12Q1` (IGBP land cover) to exclude oceans/water bodies, forests, wetlands, croplands, and urban areas, alongside multi-year peak vegetation filtering (`MODIS/061/MOD13A3`, $\text{NDVI}_{\max} < 0.20$). High-latitude polar terrain ($\text{lat} < -60^\circ$ or $\text{lat} > 75^\circ$), where MODIS IGBP/NDVI products are masked, is explicitly exempted and passed to Stage 2.

### Stage 2: High-Resolution Multi-Sensor Telemetry ($30\text{ m} \text{ to } 11\text{ km}$)
Surviving discovery cells and the 31 Flow A calibration anchors are processed in spatially sorted, self-splitting batches with incremental disk checkpointing (`cache/stage2_partial_checkpoint.csv`):

1. **Morphometry & Multi-Scale Slope (`NASA/NASADEM_HGT/001` + `COPERNICUS/DEM/GLO30_2024_1`):**
   * Seamless $30\text{ m}$ global elevation mosaic combining SRTM NASADEM with Copernicus GLO-30 at high latitudes ($|\text{lat}| > 56^\circ$).
   * Evaluates pixel slope ($30\text{ m}$), focal-smoothed slopes ($90\text{ m}$ and $270\text{ m}$), 75th-percentile traverse slope, detrended RMS surface roughness, safe-landing fraction ($\text{slope} \le 5^\circ$), and ultra-flat fraction ($\text{slope} \le 2^\circ$) within an isotropic metric circle ($R = 480\text{ m}$, $\approx 1\text{ km}$ diameter) to avoid high-latitude longitudinal distortion.
2. **Optical/SWIR Reflectance & Mineral Proxies (`LANDSAT/LC08/C02/T1_L2` + `LC09`):**
   * Cloud- and shadow-masked (`QA_PIXEL`) multi-year surface reflectance median composite (`SR_B2` through `SR_B7`).
   * Computes strict abiotic $\text{NDVI}$ ($< 0.15$), $\text{NDWI}$, $\text{NDSI}$, broadband Albedo ($A$), Ferric Oxide Index ($\text{Red}/\text{Blue}$), Mafic/Basalt Index, and spectral proxies for $\text{FeO}$ ($\text{wt}\%$), $\text{TiO}_2$ ($\text{wt}\%$), evaporite salts ($\text{wt}\%$), and surface water ice ($\text{wt}\%$).
3. **Diurnal Thermophysics (`MODIS/061/MYD11A2`):**
   * Extracts multi-year daytime ($13{:}30$) and nighttime ($01{:}30$) Land Surface Temperature statistics ($T_{\max}$ 95th percentile, $T_{\min}$ 5th percentile, $T_{\text{mean}}$, diurnal swing $\Delta T = T_{\text{day}} - T_{\text{night}}$) and clear-sky solar visibility percentage.
   * Computes Apparent Thermal Inertia:
     $$\text{ATI} = \frac{1 - A}{\Delta T}$$
     and auto-calibrates the scaling factor $k$ against Flow A catalog `Thermal_Inertia` values via median ratio matching across the 31 anchor sites.
4. **Atmosphere, Solar Radiation & Winds (`ECMWF/ERA5_LAND/MONTHLY_AGGR`):**
   * Extracts 10 m mean and peak monthly wind speeds, barometric surface pressure, downward shortwave solar radiation (UV flux proxy), and top-layer volumetric soil moisture (`swvl1`).
   * Below $-60^\circ$ latitude (where ERA5-Land is masked over Antarctica), applies an exponential barometric scale-height fallback ($H = 8434.5\text{ m}$) and latitude/elevation radiative fallback.
5. **Regolith Geotechnics & Aeolian Deflation (`OpenLandMap SOL` 250 m):**
   * Combines sand/clay weight fractions and soil bulk density ($0\text{--}15\text{ cm}$ and $0\text{--}60\text{ cm}$) with ERA5-Land soil moisture to estimate internal friction angle ($\phi$), soil cohesion ($c$), Terzaghi/Vesic strip-footing bearing capacity ($q_u$), and aeolian deflation susceptibility.

### Stage 3: Latent Space Projection & Similarity Scoring (`run_flow_B_discovery.py`)
* **Standardization & Outlier Clipping:**
  $$Z_{ij} = \text{clip}\left( w_j \frac{X_{ij} - \mu_j}{\sigma_j},\; -z_{\text{clip}},\; +z_{\text{clip}} \right)$$
  with default $z_{\text{clip}} = 4.5$ to prevent extreme terrestrial-to-lunar geotechnical scale disparities (e.g., Earth consolidated soil bearing capacity vs. lunar regolith) from dominating Euclidean distances.
* **SVD/PCA & K-Means Archetypes:** Singular Value Decomposition is fitted strictly on the standardized Flow A calibration matrix $[Z_{\text{Earth}}; Z_{\text{Target}}]$; global discovery cells are projected onto the eigenvectors $V$ ($S_g = Z_g V$). K-Means ($K=3$, `random_state=42`) partitions sites into data-driven geological archetypes.
* **Similarity Score & Radar Sub-Scores:**
  * Latent Euclidean distance $d = \| S_g - S_{\text{target}} \|_2$ is benchmarked against the median distance $d_{\text{median}}$ and minimum distance $d_{\text{best}}$ of Flow A catalog analogs:
    $$\text{Score}_{\text{latent}} = 100 \cdot \exp\left(-\ln(2)\,\frac{d}{d_{\text{median}}}\right)$$
    where a score $> 50\%$ outperforms the median Flow A catalog analog, and $d / d_{\text{best}} < 1.0$ indicates a closer multivariate match than *any* previously catalogued site.
  * Per-category similarity sub-scores ($0\text{--}100$) are computed across **Morphometry**, **Thermophysics**, **Geochemistry**, and **Environment** to power the 4-axis Radar Chart.

---

## 4. Installation & Quickstart (Ubuntu / Linux)

### 4.1 Environment Setup
```bash
cd ~/Perseus_NASA_Space_Apps_challenge_2026/flow_B_global_discovery
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
```

Authenticate your Google Cloud / Earth Engine account once:
```bash
earthengine authenticate
```

### 4.2 Running the Full Discovery Pipeline
Replace `<YOUR_GCP_PROJECT_ID>` with your Earth Engine-registered Google Cloud Project ID (e.g., `perseus-space-apps-2026`):

```bash
# 1. Extract global abiotic satellite grid (use --mode smoke/fast for quick tests, --mode full for 0.25 deg global)
python3 gee_global_extract.py --project <YOUR_GCP_PROJECT_ID> --mode full --batch-size 150 --workers 6

# 2. Run Lunar & Martian discovery scoring across all catalog targets
python3 run_flow_B_discovery.py --all-targets

# 3. Generate the standalone interactive Leaflet map
python3 generate_local_viewer.py
```

### 4.3 Customizing Engineering & Scientific Parameters
You can export and edit all engineering thresholds, category weights, and hard filters (`lander` slope $\le 5^\circ$, `rover` slope $\le 12^\circ$, diurnal $\Delta T$ windows):
```bash
python3 run_flow_B_discovery.py --write-default-params custom_params.json
python3 run_flow_B_discovery.py --params custom_params.json --hard-filter lander --all-targets
```

---

## 5. Generated Deliverables (`../results/`)

* `flowB_lunar_heatmap.json` & `flowB_mars_heatmap.json`: Full metadata, Flow A anchor validation diagnostics, PCA coordinates, 4-axis Radar sub-scores, engineering flags, and top scored discovery candidates.
* `flowB_lunar_discoveries.geojson` & `flowB_mars_discoveries.geojson`: GIS-ready GeoJSON feature collections of the top-$N$ newly discovered analog locations.
* `flowB_exclusion_report.json`: Auditable verification log proving zero overlap with the 31 Flow A sites and reporting the minimum distance ($> 50\text{ km}$) of all surviving candidates.
* `flowB_lunar_feature_parity_report.csv` & `flowB_mars_feature_parity_report.csv`: Site-by-site Pearson $r$ and Spearman $\rho$ correlation benchmarks comparing satellite-extracted values against Flow A catalog ground truth.
* `perseus_local_viewer.html`: Self-contained interactive satellite map ready for local inspection or GitHub Pages deployment.

---

## 6. Known Scientific Limitations & Transparency Notes

In accordance with NASA open-science best practices, the following methodological boundaries are explicitly documented in `cache/extraction_metadata.json`:
1. **Sub-Decameter Micro-Topography:** Global $1\text{ m}$ and $5\text{ m}$ baseline slopes (`Slope_1m`, `Slope_5m`) and boulder-field counts are unavailable globally and are intentionally omitted rather than synthetically extrapolated; Flow B restricts its feature intersection to verified $30\text{ m}\text{--}270\text{ m}$ DEM metrics.
2. **Spectral Proxies vs. Laboratory Assays:** Columns such as `FeO_wt_pct`, `TiO2_wt_pct`, `Salts_Evaporites_wt_pct`, and `Water_Ice_wt_pct` represent multi-band optical/SWIR spectral-index proxies rather than XRF/XRD laboratory measurements.
3. **Antarctic Pedological Coverage:** `OpenLandMap` does not cover latitudes south of $-60^\circ$; Antarctic discovery cells are retained via `Soil_Data_Available = 0` and evaluated on their morphometric, thermal, optical, and barometric/radiative attributes.
