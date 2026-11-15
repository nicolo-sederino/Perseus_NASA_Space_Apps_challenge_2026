# SYBIL — Planetary Analogue Discovery & Mission Test Planning System

> **Project Perseus — Flow A (Catalog-Driven Unsupervised ML & Multi-Constraint Engineering Decision Engine)**  
> *NASA Space Apps Challenge 2026 | Challenge: "Identify Earth Locations that Analog the Permanent Moon Base Locations and Mars"*

```text
==========================================================================
                      _____ __   __ ____   ___  _                         
                     / ____|\ \ / /|  _ \ |_ _|| |                        
                    | (___   \ V / | |_) | | | | |                        
                     \___ \   | |  |  _ <  | | | |                        
                     ____) |  | |  | |_) | | | | |___                     
                    |_____/   |_|  |____/ |___||_____|                    
                                                                          
      SYBIL: The oracle and support system for the mission test planning  
==========================================================================
```

## 1. Executive Overview & Challenge Alignment

Selecting future landing sites and permanent surface base locations on the Moon and Mars requires balancing terrain safety, regolith geotechnics, thermal extremes, illumination, direct-to-Earth (DTE) telecommunications, radiation shielding, and In-Situ Resource Utilization (ISRU). While terrestrial analogue sites allow mission teams to validate hardware and operational procedures prior to flight, existing analogue campaigns are often selected qualitatively and lack quantitative, mission-specific characterization.

**SYBIL** is the catalog-driven analytical core of **Project Perseus** (operating alongside **IRIS**, our global Earth-Observation discovery engine). Implemented in pure MATLAB with **zero external toolbox dependencies**, SYBIL ingests open planetary and terrestrial datasets from NASA and international space agency partners to:

1. **Quantify Geophysical & Engineering Similarity:** Project terrestrial analogue sites and lunar/Martian target regions into a joint standardized latent space via **Singular Value Decomposition (SVD) Principal Component Analysis (PCA)** across **24 lunar features** and **15 Martian features**.
2. **Discover Geomorphological Families Unsupervisedly:** Partition terrestrial sites into $k = 5$ natural geological and environmental clusters using a custom implementation of **Lloyd's K-Means algorithm** with automated regex-based semantic cluster labeling.
3. **Perform Bidirectional & Dual-Metric Matching:** Resolve both **Mission-Driven queries** (*Target $\to$ Best Earth Analogue*, Streams A & C) and **Site-Driven queries** (*Earth Analogue $\to$ Best Planetary Target*, Streams B & D), combining Euclidean distance with Cosine Similarity for high-dimensional Martian atmospheric and radiative regimes.
4. **Plan Multi-Phase Engineering Campaigns (Multi-Test Decision Engine):** Evaluate terrestrial sites against multi-constraint Boolean engineering masks and continuous $0\text{--}100$ scoring functions across **6 Lunar operational test scenarios** and **7 Martian operational test scenarios**, supporting simultaneous multi-test intersections (`AND`) and a **Best-Fit Fallback** mode when combining physically opposing terrains.

---

## 2. Repository Structure & Quickstart Guide

### 2.1 Module Contents (`SYBIL/`)

| File | Role & Description |
| :--- | :--- |
| `SYBIL_app.m` | **Interactive UI Control Center (Pro UI Edition):** Single-window MATLAB `uifigure` application featuring real-time Moon/Mars switching, target dropdowns, telemetry readout, collapsible legends, interactive 3D PCA cube, polar Radar Chart, `geoaxes` Earth topographic map, 2D planetary map, and point-and-click Multi-Test Mission Planner. |
| `Sybil_Machine_Learning.m` | **Full Diagnostic Pipeline & Dual-Dashboard Script:** Executes the complete analytical pipeline, prints exhaustive bidirectional matching tables (Streams A, B, C, and D) to the Command Window, renders maximized dashboards (`fig1` for Moon, `fig2` for Mars), and launches an interactive CLI `while` loop for multi-test queries. |
| `moon_map.jpg` | Local equirectangular lunar surface basemap (`loadPlanetMap` automatically fetches or procedurally synthesizes the texture if missing). |
| `mars_map.jpg` | Local equirectangular Martian true-color surface basemap (with automatic online/procedural fallback). |

### 2.2 Prerequisites & Execution

* **Software:** MATLAB R2021a or newer (Base MATLAB only; no *Statistics and Machine Learning Toolbox* or *Mapping Toolbox* required).
* **Required Input Catalogs:** Ensure the four schema-agnostic CSV catalogs from `../data/` are accessible in the active working directory (or add `../data` to your MATLAB path):
  * `Lunar_Targets_Catalog.csv`
  * `Earth_Lunar_Analogs_Catalog.csv`
  * `Mars_Targets_Catalog.csv`
  * `Earth_Mars_Analogs_Catalog.csv`

```matlab
% 1. Add the data directory to the MATLAB path (if running from SYBIL/)
addpath('../data');

% 2a. Launch the Interactive UI Control Center (Recommended)
SYBIL_app

% 2b. Or run the Full CLI Diagnostic Report & Dual-Figure Pipeline
Sybil_Machine_Learning
```

---

## 3. Data-Driven & Schema-Agnostic Architecture

SYBIL contains **zero hardcoded site names, row indices, or column positions**. Both `SYBIL_app.m` and `Sybil_Machine_Learning.m` dynamically inspect the input CSV schemas at runtime:

1. **Deterministic Environment Setup:** Clears stale graphics handles and defines `meta_cols` (`Site_Name`, `Country`, `Category`, `Latitude`, `Longitude`, `Region_State`, `Geological_Setting`, `Best_Lunar_Match`, `Primary_Use`, `Analog_Category`, `Operational_Status`, `Dominant_Mineral`).
   > **Engineering Note on Spatial Coordinates:** Although `Latitude` and `Longitude` are floating-point numbers, they are explicitly placed inside `meta_cols`. Excluding GPS coordinates from the feature matrices prevents the unsupervised ML algorithms from grouping sites by geographic proximity (e.g., clustering all North American sites together) rather than by geophysical, thermal, and geotechnical equivalence.
2. **Polymorphic Taxonomy Ingestion:** Dynamically inspects `T_earth.Properties.VariableNames` to extract site classifications whether stored under `Category`, `Analog_Category`, or `Geological_Setting`, falling back gracefully via `repmat` if absent.
3. **Order-Preserving Feature Alignment (`'stable'`):** Extracts shared quantitative features via set operations:

$$\mathcal{F}_{\text{Planet}} = \left(\text{Columns}(T_{\text{planet}}) \cap \text{Columns}(T_{\text{earth}})\right) \setminus \mathcal{M}_{\text{meta}}$$

Passing the `'stable'` flag to `intersect` and `setdiff` preserves the physical domain ordering of the CSV columns ($\text{Topography} \to \text{Soil Mechanics} \to \text{Geochemistry} \to \text{Thermal/Radiative Environment} \to \text{Subsurface Morphology}$).

---

## 4. Mathematical & Machine Learning Formulation

### 4.1 Joint Z-Score Standardization

In planetary and terrestrial catalogs, physical parameters span over six orders of magnitude—for example, `Safe_Area_m2` reaches $800{,}000\text{ m}^2$, whereas `Boulder_Area_frac` varies between $0.01$ and $0.08$. Without normalization, a $1\%$ variation in safe area ($\Delta = 8{,}000$) would dominate squared Euclidean distance by a factor of $\sim 10^{11}$ relative to boulder coverage.

SYBIL concatenates the terrestrial and planetary matrices $X_{\text{all}} = [X_{\text{Earth}}; X_{\text{Planet}}] \in \mathbb{R}^{N_{\text{tot}} \times d}$ and transforms each feature $j \in \{1, \dots, d\}$ into a dimensionless standardized variable (with a zero-variance guard $\sigma_j = 0 \implies \sigma_j \leftarrow 10^{-5}$):

$$\mu_j = \frac{1}{N_{\text{tot}}}\sum_{i=1}^{N_{\text{tot}}} X_{i,j}, \qquad \sigma_j = \sqrt{\frac{1}{N_{\text{tot}}-1}\sum_{i=1}^{N_{\text{tot}}} (X_{i,j} - \mu_j)^2}, \qquad Z_{i,j} = \frac{X_{i,j} - \mu_j}{\sigma_j}$$

### 4.2 Dimensionality Reduction via Economy SVD-PCA

Given the mean-centered standardized matrix $Z_c = Z - \bar{Z}$, SYBIL computes the economy Singular Value Decomposition `[~, S, V] = svd(Z_c, 'econ')`:

$$Z_c = U \Sigma V^T$$

* $V \in \mathbb{R}^{d \times d}$ is the orthogonal eigenvector matrix ($V^T V = I$) defining the principal component axes ($\text{PC1}, \dots, \text{PC}d$).
* $\Sigma$ (`S`) is the diagonal matrix of singular values $\sigma_{\text{svd},1} \ge \sigma_{\text{svd},2} \ge \dots \ge \sigma_{\text{svd},d} \ge 0$.
* All sites are projected into the orthogonal latent space via `score_all = Z * V`.

**Full-Space Distance Invariance vs. 3D Visualization:** Because $V$ is orthogonal, Euclidean distances computed across all $d$ components (`score_Earth_full`, `score_Planet_full`) are strictly invariant to the coordinate transformation:

$$\|s_e - s_p\|_2 = \|(z_e - z_p)V\|_2 = \|z_e - z_p\|_2$$

Consequently, SYBIL separates the 3-component projection (`score_*_3D`, capturing $>68\%$ of variance on the Moon for 3D visual inspection) from the full $d$-dimensional projection used for exact analogue matching. The percentage of explained variance for each axis $k$ is computed from the covariance eigenvalues $\lambda_k = S_{k,k}^2 / (N_{\text{tot}} - 1)$:

$$\text{Explained}_k(\%) = \frac{\lambda_k}{\sum_{j=1}^{d} \lambda_j} \times 100$$

### 4.3 Unsupervised K-Means Clustering ($k = 5$) & Semantic Labeling

To discover natural geomorphological groupings among terrestrial sites without imposing manual labels, SYBIL implements Lloyd's iterative K-Means algorithm (`rng(42)`, $100$ iterations) minimizing the Within-Cluster Sum of Squares (WCSS):

$$J = \sum_{c=1}^{k} \sum_{i \in C_c} \|z_i - \mu_c\|_2^2$$

Partitioning at $k = 5$ separates terrestrial sites into five physical families:
1. **Lava tubes & subsurface cavities** (e.g., *Craters of the Moon*, *Lava River Cave*, *Nāhuku*, *Natal Cave*)
2. **Basaltic volcanic plains & lunar maria analogues** (e.g., *Black Point Lava Flow*, *Harrat Khaybar*, *Afar*, *Payenia*)
3. **Anorthositic highland complexes** (e.g., *Miyake-jima*, *Nusfjord*)
4. **Extreme polar & hyper-arid deserts** (e.g., *McMurdo Dry Valleys*, *Atacama Desert*)
5. **Impact craters, calderas & collapse structures** (e.g., *Haughton Crater*, *Askja*, *Lonar*, *Barda Negra*)

**Automated Semantic Cluster Label Generator:** Because unsupervised clustering outputs numeric IDs ($1\text{--}5$), SYBIL dynamically strips parenthetical subtypes via `regexprep(cat, '\s*\(.*?\)', '')`, counts category frequencies via `unique` and `accumarray`, and constructs human-readable legend titles formatted as `C1: Dominant Category 1 / Dominant Category 2`.

### 4.4 Bidirectional & Dual-Metric Matching (Streams A, B, C & D)

SYBIL evaluates correspondence in both directions:
* **Mission-Driven Streams (A: Moon $\to$ Earth; C: Mars $\to$ Earth):** Identifies the closest terrestrial twin $e^*(m)$ for each planetary target $m$:

$$e^*(m) = \arg\min_{e \in \{1, \dots, N_{\text{Earth}}\}} \|s_{\text{Earth}}(e) - s_{\text{Planet}}(m)\|_2$$

* **Site-Driven Inverse Streams (B: Earth $\to$ Moon; D: Earth $\to$ Mars):** Identifies which planetary region $m^*(e)$ each terrestrial facility best simulates:

$$m^*(e) = \arg\min_{m \in \{1, \dots, N_{\text{Planet}}\}} \|s_{\text{Planet}}(m) - s_{\text{Earth}}(e)\|_2$$

#### Dual-Metric Matching on Mars: Euclidean Distance + Cosine Similarity
In the Martian module, Euclidean distance is paired with **Cosine Similarity** to capture directional alignment across atmospheric, radiative, and geological vectors:

$$d_{\text{Eucl}}(u, v) = \sqrt{\sum_{j=1}^{15}(u_j - v_j)^2}, \qquad S_{\text{Cos}}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

* **Multidimensional Simultaneity (*Kilimanjaro Caldera & Glaciers, Tanzania*):** In Stream C, Kilimanjaro ($5{,}895\text{ m}$ a.s.l. near the equator) ranks as the top match for *Eastern Hellas Basin* ($d_{\text{Eucl}} = 2.862$, $S_{\text{Cos}} = 0.8020$) because it simultaneously combines depressed barometric pressure ($\sim 500\text{ mbar}$), intense UV flux ($\text{UV} = 11$), sublimating ice ($65\%$), and a basaltic volcanic substrate ($80\%$).
* **Value of Inverse Stream D (*Icelandic Volcanic Sites*):** Low-elevation, high-latitude Icelandic sites (*Krýsuvík*, *Námafjall*, *Mælifellssandur*, $150\text{--}580\text{ m}$ a.s.l.) exhibit near-sea-level pressure ($945\text{--}995\text{ mbar}$) and moderate UV index ($4.2\text{--}4.5$), placing them just behind high-altitude volcanoes (*Craters of the Moon*, *Mount Etna*) in Stream C's top-1 ranking. Inverse Stream D explicitly reveals each Icelandic site's optimal Martian counterpart (e.g., *Mælifellssandur* $\to$ *Elysium Planitia / Pavonis Mons*).

---

## 5. Planetary Engineering Parameter Matrices

### 5.1 Lunar Module — 24 Quantitative Features

Grounded in Apollo Lunar Module landing-gear specifications (mean slope $\le 6^\circ$, effective slope $\le 12^\circ$, footprint relief $\le 0.61\text{ m}$, pad pressure $\approx 4.6\text{ kPa}$), Lunar Reconnaissance Orbiter (LOLA, LROC NAC, Diviner, Mini-RF) observations, LCROSS impact data ($5.6 \pm 2.9\text{ wt}\%$ water ice at Cabeus), and Artemis IV south-polar reconnaissance studies:

| Physical Domain | CSV Feature Name(s) | Units | Engineering Significance |
| :--- | :--- | :--- | :--- |
| **Topography & Slopes** | `Slope_1m`, `Slope_5m`, `Slope_25m` | $\text{deg } (^\circ)$ | Multi-baseline terrain inclination for lander footpads ($1\text{ m}$), rover stability ($5\text{ m}$), and base layout ($25\text{ m}$). |
| **Attitude** | `Effective_Attitude` | $\text{deg } (^\circ)$ | Combined effective touchdown attitude incorporating local relief and footpad penetration. |
| **Surface Obstacles** | `Boulder_Height_m`, `Boulder_Area_frac` | $\text{m}$, $\text{frac}$ | Exposed rock height (engine nozzle/chassis clearance) and projected boulder area coverage. |
| **Site Operability** | `Safe_Area_m2`, `Hab_Sep_m` | $\text{m}^2$, $\text{m}$ | Contiguous hazard-free touchdown zone and Plume-Surface Interaction (PSI) habitat-to-pad separation. |
| **Mobility** | `Traverse_Slope_deg` | $\text{deg } (^\circ)$ | Mean inclination along rover transit corridors. |
| **Regolith Geotechnics** | `Density_0_15cm_kg_m3`, `Density_0_60cm_kg_m3` | $\text{kg/m}^3$ | Depth-averaged regolith bulk density ($1{,}500\text{ kg/m}^3$ top $15\text{ cm}$; $1{,}660\text{ kg/m}^3$ top $60\text{ cm}$) governing sinkage and excavation. |
| **Soil Mechanics** | `Cohesion_kPa`, `Friction_Angle_deg`, `Bearing_Capacity_kPa` | $\text{kPa}$, $\text{deg}$, $\text{kPa}$ | Mohr-Coulomb shear parameters ($c, \phi$) for wheel traction and ultimate soil bearing capacity for heavy habitats. |
| **ISRU Geochemistry** | `FeO_wt_pct`, `TiO2_wt_pct` | $\text{wt}\%$ | Iron and titanium oxide abundance (ilmenite feedstock for hydrogen/carbothermal $\text{O}_2$ reduction). |
| **Thermal Regime** | `T_min_K`, `T_max_K`, `Thermal_Inertia` | $\text{K}$, $\text{TIU}$ | Diurnal thermal cycling bounds ($100\text{--}400\text{ K}$ equatorial; $<30\text{ K}$ PSRs) and apparent thermal inertia ($\text{J}\cdot\text{m}^{-2}\cdot\text{K}^{-1}\cdot\text{s}^{-1/2}$). |
| **Power & Telecom** | `Sun_Visibility_pct`, `DTE_pct` | $\%$ | Solar illumination fraction (across the $18.6\text{-year}$ lunar nodal cycle) and Direct-to-Earth radio line-of-sight availability. |
| **Volatiles & Cavities** | `Water_Ice_wt_pct`, `Pit_Diameter_m`, `Pit_Depth_m` | $\text{wt}\%$, $\text{m}$ | Polar water-ice concentration and morphometry of volcanic skylights and lava tubes (e.g., Mare Tranquillitatis & Marius Hills). |

### 5.2 Martian Module — 15 Quantitative Features

Grounded in Amazonian Mars physics ($6.36\text{ mbar}$ mean pressure, $95.1\%\ \text{CO}_2$, $3.73\text{ m/s}^2$ gravity, $24\text{ h } 37\text{ min}$ sol, $0.67\text{ mSv/day}$ GCR dose rate), Mars Science Laboratory (REMS), Mars Reconnaissance Orbiter (SHARAD/HiRISE/CRISM), and Martian debris-covered glacier inventories (lobate debris aprons, lineated valley fill, and concentric crater fill in the $30^\circ\text{--}50^\circ$ latitude belts):

* **Atmospheric & Radiative Parameters:** Elevation (`Altitude_m` / `Elevation_m`), surface barometric pressure (`Surface_Pressure_mbar`), ultraviolet flux index (`UV_Flux_Index`), mean and peak gust wind speeds (`Wind_Mean_mps`, `Wind_Gust_mps`).
* **Thermal & Hydrological Parameters:** Minimum and maximum surface temperatures (`T_min_K`, `T_max_K`), relative humidity (`Humidity_pct`), and subsurface water-ice/permafrost mass fraction (`Water_Ice_wt_pct`).
* **Geological, Geochemical & Geotechnical Parameters:** Mafic basaltic index (`Mafic_Basalt_Idx`), evaporite/sulfate/chloride salt content (`Salts_Evaporites_wt_pct`), eolian deflation index (`Eolian_Deflation_Idx`), mean and maximum slopes (`Mean_Slope_deg`, `Max_Slope_deg`), soil cohesion (`Soil_Cohesion_kPa`), internal friction angle (`Friction_Angle_deg`), and thermal inertia (`Thermal_Inertia`).

---

## 6. Multi-Test Decision Engine (Multi-Feature Engineering)

Beyond statistical similarity, mission planners require deterministic pass/fail screening against operational hardware limits. SYBIL implements a rule-based expert system (`evaluateMoonTests` and `evaluateMarsTests` in `SYBIL_app.m`; Part III interactive `while` loop in `Sybil_Machine_Learning.m`) that evaluates Boolean suitability masks $M_{i,k} \in \{0, 1\}$ and normalized engineering scores $S_{i,k} \in [0, 100]$ across $N$ terrestrial sites and $K$ user-selected mission tests.

### 6.1 Lunar Operational Test Scenarios (Tests 1–6)

1. **Test 1 — Spacecraft Landing / Lunar Touchdown (`Landing`):**
   * *Engineering Rationale:* Prevents lander tip-over, ensures engine nozzle clearance above rocks, verifies footpad bearing support, and enforces Plume-Surface Interaction (PSI) ejecta safety separation.
   * *Boolean Mask ($m_1$):*
     $$\text{Area} \ge 200{,}000\text{ m}^2 \;\land\; \theta_{5\text{m}} \le 5.0^\circ \;\land\; \theta_{1\text{m}} \le 4.0^\circ \;\land\; D_{\text{sep}} \ge 2{,}000\text{ m} \;\land\; h_{\text{boulders}} \le 0.35\text{ m} \;\land\; f_{\text{boulders}} \le 0.05 \;\land\; C_b \ge 5.0\text{ kPa}$$
   * *Scoring Equation ($s_1$):*
     $$s_1 = 30 \cdot \frac{\text{Area}}{\max(\text{Area})} + 25 \cdot \left(1 - \frac{\theta_{5\text{m}}}{10}\right) + 25 \cdot \left(1 - \frac{f_{\text{boulders}}}{0.10}\right) + 20 \cdot \frac{C_b}{\max(C_b)}$$

2. **Test 2 — Lunar Rover Mobility (`Rover`):**
   * *Engineering Rationale:* Guarantees wheel-regolith traction under the Mohr-Coulomb shear failure criterion ($\tau_{\max} = c + \sigma \tan\phi$), maintains clearance beneath rocker-bogie suspensions ($0.38\text{ m}$), and prevents wheel sinkage in low-thermal-inertia dust.
   * *Boolean Mask ($m_2$):*
     $$\theta_{\text{trav}} \le 8.0^\circ \;\land\; \phi \ge 38.0^\circ \;\land\; h_{\text{boulders}} \le 0.38\text{ m} \;\land\; c \ge 0.44\text{ kPa} \;\land\; I_{\text{therm}} \ge 200\text{ TIU}$$
   * *Scoring Equation ($s_2$):*
     $$s_2 = 35 \cdot \left(1 - \frac{\theta_{\text{trav}}}{15}\right) + 25 \cdot \left(1 - \frac{h_{\text{boulders}}}{0.50}\right) + 20 \cdot \frac{c}{\max(c)} + 20 \cdot \frac{I_{\text{therm}}}{\max(I_{\text{therm}})}$$

3. **Test 3 — Full Lunar Base (`Full Base`):**
   * *Engineering Rationale:* Screens for an extended surface outpost requiring low $25\text{ m}$-baseline inclination, high bearing capacity for multi-module habitats, sustained solar illumination ($\ge 70\%$), near-continuous Direct-to-Earth (DTE) telemetry ($\ge 90\%$), and safe pad-to-habitat separation.
   * *Boolean Mask ($m_3$):*
     $$\theta_{25\text{m}} \le 4.6^\circ \;\land\; \text{Area} \ge 250\text{ m}^2 \;\land\; C_b \ge 5.5\text{ kPa} \;\land\; \text{Sun}_{\text{vis}} \ge 70\% \;\land\; \text{DTE} \ge 90\% \;\land\; D_{\text{sep}} \ge 2{,}000\text{ m}$$
   * *Scoring Equation ($s_3$):*
     $$s_3 = 30 \cdot \frac{C_b}{\max(C_b)} + 30 \cdot \frac{\text{Sun}_{\text{vis}}}{100} + 25 \cdot \frac{D_{\text{sep}}}{\max(D_{\text{sep}})} + 15 \cdot \left(1 - \frac{\theta_{25\text{m}}}{10}\right)$$

4. **Test 4 — Single Lunar Habitat Module for 2–4 Crew (`2-4p Habitat`):**
   * *Engineering Rationale:* Evaluates the exact same 6 parameters and scoring function ($s_4 = s_3$) as Test 3 while relaxing topographic and footprint thresholds for an initial standalone habitat module ($\ge 25\text{ m}^2$):
     $$\theta_{25\text{m}} < 7.0^\circ \;\land\; \text{Area} \ge 25\text{ m}^2 \;\land\; C_b \ge 5.0\text{ kPa} \;\land\; \text{Sun}_{\text{vis}} \ge 60\% \;\land\; \text{DTE} \ge 90\% \;\land\; D_{\text{sep}} \ge 1{,}400\text{ m}$$

5. **Test 5 — Spacesuit (EVA) Testing & Subsurface Lava Tubes (`EVA/Lava Tubes`):**
   * *Engineering Rationale:* Validates spacesuit mobility, subterranean communications, and radiation/micrometeorite shielding inside deep volcanic skylights and lava tubes subject to strong diurnal thermal gradients ($\Delta T = T_{\max} - T_{\min}$).
   * *Boolean Mask ($m_5$):*
     $$Z_{\text{pit}} \ge 20\text{ m} \;\land\; D_{\text{pit}} \ge 35\text{ m} \;\land\; c \ge 0.44\text{ kPa} \;\land\; \Delta T \ge 35\text{ K}$$
   * *Scoring Equation ($s_5$):*
     $$s_5 = 40 \cdot \min\left(\frac{Z_{\text{pit}}}{50}, 1\right) + 35 \cdot \min\left(\frac{\Delta T}{60}, 1\right) + 25 \cdot \frac{c}{\max(c)}$$

6. **Test 6 — Power Systems & Lunar ISRU (`Power/ISRU`):**
   * *Engineering Rationale:* Couples high solar irradiance ($\ge 70\%$) with extractable feedstock—either ilmenite-rich mare basalt ($\text{FeO} \ge 13.5\text{ wt}\%$ and $\text{TiO}_2 \ge 3.0\text{ wt}\%$) for oxygen reduction or polar water ice ($> 0\text{ wt}\%$)—within excavatable subsurface regolith ($\rho_{0\text{--}60} \le 1{,}710\text{ kg/m}^3$).
   * *Boolean Mask ($m_6$):*
     $$\text{Sun}_{\text{vis}} \ge 70\% \;\land\; \Big((\text{FeO} \ge 13.5 \;\land\; \text{TiO}_2 \ge 3.0) \;\lor\; \text{Water\_Ice} > 0\Big) \;\land\; \rho_{0\text{--}60} \le 1{,}710\text{ kg/m}^3 \;\land\; \theta_{5\text{m}} \le 6.5^\circ$$
   * *Scoring Equation ($s_6$):*
     $$s_6 = 45 \cdot \frac{\text{Sun}_{\text{vis}}}{100} + 35 \cdot \min\left(\max\left(\frac{\text{FeO}}{16}, \frac{\text{Water\_Ice}}{5}\right), 1\right) + 20 \cdot \left(1 - \frac{\theta_{5\text{m}}}{10}\right)$$

### 6.2 Martian Operational Test Scenarios (Tests 1–7)

In the Martian module, semantic category filtering (`contains(cat_m, ...)`) is combined with adaptive `ismember(..., vars_m)` column guards so the engine remains resilient across heterogeneous catalog schemas:

| ID | Test Scenario | Geological & Environmental Constraints |
| :- | :--- | :--- |
| **1** | **Landing / Martian EDL** | Desert basins, salt flats, plains, or plateaus; $\text{Area} \ge 500{,}000\text{ m}^2$; $I_{\text{therm}} \ge 150\text{ TIU}$ (prevents radar altimetry attenuation in thick dust); $\text{Albedo} \le 0.26$; $\theta \le 10^\circ$. |
| **2** | **Martian Rover Mobility** | Volcanic fields, deserts, dunes, or plateaus (excluding coastal glacial zones); $f_{\text{boulders}} \le 0.07$ ($7\%$ ExoMars standard); $I_{\text{therm}} \ge 150\text{ TIU}$; $\theta_{\text{trav}} \le 20^\circ$. |
| **3** | **Full Martian Base** | Compact salt/sulfate, volcanic, desert, or plateau substrates (excluding mobile dunes, glaciers, active fumaroles, and extreme peaks); $\text{Area} \ge 500\text{ m}^2$; $\theta \le 5^\circ$; $I_{\text{therm}} \ge 180\text{ TIU}$; $C_b \ge 5.5\text{ kPa}$. |
| **4** | **Single Habitat (2–4 Crew)** | Same lithological filter as Test 3 (allowing high-altitude plateaus); $\text{Area} \ge 40\text{ m}^2$; $\theta \le 8^\circ$; $I_{\text{therm}} \ge 160\text{ TIU}$; $C_b \ge 4.5\text{ kPa}$. |
| **5** | **Drone / UAV Testing** | High-altitude mountains and plateaus ($\text{Elevation} \ge 2{,}500\text{ m}$) replicating Mars's low-Reynolds-number aerodynamic regime, with wind speed $\le 10\text{ m/s}$. |
| **6** | **Extreme EVA Testing** | Cold, polar, Arctic, or hyper-arid environments with $T_{\min} \le 255\text{ K}$ ($-18^\circ\text{C}$) and relative humidity $\le 15\%$. |
| **7** | **Power Systems & ISRU** | Polar, periglacial, salt-crust, or gypsum environments (excluding un-drillable massive basalt flows) or subsurface water ice $> 0\text{ wt}\%$. |

### 6.3 Multi-Test Combination Logic (`AND` Intersection & Best-Fit Fallback)

When $K$ operational tests are evaluated simultaneously (e.g., entering `1 2 4` in the CLI or using `Ctrl+Click` in the UI listbox), SYBIL constructs the Boolean suitability matrix $M \in \{0, 1\}^{N \times K}$ and the normalized score matrix $S \in [0, 100]^{N \times K}$ across all $N$ terrestrial sites:

```math
\mathrm{mask}_{\mathrm{all}}(i) = \bigwedge_{k=1}^{K} M(i,k), \qquad \bar{S}(i) = \frac{1}{K}\sum_{k=1}^{K} S(i,k)
```

## 7. Interactive UI Control Center & Callback Architecture

### 7.1 `SYBIL_app.m` (App Designer `uifigure` Control Center)

`SYBIL_app.m` encapsulates the entire dual-planet engine inside a dark-themed single-window mission control dashboard (`1500 × 880 px`):
* **Left Control Column (`330 px`):** Celestial body toggle (`MOON (24 Feat.)` / `MARS (15 Feat.)`), target selector dropdown, live telemetry card (coordinates, best Earth match, geological category, PCA Euclidean distance), multi-select test planner listbox, `RUN MULTI-TEST SITE FINDER` and `Clear Test Stars from Map` controls, and a real-time ranking log.
* **Right `2 × 2` Visualization Grid:**
  1. **Top-Left — Cubic 3D PCA Latent Space:** Interactive 3D scatter cube displaying K-Means clusters (`C1`–`C5`), planetary target stars, and best-match rings with calibrated axis margins (`PC1`, `PC2`, `PC3`).
  2. **Top-Right — Comparative Geophysical Radar Chart (`polaraxes`):** Closed spider polygon overlaying the active planetary target against its matched Earth analogue, with dual-layer `DataTipRows` showing both raw SI values and normalized $[0, 100\%]$ levels.
     * *Domain-Adapted Radar Scaling:* Lunar radar axes use joint min-max scaling across Earth + Moon (`[X_E; X_P]`), whereas Martian radar axes use **intra-domain normalization** (scaling Earth against Earth and Mars against Mars) so extreme Martian barometric baselines ($\sim 6.36\text{ mbar}$ vs. $1013\text{ mbar}$ on Earth) do not compress terrestrial atmospheric variations to zero.
  3. **Bottom-Left — Earth Topographic Map (`geoaxes`):** Displays all terrestrial sites, highlighted best AI matches with collision-avoiding text labels, and dynamic green/cyan stars (`TestLayer`) for multi-test search results.
  4. **Bottom-Right — Planetary Surface Map (`uiaxes`):** Equirectangular Moon/Mars surface map with clickable target markers and smart label placement.
* **Collapsible Legends:** Each quadrant features a `Legend` toggle button (`onToggleLegendTag`) and interactive legend item click handlers so dense labels never obscure data points.

### 7.2 Local Callback Subsystem

Four modular local functions enable real-time **Click-to-Radar** exploration across 3D, 2D, and geographic coordinate systems:
* `shortName` / `shortSiteName`: Strips parenthetical descriptors (e.g., `"Olympus Mons (Shield Volcano & Basal Scarp)"` $\to$ `"Olympus Mons"`) to prevent Radar Chart titles from overflowing adjacent panels.
* `updateRadarPlot` / `updateSelectedTargetView`: Closes the polar polygon by appending the first vertex (`[vec(:)', vec(1)]`) across `linspace(0, 360, num_f + 1)` angles and attaches custom SI + normalized tooltips.
* `findClickedIdx` / `findClickedIndex`: Polymorphically inspects `evt.IntersectionPoint` using `isprop(src, 'LatitudeData')` and `isprop(src, 'ZData')` so a single nearest-neighbor Euclidean lookup resolves clicks across 3D scatter cubes, 2D planetary maps, and geographic maps.
* `onClickPlanet` / `onClickEarth` (`clickPlanetPoint` / `clickEarthPoint`): Maps cluster-local marker indices back to global catalog row indices via `idx_map(raw_idx)` and refreshes the Radar Chart and telemetry readout instantaneously.
* **GeographicAxes Handle Preservation (`Sybil_Machine_Learning.m`):** Calling `geoscatter` inside a standard `subplot` replaces the Cartesian `Axes` with a `GeographicAxes` object; `Sybil_Machine_Learning.m` explicitly captures `ax3 = gca` immediately after `geoscatter` so subsequent multi-test star overlays never reference a deleted handle.

---

## 8. Terrestrial Analogue Portfolio & Fidelity Boundaries

SYBIL's catalogs synthesize both heritage NASA/ESA testbeds and newly characterized international sites:
* **Lunar Maria & Basaltic Regolith Mechanics:** *Black Point Lava Flow* (USA — BP-1 simulant source), *Mount Etna* (Italy — Cisternazza crater & lava tubes, ESA ARCHES/Analog-1), *Jingpo Lake Volcanic Field* (China — $>20\text{ km}$ lava tube & skylight continuum matching *Marius Hills*), *Askja Caldera* (Iceland — Apollo 15/17 astronaut training), *Pinacate Volcanic Field* (Mexico), and *Moses Lake Sand Dunes* (USA).
* **Lunar Highlands & Anorthositic Crust:** *Nusfjord / Flakstadøy* (Lofoten Islands, Norway — glacially exposed pure anorthosite used in ESA PANGAEA) and *Miyake-jima* (Japan — megacrystic high-calcic $\sim 95\%$ anorthite crustal material analogue).
* **Lunar Lava Tubes & Subsurface Shelters:** *Craters of the Moon* (Idaho, USA), *Lava River Cave* (Arizona, USA), *Nāhuku* (Hawai'i, USA), *Terceira Island / Natal Cave* (Azores, Portugal — CAMões mission), *Jeju Island* (South Korea), and *Chongqing Karst Caves* (China — structural layout/human-factors proxy).
* **Lunar & Martian Polar Cryosphere / Isolation:** *Haughton Impact Crater* (Devon Island, Canada — TRIDENT drill testing), *McMurdo Dry Valleys* (Antarctica), and *Stenkul Fjord* (Ellesmere Island, Canada — sub-zero hypersaline permafrost springs).
* **Martian Eolian, Evaporitic, Radiative & Volcanic Regimes:** *Qaidam Basin & Lenghu Desert* (China — yardangs matching *Medusae Fossae*), *Ustyurt Plateau* (Kazakhstan/Uzbekistan), *Trans-Altai Gobi Desert* (Mongolia), *Badwater Basin* (Death Valley, USA), *Salar de Arizaro* & *Salar de Antofalla* (Argentina), *Etosha Pan* (Namibia), *Atacama Desert* (Chile), *Llullaillaco Volcano* (Chile/Argentina, $6{,}739\text{ m}$ UV/hypobaric), *Eastern Pamir Plateau* (Tajikistan), *Kilimanjaro Caldera* (Tanzania), *Bardenas Reales* (Spain), and *Harrat Khaybar* (Saudi Arabia).

> **Engineering Fidelity Limits:** No terrestrial site simultaneously replicates lunar/Martian gravity ($1/6\,g$ or $0.38\,g$), ultra-high vacuum or $6.36\text{ mbar}$ $\text{CO}_2$ atmosphere, unshielded galactic cosmic radiation, or $25\text{ K}$ permanently shadowed cold traps. SYBIL treats terrestrial analogues as **task-specific process testbeds** rather than total environmental replicas.

---

## 9. NASA Space Apps Challenge 2026 Compliance & References

### 9.1 Judging Criteria Alignment

| Criterion | How SYBIL Delivers |
| :--- | :--- |
| **Impact** | Replaces subjective, single-site analogue selection with quantitative multi-test mission planning, enabling mission teams to identify multi-role Earth testbeds or optimal complementary site pairs. |
| **Creativity** | Combines schema-agnostic SVD-PCA, zero-toolbox Lloyd's K-Means with automated regex cluster naming, intra-domain Martian radar normalization, bidirectional matching, and Best-Fit Fallback logic. |
| **Validity** | Grounded in peer-reviewed planetary engineering thresholds (Apollo LM landing-gear specs, LRO LOLA/Diviner/Mini-RF, LCROSS, MSL REMS, SHARAD/HiRISE) with full mathematical transparency. |
| **Relevance** | Directly answers the 2026 challenge by integrating open NASA and Space Agency Partner datasets across deserts, polar regions, volcanic fields, and lava tubes into an interactive exploration tool. |
| **Presentation** | Provides both a dark-mode interactive `uifigure` Control Center (`SYBIL_app.m`) with real-time Click-to-Radar inspection and automated CLI diagnostic reports (`Sybil_Machine_Learning.m`). |

### 9.2 Primary NASA & Space Agency Partner Data Sources

1. **NASA Lunar Reconnaissance Orbiter (LRO):** LOLA global topography and multiscale slope/roughness models; LROC NAC boulder size-frequency distributions and pit/skylight stereo morphometry (*Marius Hills*, *Mare Tranquillitatis*); Diviner Lunar Radiometer thermophysical properties (Hayne et al., 2017) and polar cold-trap temperatures; Mini-RF radar observations.
2. **NASA LCROSS & Moon Mineralogy Mapper ($\text{M}^3$ / Chandrayaan-1 ISRO):** Cabeus polar water-ice concentration ($5.6 \pm 2.9\text{ wt}\%$) and polar surface water-ice spectral detections.
3. **NASA Apollo & Artemis Architecture Baselines:** *Lunar Sourcebook* (Carrier et al., 1991), *Apollo LM Landing Gear Subsystem* (NASA TN D-6850), *Moon to Mars Architecture: Lunar Site Selection* (NASA, 2024), *Artemis III Science Definition Team Report* (NASA/SP-20205009602), and *Lunar Regolith Simulant User's Guide* (NASA/TM-20240011783).
4. **NASA Mars Exploration & Planetary Data:** *Mars Fact Sheet* (NSSDCA), Mars Science Laboratory (MSL) REMS environmental monitoring, Mars Reconnaissance Orbiter (MRO) HiRISE/CRISM/SHARAD mid-latitude debris-covered glacier datasets, and Phoenix perchlorate soil chemistry.
5. **International Space Agency Partner & Academic Baselines:** ESA PANGAEA / PANGAEA-X (*Nusfjord*, *Lanzarote*, *Nördlinger Ries*), ESA/DLR *ARCHES* & *LUNA* analogue baselines (*Mount Etna*), CSA lunar geology training (*Kamestastin / Mistastin*), JAXA SELENE/Kaguya terrain data, and recent peer-reviewed morphometric/mineralogical studies (*Jingpo Lake*, *Miyake-jima*, *CAMões Terceira*).
