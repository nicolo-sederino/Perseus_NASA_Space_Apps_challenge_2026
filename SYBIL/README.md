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

* **Atmospheric & Radiative Parameters:** Elevation (`Elevation_m`), surface barometric pressure (`Surface_Pressure_mbar`), ultraviolet flux index (`UV_Flux_Index`), and mean/peak wind speed (`Wind_Speed_ms`).
* **Thermal & Hydrological Parameters:** Mean, minimum, and maximum temperatures (`Mean_Temp_C` / `T_min_K`), diurnal thermal swing, relative humidity (`Humidity_pct`), and water-ice/permafrost mass fraction (`Water_Ice_wt_pct`, exceeding $80\%$ in mid-latitude debris-covered glaciers such as *Deuteronilus Mensae* and *Eastern Hellas*).
* **Geological & Geotechnical Parameters:** Basaltic abundance index, sulfate/evaporite and perchlorate content ($0.4\text{--}0.6\text{ wt}\%$ on Mars), mean/traverse slope (`Mean_Slope_deg` / `Slope_deg` / `Traverse_Slope_deg`), soil cohesion (`Soil_Cohesion_kPa`, $0.2\text{--}10\text{ kPa}$), internal friction angle (`Friction_Angle_deg`, $18^\circ\text{--}39^\circ$), albedo, and thermal inertia (`Thermal_Inertia`).

---

## 6. Multi-Test Mission Planner & Rule-Based Expert Engine

Part III (`evaluateMoonTests` and `evaluateMarsTests`) translates operational test requirements into Boolean suitability masks $m_k \in \{0, 1\}^N$ and continuous engineering scores $s_k \in [0, 100]^N$.

### 6.1 Lunar Engineering Test Scenarios (Tests 1–6)

1. **Test 1 — Spacecraft Landing / Lunar Touchdown:**
   * *Engineering Rationale:* Prevents lander tip-over, ensures engine nozzle clearance above rocks, verifies footpad bearing capacity, and enforces Plume-Surface Interaction (PSI) ejecta separation.
   * *Boolean Mask ($m_1$):*

$$\text{Safe\_Area} \ge 200{,}000\text{ m}^2 \;\land\; \theta_{5\text{m}} \le 5.0^\circ \;\land\; \theta_{1\text{m}} \le 4.0^\circ \;\land\; D_{\text{sep}} \ge 2{,}000\text{ m} \;\land\; h_{\text{boulder}} \le 0.35\text{ m} \;\land\; f_{\text{boulder}} \le 0.05 \;\land\; C_b \ge 5.0\text{ kPa}$$

   * *Score ($s_1$):*

$$s_1 = 30\frac{\text{Safe\_Area}}{\max(\text{Safe\_Area})} + 25\left(1 - \frac{\theta_{5\text{m}}}{10}\right) + 25\left(1 - \frac{f_{\text{boulder}}}{0.10}\right) + 20\frac{C_b}{\max(C_b)}$$

2. **Test 2 — Lunar Rover Mobility:**
   * *Engineering Rationale:* Ensures wheel traction via the Mohr-Coulomb criterion ($\tau_{\max} = c + \sigma \tan\phi$), maintains chassis clearance ($\le 0.38\text{ m}$), and prevents sinkage in low-thermal-inertia dust.
   * *Boolean Mask ($m_2$):*

$$\theta_{\text{trav}} \le 8.0^\circ \;\land\; \phi \ge 38.0^\circ \;\land\; h_{\text{boulder}} \le 0.38\text{ m} \;\land\; c \ge 0.44\text{ kPa} \;\land\; I_{\text{therm}} \ge 200$$

   * *Score ($s_2$):*

$$s_2 = 35\left(1 - \frac{\theta_{\text{trav}}}{15}\right) + 25\left(1 - \frac{h_{\text{boulder}}}{0.50}\right) + 20\frac{c}{\max(c)} + 20\frac{I_{\text{therm}}}{\max(I_{\text{therm}})}$$

3. **Test 3 — Full Lunar Base (Extended Outpost):**
   * *Engineering Rationale:* Permanent surface bases require gentle $25\text{ m}$ baseline terrain, high bearing capacity, prolonged solar power availability, continuous DTE communications, and safe pad separation.
   * *Boolean Mask ($m_3$):*

$$\theta_{25\text{m}} \le 4.6^\circ \;\land\; \text{Safe\_Area} \ge 250\text{ m}^2 \;\land\; C_b \ge 5.5\text{ kPa} \;\land\; \text{Sun}_{\text{vis}} \ge 70\% \;\land\; \text{DTE} \ge 90\% \;\land\; D_{\text{sep}} \ge 2{,}000\text{ m}$$

   * *Score ($s_3$):*

$$s_3 = 30\frac{C_b}{\max(C_b)} + 30\frac{\text{Sun}_{\text{vis}}}{100} + 25\frac{D_{\text{sep}}}{\max(D_{\text{sep}})} + 15\left(1 - \frac{\theta_{25\text{m}}}{10}\right)$$

4. **Test 4 — Single Lunar Habitat Module (2–4 Crew):**
   * *Engineering Rationale:* Evaluates the exact same six parameters and scoring equation ($s_4 = s_3$) as Test 3, relaxing slope, area ($\ge 25\text{ m}^2$), illumination ($\ge 60\%$), and separation ($\ge 1{,}400\text{ m}$) thresholds to match an initial small-footprint crew module.
   * *Boolean Mask ($m_4$):*

$$\theta_{25\text{m}} < 7.0^\circ \;\land\; \text{Safe\_Area} \ge 25\text{ m}^2 \;\land\; C_b \ge 5.0\text{ kPa} \;\land\; \text{Sun}_{\text{vis}} \ge 60\% \;\land\; \text{DTE} \ge 90\% \;\land\; D_{\text{sep}} \ge 1{,}400\text{ m}$$

5. **Test 5 — Spacesuit (EVA) Testing & Lava Tubes:**
   * *Engineering Rationale:* Validates EVA operations and natural radiation/micrometeorite shielding inside deep volcanic skylights and lava tubes experiencing large diurnal thermal swings ($\Delta T = T_{\max} - T_{\min}$).
   * *Boolean Mask ($m_5$):*

$$Z_{\text{pit}} \ge 20\text{ m} \;\land\; D_{\text{pit}} \ge 35\text{ m} \;\land\; c \ge 0.44\text{ kPa} \;\land\; (T_{\max} - T_{\min}) \ge 35\text{ K}$$

   * *Score ($s_5$):*

$$s_5 = 40\min\left(\frac{Z_{\text{pit}}}{50}, 1\right) + 35\min\left(\frac{\Delta T}{60}, 1\right) + 25\frac{c}{\max(c)}$$

6. **Test 6 — Power Systems & Lunar ISRU:**
   * *Engineering Rationale:* Couples photovoltaic solar availability ($\ge 70\%$) with extractable feedstock (ilmenite-rich regolith with $\text{FeO} \ge 13.5\text{ wt}\%$ and $\text{TiO}_2 \ge 3.0\text{ wt}\%$, or polar water ice $> 0\text{ wt}\%$) and excavatable subsurface bulk density ($\rho_{0\text{--}60\text{cm}} \le 1{,}710\text{ kg/m}^3$).
   * *Boolean Mask ($m_6$):*

$$\text{Sun}_{\text{vis}} \ge 70\% \;\land\; \left((\text{FeO} \ge 13.5 \land \text{TiO}_2 \ge 3.0) \lor \text{Ice} > 0\right) \;\land\; \rho_{0\text{--}60\text{cm}} \le 1{,}710\text{ kg/m}^3 \;\land\; \theta_{5\text{m}} \le 6.5^\circ$$

   * *Score ($s_6$):*

$$s_6 = 45\frac{\text{Sun}_{\text{vis}}}{100} + 35\min\left(\max\left(\frac{\text{FeO}}{16}, \frac{\text{Ice}}{5}\right), 1\right) + 20\left(1 - \frac{\theta_{5\text{m}}}{10}\right)$$

### 6.2 Martian Engineering Test Scenarios (Tests 1–7)

Using adaptive `ismember(..., vars_m)` column guards and semantic category filtering on `T_mars_earth`:

1. **Test 1 — Spacecraft Landing / Martian EDL:** Desert basins, salt flats (playas), plains, or plateaus with $\text{Safe\_Area} \ge 500{,}000\text{ m}^2$, $\text{Thermal\_Inertia} \ge 150\text{ TIU}$ (avoiding thick radar-attenuating dust), $\text{Albedo} \le 0.26$, and $\text{Slope} \le 10^\circ$.
2. **Test 2 — Rover Mobility:** Volcanic, desert, dune, or plateau terrains (excluding coastal/glacio-marine settings) with $\text{Boulder\_Area\_frac} \le 0.07$, $\text{Thermal\_Inertia} \ge 150\text{ TIU}$, and $\text{Traverse\_Slope} \le 20^\circ$.
3. **Test 3 — Full Martian Base:** Compact salt, volcanic, desert, or plateau substrates (excluding mobile dunes, glacio-marine settings, active fumaroles, and extreme high-altitude peaks) with $\text{Safe\_Area} \ge 500\text{ m}^2$, $\text{Slope} \le 5^\circ$, $\text{Thermal\_Inertia} \ge 180\text{ TIU}$, and $\text{Bearing\_Capacity} \ge 5.5\text{ kPa}$.
4. **Test 4 — Single Martian Habitat Module (2–4 Crew):** Evaluates the exact same physical parameters as Test 3 while relaxing constraints to $\text{Safe\_Area} \ge 40\text{ m}^2$, $\text{Slope} \le 8^\circ$, $\text{Thermal\_Inertia} \ge 160\text{ TIU}$, and $\text{Bearing\_Capacity} \ge 4.5\text{ kPa}$, permitting high-altitude volcanic plateaus.
5. **Test 5 — Drone / UAV Testing (Ingenuity-Class Aerodynamics):** High-altitude mountain ridges and plateaus ($\text{Elevation} \ge 2{,}500\text{ m}$ a.s.l.) where low air density approximates Mars's low-Reynolds-number rotorcraft regime, with $\text{Wind\_Speed} \le 10\text{ m/s}$.
6. **Test 6 — Extreme EVA Testing:** Cold, polar, Arctic, or hyper-arid deserts with $T_{\min} \le 255\text{ K}$ ($-18^\circ\text{C}$) and $\text{Humidity} \le 15\%$.
7. **Test 7 — Power Systems & Martian ISRU:** Polar, periglacial, salt-crust, or gypsum environments (excluding massive un-drillable basalt flows) with subsurface water ice ($\text{Water\_Ice} > 0\text{ wt}\%$) or hydrated evaporites.

### 6.3 Multi-Test Combination Logic (`AND` Intersection & Best-Fit Fallback)

When $K$ tests are selected simultaneously, SYBIL constructs $M \in \{0, 1\}^{N \times K}$ and $S \in [0, 100]^{N \times K}$:

$$\text{mask\_all}(i) = \bigwedge_{k=1}^{K} M_{i,k}, \qquad \overline{S}_i = \frac{1}{K}\sum_{k=1}^{K} S_{i,k}$$

* **Exact Conjunction (`any(mask_all) == true`):** Extracts all terrestrial sites passing $100\%$ of selected tests and ranks them by descending combined mean score $\overline{S}_i$.
* **Best-Fit Fallback (`any(mask_all) == false`):** If the user combines physically mutually exclusive tests (e.g., *Test 1: Flat Open-Plain Landing* + *Test 5: Deep Lava Tube Skylight*), no single site can satisfy both simultaneously. Instead of returning an empty result, SYBIL computes $P_i = \sum_{k=1}^{K} M_{i,k}$, identifies $P_{\max} = \max_i(P_i)$ (e.g., $2/3$ tests passed), and highlights the Top 5 multi-role sites ranked by $\overline{S}_i$.

---

## 7. Interactive Visualization & UI Architecture

Both `SYBIL_app.m` and `Sybil_Machine_Learning.m` implement a synchronized 4-quadrant analytical dashboard:

1. **Cubic 3D PCA Latent Space (`ax1` / `App.Ax3D`):** Equal-aspect (`pbaspect([1 1 1])`, `axis vis3d`) 3D scatter cube displaying terrestrial sites colored by K-Means cluster ($k=5$), planetary targets (pentagrams), and optimal analogue rings.
2. **Comparative Geophysical Signature — Polar Radar Chart (`polaraxes`):** Topologically closed $360^\circ$ polygon (`linspace(0, 360, num_f + 1)`) comparing the selected planetary target against its matched Earth analogue.
   * *Dual-Layer DataTips:* Hovering over any vertex displays both the **Actual Physical Value in SI units** and the **Normalized Level ($0\text{--}100\%$)**.
   * *Domain-Aware Scaling:* On the Moon, joint min-max scaling (`[X_E; X_P]`) highlights direct physical overlap; on Mars, intra-domain scaling (`X_E` and `X_P` normalized within their respective planetary bounds) prevents extreme planetary pressure/temperature offsets ($6\text{ mbar}$ vs. $1{,}013\text{ mbar}$) from collapsing terrestrial profiles into the origin.
3. **Earth Topographic Map (`geoaxes`):** Interactive geographic basemap (`colorterrain`) plotting all terrestrial catalog sites, red best-match markers, green/cyan multi-test winner stars, and collision-avoiding text labels (`placed_e`).
   > **Technical Fix for `GeographicAxes` Lifecycle:** Because calling `geoscatter` on a standard Cartesian `subplot` destroys the underlying `Axes` handle and replaces it with a `GeographicAxes` instance, SYBIL explicitly captures `ax3 = gca;` immediately after `geoscatter` so subsequent multi-test layers plot cleanly without throwing `Attempt to plot into deleted Axes`.
4. **2D Planetary Surface Target Map:** Equirectangular projection ($[-180^\circ, 180^\circ] \times [-90^\circ, 90^\circ]$) overlaid on `moon_map.jpg` or `mars_map.jpg` with anti-overlap label placement (`placed_boxes`).
5. **Polymorphic Click-to-Radar Callbacks (`findClickedIdx`):** A unified click handler inspects `isprop(src, 'LatitudeData')` and `isprop(src, 'ZData')` to resolve mouse clicks across 3D scatter cubes, 2D planetary maps, and geographic maps, dynamically refreshing the Radar Chart and telemetry panel in real time.

---

## 8. Terrestrial Analogue Portfolio & Physical Fidelity Boundaries

SYBIL's catalogs synthesize established NASA/ESA analogue testbeds alongside newly characterized international sites:

* **Lunar Maria & Basaltic Plains:** *Black Point Lava Flow* (Arizona, BP-1 simulant), *Mount Etna* (Italy, ESA/DLR ARCHES & Analog-1), *Jingpo Lake Volcanic Field* (NE China, Marius Hills rille/skylight continuum), *Askja* (Iceland), *Pinacate* (Mexico), *Potrillo Volcanic Field* (New Mexico).
* **Lunar Highlands & Anorthositic Crust:** *Nusfjord / Flakstadøy* (Lofoten Islands, Norway, ESA PANGAEA), *Miyake-jima* (Japan, high-calcic $\sim 95\%$ anorthite megacrysts), *Kamestastin / Mistastin* (Labrador, Canada, impact-shocked anorthosite), *Nördlinger Ries* (Germany, impact breccia/suevite).
* **Lunar & Martian Lava Tubes / Subsurface Shelters:** *Craters of the Moon* (Idaho), *Lava River Cave* (Arizona), *Nāhuku* (Hawai'i), *Natal Cave, Terceira* (Azores, CAMões mission), *Lanzarote / Corona Tube* (Canary Islands, HULC camera trials), *Jingpo Lake* ($>20\text{ km}$ tube network).
* **Martian Eolian, Evaporitic, Radiative & Astrobiological Analogues:** *Qaidam Basin & Lenghu* (China, Medusae Fossae yardangs), *Ustyurt Plateau* (Kazakhstan/Uzbekistan, wind deflation pavement), *Trans-Altai Gobi* (Mongolia, thermoclasm), *Badwater Basin* (Death Valley, halite/sulfate polygons), *Salar de Arizaro & Salar de Antofalla* (Argentina), *Etosha Pan* (Namibia), *Llullaillaco Volcano* ($>6{,}700\text{ m}$, extreme UV & low pressure), *Kilimanjaro Caldera* ($5{,}895\text{ m}$), *Eastern Pamir Plateau* (Tajikistan), *Haughton Crater* & *Stenkul Fjord* (Canadian Arctic), *McMurdo Dry Valleys* (Antarctica), *Atacama Desert* (Chile), *Harrat Khaybar* (Saudi Arabia).

**Physical Fidelity Limitations:** No terrestrial location simultaneously reproduces lunar/Martian gravity ($1.62\text{ m/s}^2$ and $3.73\text{ m/s}^2$), ultra-high lunar vacuum, unshielded galactic cosmic rays ($1.369\text{ mSv/day}$ on the Moon; $0.67\text{ mSv/day}$ on Mars), or cryogenic PSR temperatures ($<30\text{ K}$). SYBIL explicitly enforces domain-specific test constraints so analogue sites are validated strictly for the physical processes they faithfully represent.

---

## 9. Judging Alignment, Open Data Attribution & Transparency

* **Impact & Relevance:** Directly solves the challenge by turning qualitative site lists into a quantitative, bidirectional engineering decision tool grounded in NASA and partner agency measurements.
* **Creativity:** Introduces bidirectional matching (Streams A–D), dual Euclidean/Cosine Martian metrics, automated semantic cluster naming, and a Multi-Test Decision Engine with Best-Fit Fallback.
* **Validity & Presentation:** Built on peer-reviewed planetary geotechnics, thermophysics, and mission-architecture standards, delivered through a zero-dependency interactive GUI (`SYBIL_app.m`) and complete mathematical documentation.
* **NASA & Space Agency Partner Data Attribution:** Utilizes open data and mission results from **NASA** (LRO LOLA, LROC NAC, Diviner, Mini-RF, LCROSS, LADEE, Apollo Lunar Sourcebook, Mars Fact Sheet, MSL REMS, MRO HiRISE/SHARAD/CRISM), **ESA** (PANGAEA, ARCHES, Mars Express, LUNA), **CSA** (Kamestastin lunar training), **JAXA** (Kaguya/SELENE), **DLR** (Deuteronilus Mensae & ARCHES), **ISRO** (Chandrayaan-1 $\text{M}^3$), and **ASI**.
* **Super Intelligence (SI) Disclosure:** Generative AI/LLM tools were used solely to assist with technical documentation formatting and UI code refactoring; all scientific methodology, feature engineering, scoring equations, and planetary datasets were designed and curated by the Project Perseus team.

---

## 10. Primary References

1. **Rogers, W. F. (1972).** *Apollo Experience Report—Lunar Module Landing Gear Subsystem.* NASA TN D-6850 (NTRS 19720018253).
2. **Carrier, W. D. III, Olhoeft, G. R., & Mendell, W. (1991).** *Physical Properties of the Lunar Surface.* Chapter 9 in *Lunar Sourcebook*, Cambridge University Press / LPI.
3. **NASA (2024).** *Lunar Site Selection.* Moon to Mars Architecture White Paper; *NASA Provides Update on Artemis III Moon Landing Regions* (2024); *Artemis IV* (2026).
4. **Hayne, P. O., et al. (2017).** *Global Regolith Thermophysical Properties of the Moon from the Diviner Lunar Radiometer Experiment.* *J. Geophys. Res. Planets*, 122, 2371–2400.
5. **Mazarico, E., et al. (2011) & Gläser, P., et al. (2018).** *Illumination Conditions of the Lunar Polar Regions Using LOLA Topography.* *Icarus*, 211, 1066–1081; *Planet. Space Sci.*, 162, 170–178.
6. **Li, S., et al. (2018) & Colaprete, A., et al. (2010).** *Direct Evidence of Surface Exposed Water Ice in the Lunar Polar Regions.* *PNAS*, 115(36); *Detection of Water in the LCROSS Ejecta Plume.* *Science*, 330, 463–468.
7. **Carrer, L., et al. (2024).** *Radar Evidence of an Accessible Cave Conduit on the Moon Below the Mare Tranquillitatis Pit.* *Nature Astronomy*, 8, 1119–1126.
8. **Haberle, R. M., et al. (2017) & Martínez, G. M., et al. (2017).** *The Atmosphere and Climate of Mars.* Cambridge University Press; *Space Science Reviews*, 212, 503–538.
9. **Hecht, M. H., et al. (2009) & Navarro-González, R., et al. (2003).** *Detection of Perchlorate and the Soluble Chemistry of Martian Soil at the Phoenix Landing Site.* *Science*, 325, 64–67; *Science*, 302, 1018–1021.
10. **Levy, J. S., et al. (2014), Karlsson, N. B., et al. (2015), & Steinberg, et al. (2025).** Martian debris-covered glacier inventory, radar ice fraction ($>80\%$), and global water equivalent volume ($1.55 \times 10^5\text{ km}^3$).
11. **Wang, X., et al. (2011), Cabrol, N. A., et al. (2014), & Camp, V. E., et al. (1991).** Terrestrial Mars analogues: Qaidam Basin yardangs, high-altitude Andean volcanoes, and Harrat Khaybar volcanic plains.
