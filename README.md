# Perseus: Planetary Analogue Discovery System (Moon & Mars)
**NASA Space Apps Challenge 2026 — Challenge: *Identify Earth Locations that Analog the Permanent Moon Base Locations and Mars***

🔗 **Live Interactive Web Application:** `[INSERT_PUBLIC_WEB_URL_HERE]`  
🎥 **Project Demo (30s Video / 7-Slide Deck):** `[INSERT_PUBLIC_DEMO_LINK_HERE]`

---

## 1. Overview & The Engineering Problem
Selecting future landing sites and permanent surface bases on the Moon and Mars requires matching complex planetary mission constraints—topography, geology, thermophysics, radiation, and resource availability—with terrestrial testbeds. However, **no single site on Earth is a universal planetary analogue**:
* An environment ideal for rover traction on loose basaltic lapilli (e.g., Mount Etna) cannot replicate cryogenic temperatures or ultra-high vacuum.
* A hyperarid desert (e.g., Atacama Desert) replicates UV stress and low biomass, yet lacks permafrost or basaltic lava tubes.

**Perseus** bridges this gap by transforming static, poorly characterized location lists into a **Task-Specific, Bidirectional Planetary Analogue Discovery System**. Instead of searching for a generic "best site," mission planners, engineers, and researchers filter and match environments based on concrete operational objectives (*Rover Mobility & Abrasion, Subsurface Lava-Tube Habitation, ISRU Water-Ice Extraction, High-Altitude Radiative Stress, and Astrobiology*).

---

## 2. How It Works (Methodology & Architecture)
Our platform pairs an unsupervised machine learning analytical engine (`core_matlab/ProvaHackaton7.m`) with an interactive web visualization dashboard (`app/`):

1. **Multi-Domain Quantitative Catalogs (`data/`):** Four curated datasets (`Lunar_Targets_Catalog.csv`, `Earth_Lunar_Analogs_Catalog.csv`, `Mars_Targets_Catalog.csv`, `Earth_Mars_Analogs_Catalog.csv`) encoding SI geomorphological, thermophysical, geotechnical, and geochemical parameters.
2. **Domain-Adapted Standardization:** Features are normalized via Z-score. For Mars, domain-adapted scaling decouples extreme barometric baselines (~6.36 mbar vs. 1013 mbar on Earth) so atmospheric pressure differences do not mask critical surface geology and geochemistry.
3. **Dimensionality Reduction & Clustering (SVD / PCA & K-Means):** Singular Value Decomposition projects Earth, Moon, and Mars sites into a shared 3D principal component cube, while K-Means clustering groups terrestrial environments by operational regime.
4. **Bidirectional Dual-Metric Matching:**
   * **Planet $\rightarrow$ Earth:** Select a lunar or Martian target (e.g., *Marius Hills, Shackleton Rim, Mare Tranquillitatis, Deuteronilus Mensae, Gale Crater*) to rank the closest terrestrial testbeds via Euclidean distance in PCA space and Cosine Similarity.
   * **Earth $\rightarrow$ Planet:** Click any terrestrial location—combining historical testbeds (*Black Point Lava Flow, Haughton Crater, Badwater Basin, Nördlinger Ries*) with newly characterized international analogs (*Mount Etna, Nusfjord, Terceira, Miyake-jima, Lanzarote, Salar de Arizaro, Bardenas Reales*)—to discover which planetary environment it best simulates.
5. **Engineering Fidelity & Limitation Warnings:** Every match explicitly flags unsimulated environmental gaps (e.g., $1/6g$ or $0.38g$ gravity, ultra-high vacuum, cosmic radiation), guiding mission teams on when to pair field campaigns with cryogenic vacuum chambers or regolith simulants.

---

## 3. NASA & Space Agency Partner Data Integration
To ensure scientific validity and relevance, our catalogs integrate open datasets and peer-reviewed retrievals from NASA and 2026 Space Agency Partners:

| Domain | Missions, Instruments & Open Datasets | Extracted Parameters |
| :--- | :--- | :--- |
| **Moon** | **NASA LRO** (*Diviner*, *LOLA*, *LROC NAC*, *Mini-RF*), **LCROSS**, **ISRO Chandrayaan-1** (*M³*), **JAXA Kaguya** (*TC/MI*) | Multiscale slopes (1 m / 5 m / 25 m), Hurst roughness, boulder size-frequency, PSR temperatures (<110 K), thermal inertia, FeO/TiO₂ wt%, water-ice absorption bands, skylight & conduit geometry |
| **Mars** | **NASA MRO** (*HiRISE*, *SHARAD*), **MSL Curiosity** (*REMS/RAD*), **Phoenix**, **Mars Odyssey**, **NASA Mars Fact Sheet** | Diurnal thermal swings, surface pressure cycles (~6.36 mbar), buried glacier ice fraction (>80%) & depth, perchlorate/sulfate/borate chemistry, regolith internal friction angle |
| **Earth & Partners** | **NASA** (*Landsat TIRS*, *ECOSTRESS*), **Copernicus DEM**, **ESA** (*PANGAEA*, *LUNA*, *Etna*, *Lanzarote*, *Nusfjord*), **CSA** (*Haughton*, *Kamestastin*), **JAXA** (*Miyake-jima*), **CONAE** (*Andean Salars*) | Lava-tube morphometry, anorthositic/basaltic mineralogy, hyperarid evaporite crusts, wind yardang deflation, high-altitude UV radiative thinning |

---

## 4. Tech Stack & 2026 Global Collaborator Resources
Built leveraging open-source tools and Global Offers categories:
* **Data Visualization Tools:** Interactive 3D PCA Latent Space Cube, Comparative Geophysical Radar Charts (with real SI unit tooltips), and Dual Planetary/Terrestrial 2D Maps.
* **Cloud Computing & Website Building Resources:** Public, zero-login web deployment connected directly to GitHub for instant browser access.
* **APIs & Open Data Repositories:** Direct integration of open planetary basemaps, topographic elevation models, and NASA/ESA open science archives.
* **Collaboration Tools:** GitHub version control and modular CSV/Markdown documentation for multidisciplinary engineering, geology, and software workflows.

---

## 5. Repository Structure
```text
├── data/                  # 4 quantitative CSV catalogs + DATA_DICTIONARY.md (SI units & sources)
├── core_matlab/           # Original MATLAB analytical engine (ProvaHackaton7.m) & planetary basemaps
├── app/                   # Interactive web application (Python / Streamlit / Plotly)
└── docs/                  # Lunar & Martian engineering assessment frameworks and literature syntheses
└── docs/                  # Engineering assessment frameworks & Mars/Moon literature syntheses[cite: 1, 3, 4]
