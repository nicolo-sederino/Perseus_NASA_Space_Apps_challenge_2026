#!/usr/bin/env python3
"""
Team Perseus - Interactive Discovery Map Generator for NASA Space Apps 2026
Renders IRIS (Flow B) planetary analog discoveries against SYBIL (Flow A) exclusion zones,
highlighting the Top 5 sites per (Planet + Landing Site) configuration and enabling direct
client-side export in CSV, GeoJSON, and JSON formats.
"""

import json
from pathlib import Path

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Team Perseus - NASA Space Apps 2026 | IRIS Global Discovery Map</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; background: #0b0f19; color: #f3f4f6; }
        #header { background: #111827; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #3b82f6; flex-wrap: wrap; gap: 10px; }
        h1 { margin: 0; font-size: 15.5px; color: #60a5fa; letter-spacing: 0.4px; }
        .sub { font-size: 11.5px; color: #9ca3af; margin-top: 2px; }
        .controls { display: flex; gap: 7px; align-items: center; flex-wrap: wrap; }
        button, select { background: #1f2937; color: #f9fafb; border: 1px solid #3b82f6; padding: 6px 11px; cursor: pointer; border-radius: 6px; font-weight: 600; font-size: 12px; transition: all 0.18s; }
        button:hover, select:hover { background: #374151; }
        button.active { background: #2563eb; border-color: #93c5fd; }
        button.excl-btn { border-color: #ef4444; }
        button.excl-btn.active { background: #b91c1c; border-color: #fca5a5; }
        .dl-group { display: flex; gap: 4px; align-items: center; border-left: 1px solid #374151; padding-left: 8px; margin-left: 2px; }
        .dl-btn { border-color: #10b981; color: #d1fae5; background: #064e3b; font-size: 11.5px; padding: 6px 9px; }
        .dl-btn:hover { background: #047857; }
        #map { height: calc(100vh - 62px); width: 100%; background: #0b0f19; }

        /* Top 5 Floating Leaderboard */
        #top5-panel {
            position: absolute; bottom: 24px; right: 14px; z-index: 1000;
            background: rgba(17, 24, 39, 0.94); border: 1px solid #fbbf24;
            border-radius: 8px; padding: 10px 12px; width: 285px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6); backdrop-filter: blur(4px);
        }
        #top5-panel h4 { margin: 0 0 7px 0; font-size: 12.5px; color: #fbbf24; display: flex; justify-content: space-between; align-items: center; }
        .top5-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: 5px 7px; margin-bottom: 4px; border-radius: 5px;
            background: rgba(31, 41, 55, 0.85); cursor: pointer; font-size: 11.5px;
            border-left: 3px solid #fbbf24; transition: background 0.15s;
        }
        .top5-item:hover { background: rgba(55, 65, 81, 0.95); }
        .top5-rank { font-weight: 800; color: #fbbf24; margin-right: 6px; }
        .top5-name { color: #f3f4f6; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 155px; }
        .top5-score { color: #34d399; font-weight: 700; font-size: 11.5px; }

        /* Map Legend */
        #legend-panel {
            position: absolute; bottom: 24px; left: 14px; z-index: 1000;
            background: rgba(17, 24, 39, 0.92); border: 1px solid #374151;
            border-radius: 8px; padding: 9px 12px; font-size: 11.5px; line-height: 1.6;
        }
        .leg-row { display: flex; align-items: center; gap: 7px; }
        .dot { width: 11px; height: 11px; border-radius: 50%; display: inline-block; }

        /* Popups */
        .popup-box { font-size: 12.5px; color: #111827; line-height: 1.45; min-width: 255px; }
        .popup-box h3 { margin: 0 0 6px 0; color: #1d4ed8; font-size: 14.5px; }
        .badge-top5 { background: #fef3c7; color: #92400e; border: 1px solid #f59e0b; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 800; }
        .badge { background: #dbeafe; color: #1e40af; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }
        .badge-ok { background: #d1fae5; color: #065f46; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }
        .badge-no { background: #fee2e2; color: #991b1b; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }
    </style>
</head>
<body>
    <div id="header">
        <div>
            <h1>NASA Space Apps Challenge 2026 &mdash; Team Perseus | IRIS Global Analog Discovery</h1>
            <div class="sub" id="target-info">Loading IRIS planetary target metadata...</div>
        </div>
        <div class="controls">
            <button id="btn-lunar" class="active" onclick="switchPlanet('lunar')">IRIS: Moon</button>
            <button id="btn-mars" onclick="switchPlanet('mars')">IRIS: Mars</button>
            <select id="target-select" onchange="onTargetChange()"></select>
            <button id="btn-excl" class="excl-btn active" onclick="toggleExclusion()">SYBIL 50km Exclusion</button>
            <div class="dl-group" title="Download scored IRIS sites for the selected Planet + Landing Site">
                <button class="dl-btn" onclick="downloadData('csv')">&#8681; CSV</button>
                <button class="dl-btn" onclick="downloadData('geojson')">&#8681; GeoJSON</button>
                <button class="dl-btn" onclick="downloadData('json')">&#8681; JSON</button>
            </div>
        </div>
    </div>

    <div id="map"></div>

    <div id="legend-panel">
        <div style="font-weight:700; color:#93c5fd; margin-bottom:3px;">Map Legend (IRIS vs SYBIL)</div>
        <div class="leg-row"><span class="dot" style="background:#fbbf24; border:2px solid #fff;"></span> <b>Top 5 IRIS Discoveries (#1&ndash;#5)</b></div>
        <div class="leg-row"><span class="dot" style="background:#10b981;"></span> High Similarity (&gt; 60%)</div>
        <div class="leg-row"><span class="dot" style="background:#38bdf8;"></span> Moderate Similarity (45%&ndash;60%)</div>
        <div class="leg-row"><span class="dot" style="background:#f97316;"></span> Lower Similarity (&lt; 45%)</div>
        <div class="leg-row"><span class="dot" style="background:rgba(239,68,68,0.25); border:1.5px dashed #ef4444;"></span> SYBIL Excluded Site (50 km)</div>
    </div>

    <div id="top5-panel">
        <h4><span>&#9733; Top 5 IRIS Sites</span><span style="font-size:10.5px; color:#9ca3af; font-weight:normal;">Click to zoom</span></h4>
        <div id="top5-list"></div>
    </div>

    <script>
        const lunarData = __LUNAR_JSON__;
        const marsData = __MARS_JSON__;
        const exclData = __EXCL_JSON__;
        let currentPlanet = 'lunar';
        let markerByRank = {};

        const map = L.map('map', { preferCanvas: true }).setView([20, 10], 2);

        const satelliteLayer = L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            { attribution: 'Tiles &copy; Esri &mdash; Esri, Maxar, NASA/USGS', maxZoom: 18 }
        );
        const darkCanvasLayer = L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
            { attribution: 'Tiles &copy; Esri', maxZoom: 16 }
        );
        satelliteLayer.addTo(map);
        L.control.layers(
            { "Satellite Imagery (True Color)": satelliteLayer, "Dark Tactical Base": darkCanvasLayer },
            null,
            { position: 'topright' }
        ).addTo(map);

        let discoveryLayer = L.layerGroup().addTo(map);
        let exclusionLayer = L.layerGroup().addTo(map);
        let showExclusion = true;

        function renderExclusionZones() {
            exclusionLayer.clearLayers();
            const radiusMeters = (exclData.buffer_km || 50) * 1000;
            (exclData.flow_a_sites || []).forEach(s => {
                let circle = L.circle([s.latitude, s.longitude], {
                    radius: radiusMeters, color: '#ef4444', weight: 1.5, dashArray: '5, 5', fillColor: '#ef4444', fillOpacity: 0.12
                });
                circle.bindPopup(`<div class="popup-box">
                    <h3 style="color:#b91c1c;">Excluded SYBIL Catalog Site</h3>
                    <b>Site:</b> ${s.site_name}<br>
                    <b>Exclusion Radius:</b> ${exclData.buffer_km || 50} km<br>
                    <i>All cells inside this SYBIL buffer are excluded from IRIS discovery.</i>
                </div>`);
                exclusionLayer.addLayer(circle);
            });
        }

        function toggleExclusion() {
            showExclusion = !showExclusion;
            document.getElementById('btn-excl').classList.toggle('active', showExclusion);
            if (showExclusion) exclusionLayer.addTo(map); else map.removeLayer(exclusionLayer);
        }

        function getColor(p) {
            if (p.global_rank <= 5) return '#fbbf24';
            const s = p.similarity_score;
            return s > 60 ? '#10b981' : s > 45 ? '#38bdf8' : s > 30 ? '#f97316' : '#ef4444';
        }

        function populateTargetSelect(dataset) {
            const sel = document.getElementById('target-select');
            sel.innerHTML = '';
            const meta = dataset.metadata || {};
            const targets = meta.available_targets || [meta.reference_target || 'Default'];
            targets.forEach(t => {
                const opt = document.createElement('option');
                opt.value = t;
                opt.innerText = 'Target: ' + t.replaceAll('_', ' ');
                if (t === meta.reference_target) opt.selected = true;
                sel.appendChild(opt);
            });
        }

        function getActiveSelection() {
            const dataset = currentPlanet === 'lunar' ? lunarData : marsData;
            const meta = dataset.metadata || {};
            const chosenTarget = document.getElementById('target-select').value || meta.reference_target || 'Target';
            const tMap = dataset.targets_map || {};
            const points = (tMap[chosenTarget] && tMap[chosenTarget].length) ? tMap[chosenTarget] : (dataset.grid_points || []);
            return { dataset, meta, chosenTarget, points };
        }

        function zoomToRank(rank) {
            const m = markerByRank[rank];
            if (!m) return;
            const ll = m.getLatLng();
            map.flyTo(ll, 6, { duration: 1.0 });
            setTimeout(() => m.openPopup(), 450);
        }

        function renderTop5Panel(points) {
            const listEl = document.getElementById('top5-list');
            listEl.innerHTML = '';
            points.slice(0, 5).forEach(p => {
                const div = document.createElement('div');
                div.className = 'top5-item';
                div.onclick = () => zoomToRank(p.global_rank);
                div.innerHTML = `
                    <div style="display:flex; align-items:center; overflow:hidden;">
                        <span class="top5-rank">#${p.global_rank}</span>
                        <span class="top5-name" title="${p.site_name} (${p.region})">${p.site_name}</span>
                    </div>
                    <span class="top5-score">${p.similarity_score}%</span>
                `;
                listEl.appendChild(div);
            });
        }

        function renderCurrentSelection() {
            discoveryLayer.clearLayers();
            markerByRank = {};
            const { meta, chosenTarget, points } = getActiveSelection();

            document.getElementById('target-info').innerText =
                `IRIS Target: ${chosenTarget.replaceAll('_', ' ')} | Evaluated Abiotic Cells: ${meta.total_evaluated_sites || points.length} | Excluded SYBIL Sites: ${meta.flow_a_sites_excluded_from_search || 44}`;

            // Render secondary points first, then Top 5 on top so they are never occluded
            const ordered = [...points].sort((a, b) => b.global_rank - a.global_rank);

            ordered.forEach(p => {
                const isTop5 = p.global_rank <= 5;
                const color = getColor(p);
                const marker = L.circleMarker([p.lat, p.lon], {
                    radius: isTop5 ? 10.5 : (p.global_rank <= 25 ? 6.5 : 4.8),
                    color: isTop5 ? '#ffffff' : color,
                    weight: isTop5 ? 2.5 : 1,
                    fillColor: color,
                    fillOpacity: isTop5 ? 0.96 : 0.78
                });

                const eng = p.engineering || {};
                const tele = p.telemetry || {};
                const topBadge = isTop5 ? `<span class="badge-top5">&#9733; TOP ${p.global_rank} IRIS ANALOG</span> ` : '';
                const landerBadge = eng.lander_suitable ? '<span class="badge-ok">Lander OK</span>' : '<span class="badge-no">Slope &gt; 5&deg;</span>';
                const roverBadge = eng.rover_suitable ? '<span class="badge-ok">Rover OK</span>' : '<span class="badge-no">Steep</span>';

                const popupContent = `<div class="popup-box">
                    <h3>#${p.global_rank} &mdash; ${p.site_name}</h3>
                    ${topBadge}<span class="badge">New IRIS Site (${eng.nearest_flow_a_site_km || '?'} km from SYBIL)</span><br>
                    <b>Region / Biome:</b> ${p.region}<br>
                    <b>Suitability:</b> ${landerBadge} ${roverBadge}<br>
                    <b>Similarity Score:</b> <span style="color:#2563eb; font-weight:bold;">${p.similarity_score}%</span>
                    (d/d<sub>best,SYBIL</sub> = <b>${p.distance_vs_flow_a_best}</b>)<br>
                    <b>Cluster Archetype:</b> ${p.cluster_label}<br>
                    <hr style="margin:6px 0; border:0; border-top:1px solid #ddd;">
                    <b>Radar Sub-Scores (0-100):</b><br>
                    &bull; Morphometry: <b>${p.radar_subscores.Morphometry}</b> | Thermophysics: <b>${p.radar_subscores.Thermophysics}</b><br>
                    &bull; Geochemistry: <b>${p.radar_subscores.Geochemistry}</b> | Environment: <b>${p.radar_subscores.Environment}</b><br>
                    <hr style="margin:6px 0; border:0; border-top:1px solid #ddd;">
                    <b>Key Telemetry:</b><br>
                    &bull; Altitude: ${tele.Altitude_m ?? 'N/A'} m | Slope: ${tele.Mean_Slope_deg ?? 'N/A'}&deg;<br>
                    &bull; Diurnal &Delta;T: ${tele.Diurnal_Delta_T_K ?? 'N/A'} K | T<sub>min</sub>: ${tele.T_min_K ?? 'N/A'} K
                </div>`;
                marker.bindPopup(popupContent);
                discoveryLayer.addLayer(marker);
                if (isTop5) marker.bringToFront();
                markerByRank[p.global_rank] = marker;
            });

            renderTop5Panel(points);
        }

        function triggerBlobDownload(content, filename, mimeType) {
            const blob = new Blob([content], { type: mimeType });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        function downloadData(format) {
            const { meta, chosenTarget, points } = getActiveSelection();
            const safeTarget = chosenTarget.replace(/[^a-zA-Z0-9_-]+/g, '_');
            const baseName = `IRIS_${currentPlanet}_${safeTarget}`;

            if (format === 'json') {
                const payload = {
                    system: "IRIS (Global Discovery Engine) calibrated against SYBIL",
                    planetary_body: currentPlanet,
                    reference_target: chosenTarget,
                    metadata: meta,
                    discovered_sites: points
                };
                triggerBlobDownload(JSON.stringify(payload, null, 2), `${baseName}.json`, 'application/json');
                return;
            }

            if (format === 'geojson') {
                const fc = {
                    type: "FeatureCollection",
                    metadata: {
                        system: "IRIS Global Analog Discovery",
                        calibration_baseline: "SYBIL",
                        planetary_body: currentPlanet,
                        reference_target: chosenTarget
                    },
                    features: points.map(p => ({
                        type: "Feature",
                        geometry: { type: "Point", coordinates: [p.lon, p.lat] },
                        properties: {
                            iris_rank: p.global_rank,
                            site_name: p.site_name,
                            region: p.region,
                            similarity_score_pct: p.similarity_score,
                            distance_vs_sybil_best: p.distance_vs_flow_a_best,
                            cluster_label: p.cluster_label,
                            ...p.radar_subscores,
                            ...p.engineering,
                            ...p.telemetry
                        }
                    }))
                };
                triggerBlobDownload(JSON.stringify(fc, null, 2), `${baseName}.geojson`, 'application/geo+json');
                return;
            }

            if (format === 'csv') {
                if (!points.length) return;
                const teleKeys = Object.keys(points[0].telemetry || {});
                const headers = [
                    "iris_rank", "site_name", "region", "latitude", "longitude",
                    "similarity_score_pct", "latent_distance", "cosine_similarity",
                    "distance_vs_sybil_best", "cluster_label",
                    "morphometry_score", "thermophysics_score", "geochemistry_score", "environment_score",
                    "lander_suitable", "rover_suitable", "nearest_sybil_site_km",
                    ...teleKeys
                ];
                const esc = v => {
                    if (v === null || v === undefined) return '';
                    const s = String(v);
                    return (s.includes(',') || s.includes('"') || s.includes('\n')) ? `"${s.replaceAll('"', '""')}"` : s;
                };
                const rows = [headers.join(',')];
                points.forEach(p => {
                    const r = p.radar_subscores || {};
                    const e = p.engineering || {};
                    const t = p.telemetry || {};
                    const vals = [
                        p.global_rank, p.site_name, p.region, p.lat, p.lon,
                        p.similarity_score, p.latent_distance, p.cosine_similarity,
                        p.distance_vs_flow_a_best, p.cluster_label,
                        r.Morphometry, r.Thermophysics, r.Geochemistry, r.Environment,
                        e.lander_suitable, e.rover_suitable, e.nearest_flow_a_site_km,
                        ...teleKeys.map(k => t[k])
                    ];
                    rows.push(vals.map(esc).join(','));
                });
                triggerBlobDownload(rows.join('\n'), `${baseName}.csv`, 'text/csv;charset=utf-8;');
            }
        }

        function switchPlanet(type) {
            currentPlanet = type;
            document.getElementById('btn-lunar').classList.toggle('active', type === 'lunar');
            document.getElementById('btn-mars').classList.toggle('active', type === 'mars');
            populateTargetSelect(type === 'lunar' ? lunarData : marsData);
            renderCurrentSelection();
        }

        function onTargetChange() {
            renderCurrentSelection();
        }

        renderExclusionZones();
        populateTargetSelect(lunarData);
        renderCurrentSelection();
    </script>
</body>
</html>
"""


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    lunar_json_path = results_dir / "flowB_lunar_heatmap.json"
    mars_json_path = results_dir / "flowB_mars_heatmap.json"
    excl_json_path = results_dir / "flowB_exclusion_report.json"

    lunar_data = json.loads(lunar_json_path.read_text(encoding="utf-8")) if lunar_json_path.exists() else {"grid_points": [], "metadata": {}, "targets_map": {}}
    mars_data = json.loads(mars_json_path.read_text(encoding="utf-8")) if mars_json_path.exists() else {"grid_points": [], "metadata": {}, "targets_map": {}}
    excl_data = json.loads(excl_json_path.read_text(encoding="utf-8")) if excl_json_path.exists() else {"flow_a_sites": [], "buffer_km": 50.0}

    html_content = (
        HTML_TEMPLATE
        .replace("__LUNAR_JSON__", json.dumps(lunar_data))
        .replace("__MARS_JSON__", json.dumps(mars_data))
        .replace("__EXCL_JSON__", json.dumps(excl_data))
    )

    output_html = results_dir / "perseus_local_viewer.html"
    output_html.write_text(html_content, encoding="utf-8")
    print(f"IRIS satellite discovery map successfully generated at: {output_html}")


if __name__ == "__main__":
    main()
