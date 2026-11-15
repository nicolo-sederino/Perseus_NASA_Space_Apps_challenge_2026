#!/usr/bin/env python3
"""
Team Perseus - Interactive Discovery Map Generator for NASA Space Apps 2026
Supports dynamic switching across all 5 Lunar and 10 Martian reference targets,
4-axis Radar sub-scores, engineering flags, and Flow A 50km exclusion zones.
"""

import json
from pathlib import Path


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

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Team Perseus - NASA Space Apps 2026 | Global Discovery Map</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #0b0f19; color: #eee; }}
        #header {{ background: #111827; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #3b82f6; flex-wrap: wrap; gap: 10px; }}
        h1 {{ margin: 0; font-size: 16px; color: #60a5fa; letter-spacing: 0.4px; }}
        .sub {{ font-size: 12px; color: #9ca3af; margin-top: 2px; }}
        .controls {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
        button, select {{ background: #1f2937; color: white; border: 1px solid #3b82f6; padding: 7px 12px; cursor: pointer; border-radius: 6px; font-weight: bold; font-size: 12.5px; transition: all 0.2s; }}
        button:hover, select:hover {{ background: #374151; }}
        button.active {{ background: #2563eb; border-color: #60a5fa; }}
        button.excl-btn {{ border-color: #ef4444; }}
        button.excl-btn.active {{ background: #b91c1c; border-color: #f87171; }}
        #map {{ height: calc(100vh - 64px); width: 100%; background: #0b0f19; }}
        .popup-box {{ font-size: 12.5px; color: #111; line-height: 1.45; min-width: 245px; }}
        .popup-box h3 {{ margin: 0 0 6px 0; color: #1d4ed8; font-size: 14.5px; }}
        .badge {{ background: #dbeafe; color: #1e40af; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
        .badge-ok {{ background: #d1fae5; color: #065f46; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
        .badge-no {{ background: #fee2e2; color: #991b1b; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
    </style>
</head>
<body>
    <div id="header">
        <div>
            <h1>NASA Space Apps Challenge 2026 &mdash; Team Perseus (Flow B Global Discovery)</h1>
            <div class="sub" id="target-info">Loading target metadata...</div>
        </div>
        <div class="controls">
            <button id="btn-lunar" class="active" onclick="switchPlanet('lunar')">Moon Analogs</button>
            <button id="btn-mars" onclick="switchPlanet('mars')">Mars Analogs</button>
            <select id="target-select" onchange="onTargetChange()"></select>
            <button id="btn-excl" class="excl-btn active" onclick="toggleExclusion()">Flow A 50km Exclusion Zones</button>
        </div>
    </div>
    <div id="map"></div>

    <script>
        const lunarData = {json.dumps(lunar_data)};
        const marsData = {json.dumps(mars_data)};
        const exclData = {json.dumps(excl_data)};
        let currentPlanet = 'lunar';

        const map = L.map('map', {{ preferCanvas: true }}).setView([20, 10], 2);

        const satelliteLayer = L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
            {{ attribution: 'Tiles &copy; Esri &mdash; Esri, Maxar, NASA/USGS', maxZoom: 18 }}
        );
        const darkCanvasLayer = L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}',
            {{ attribution: 'Tiles &copy; Esri', maxZoom: 16 }}
        );
        satelliteLayer.addTo(map);
        L.control.layers({{ "Satellite Imagery (True Color)": satelliteLayer, "Dark Tactical Base": darkCanvasLayer }}, null, {{ position: 'topright' }}).addTo(map);

        let discoveryLayer = L.layerGroup().addTo(map);
        let exclusionLayer = L.layerGroup().addTo(map);
        let showExclusion = true;

        function renderExclusionZones() {{
            exclusionLayer.clearLayers();
            const radiusMeters = (exclData.buffer_km || 50) * 1000;
            (exclData.flow_a_sites || []).forEach(s => {{
                let circle = L.circle([s.latitude, s.longitude], {{
                    radius: radiusMeters, color: '#ef4444', weight: 1.5, dashArray: '5, 5', fillColor: '#ef4444', fillOpacity: 0.12
                }});
                circle.bindPopup(`<div class="popup-box">
                    <h3 style="color:#b91c1c;">Excluded Flow A Catalog Site</h3>
                    <b>Site:</b> ${{s.site_name}}<br>
                    <b>Exclusion Radius:</b> ${{exclData.buffer_km || 50}} km<br>
                    <i>All cells inside this buffer are excluded from Flow B discovery.</i>
                </div>`);
                exclusionLayer.addLayer(circle);
            }});
        }}

        function toggleExclusion() {{
            showExclusion = !showExclusion;
            document.getElementById('btn-excl').classList.toggle('active', showExclusion);
            if (showExclusion) exclusionLayer.addTo(map); else map.removeLayer(exclusionLayer);
        }}

        function getColor(score) {{
            return score > 65 ? '#10b981' : score > 48 ? '#f59e0b' : score > 30 ? '#f97316' : '#ef4444';
        }}

        function populateTargetSelect(dataset) {{
            const sel = document.getElementById('target-select');
            sel.innerHTML = '';
            const meta = dataset.metadata || {{}};
            const targets = meta.available_targets || [meta.reference_target || 'Default'];
            targets.forEach(t => {{
                const opt = document.createElement('option');
                opt.value = t;
                opt.innerText = 'Target: ' + t.replaceAll('_', ' ');
                if (t === meta.reference_target) opt.selected = true;
                sel.appendChild(opt);
            }});
        }}

        function renderCurrentSelection() {{
            discoveryLayer.clearLayers();
            const dataset = currentPlanet === 'lunar' ? lunarData : marsData;
            const meta = dataset.metadata || {{}};
            const chosenTarget = document.getElementById('target-select').value || meta.reference_target;
            const tMap = dataset.targets_map || {{}};
            const points = (tMap[chosenTarget] && tMap[chosenTarget].length) ? tMap[chosenTarget] : (dataset.grid_points || []);

            document.getElementById('target-info').innerText =
                `Reference Target: ${{chosenTarget.replaceAll('_', ' ')}} | Total Evaluated Abiotic Cells: ${{meta.total_evaluated_sites || points.length}} | Excluded Flow A Sites: ${{meta.flow_a_sites_excluded_from_search || 31}}`;

            points.forEach(p => {{
                let color = getColor(p.similarity_score);
                let marker = L.circleMarker([p.lat, p.lon], {{
                    radius: p.global_rank <= 25 ? 7.5 : 5,
                    color: p.global_rank <= 25 ? '#ffffff' : color,
                    weight: p.global_rank <= 25 ? 2 : 1,
                    fillColor: color,
                    fillOpacity: 0.88
                }});

                let eng = p.engineering || {{}};
                let tele = p.telemetry || {{}};
                let landerBadge = eng.lander_suitable ? '<span class="badge-ok">Lander OK</span>' : '<span class="badge-no">Slope &gt; 5&deg;</span>';
                let roverBadge = eng.rover_suitable ? '<span class="badge-ok">Rover OK</span>' : '<span class="badge-no">Steep</span>';

                let popupContent = `<div class="popup-box">
                    <h3>#${{p.global_rank}} &mdash; ${{p.site_name}}</h3>
                    <b>Region / Biome:</b> ${{p.region}}<br>
                    <b>Status:</b> <span class="badge">New Discovery (${{eng.nearest_flow_a_site_km || '?'}} km from Flow A)</span><br>
                    <b>Suitability:</b> ${{landerBadge}} ${{roverBadge}}<br>
                    <b>Similarity Score:</b> <span style="color:#2563eb; font-weight:bold;">${{p.similarity_score}}%</span>
                    (d/d<sub>best,FlowA</sub> = <b>${{p.distance_vs_flow_a_best}}</b>)<br>
                    <b>Cluster Archetype:</b> ${{p.cluster_label}}<br>
                    <hr style="margin:6px 0; border:0; border-top:1px solid #ddd;">
                    <b>Radar Sub-Scores (0-100):</b><br>
                    &bull; Morphometry: <b>${{p.radar_subscores.Morphometry}}</b> | Thermophysics: <b>${{p.radar_subscores.Thermophysics}}</b><br>
                    &bull; Geochemistry: <b>${{p.radar_subscores.Geochemistry}}</b> | Environment: <b>${{p.radar_subscores.Environment}}</b><br>
                    <hr style="margin:6px 0; border:0; border-top:1px solid #ddd;">
                    <b>Key Telemetry:</b><br>
                    &bull; Altitude: ${{tele.Altitude_m ?? 'N/A'}} m | Slope: ${{tele.Mean_Slope_deg ?? 'N/A'}}&deg;<br>
                    &bull; Diurnal &Delta;T: ${{tele.Diurnal_Delta_T_K ?? 'N/A'}} K | T<sub>min</sub>: ${{tele.T_min_K ?? 'N/A'}} K
                </div>`;
                marker.bindPopup(popupContent);
                discoveryLayer.addLayer(marker);
            }});
        }}

        function switchPlanet(type) {{
            currentPlanet = type;
            document.getElementById('btn-lunar').classList.toggle('active', type === 'lunar');
            document.getElementById('btn-mars').classList.toggle('active', type === 'mars');
            populateTargetSelect(type === 'lunar' ? lunarData : marsData);
            renderCurrentSelection();
        }}

        function onTargetChange() {{
            renderCurrentSelection();
        }}

        renderExclusionZones();
        populateTargetSelect(lunarData);
        renderCurrentSelection();
    </script>
</body>
</html>
"""

    output_html = results_dir / "perseus_local_viewer.html"
    output_html.write_text(html_content, encoding="utf-8")
    print(f"Satellite discovery map successfully generated at: {output_html}")


if __name__ == "__main__":
    main()
