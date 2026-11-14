#!/usr/bin/env python3
"""
Team Perseus - Local Map Viewer Generator for NASA Space Apps 2026
Generates an interactive standalone HTML map (results/perseus_local_viewer.html)
using keyless Satellite Imagery & Dark Terrain basemaps.
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

    lunar_data = (
        json.loads(lunar_json_path.read_text(encoding="utf-8"))
        if lunar_json_path.exists()
        else {"grid_points": []}
    )
    mars_data = (
        json.loads(mars_json_path.read_text(encoding="utf-8"))
        if mars_json_path.exists()
        else {"grid_points": []}
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Team Perseus - NASA Space Apps 2026 | Local Discovery Viewer</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #0b0f19; color: #eee; }}
        #header {{ background: #111827; padding: 14px 24px; display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #3b82f6; }}
        h1 {{ margin: 0; font-size: 17px; color: #60a5fa; letter-spacing: 0.4px; }}
        .controls {{ display: flex; gap: 10px; }}
        button {{ background: #1f2937; color: white; border: 1px solid #3b82f6; padding: 8px 16px; cursor: pointer; border-radius: 6px; font-weight: bold; transition: all 0.2s; }}
        button:hover {{ background: #374151; }}
        button.active {{ background: #2563eb; border-color: #60a5fa; }}
        #map {{ height: calc(100vh - 60px); width: 100%; background: #0b0f19; }}
        .popup-box {{ font-size: 13px; color: #111; line-height: 1.45; min-width: 220px; }}
        .popup-box h3 {{ margin: 0 0 6px 0; color: #1d4ed8; font-size: 15px; }}
        .badge {{ background: #e5e7eb; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>NASA Space Apps Challenge 2026 &mdash; Team Perseus (Local Discovery Viewer)</h1>
        <div class="controls">
            <button id="btn-lunar" class="active" onclick="switchLayer('lunar')">Flow B: Lunar Analogs</button>
            <button id="btn-mars" onclick="switchLayer('mars')">Flow B: Mars Analogs</button>
        </div>
    </div>
    <div id="map"></div>

    <script>
        const lunarPoints = {json.dumps(lunar_data.get("grid_points", []))};
        const marsPoints = {json.dumps(mars_data.get("grid_points", []))};

        const map = L.map('map').setView([20, 10], 2);

        // Keyless Satellite Imagery & Dark Basemap layers (work locally with zero API keys)
        const satelliteLayer = L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
            {{
                attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, NASA/USGS',
                maxZoom: 18
            }}
        );

        const darkCanvasLayer = L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}',
            {{
                attribution: 'Tiles &copy; Esri &mdash; Esri, HERE, Garmin, FAO, NOAA',
                maxZoom: 16
            }}
        );

        satelliteLayer.addTo(map);

        L.control.layers(
            {{
                "Satellite Imagery (True Color)": satelliteLayer,
                "Dark Tactical Base": darkCanvasLayer
            }},
            null,
            {{ position: 'topright' }}
        ).addTo(map);

        let currentLayerGroup = L.layerGroup().addTo(map);

        function getColor(score) {{
            return score > 75 ? '#10b981' : score > 55 ? '#f59e0b' : score > 35 ? '#f97316' : '#ef4444';
        }}

        function renderPoints(points) {{
            currentLayerGroup.clearLayers();
            points.forEach(p => {{
                let color = getColor(p.similarity_score);
                let marker = L.circleMarker([p.lat, p.lon], {{
                    radius: p.is_known_site ? 9 : 6,
                    color: '#ffffff',
                    weight: p.is_known_site ? 2.5 : 1.2,
                    fillColor: color,
                    fillOpacity: 0.9
                }});

                let popupContent = `<div class="popup-box">
                    <h3>${{p.site_name}}</h3>
                    <b>Region / Biome:</b> ${{p.region}}<br>
                    <b>Status:</b> <span class="badge">${{p.is_known_site ? 'Flow A Calibration Anchor' : 'New Discovery Candidate'}}</span><br>
                    <b>Global Rank:</b> #${{p.global_rank}}<br>
                    <b>Similarity Score:</b> <span style="color:#2563eb; font-weight:bold;">${{p.similarity_score}}%</span><br>
                    <b>Cluster Archetype:</b> ${{p.cluster_label}}<br>
                    <hr style="margin:6px 0; border:0; border-top:1px solid #ddd;">
                    <b>Radar Sub-Scores (0-100):</b><br>
                    &bull; Morphometry: <b>${{p.radar_subscores.Morphometry}}</b><br>
                    &bull; Thermophysics: <b>${{p.radar_subscores.Thermophysics}}</b><br>
                    &bull; Geochemistry: <b>${{p.radar_subscores.Geochemistry}}</b><br>
                    &bull; Environment: <b>${{p.radar_subscores.Environment}}</b><br>
                    <hr style="margin:6px 0; border:0; border-top:1px solid #ddd;">
                    <b>Key Telemetry:</b><br>
                    &bull; Altitude: ${{p.telemetry.Altitude_m}} m<br>
                    &bull; Mean Slope: ${{p.telemetry.Mean_Slope_deg}}&deg;<br>
                    &bull; Diurnal &Delta;T: ${{p.telemetry.Diurnal_Delta_T_K}} K
                </div>`;
                marker.bindPopup(popupContent);
                currentLayerGroup.addLayer(marker);
            }});
        }}

        function switchLayer(type) {{
            document.getElementById('btn-lunar').classList.toggle('active', type === 'lunar');
            document.getElementById('btn-mars').classList.toggle('active', type === 'mars');
            renderPoints(type === 'lunar' ? lunarPoints : marsPoints);
        }}

        renderPoints(lunarPoints);
    </script>
</body>
</html>
"""

    output_html = results_dir / "perseus_local_viewer.html"
    output_html.write_text(html_content, encoding="utf-8")
    print(f"Satellite local map successfully generated at: {output_html}")


if __name__ == "__main__":
    main()
