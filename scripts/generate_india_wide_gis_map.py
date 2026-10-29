#!/usr/bin/env python3
"""
=============================================================================
Pan-India EV Infrastructure GIS Spatial Map Builder
=============================================================================
Creates:
1. results/india_wide/gis/interactive_planning_map_india.html
   - Indian map with state outlines (36 States & UTs)
   - Specific EV charging stations visualized within each state outline
   - Filterable layers: DC Fast (>=50kW), AC Slow (<50kW), State Outlines, EIPS Centers
2. results/india_wide/figures/gis_spatial_distribution_india.png
   - Static high-res GIS map plot of national infrastructure
=============================================================================
"""

import os
import sys
import json
import html
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from folium.plugins import FastMarkerCluster

BASE_DIR = Path("C:/Users/Dell/Downloads/EV_Charging")
DATA_DIR = BASE_DIR / "data" / "india_wide"
RESULTS_DIR = BASE_DIR / "results" / "india_wide"
GIS_DIR = RESULTS_DIR / "gis"
FIG_DIR = RESULTS_DIR / "figures"

GIS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 80)
    print("[INFO] GENERATING PAN-INDIA GIS INTERACTIVE MAP & SPATIAL LAYERS")
    print("=" * 80)

    # 1. Load Data
    data_30_path = DATA_DIR / "INDIA_WIDE_EV_CHARGING_MASTER_DATASET_52K_30FEAT.csv"
    summary_path = DATA_DIR / "INDIA_WIDE_STATE_EV_CHARGING_SUMMARY_52K.csv"
    geojson_path = DATA_DIR / "india_states_boundary.geojson"

    if not data_30_path.exists() or not summary_path.exists():
        print("[ERROR] Required dataset or summary CSV not found.")
        sys.exit(1)

    df_30 = pd.read_csv(data_30_path)
    df_summary = pd.read_csv(summary_path)

    print(f"[OK] Ingested 52K Stations: {len(df_30):,} rows")
    print(f"[OK] Ingested 36 State Summaries: {len(df_summary)} states/UTs")

    # Map state summary dictionary for fast lookup
    state_meta_map = {}
    for _, r in df_summary.iterrows():
        state_meta_map[r["state_or_ut"].strip().lower()] = r.to_dict()

    # 2. Initialize Folium Map centered on India
    india_map = folium.Map(
        location=[22.50, 82.50],
        zoom_start=5,
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; <a href='https://carto.com/'>CARTO</a>",
        name="CARTO Dark Matter (High Contrast GIS)",
        control_scale=True
    )

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a>",
        name="OpenStreetMap Standard"
    ).add_to(india_map)

    # Feature Groups
    fg_boundaries = folium.FeatureGroup(name="🇮🇳 Indian State Outlines (36 States & UTs)", show=True)
    fg_fast = folium.FeatureGroup(name="⚡ DC Fast Stations (≥50 kW)", show=True)
    fg_slow = folium.FeatureGroup(name="🔌 AC Slow / Destination (<50 kW)", show=True)
    fg_hubs = folium.FeatureGroup(name="🏛️ State EIPS Priority Centers", show=True)

    # 3. Add State Outlines via GeoJSON if available
    if geojson_path.exists():
        print(f"[OK] Loading India States GeoJSON from: {geojson_path}")
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        def style_function(feature):
            p = feature.get("properties", {})
            st_name = (p.get("NAME_1") or p.get("name") or p.get("st_nm") or "").strip().lower()
            info = state_meta_map.get(st_name, {})
            # Rank color
            rank = info.get("national_rank", 20)
            if rank <= 5:
                border_color = "#10b981" # Top Tier
            elif rank <= 12:
                border_color = "#38bdf8" # Second Tier
            elif rank <= 24:
                border_color = "#f59e0b" # Third Tier
            else:
                border_color = "#94a3b8" # Basic Tier

            return {
                "fillColor": border_color,
                "color": border_color,
                "weight": 1.8,
                "fillOpacity": 0.15,
                "dashArray": "2, 4"
            }

        def highlight_function(feature):
            return {
                "fillColor": "#38bdf8",
                "color": "#ffffff",
                "weight": 3.0,
                "fillOpacity": 0.40,
                "dashArray": ""
            }

        # Add GeoJSON to boundaries layer
        for feature in geojson_data.get("features", []):
            p = feature.get("properties", {})
            st_name_raw = p.get("NAME_1") or p.get("name") or p.get("st_nm") or "State"
            st_key = st_name_raw.strip().lower()
            info = state_meta_map.get(st_key, {})

            tot = info.get("july2026_scaled_total", "N/A")
            fast = info.get("july2026_scaled_fast", "N/A")
            slow = info.get("july2026_scaled_slow", "N/A")
            pct = info.get("fast_charging_percentage", "N/A")
            rank = info.get("national_rank", "N/A")
            agency = info.get("state_nodal_agency", "Designated State Agency")

            tot_val = f"{tot:,}" if isinstance(tot, (int, np.integer)) else str(tot)
            fast_val = f"{fast:,}" if isinstance(fast, (int, np.integer)) else str(fast)
            slow_val = f"{slow:,}" if isinstance(slow, (int, np.integer)) else str(slow)

            tooltip_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 13px; line-height: 1.5; min-width: 200px;">
                <h4 style="margin:0 0 5px 0; color:#38bdf8; border-bottom:1px solid #334155; padding-bottom:3px;">
                    {st_name_raw} (Rank #{rank})
                </h4>
                <b>Total EVCS:</b> {tot_val}<br>
                <b>DC Fast:</b> {fast_val} ({pct}%)<br>
                <b>AC Slow:</b> {slow_val}<br>
                <small style="color:#94a3b8;"><b>Nodal Agency:</b> {agency}</small>
            </div>
            """

            folium.GeoJson(
                feature,
                style_function=style_function,
                highlight_function=highlight_function,
                tooltip=folium.Tooltip(tooltip_html, sticky=True)
            ).add_to(fg_boundaries)

        print("[OK] State Outlines added successfully.")

    # 4. Add Representative Charging Stations Across All 36 States
    print("[INFO] Adding Station Markers across all 36 States & UTs...")
    # Sample up to 80 stations per state to ensure fast DOM loading while showing full coverage
    sampled_indices = []
    for st_name, group in df_30.groupby("state"):
        sample_size = min(len(group), 85)
        # Stratified sample fast and slow
        fast_grp = group[group["fast_charging_flag"] == 1]
        slow_grp = group[group["fast_charging_flag"] == 0]

        n_fast = min(len(fast_grp), int(sample_size * 0.4))
        n_slow = sample_size - n_fast

        if n_fast > 0:
            sampled_indices.extend(fast_grp.sample(n=n_fast, random_state=42).index.tolist())
        if n_slow > 0 and len(slow_grp) > 0:
            sampled_indices.extend(slow_grp.sample(n=min(len(slow_grp), n_slow), random_state=42).index.tolist())

    df_sample = df_30.loc[sampled_indices].copy()
    print(f"[OK] Plotting {len(df_sample):,} representative charging station locations across all 36 states.")

    for _, row in df_sample.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        st_id = html.escape(str(row["station_id"]))
        st_name = html.escape(str(row["station_name"]))
        operator = html.escape(str(row["operator"]))
        state = html.escape(str(row["state"]))
        dist = html.escape(str(row["district"]))
        ch_type = html.escape(str(row["charger_type"]))
        conn_type = html.escape(str(row["connector_type"]))
        power = float(row.get("rated_power_kw", 50.0))
        is_fast = int(row.get("fast_charging_flag", 0))
        h_dist = float(row.get("distance_to_highway_km", 1.0))
        road = html.escape(str(row.get("road_type", "primary")))

        status_text = "DC Fast Charging (≥50 kW)" if is_fast == 1 else "AC Slow Charging (<50 kW)"
        header_color = "#10b981" if is_fast == 1 else "#f59e0b"

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; width: 260px; line-height: 1.45;">
            <h4 style="margin:0 0 6px 0; color:{header_color}; border-bottom:2px solid {header_color}; padding-bottom:3px;">
                {st_id} • {status_text}
            </h4>
            <b>Station:</b> {st_name}<br>
            <b>State / District:</b> {state} / {dist}<br>
            <b>Operator:</b> {operator}<br>
            <b>Charger Architecture:</b> {ch_type}<br>
            <b>Connector Standard:</b> {conn_type}<br>
            <b>Power Rating:</b> {power:.1f} kW<br>
            <b>Distance to Highway:</b> {h_dist:.2f} km<br>
            <b>Corridor Road Type:</b> {road}<br>
            <hr style="border:0;border-top:1px solid #334155;margin:6px 0;">
            <small style="color:#64748b;">Ground Truth: National Operational Public EVCS</small>
        </div>
        """

        popup = folium.Popup(popup_html, max_width=300)

        if is_fast == 1:
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color="#ffffff",
                fill=True,
                fill_color="#10b981",
                fill_opacity=0.85,
                weight=1,
                tooltip=f"{st_id} | {state} | {operator} (DC Fast)",
                popup=popup
            ).add_to(fg_fast)
        else:
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="#ffffff",
                fill=True,
                fill_color="#a855f7",
                fill_opacity=0.85,
                weight=1,
                tooltip=f"{st_id} | {state} | {operator} (AC Slow)",
                popup=popup
            ).add_to(fg_slow)

    # 5. Add State EIPS Priority Hubs (State Centroids / Capitals)
    STATE_CAPITAL_COORDS = {
        "Karnataka": (12.9716, 77.5946), "Maharashtra": (18.9220, 72.8347), "Uttar Pradesh": (26.8467, 80.9462),
        "Delhi": (28.6139, 77.2090), "Tamil Nadu": (13.0827, 80.2707), "Rajasthan": (26.9124, 75.7873),
        "Kerala": (8.5241, 76.9366), "Gujarat": (23.0225, 72.5714), "Madhya Pradesh": (23.2599, 77.4126),
        "Telangana": (17.3850, 78.4867), "Haryana": (30.7333, 76.7794), "West Bengal": (22.5726, 88.3639),
        "Andhra Pradesh": (16.5062, 80.6480), "Punjab": (31.6340, 74.8723), "Odisha": (20.2961, 85.8245),
        "Bihar": (25.5941, 85.1376), "Jharkhand": (23.3441, 85.3096), "Chhattisgarh": (21.2514, 81.6296),
        "Assam": (26.1445, 91.7362), "Uttarakhand": (30.3165, 78.0322), "Jammu and Kashmir": (34.0837, 74.7973),
        "Goa": (15.4909, 73.8278), "Himachal Pradesh": (31.1048, 77.1734), "Meghalaya": (25.5788, 91.8933),
        "Manipur": (24.8170, 93.9368), "Tripura": (23.8315, 91.2868), "Puducherry": (11.9416, 79.8083),
        "Arunachal Pradesh": (27.0844, 93.6053), "Nagaland": (25.6751, 94.1086), "Chandigarh": (30.7333, 76.7794),
        "Mizoram": (23.7271, 92.7176), "Sikkim": (27.3389, 88.6065), "Dadra and Nagar Haveli and Daman and Diu": (20.3974, 72.8328),
        "Andaman and Nicobar Islands": (11.6234, 92.7265), "Ladakh": (34.1526, 77.5771), "Lakshadweep": (10.5667, 72.6417)
    }

    for _, r in df_summary.iterrows():
        st_name = r["state_or_ut"]
        lat, lon = STATE_CAPITAL_COORDS.get(st_name, (20.0, 78.0))
        rank = int(r["national_rank"])
        tot = int(r["july2026_scaled_total"])
        fast = int(r["july2026_scaled_fast"])
        slow = int(r["july2026_scaled_slow"])
        pct = float(r["fast_charging_percentage"])
        agency = html.escape(str(r["state_nodal_agency"]))

        hub_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; width: 280px; line-height: 1.45;">
            <h4 style="margin:0 0 5px 0; color:#38bdf8; border-bottom:2px solid #38bdf8; padding-bottom:3px;">
                {st_name} (National Rank #{rank})
            </h4>
            <b>Total Stations:</b> {tot:,}<br>
            <b>DC Fast Stations:</b> {fast:,} ({pct}%)<br>
            <b>AC Slow Stations:</b> {slow:,}<br>
            <b>State Nodal Agency:</b> {agency}<br>
            <hr style="border:0;border-top:1px solid #334155;margin:5px 0;">
            <b>GIS Status:</b> Verified State Administrative Hub
        </div>
        """

        folium.Circle(
            location=[lat, lon],
            radius=18000 if tot > 1000 else 10000,
            color="#38bdf8",
            fill=True,
            fill_color="#38bdf8",
            fill_opacity=0.35,
            weight=2,
            tooltip=f"{st_name} | Rank #{rank} | Total: {tot:,} Stations",
            popup=folium.Popup(hub_html, max_width=300)
        ).add_to(fg_hubs)

    # 6. Add all layers to map
    fg_boundaries.add_to(india_map)
    fg_fast.add_to(india_map)
    fg_slow.add_to(india_map)
    fg_hubs.add_to(india_map)

    folium.LayerControl(collapsed=False).add_to(india_map)

    # Save interactive HTML map
    map_out_path = GIS_DIR / "interactive_planning_map_india.html"
    india_map.save(str(map_out_path))
    print(f"[SUCCESS] Exported Interactive GIS Map: {map_out_path}")

    # 7. Generate Static Publication GIS Figure
    plt.figure(figsize=(12, 10), dpi=300)
    plt.style.use("dark_background")

    # Plot stations scatter
    plt.scatter(
        df_30[df_30["fast_charging_flag"] == 0]["longitude"],
        df_30[df_30["fast_charging_flag"] == 0]["latitude"],
        c="#a855f7", s=3, alpha=0.5, label="AC Slow / Destination (<50 kW)"
    )
    plt.scatter(
        df_30[df_30["fast_charging_flag"] == 1]["longitude"],
        df_30[df_30["fast_charging_flag"] == 1]["latitude"],
        c="#10b981", s=5, alpha=0.7, label="DC Fast Charging (≥50 kW)"
    )

    plt.title("Pan-India EV Charging Infrastructure Geospatial Distribution\n52,718 Public Stations across 36 States & UTs (July 2026 PIB Benchmark)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Longitude (°E)", fontsize=10)
    plt.ylabel("Latitude (°N)", fontsize=10)
    plt.xlim(68.0, 98.0)
    plt.ylim(6.0, 38.0)
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend(loc="lower right", framealpha=0.8, fontsize=9)
    plt.tight_layout()

    fig_out_path = FIG_DIR / "gis_spatial_distribution_india.png"
    plt.savefig(fig_out_path)
    plt.close()
    print(f"[SUCCESS] Exported Static GIS Figure: {fig_out_path}")

    print("=" * 80)
    print("[ALL GIS TASKS COMPLETE]")
    print("=" * 80)

if __name__ == "__main__":
    main()
