#!/usr/bin/env python3
"""
=============================================================================
Step 7: GIS Spatial Analysis & Interactive Geospatial Dashboard
=============================================================================
Project: Explainable AI Framework for Smart EV Charging Infrastructure Planning
Domain: Telangana, India (33 Administrative Districts)

Inputs:
- Frozen 166-station master dataset:
  data/final/FINAL_PROJECT_DATASET.csv
- Step 6 EIPS district planning output:
  results/planning/district_eips_rankings.csv

Outputs:
- results/gis/station_gis_layer.geojson
- results/gis/district_eips_layer.geojson
- results/gis/interactive_planning_map.html
- results/figures/gis_spatial_distribution.png

Important terminology:
- fast_charging_flag = observed infrastructure classification.
- NaN/unknown fast_charging_flag records are preserved as unlabeled/unknown.
- EIPS = district-level planning score; it is not an ML performance metric.
=============================================================================
"""

import os
import json
import html
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium

warnings.filterwarnings("ignore")


# ========================= CONFIGURATION =========================

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_FINAL_DIR = BASE_DIR / "data" / "final"
RESULTS_DIR = BASE_DIR / "results"
PLANNING_DIR = RESULTS_DIR / "planning"
GIS_DIR = RESULTS_DIR / "gis"
FIGURES_DIR = RESULTS_DIR / "figures"

for directory in [GIS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

MASTER_DATASET = DATA_FINAL_DIR / "FINAL_PROJECT_DATASET.csv"
EIPS_DATASET = PLANNING_DIR / "district_eips_rankings.csv"

EXPECTED_MASTER_ROWS = 166
EXPECTED_MASTER_COLUMNS = 78
EXPECTED_DISTRICTS = 33

# Frozen master dataset SHA-256 recorded in the project configuration.
EXPECTED_MASTER_SHA256 = (
    "a1999524323860397e7255b5ff7884d7f77394b09abd69a168a306bf7eae9788"
)

REQUIRED_STATION_COLUMNS = [
    "station_id",
    "station_name",
    "district",
    "latitude",
    "longitude",
    "operator",
    "charger_type",
    "connector_type",
    "charging_power_kw",
    "fast_charging_flag",
    "distance_to_highway_km",
    "distance_to_city_center_km",
    "nearby_poi_count",
]

REQUIRED_EIPS_COLUMNS = [
    "rank",
    "district",
    "priority_tier",
    "EIPS",
    "district_ev_penetration_rate",
    "infrastructure_gap_index",
    "total_stations",
    "fast_stations",
    "slow_stations",
    "unknown_label_stations",
    "nh_corridors_crossing_count",
    "spatial_suitability_score",
    "grid_readiness_score",
    "grid_data_observed_flag",
    "eips_explanation",
]


# Telangana district reference centroids used for visualization.
DISTRICT_CENTROIDS = {
    "Adilabad": (19.6641, 78.5320),
    "Bhadradri Kothagudem": (17.5558, 80.6177),
    "Hyderabad": (17.3850, 78.4867),
    "Jagitial": (18.7946, 78.9127),
    "Jangaon": (17.7249, 79.1557),
    "Jayashankar Bhupalapally": (18.4348, 79.8661),
    "Jogulamba Gadwal": (16.2330, 77.8080),
    "Kamareddy": (18.3248, 78.3375),
    "Karimnagar": (18.4386, 79.1288),
    "Khammam": (17.2473, 80.1514),
    "Kumuram Bheem Asifabad": (19.3588, 79.2891),
    "Kumurambheem Asifabad": (19.3588, 79.2891),
    "Mahabubabad": (17.5985, 80.0038),
    "Mahabubnagar": (16.7433, 78.0035),
    "Mancherial": (18.8679, 79.4639),
    "Medak": (18.0478, 78.2618),
    "Medchal-Malkajgiri": (17.6297, 78.5583),
    "Mulugu": (18.1925, 79.9400),
    "Nagarkurnool": (16.4856, 78.3196),
    "Nalgonda": (17.0577, 79.2684),
    "Narayanpet": (16.7410, 77.4988),
    "Nirmal": (19.0964, 78.3430),
    "Nizamabad": (18.6725, 78.0941),
    "Peddapalli": (18.6161, 79.3739),
    "Rajanna Sircilla": (18.3888, 78.8037),
    "Ranga Reddy": (17.2023, 78.3578),
    "Sangareddy": (17.6190, 78.0815),
    "Siddipet": (18.1018, 78.8520),
    "Suryapet": (17.1439, 79.6239),
    "Vikarabad": (17.3366, 77.9048),
    "Wanaparthy": (16.3624, 78.0628),
    "Warangal Rural": (17.9689, 79.6000),
    "Warangal Urban": (17.9784, 79.5941),
    "Yadadri Bhuvanagiri": (17.5108, 78.8845),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_float(value, default=None):
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=None):
    if pd.isna(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_text(value, default="Unknown"):
    if pd.isna(value):
        return default
    text_value = str(value).strip()
    return text_value if text_value else default


def station_status(flag):
    numeric_flag = safe_float(flag)
    if numeric_flag == 1.0:
        return "DC Fast (>= 50 kW)"
    if numeric_flag == 0.0:
        return "AC / Non-Fast (observed)"
    return "Unlabeled / Unknown"


def priority_style(rank):
    rank = safe_int(rank, 999)
    if rank <= 10:
        return "#dc2626", 9
    if rank <= 22:
        return "#d97706", 7
    return "#059669", 5


def load_gis_data():
    print("=" * 80)
    print("[INFO] STEP 7: STARTING GIS SPATIAL ANALYSIS & DASHBOARD GENERATION")
    print("=" * 80)

    if not MASTER_DATASET.exists():
        raise FileNotFoundError(
            f"[ERROR] Frozen master station dataset not found at {MASTER_DATASET}"
        )
    if not EIPS_DATASET.exists():
        raise FileNotFoundError(
            f"[ERROR] EIPS rankings not found at {EIPS_DATASET}. "
            "Run 06_planning_eips.py first."
        )

    master_hash = sha256_file(MASTER_DATASET)
    if master_hash != EXPECTED_MASTER_SHA256:
        raise ValueError(
            "[ERROR] Frozen master dataset SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_MASTER_SHA256}\n"
            f"Actual:   {master_hash}"
        )

    df_st = pd.read_csv(MASTER_DATASET)
    df_eips = pd.read_csv(EIPS_DATASET)

    missing_station = [
        column for column in REQUIRED_STATION_COLUMNS if column not in df_st.columns
    ]
    missing_eips = [
        column for column in REQUIRED_EIPS_COLUMNS if column not in df_eips.columns
    ]

    if missing_station:
        raise ValueError(
            f"[ERROR] Missing station columns: {missing_station}"
        )
    if missing_eips:
        raise ValueError(
            f"[ERROR] Missing EIPS columns: {missing_eips}"
        )

    if df_st.shape != (EXPECTED_MASTER_ROWS, EXPECTED_MASTER_COLUMNS):
        raise ValueError(
            "[ERROR] Frozen master shape mismatch: "
            f"expected {EXPECTED_MASTER_ROWS}x{EXPECTED_MASTER_COLUMNS}, "
            f"got {df_st.shape[0]}x{df_st.shape[1]}"
        )

    station_districts = set(df_st["district"].dropna().astype(str).str.strip())
    eips_districts = set(df_eips["district"].dropna().astype(str).str.strip())

    if len(station_districts) != EXPECTED_DISTRICTS:
        raise ValueError(
            f"[ERROR] Station dataset contains {len(station_districts)} districts; "
            f"expected {EXPECTED_DISTRICTS}."
        )
    if len(eips_districts) != EXPECTED_DISTRICTS:
        raise ValueError(
            f"[ERROR] EIPS output contains {len(eips_districts)} districts; "
            f"expected {EXPECTED_DISTRICTS}."
        )
    if station_districts != eips_districts:
        raise ValueError(
            "[ERROR] Station and EIPS district sets do not match."
        )

    missing_centroids = sorted(station_districts - set(DISTRICT_CENTROIDS))
    if missing_centroids:
        raise ValueError(
            f"[ERROR] Missing district centroid definitions: {missing_centroids}"
        )

    # Normalize and validate coordinates before mapping.
    df_st["latitude"] = pd.to_numeric(df_st["latitude"], errors="coerce")
    df_st["longitude"] = pd.to_numeric(df_st["longitude"], errors="coerce")
    if df_st[["latitude", "longitude"]].isna().any().any():
        raise ValueError("[ERROR] Station coordinates contain missing/non-numeric values.")

    df_st["fast_charging_flag"] = pd.to_numeric(
        df_st["fast_charging_flag"], errors="coerce"
    )

    df_eips["rank"] = pd.to_numeric(df_eips["rank"], errors="coerce")
    df_eips["EIPS"] = pd.to_numeric(df_eips["EIPS"], errors="coerce")
    df_eips["infrastructure_gap_index"] = pd.to_numeric(
        df_eips["infrastructure_gap_index"], errors="coerce"
    )

    if df_eips["rank"].isna().any() or df_eips["EIPS"].isna().any():
        raise ValueError("[ERROR] EIPS rank or score contains missing/non-numeric values.")

    df_eips = df_eips.sort_values("rank").reset_index(drop=True)

    if df_eips["rank"].tolist() != list(range(1, EXPECTED_DISTRICTS + 1)):
        raise ValueError("[ERROR] EIPS ranks are not a complete sequence from 1 to 33.")

    print(
        f"[INFO] Loaded frozen master: {len(df_st)} stations x "
        f"{df_st.shape[1]} columns"
    )
    print(
        "[INFO] Fast/slow/unknown labels: "
        f"{int((df_st['fast_charging_flag'] == 1).sum())}/"
        f"{int((df_st['fast_charging_flag'] == 0).sum())}/"
        f"{int(df_st['fast_charging_flag'].isna().sum())}"
    )
    print(f"[INFO] Loaded EIPS rankings: {len(df_eips)} districts")
    print(
        f"[INFO] Districts with observed TGSPDCL context: "
        f"{int(pd.to_numeric(df_eips['grid_data_observed_flag'], errors='coerce').fillna(0).sum())}"
    )

    return df_st, df_eips


def export_geojson_layers(df_st, df_eips):
    print("\n" + "=" * 80)
    print("[INFO] GENERATING SPATIAL GEOJSON LAYERS")
    print("=" * 80)

    eips_by_district = df_eips.set_index("district").to_dict("index")
    station_features = []

    for _, row in df_st.iterrows():
        district = safe_text(row["district"])
        context = eips_by_district[district]
        flag = safe_float(row["fast_charging_flag"])

        station_properties = {
            "station_id": safe_text(row["station_id"]),
            "station_name": safe_text(row["station_name"]),
            "district": district,
            "operator": safe_text(row["operator"]),
            "charger_type": safe_text(row["charger_type"]),
            "connector_type": safe_text(row["connector_type"]),
            "charging_power_kw": safe_float(row["charging_power_kw"]),
            "fast_charging_flag": safe_int(flag),
            "fast_charging_desc": station_status(flag),
            "distance_to_highway_km": safe_float(row["distance_to_highway_km"]),
            "distance_to_city_center_km": safe_float(row["distance_to_city_center_km"]),
            "nearby_poi_count": safe_int(row["nearby_poi_count"]),
            "district_eips": safe_float(context["EIPS"]),
            "district_eips_rank": safe_int(context["rank"]),
            "district_priority_tier": safe_text(context["priority_tier"]),
            "district_ev_penetration_rate": safe_float(
                context["district_ev_penetration_rate"]
            ),
            "infrastructure_gap_index": safe_float(
                context["infrastructure_gap_index"]
            ),
            "grid_readiness_score": safe_float(
                context["grid_readiness_score"]
            ),
            "grid_data_observed_flag": safe_int(
                context["grid_data_observed_flag"]
            ),
        }

        station_features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        float(row["longitude"]),
                        float(row["latitude"]),
                    ],
                },
                "properties": station_properties,
            }
        )

    station_geojson = {
        "type": "FeatureCollection",
        "name": "telangana_ev_stations",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": station_features,
    }

    station_layer_path = GIS_DIR / "station_gis_layer.geojson"
    with station_layer_path.open("w", encoding="utf-8") as file:
        json.dump(station_geojson, file, indent=2, ensure_ascii=False)

    district_features = []

    for _, row in df_eips.iterrows():
        district = safe_text(row["district"])
        lat, lon = DISTRICT_CENTROIDS[district]

        district_features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
                "properties": {
                    "rank": safe_int(row["rank"]),
                    "district": district,
                    "priority_tier": safe_text(row["priority_tier"]),
                    "EIPS": safe_float(row["EIPS"]),
                    "ev_penetration_rate": safe_float(
                        row["district_ev_penetration_rate"]
                    ),
                    "infrastructure_gap_index": safe_float(
                        row["infrastructure_gap_index"]
                    ),
                    "total_stations": safe_int(row["total_stations"]),
                    "fast_stations": safe_int(row["fast_stations"]),
                    "slow_stations": safe_int(row["slow_stations"]),
                    "unknown_label_stations": safe_int(row["unknown_label_stations"]),
                    "nh_crossings": safe_int(row["nh_corridors_crossing_count"]),
                    "spatial_suitability_score": safe_float(
                        row["spatial_suitability_score"]
                    ),
                    "grid_readiness_score": safe_float(
                        row["grid_readiness_score"]
                    ),
                    "grid_data_observed_flag": safe_int(
                        row["grid_data_observed_flag"]
                    ),
                    "explanation": safe_text(row["eips_explanation"]),
                },
            }
        )

    district_geojson = {
        "type": "FeatureCollection",
        "name": "telangana_district_eips",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": district_features,
    }

    district_layer_path = GIS_DIR / "district_eips_layer.geojson"
    with district_layer_path.open("w", encoding="utf-8") as file:
        json.dump(district_geojson, file, indent=2, ensure_ascii=False)

    print(
        f"[OK] Station GeoJSON saved: {station_layer_path} "
        f"({len(station_features)} features)"
    )
    print(
        f"[OK] District EIPS GeoJSON saved: {district_layer_path} "
        f"({len(district_features)} features)"
    )

    return station_geojson, district_geojson


def build_interactive_folium_map(df_st, df_eips):
    print("\n" + "=" * 80)
    print("[INFO] BUILDING INTERACTIVE LEAFLET / FOLIUM PLANNING MAP")
    print("=" * 80)

    map_object = folium.Map(
        location=[17.80, 79.10],
        zoom_start=8,
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; <a href='https://carto.com/'>CARTO</a>",
        name="CARTO Dark Matter",
        control_scale=True,
    )

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a>",
        name="OpenStreetMap Standard",
    ).add_to(map_object)

    fg_fast = folium.FeatureGroup(
        name="DC Fast Stations (>=50 kW)",
        show=True,
    )
    fg_slow = folium.FeatureGroup(
        name="AC / Non-Fast Stations (observed)",
        show=True,
    )
    fg_unknown = folium.FeatureGroup(
        name="Unlabeled / Unknown Stations",
        show=True,
    )
    fg_districts = folium.FeatureGroup(
        name="District EIPS Priorities",
        show=True,
    )

    eips_by_district = df_eips.set_index("district").to_dict("index")

    for _, row in df_st.iterrows():
        district = safe_text(row["district"])
        station_id = html.escape(safe_text(row["station_id"]))
        station_name = html.escape(safe_text(row["station_name"]))
        operator = html.escape(safe_text(row["operator"]))
        charger_type = html.escape(safe_text(row["charger_type"]))
        connector_type = html.escape(safe_text(row["connector_type"]))
        flag = safe_float(row["fast_charging_flag"])
        power = safe_float(row["charging_power_kw"])
        highway_distance = safe_float(row["distance_to_highway_km"])
        city_distance = safe_float(row["distance_to_city_center_km"])
        poi_count = safe_int(row["nearby_poi_count"])

        context = eips_by_district[district]
        eips = safe_float(context["EIPS"])
        rank = safe_int(context["rank"])
        tier = html.escape(safe_text(context["priority_tier"]))

        def fmt_number(value, suffix=""):
            return "Unknown" if value is None else f"{value:.2f}{suffix}"

        power_text = "Unknown" if power is None else f"{power:.1f} kW"
        status_text = html.escape(station_status(flag))

        popup_html = f"""
        <div style="
            font-family: Arial, sans-serif;
            font-size: 12px;
            width: 255px;
            line-height: 1.45;
        ">
            <h4 style="
                margin: 0 0 7px 0;
                color: #1d4ed8;
                border-bottom: 2px solid #60a5fa;
                padding-bottom: 4px;
            ">{station_id}</h4>
            <b>Name:</b> {station_name}<br>
            <b>District:</b> {html.escape(district)}<br>
            <b>Operator:</b> {operator}<br>
            <b>Charger Type:</b> {charger_type}<br>
            <b>Connector:</b> {connector_type}<br>
            <b>Power:</b> {power_text}<br>
            <b>Highway Distance:</b> {fmt_number(highway_distance, " km")}<br>
            <b>City Center Distance:</b> {fmt_number(city_distance, " km")}<br>
            <b>Nearby POIs:</b> {poi_count if poi_count is not None else "Unknown"}<br>
            <b>Observed Classification:</b> {status_text}<br>
            <hr style="border:0;border-top:1px solid #d1d5db;margin:7px 0;">
            <b>District EIPS:</b> {eips:.2f}/100<br>
            <b>District EIPS Rank:</b> #{rank}<br>
            <b>Priority Tier:</b> {tier}
        </div>
        """

        popup = folium.Popup(popup_html, max_width=290)

        lat = float(row["latitude"])
        lon = float(row["longitude"])

        if flag == 1.0:
            marker = folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="#10b981",
                fill=True,
                fill_color="#10b981",
                fill_opacity=0.85,
                weight=1,
                tooltip=f"{safe_text(row['station_id'])} | {district} | DC Fast",
                popup=popup,
            )
            marker.add_to(fg_fast)

        elif flag == 0.0:
            marker = folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color="#f59e0b",
                fill=True,
                fill_color="#f59e0b",
                fill_opacity=0.85,
                weight=1,
                tooltip=f"{safe_text(row['station_id'])} | {district} | AC / Non-Fast",
                popup=popup,
            )
            marker.add_to(fg_slow)

        else:
            marker = folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="#9ca3af",
                fill=True,
                fill_color="#9ca3af",
                fill_opacity=0.75,
                weight=1,
                tooltip=f"{safe_text(row['station_id'])} | {district} | Unknown",
                popup=popup,
            )
            marker.add_to(fg_unknown)

    # District circles use infrastructure gap for radius so the map symbol
    # reflects the underlying gap criterion rather than rank alone.
    gap_values = pd.to_numeric(
        df_eips["infrastructure_gap_index"], errors="coerce"
    ).fillna(0.0)
    gap_min = float(gap_values.min())
    gap_max = float(gap_values.max())

    for _, row in df_eips.iterrows():
        district = safe_text(row["district"])
        lat, lon = DISTRICT_CENTROIDS[district]
        rank = safe_int(row["rank"])
        eips = safe_float(row["EIPS"], 0.0)
        gap = safe_float(row["infrastructure_gap_index"], 0.0)

        tier_color, base_radius = priority_style(rank)

        if gap_max > gap_min:
            gap_scaled = (gap - gap_min) / (gap_max - gap_min)
        else:
            gap_scaled = 0.0

        radius = base_radius + 7.0 * gap_scaled

        observed_grid = safe_int(row["grid_data_observed_flag"], 0) == 1
        grid_status = (
            "Observed TGSPDCL district context"
            if observed_grid
            else "No observed TGSPDCL district throughput value"
        )

        explanation = html.escape(safe_text(row["eips_explanation"]))
        district_popup = f"""
        <div style="
            font-family: Arial, sans-serif;
            font-size: 12px;
            width: 280px;
            line-height: 1.45;
        ">
            <h4 style="
                margin: 0 0 7px 0;
                color: #991b1b;
                border-bottom: 2px solid {tier_color};
                padding-bottom: 4px;
            ">{html.escape(district)} (Rank #{rank})</h4>
            <b>EIPS:</b> <span style="font-size:14px;font-weight:bold;">{eips:.2f}/100</span><br>
            <b>Priority Tier:</b> {html.escape(safe_text(row["priority_tier"]))}<br>
            <b>Infrastructure Gap:</b> {gap:.4f}<br>
            <b>EV Adoption Context:</b> {safe_float(row["district_ev_penetration_rate"], 0.0):.2f} EV/k-pop<br>
            <b>Existing Fast Stations:</b> {safe_int(row["fast_stations"], 0)}<br>
            <b>Total Station Records:</b> {safe_int(row["total_stations"], 0)}<br>
            <b>Unknown Labels:</b> {safe_int(row["unknown_label_stations"], 0)}<br>
            <b>NH Crossings:</b> {safe_int(row["nh_corridors_crossing_count"], 0)}<br>
            <b>Grid Context:</b> {html.escape(grid_status)}<br>
            <hr style="border:0;border-top:1px solid #d1d5db;margin:7px 0;">
            <div style="font-size:11px;color:#374151;">{explanation}</div>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=tier_color,
            fill=True,
            fill_color=tier_color,
            fill_opacity=0.45,
            weight=2,
            tooltip=f"{district} | EIPS {eips:.1f} | Rank #{rank}",
            popup=folium.Popup(district_popup, max_width=320),
        ).add_to(fg_districts)

    for feature_group in [
        fg_fast,
        fg_slow,
        fg_unknown,
        fg_districts,
    ]:
        feature_group.add_to(map_object)

    folium.LayerControl(collapsed=False).add_to(map_object)

    total = len(df_st)
    fast_count = int((df_st["fast_charging_flag"] == 1).sum())
    slow_count = int((df_st["fast_charging_flag"] == 0).sum())
    unknown_count = int(df_st["fast_charging_flag"].isna().sum())
    observed_grid_count = int(
        pd.to_numeric(df_eips["grid_data_observed_flag"], errors="coerce")
        .fillna(0)
        .sum()
    )

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 1000;
        background: rgba(17,24,39,0.92);
        color: white;
        padding: 14px 17px;
        border-radius: 9px;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 4px 16px rgba(0,0,0,0.35);
        font-family: Arial, sans-serif;
        font-size: 12px;
        line-height: 1.55;
        min-width: 270px;
    ">
        <div style="font-size:14px;font-weight:bold;margin-bottom:7px;">
            EV Infrastructure GIS Dashboard
        </div>
        <div style="color:#d1d5db;margin-bottom:7px;">
            {total} station records | {EXPECTED_DISTRICTS} districts
        </div>
        <div>
            <span style="color:#10b981;">●</span> DC Fast observed: {fast_count}<br>
            <span style="color:#f59e0b;">●</span> AC / Non-Fast observed: {slow_count}<br>
            <span style="color:#9ca3af;">●</span> Unlabeled / Unknown: {unknown_count}
        </div>
        <hr style="border:0;border-top:1px solid rgba(255,255,255,0.2);margin:7px 0;">
        <div><b>EIPS district priority:</b></div>
        <div><span style="color:#dc2626;">●</span> Tier 1: Ranks 1-10</div>
        <div><span style="color:#d97706;">●</span> Tier 2: Ranks 11-22</div>
        <div><span style="color:#059669;">●</span> Tier 3: Ranks 23-33</div>
        <div style="margin-top:7px;color:#d1d5db;">
            Circle size reflects infrastructure gap.
        </div>
        <div style="margin-top:5px;color:#fca5a5;">
            EIPS is a planning score, not an ML performance metric.
        </div>
        <div style="margin-top:5px;color:#d1d5db;">
            TGSPDCL context observed in {observed_grid_count}/{EXPECTED_DISTRICTS} districts.
        </div>
    </div>
    """

    title_html = """
    <div style="
        position: fixed;
        top: 12px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 999;
        background: rgba(17,24,39,0.92);
        color: white;
        padding: 9px 16px;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 3px 12px rgba(0,0,0,0.25);
        font-family: Arial, sans-serif;
        font-size: 14px;
        font-weight: bold;
    ">
        Telangana EV Charging Infrastructure — GIS & EIPS Planning View
    </div>
    """

    map_object.get_root().html.add_child(folium.Element(legend_html))
    map_object.get_root().html.add_child(folium.Element(title_html))

    map_path = GIS_DIR / "interactive_planning_map.html"
    map_object.save(str(map_path))

    print(f"[OK] Interactive planning map saved: {map_path}")
    return map_path


def generate_static_gis_plot(df_st, df_eips):
    print("\n" + "=" * 80)
    print("[INFO] GENERATING PUBLICATION-READY GIS SPATIAL VISUALIZATION")
    print("=" * 80)

    figure, axis = plt.subplots(figsize=(12, 10), dpi=300)

    for _, row in df_eips.iterrows():
        district = safe_text(row["district"])
        lat, lon = DISTRICT_CENTROIDS[district]
        rank = safe_int(row["rank"], 999)
        eips = safe_float(row["EIPS"], 0.0)
        color, _ = priority_style(rank)

        axis.scatter(
            lon,
            lat,
            s=110 + eips * 10,
            color=color,
            alpha=0.22,
            edgecolors=color,
            linewidths=1.2,
            zorder=1,
        )

        if rank <= 8 or district in [
            "Hyderabad",
            "Medchal-Malkajgiri",
            "Warangal Urban",
        ]:
            axis.text(
                lon,
                lat + 0.07,
                f"{district}\n(#{rank})",
                fontsize=7.5,
                ha="center",
                fontweight="bold",
                color="#1e293b",
                zorder=3,
            )

    fast_stations = df_st[df_st["fast_charging_flag"] == 1]
    slow_stations = df_st[df_st["fast_charging_flag"] == 0]
    unknown_stations = df_st[df_st["fast_charging_flag"].isna()]

    axis.scatter(
        fast_stations["longitude"],
        fast_stations["latitude"],
        c="#10b981",
        s=35,
        label=f"DC Fast observed (n={len(fast_stations)})",
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )

    axis.scatter(
        slow_stations["longitude"],
        slow_stations["latitude"],
        c="#f59e0b",
        s=30,
        label=f"AC / Non-Fast observed (n={len(slow_stations)})",
        alpha=0.9,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )

    axis.scatter(
        unknown_stations["longitude"],
        unknown_stations["latitude"],
        c="#94a3b8",
        s=20,
        label=f"Unlabeled / Unknown (n={len(unknown_stations)})",
        alpha=0.7,
        edgecolors="gray",
        linewidths=0.5,
        zorder=4,
    )

    axis.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    axis.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    axis.set_title(
        "Telangana EV Charging Infrastructure Spatial Distribution and EIPS Priorities",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(loc="lower right", frameon=True, fontsize=9)

    axis.set_xlim(77.0, 81.6)
    axis.set_ylim(15.8, 20.0)

    figure.tight_layout()

    output_path = FIGURES_DIR / "gis_spatial_distribution.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"[OK] Static GIS visualization saved: {output_path}")


def write_step7_metadata(df_st, df_eips):
    metadata = {
        "project_title": "Explainable AI Framework for Smart EV Charging Infrastructure Planning",
        "step": 7,
        "master_dataset": str(MASTER_DATASET.relative_to(BASE_DIR)),
        "master_dataset_sha256": sha256_file(MASTER_DATASET),
        "station_records": int(len(df_st)),
        "districts": int(df_st["district"].nunique()),
        "fast_observed": int((df_st["fast_charging_flag"] == 1).sum()),
        "slow_observed": int((df_st["fast_charging_flag"] == 0).sum()),
        "unknown_fast_charging_labels": int(df_st["fast_charging_flag"].isna().sum()),
        "eips_districts": int(len(df_eips)),
        "eips_output": str(EIPS_DATASET.relative_to(BASE_DIR)),
        "districts_with_observed_tgspdcl_context": int(
            pd.to_numeric(df_eips["grid_data_observed_flag"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "notes": [
            "Unknown fast-charging labels are retained and never imputed.",
            "EIPS is a district-level planning score, not an ML performance metric.",
            "District circles are centered on the project centroid reference table.",
            "Circle size in the interactive map is scaled from infrastructure_gap_index.",
        ],
    }

    metadata_path = GIS_DIR / "gis_run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"[OK] GIS run metadata saved: {metadata_path}")


def main():
    df_stations, df_eips = load_gis_data()
    export_geojson_layers(df_stations, df_eips)
    build_interactive_folium_map(df_stations, df_eips)
    generate_static_gis_plot(df_stations, df_eips)
    write_step7_metadata(df_stations, df_eips)

    print("\n" + "=" * 80)
    print("[OK] STEP 7 GIS ANALYSIS & DASHBOARD GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
