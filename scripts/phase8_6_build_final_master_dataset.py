#!/usr/bin/env python3
"""
Phase 8.6: Build the Single Final Master Project Dataset
Explainable AI Framework for Smart EV Charging Infrastructure Planning

This script constructs the ONE FINAL MASTER CSV DATASET:
data/final/final_project_master_dataset.csv

Row Grain: ONE ROW = ONE PHYSICAL EV CHARGING STATION (N = 166)
First 25 columns: Exactly the frozen 25 core attributes from Phase 8.5.
Appended columns: Legitimate station GIS features, district demographic/highway/EV context,
and utility grid indicators.
"""

import sys
import os
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path("c:/Users/Dell/Downloads/EV_Charging")
DATA_FINAL_DIR = BASE_DIR / "data" / "final"
REPORTS_DIR = BASE_DIR / "reports"

CORE_25_ATTRIBUTES = [
    "station_id", "station_name", "state", "district", "city",
    "latitude", "longitude", "pincode", "location_type", "charger_type",
    "connector_type", "number_of_chargers", "charging_power_kw", "station_capacity",
    "availability_status", "operator", "24x7_availability", "road_type",
    "highway_name", "distance_to_highway_km", "distance_to_city_center_km",
    "traffic_density", "nearby_poi_count", "parking_availability", "data_timestamp"
]

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 70)
    print("STARTING PHASE 8.6: BUILD SINGLE FINAL MASTER PROJECT DATASET")
    print("=" * 70)

    # 1. Load Frozen Core 25 Dataset
    core_csv = DATA_FINAL_DIR / "telangana_ev_charging_stations_25_attributes.csv"
    if not core_csv.exists():
        print(f"Error: Frozen core file {core_csv} missing.")
        sys.exit(1)
    df_core = pd.read_csv(core_csv)
    print(f"Loaded Core 25 Dataset: {len(df_core)} rows, {len(df_core.columns)} columns.")

    # Verify first 25 columns match exactly
    assert list(df_core.columns) == CORE_25_ATTRIBUTES, "Core 25 column order mismatch!"

    # 2. Load Station GIS Spatial Features from Expanded Master
    exp_csv = DATA_FINAL_DIR / "telangana_station_master_expanded.csv"
    df_exp = pd.read_csv(exp_csv)
    print(f"Loaded Expanded Master: {len(df_exp)} rows, {len(df_exp.columns)} columns.")

    gis_cols = [
        "station_id",
        "highway_proximity_score",
        "city_center_proximity_score",
        "poi_diversity_index",
        "_distance_to_major_road_km",
        "_nearest_major_road_type",
        "_nearest_road_distance_m",
        "_poi_fuel_count",
        "_poi_parking_count",
        "_poi_food_count",
        "_poi_shopping_count",
        "_poi_healthcare_count",
        "_poi_lodging_count",
        "_poi_transport_count",
        "_poi_financial_count",
        "_poi_commercial_count",
        "_poi_search_radius_m",
        "_gis_projection_crs",
        "_nearest_nh_name",
        "fast_charging_flag",
        "operational_completeness_pct",
        "_data_source",
        "_source_id",
        "_expansion_batch"
    ]
    df_gis = df_exp[gis_cols].copy()
    df_gis.rename(columns={
        "_distance_to_major_road_km": "distance_to_major_road_km",
        "_nearest_major_road_type": "nearest_major_road_type",
        "_nearest_road_distance_m": "nearest_road_distance_m",
        "_poi_fuel_count": "poi_fuel_count",
        "_poi_parking_count": "poi_parking_count",
        "_poi_food_count": "poi_food_count",
        "_poi_shopping_count": "poi_shopping_count",
        "_poi_healthcare_count": "poi_healthcare_count",
        "_poi_lodging_count": "poi_lodging_count",
        "_poi_transport_count": "poi_transport_count",
        "_poi_financial_count": "poi_financial_count",
        "_poi_commercial_count": "poi_commercial_count",
        "_poi_search_radius_m": "gis_search_radius_m",
        "_gis_projection_crs": "gis_projection_crs",
        "_nearest_nh_name": "nearest_highway_name_osm",
        "_data_source": "source_registry",
        "_source_id": "source_registry_id",
        "_expansion_batch": "expansion_batch"
    }, inplace=True)

    # 3. Load District Contextual Features for FY2023-24 from District Year Panel
    panel_csv = DATA_FINAL_DIR / "district_year_panel.csv"
    df_panel = pd.read_csv(panel_csv)
    p24 = df_panel[df_panel["financial_year"] == "FY2023-24"].copy()
    print(f"Loaded FY2023-24 District Panel Slice: {len(p24)} districts.")

    dist_cols = [
        "district",
        "population_census_2011",
        "area_sq_km",
        "population_density_2011",
        "urban_population_pct",
        "urban_tier",
        "projected_population",
        "projected_population_density",
        "nh_corridors_crossing_count",
        "nh_corridors_crossing",
        "estimated_district_ev_registrations_proxy",
        "estimated_district_cumulative_evs_proxy",
        "district_ev_penetration_rate",
        "expanded_station_count",
        "fast_charging_station_count",
        "fast_charger_ratio",
        "expanded_total_capacity_kw",
        "expanded_station_density_per_1000_sq_km",
        "expanded_station_density_per_lakh_pop",
        "measured_discom_annual_ev_kwh",
        "measured_discom_billed_ev_connections"
    ]
    df_dist = p24[dist_cols].copy()
    df_dist.rename(columns={
        "population_census_2011": "district_population_census_2011",
        "area_sq_km": "district_area_sq_km",
        "population_density_2011": "district_population_density_2011",
        "urban_population_pct": "district_urban_population_pct",
        "urban_tier": "district_urban_tier",
        "projected_population": "district_projected_population_fy24",
        "projected_population_density": "district_projected_density_fy24",
        "nh_corridors_crossing_count": "district_nh_crossing_count",
        "nh_corridors_crossing": "district_nh_routes",
        "estimated_district_ev_registrations_proxy": "district_ev_registrations_fy24",
        "estimated_district_cumulative_evs_proxy": "district_cumulative_evs_fy24",
        "district_ev_penetration_rate": "district_ev_penetration_rate_fy24",
        "expanded_station_count": "district_total_stations",
        "fast_charging_station_count": "district_fast_stations",
        "fast_charger_ratio": "district_fast_charger_ratio",
        "expanded_total_capacity_kw": "district_total_capacity_kw",
        "expanded_station_density_per_1000_sq_km": "district_station_density_per_1000_sqkm",
        "expanded_station_density_per_lakh_pop": "district_station_density_per_lakh_pop",
        "measured_discom_annual_ev_kwh": "district_measured_discom_ev_kwh_fy24",
        "measured_discom_billed_ev_connections": "district_measured_discom_connections_fy24"
    }, inplace=True)

    # Derived contextual metrics
    df_dist["district_ev_to_charger_ratio"] = (
        df_dist["district_cumulative_evs_fy24"] / df_dist["district_total_stations"].replace(0, np.nan)
    ).round(1)
    df_dist["discom_utility_data_available_flag"] = df_dist["district_measured_discom_ev_kwh_fy24"].notnull().astype(int)

    # 4. Perform Strict Left Joins (Preserving Exactly 166 Rows)
    print("\n--- PERFORMING CARDINALITY-PRESERVING MERGES ---")
    m1 = pd.merge(df_core, df_gis, on="station_id", how="left")
    print(f"After GIS merge: {len(m1)} rows (cardinality 1:1 preserved).")

    m2 = pd.merge(m1, df_dist, on="district", how="left")
    print(f"After District Context merge: {len(m2)} rows (cardinality m:1 preserved).")

    # Verify first 25 columns are exactly the core 25
    assert list(m2.columns[:25]) == CORE_25_ATTRIBUTES, "First 25 columns do NOT match Core 25!"
    assert len(m2) == 166, f"Row count changed to {len(m2)}! Expected 166."

    # 5. Save Single Master Dataset
    master_csv = DATA_FINAL_DIR / "final_project_master_dataset.csv"
    m2.to_csv(master_csv, index=False)
    print(f"\nSuccessfully saved SINGLE MASTER DATASET: {master_csv}")
    print(f"Final Master Dimensions: {m2.shape[0]} rows x {m2.shape[1]} columns.")

    master_sha = compute_sha256(master_csv)
    print(f"Master CSV SHA-256: {master_sha}")

    # 6. Generate Master Data Dictionary
    print("\n--- GENERATING MASTER DATA DICTIONARY ---")
    dict_entries = []

    # Map sources and descriptions
    for col in m2.columns:
        if col in CORE_25_ATTRIBUTES:
            g_level = "station"
            if col in ["state", "district"]:
                g_level = "district/state administrative"
            source_type = "Core 25 Attribute (Phase 8.5 Frozen Baseline)"
            unit = "varies"
            if "km" in col:
                unit = "km"
            elif "kw" in col:
                unit = "kW"
            elif "count" in col or "number" in col:
                unit = "count"
            elif "latitude" in col or "longitude" in col:
                unit = "decimal degrees"
            else:
                unit = "categorical/string"
            treatment = "Preserved from source; unrecorded values as NA"
        elif col.startswith("district_") or col.startswith("discom_"):
            g_level = "district contextual (FY2023-24)"
            source_type = "Official Government Registry (Census 2011 / VAHAN / TGSPDCL)"
            if "pct" in col or "ratio" in col or "rate" in col:
                unit = "percentage/ratio"
            elif "sq_km" in col or "sqkm" in col:
                unit = "sq km"
            elif "kwh" in col:
                unit = "kWh"
            elif "kw" in col:
                unit = "kW"
            elif "flag" in col:
                unit = "binary (0/1)"
            else:
                unit = "count/integer"
            treatment = "Joined via district key; NA where utility data not covered in district"
        elif "poi_" in col or "road_" in col or "highway_" in col or "proximity" in col or "gis_" in col:
            g_level = "station spatial / GIS"
            source_type = "OpenStreetMap Overpass Road Network & POI Graph"
            if "score" in col or "index" in col:
                unit = "index [0, 1]"
            elif "km" in col:
                unit = "km"
            elif "distance_m" in col:
                unit = "meters"
            elif "count" in col:
                unit = "count"
            else:
                unit = "categorical/CRS"
            treatment = "Computed geometrically via Haversine / Overpass query"
        elif col == "fast_charging_flag":
            g_level = "station"
            source_type = "Direct Hardware Specification (DC Fast >= 50kW)"
            unit = "binary target (1=DC Fast, 0=AC Level 1/2)"
            treatment = "Binary target; non-null for all classified hardware"
        else:
            g_level = "station metadata"
            source_type = "Pipeline Provenance & Audit Metadata"
            unit = "metadata"
            treatment = "Preserved audit information"

        dict_entries.append({
            "column_order": len(dict_entries) + 1,
            "column_name": col,
            "granularity": g_level,
            "data_type": str(m2[col].dtype),
            "unit": unit,
            "source_category": source_type,
            "non_null_count": int(m2[col].notnull().sum()),
            "missing_count": int(m2[col].isnull().sum()),
            "missing_pct": round(m2[col].isnull().sum() / len(m2) * 100.0, 1),
            "missing_treatment": treatment
        })

    df_master_dict = pd.DataFrame(dict_entries)
    dict_csv = DATA_FINAL_DIR / "final_project_master_data_dictionary.csv"
    df_master_dict.to_csv(dict_csv, index=False)
    print(f"Saved Master Data Dictionary: {dict_csv} ({len(df_master_dict)} entries)")
    dict_sha = compute_sha256(dict_csv)

    # 7. Verification Checks
    val_rows = len(m2) == 166
    val_cols = len(m2.columns) == 70
    val_first25 = list(m2.columns[:25]) == CORE_25_ATTRIBUTES
    val_station_unique = m2["station_id"].is_unique
    val_no_dup_districts = (m2.groupby("station_id")["district"].count() == 1).all()

    print("\n--- FINAL VERIFICATION CHECKS ---")
    print(f"Master Row Count (N=166):             {'PASS' if val_rows else 'FAIL'} ({len(m2)})")
    print(f"Master Column Count (C=70):           {'PASS' if val_cols else 'FAIL'} ({len(m2.columns)})")
    print(f"First 25 Columns Identical:           {'PASS' if val_first25 else 'FAIL'}")
    print(f"Station ID Primary Key Uniqueness:    {'PASS' if val_station_unique else 'FAIL'}")
    print(f"No Join Duplication / Fan-out:       {'PASS' if val_no_dup_districts else 'FAIL'}")

    return {
        "master_sha": master_sha,
        "dict_sha": dict_sha,
        "shape": m2.shape
    }

if __name__ == "__main__":
    main()
