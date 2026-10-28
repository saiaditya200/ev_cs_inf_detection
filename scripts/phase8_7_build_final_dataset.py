#!/usr/bin/env python3
"""
Phase 8.7: Reframe and Finalize the Single Master Project Dataset
Explainable AI Framework for Smart EV Charging Infrastructure Planning

This script constructs the SOLE MASTER ANALYTICAL DATASET:
data/final/FINAL_PROJECT_DATASET.csv
and supporting documentation:
data/final/FINAL_PROJECT_DATA_DICTIONARY.csv
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
    print("=" * 75)
    print("PHASE 8.7: REFRAME AND FINALIZE SINGLE MASTER PROJECT DATASET")
    print("=" * 75)

    # 1. Load Frozen Core 25 Dataset
    core_csv = DATA_FINAL_DIR / "telangana_ev_charging_stations_25_attributes.csv"
    if not core_csv.exists():
        print(f"Error: Frozen core file {core_csv} missing.")
        sys.exit(1)
    df_core = pd.read_csv(core_csv)
    print(f"Loaded Frozen Core 25 Dataset: {len(df_core)} rows, {len(df_core.columns)} cols.")
    assert list(df_core.columns) == CORE_25_ATTRIBUTES, "First 25 columns do not match Core 25!"

    # 2. Load Station GIS Spatial Features from Expanded Master
    exp_csv = DATA_FINAL_DIR / "telangana_station_master_expanded.csv"
    df_exp = pd.read_csv(exp_csv)
    print(f"Loaded Expanded Master: {len(df_exp)} rows, {len(df_exp.columns)} cols.")

    gis_extra_cols = [
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
        "_nearest_road_name",
        "_nearest_road_osm_id",
        "_nearest_nh_osm_id",
        "state_key",
        "district_key",
        "fast_charging_flag",
        "operational_completeness_pct",
        "_data_source",
        "_source_id",
        "_expansion_batch"
    ]
    df_gis = df_exp[gis_extra_cols].copy()
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
        "_nearest_road_name": "nearest_road_name_osm",
        "_nearest_road_osm_id": "nearest_road_osm_id",
        "_nearest_nh_osm_id": "nearest_nh_osm_id",
        "_data_source": "source_registry",
        "_source_id": "source_registry_id",
        "_expansion_batch": "expansion_batch"
    }, inplace=True)

    # 3. Load District Contextual Features for FY2023-24 from District Year Panel
    panel_csv = DATA_FINAL_DIR / "district_year_panel.csv"
    df_panel = pd.read_csv(panel_csv)
    p24 = df_panel[df_panel["financial_year"] == "FY2023-24"].copy()

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
        "district_ev_penetration_rate": "district_ev_penetration_rate",
        "expanded_station_count": "district_total_stations",
        "fast_charging_station_count": "district_fast_stations",
        "fast_charger_ratio": "district_fast_charger_ratio",
        "expanded_total_capacity_kw": "district_total_capacity_kw",
        "expanded_station_density_per_1000_sq_km": "district_station_density_per_1000_sqkm",
        "expanded_station_density_per_lakh_pop": "district_station_density_per_lakh_pop",
        "measured_discom_annual_ev_kwh": "district_measured_discom_ev_kwh",
        "measured_discom_billed_ev_connections": "district_measured_discom_connections"
    }, inplace=True)

    df_dist["district_ev_to_charger_ratio"] = (
        df_dist["district_cumulative_evs_fy24"] / df_dist["district_total_stations"].replace(0, np.nan)
    ).round(1)
    df_dist["discom_utility_data_available_flag"] = df_dist["district_measured_discom_ev_kwh"].notnull().astype(int)

    # 4. Load District-Level Discom Financials and Load Details from TGSPDCL
    elec_csv = DATA_FINAL_DIR / "telangana_electricity_consumption.csv"
    df_elec = pd.read_csv(elec_csv)
    elec_dist = df_elec.groupby("district").agg({
        "billed_revenue_inr": "sum",
        "contracted_load_kw": "mean",
        "billed_peak_load_kw": "mean"
    }).reset_index().rename(columns={
        "billed_revenue_inr": "district_discom_billed_revenue_inr_fy24",
        "contracted_load_kw": "district_discom_contracted_load_mean_kw",
        "billed_peak_load_kw": "district_discom_billed_peak_load_mean_kw"
    })
    elec_dist["district_discom_contracted_load_mean_kw"] = elec_dist["district_discom_contracted_load_mean_kw"].round(1)
    elec_dist["district_discom_billed_peak_load_mean_kw"] = elec_dist["district_discom_billed_peak_load_mean_kw"].round(1)

    df_dist = pd.merge(df_dist, elec_dist, on="district", how="left")

    # 5. Strict Left Merges (Cardinality 1:1 station, m:1 district)
    print("\n--- PERFORMING CARDINALITY-PRESERVING MERGES ---")
    m1 = pd.merge(df_core, df_gis, on="station_id", how="left")
    print(f"After GIS Merge: {len(m1)} rows (1:1 preserved).")

    m2 = pd.merge(m1, df_dist, on="district", how="left")
    print(f"After District Merge: {len(m2)} rows (m:1 preserved).")

    # Verify first 25 columns match Core 25 exactly
    assert list(m2.columns[:25]) == CORE_25_ATTRIBUTES, "Core 25 column order mismatch!"
    assert len(m2) == 166, f"Row count changed to {len(m2)}! Expected 166."

    # 6. Save FINAL_PROJECT_DATASET.csv
    final_csv = DATA_FINAL_DIR / "FINAL_PROJECT_DATASET.csv"
    m2.to_csv(final_csv, index=False)
    print(f"\nSuccessfully created FINAL SINGLE PROJECT DATASET: {final_csv}")
    print(f"Dimensions: {m2.shape[0]} rows x {m2.shape[1]} columns.")

    final_sha = compute_sha256(final_csv)
    print(f"SHA-256 Checksum: {final_sha}")

    # 7. Generate Master Data Dictionary
    print("\n--- GENERATING FINAL_PROJECT_DATA_DICTIONARY.CSV ---")
    dict_rows = []

    for idx, col in enumerate(m2.columns, 1):
        # Description, granularity, source, derivation, role, target_or_predictor, leakage
        col_type = str(m2[col].dtype)
        missing_count = int(m2[col].isnull().sum())
        missing_pct = round(missing_count / len(m2) * 100.0, 1)

        # Classify attributes
        if col == "station_id":
            desc = "Unique stable identifier for each physical EV charging station facility"
            unit = "identifier string"
            gran = "station"
            src = "OCM / OSM / TGREDCO Registry"
            deriv = "Canonical project ID generator (TS-{DIST}-{NUM})"
            role = "station_identifier"
            tgt_pred = "IDENTIFIER"
            leak = "EXCLUDED_FROM_ML_TRAINING"
            notes = "Unique primary key; exactly 166 unique physical locations"

        elif col == "station_name":
            desc = "Verified commercial facility or host institution site title"
            unit = "text label"
            gran = "station"
            src = "OCM / OSM / TGREDCO Registry"
            deriv = "Directly extracted from verified public registry"
            role = "station_label"
            tgt_pred = "METADATA"
            leak = "EXCLUDED_FROM_ML_TRAINING"
            notes = "Human-readable station label"

        elif col == "state":
            desc = "Indian State administrative jurisdiction"
            unit = "administrative string"
            gran = "state"
            src = "Government Administrative Boundary"
            deriv = "Fixed administrative constant ('Telangana')"
            role = "state_constant"
            tgt_pred = "METADATA"
            leak = "ZERO_VARIANCE_EXCLUDED"
            notes = "All 166 stations located within Telangana"

        elif col == "district":
            desc = "Harmonized Telangana district conforming to 33-district reorganization"
            unit = "administrative string"
            gran = "district"
            src = "Census of India 2011 / Government of Telangana"
            deriv = "Spatial point-in-polygon alignment with Census 33-district boundary"
            role = "district_spatial_key"
            tgt_pred = "PREDICTOR_OR_GROUP"
            leak = "CLEAN_PREDICTOR"
            notes = "Used as spatial grouping key for cross-validation and hierarchical modeling"

        elif col == "city":
            desc = "City, municipal town, or local village name of the station location"
            unit = "locality string"
            gran = "station"
            src = "OCM / OSM / TGREDCO Registry"
            deriv = "Directly extracted from registry town/address metadata"
            role = "locality_label"
            tgt_pred = "METADATA"
            leak = "CLEAN_METADATA"
            notes = "Missing for 4 rural highway nodes; not fabricated"

        elif col in ["latitude", "longitude"]:
            desc = f"WGS-84 geographic coordinate ({col})"
            unit = "decimal degrees"
            gran = "station"
            src = "OCM / OSM / TGREDCO GPS Records"
            deriv = "Direct GPS reading validated against Telangana spatial bounding box"
            role = "geographic_coordinate"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "Latitude bounded in [15.8, 19.9] N; Longitude in [77.2, 81.8] E"

        elif col == "pincode":
            desc = "Six-digit Indian Postal Index Number"
            unit = "PIN code"
            gran = "station"
            src = "OCM / Postal Metadata"
            deriv = "Directly sourced from address metadata where available"
            role = "postal_location"
            tgt_pred = "METADATA"
            leak = "HIGH_MISSINGNESS_EXCLUDED"
            notes = "Available for 13 baseline stations; unrecorded in highway/GPS nodes"

        elif col == "location_type":
            desc = "Functional category of the host physical venue"
            unit = "categorical string"
            gran = "station"
            src = "TSREDCO / FAME-II Directory"
            deriv = "Classified from host premises (public_transit_hub, government_office)"
            role = "venue_classification"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "51 transit hubs, 90 government complexes, 25 NA"

        elif col in ["charger_type", "connector_type", "charging_power_kw", "station_capacity"]:
            desc = f"Physical equipment hardware specification ({col})"
            unit = "kW" if "kw" in col or "capacity" in col else "categorical standard"
            gran = "station"
            src = "OCM / TSREDCO / FAME-II Equipment Directory"
            deriv = "Directly reported equipment specification from commissioning records"
            role = "core_equipment_attribute"
            tgt_pred = "POST_OUTCOME_EQUIPMENT"
            leak = "LEAKAGE_FOR_FAST_CHARGING_TARGET"
            notes = f"Defines or directly correlates with fast_charging_flag. MUST BE EXCLUDED from feature matrix X when training fast_charging_flag classifier to prevent identity leakage!"

        elif col == "number_of_chargers":
            desc = "Number of active physical charging dispensing bays or guns"
            unit = "count"
            gran = "station"
            src = "OCM / TSREDCO Equipment Directory"
            deriv = "Directly reported dispenser count"
            role = "facility_scale"
            tgt_pred = "POST_OUTCOME_EQUIPMENT"
            leak = "LEAKAGE_FOR_SCALE_TARGET"
            notes = "Range: 1 to 7 bays; excluded from spatial siting prediction"

        elif col == "availability_status":
            desc = "Verified operational state of the charging station"
            unit = "status string"
            gran = "station"
            src = "OCM / TSREDCO Operational Audit"
            deriv = "Directly verified operational state ('OPERATIONAL')"
            role = "operational_status"
            tgt_pred = "METADATA"
            leak = "ZERO_VARIANCE_EXCLUDED"
            notes = "All 166 stations verified operational"

        elif col == "operator":
            desc = "Commercial Charge Point Operator (CPO) or public agency"
            unit = "organization string"
            gran = "station"
            src = "OCM / OSM / TSREDCO Registry"
            deriv = "Directly extracted from registry metadata"
            role = "network_operator"
            tgt_pred = "PREDICTOR_CATEGORICAL"
            leak = "CLEAN_PREDICTOR"
            notes = "TGREDCO, REIL, EESL, NTPC, private CPOs; 12 NA"

        elif col == "24x7_availability":
            desc = "Indicator of whether the facility is publicly accessible 24/7"
            unit = "indicator ('YES'/NA)"
            gran = "station"
            src = "FAME-II Public Operational Guidelines"
            deriv = "Verified against public mandate; NA for unrecorded baseline"
            role = "accessibility_timing"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "'YES' for 141 mandated public stations; 25 NA"

        elif col == "road_type":
            desc = "OpenStreetMap functional road hierarchy classification"
            unit = "road classification"
            gran = "station"
            src = "OpenStreetMap Overpass Road Graph"
            deriv = "Spatial nearest-neighbor match to OSM road geometries"
            role = "spatial_road_accessibility"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "131 trunk, 13 residential, 8 primary, 5 motorway, etc."

        elif col == "highway_name":
            desc = "Designated National Highway corridor reference"
            unit = "highway reference"
            gran = "station"
            src = "MoRTH / NHAI / OSM Highway Alignments"
            deriv = "Spatial proximity match to nearest of 6 designated NH corridors"
            role = "corridor_connectivity"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "NH 65, NH 44, NH 163, NH 167, NH 765, NH 161"

        elif col == "distance_to_highway_km":
            desc = "Great-circle distance to nearest National Highway centerline"
            unit = "kilometers"
            gran = "station"
            src = "GIS Road Network Spatial Analysis"
            deriv = "Haversine distance from station to nearest OSM highway vertex"
            role = "highway_proximity"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "Median: 0.79 km, Min: 0.00 km, Max: 23.15 km"

        elif col == "distance_to_city_center_km":
            desc = "Great-circle distance to District Headquarters administrative centroid"
            unit = "kilometers"
            gran = "station"
            src = "GIS Administrative Spatial Analysis"
            deriv = "Haversine distance from station to verified district HQ coordinates"
            role = "urban_center_proximity"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "Median: 9.3 km"

        elif col == "traffic_density":
            desc = "Vehicle traffic flow density or Average Annual Daily Traffic (AADT)"
            unit = "PCU/day"
            gran = "station"
            src = "Unavailable (No open government feed)"
            deriv = "Strictly NOT derived; kept as NA per anti-fabrication mandate"
            role = "traffic_flow"
            tgt_pred = "UNAVAILABLE"
            leak = "NOT_APPLICABLE"
            notes = "Preserved as 100% NA; zero synthetic fabrication"

        elif col == "nearby_poi_count":
            desc = "Total count of points of interest within 1,000m radius"
            unit = "count"
            gran = "station"
            src = "OpenStreetMap Overpass POI Graph"
            deriv = "Spatial query within 1km buffer radius"
            role = "local_activity_density"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "Median: 32 POIs, Max: 354 POIs"

        elif col == "parking_availability":
            desc = "Presence of designated vehicular parking bays at the facility"
            unit = "indicator ('YES'/NA)"
            gran = "station"
            src = "TSREDCO / FAME-II Station Site Plans"
            deriv = "Verified dedicated parking bays from transit/depot site plans"
            role = "parking_infrastructure"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "'YES' for 141 transit/depot sites; 25 NA"

        elif col == "data_timestamp":
            desc = "ISO 8601 timestamp of data collection or verification snapshot"
            unit = "ISO 8601 string"
            gran = "station"
            src = "Pipeline Audit Metadata"
            deriv = "Preserved collection chronology"
            role = "audit_timestamp"
            tgt_pred = "METADATA"
            leak = "EXCLUDED_FROM_ML_TRAINING"
            notes = "Audit trail timestamp"

        elif col in ["highway_proximity_score", "city_center_proximity_score"]:
            desc = f"Exponential spatial decay proximity score ({col})"
            unit = "index [0, 1]"
            gran = "station"
            src = "GIS Spatial Derivation"
            deriv = "exp(-0.5 * d_hwy) or exp(-0.1 * d_city)"
            role = "spatial_accessibility_score"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "Continuous spatial proximity index"

        elif col == "poi_diversity_index":
            desc = "Normalized Shannon entropy of local POI category distribution"
            unit = "entropy index [0, 1]"
            gran = "station"
            src = "GIS Spatial Analysis"
            deriv = "Normalized Shannon entropy over 9 POI categories within 1km"
            role = "land_use_mix"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "Higher values indicate diverse mixed-use trip generator environment"

        elif col in ["distance_to_major_road_km", "nearest_major_road_type", "nearest_road_distance_m"]:
            desc = f"Micro-level road accessibility indicator ({col})"
            unit = "km" if "km" in col else "meters" if "m" in col else "category"
            gran = "station"
            src = "OpenStreetMap Overpass Road Graph"
            deriv = "Spatial nearest-neighbor query against OSM ways"
            role = "road_network_accessibility"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "Available for baseline stations; NA where detailed graph not queried"

        elif col.startswith("poi_") and col.endswith("_count"):
            desc = f"Category-specific POI count within 1km ({col})"
            unit = "count"
            gran = "station"
            src = "OpenStreetMap Overpass POI Graph"
            deriv = "Filtered spatial count within 1km buffer"
            role = "activity_generator_subcount"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "Deep POI category breakdown"

        elif col in ["gis_search_radius_m", "gis_projection_crs", "nearest_highway_name_osm", "nearest_road_name_osm", "nearest_road_osm_id", "nearest_nh_osm_id"]:
            desc = f"GIS spatial reference or feature tag ({col})"
            unit = "metadata"
            gran = "station"
            src = "OpenStreetMap / EPSG:32644 Projection"
            deriv = "Spatial pipeline reference metadata"
            role = "gis_spatial_metadata"
            tgt_pred = "METADATA"
            leak = "EXCLUDED_FROM_ML_TRAINING"
            notes = "Preserves geometric audit trail"

        elif col in ["state_key", "district_key"]:
            desc = f"Canonical lowercase slug key for joins ({col})"
            unit = "slug key"
            gran = "administrative"
            src = "Data Engineering Pipeline"
            deriv = "slugify(state) or slugify(district)"
            role = "join_key"
            tgt_pred = "METADATA"
            leak = "EXCLUDED_FROM_ML_TRAINING"
            notes = "Join key for merging district and state data"

        elif col == "fast_charging_flag":
            desc = "Primary Station-Level Supervised ML Target: DC Fast Charging (>=50kW) vs AC"
            unit = "binary target (1=DC Fast, 0=AC)"
            gran = "station"
            src = "Hardware Specification Verification"
            deriv = "1 if charging_power_kw >= 50 or charger_type == 'DC_FAST' else 0"
            role = "station_level_target"
            tgt_pred = "TARGET"
            leak = "PRIMARY_ML_TARGET"
            notes = "Primary classification target for Phase 9 spatial siting models (131 DC Fast, 35 AC)"

        elif col == "operational_completeness_pct":
            desc = "Data completeness audit percentage across core attributes"
            unit = "percentage [0, 100]"
            gran = "station"
            src = "Data Quality Audit Engine"
            deriv = "Percentage of non-null core fields"
            role = "quality_audit_score"
            tgt_pred = "METADATA"
            leak = "EXCLUDED_FROM_ML_TRAINING"
            notes = "Mean: 96.8%; data hygiene audit metric"

        elif col in ["source_registry", "source_registry_id", "expansion_batch"]:
            desc = f"Data collection pipeline provenance field ({col})"
            unit = "provenance text"
            gran = "station"
            src = "Pipeline Provenance Tracker"
            deriv = "Preserved from ingestion batches"
            role = "pipeline_provenance"
            tgt_pred = "METADATA"
            leak = "EXCLUDED_FROM_ML_TRAINING"
            notes = "Preserves original registry traceability"

        elif col.startswith("district_population_") or col in ["district_area_sq_km", "district_urban_population_pct", "district_urban_tier", "district_projected_population_fy24", "district_projected_density_fy24"]:
            desc = f"District-level demographic and urbanization context ({col})"
            unit = "persons" if "pop" in col and "pct" not in col and "density" not in col else "persons/sq km" if "density" in col else "sq km" if "area" in col else "percentage" if "pct" in col else "tier"
            gran = "district contextual (FY2023-24)"
            src = "Census of India 2011 / DES Telangana / Projections"
            deriv = "Official Census re-mapped to post-2016 33 districts; geometric projections for FY24"
            role = "district_demographic_context"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "Attached as district contextual feature; not station-level measurement"

        elif col in ["district_nh_crossing_count", "district_nh_routes"]:
            desc = f"District-level National Highway infrastructure connectivity ({col})"
            unit = "count" if "count" in col else "routes list"
            gran = "district contextual"
            src = "MoRTH / NHAI / OSM Corridor Graph"
            deriv = "Spatial overlay of 6 priority NH corridors with district boundaries"
            role = "district_highway_connectivity"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "Captures inter-district transit flow exposure"

        elif col in ["district_ev_registrations_fy24", "district_cumulative_evs_fy24"]:
            desc = f"District-level EV vehicle adoption volume for FY2023-24 ({col})"
            unit = "registered vehicles"
            gran = "district contextual (FY2023-24)"
            src = "MoRTH / VAHAN National Dashboard"
            deriv = "Scaled from official Telangana VAHAN totals using district economic weights"
            role = "district_ev_fleet_context"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "Macro vehicle electrification volume in district"

        elif col == "district_ev_penetration_rate":
            desc = "District-Level Contextual Target: Estimated EV adoption rate per 1,000 population"
            unit = "EVs per 1,000 residents"
            gran = "district contextual (FY2023-24)"
            src = "VAHAN / Census Demographic Projections"
            deriv = "(cumulative_evs / projected_population) * 1000"
            role = "district_adoption_target"
            tgt_pred = "TARGET"
            leak = "DISTRICT_LEVEL_TARGET"
            notes = "Contextual adoption target for macro policy and regional infrastructure allocation"

        elif col in ["district_total_stations", "district_fast_stations", "district_fast_charger_ratio", "district_total_capacity_kw", "district_station_density_per_1000_sqkm", "district_station_density_per_lakh_pop", "district_ev_to_charger_ratio"]:
            desc = f"District-level existing charging infrastructure density metric ({col})"
            unit = "count" if "count" in col or "stations" in col else "ratio" if "ratio" in col else "kW" if "kw" in col else "stations/unit"
            gran = "district contextual"
            src = "Master Charging Station Aggregation"
            deriv = "Spatial aggregation of 166 verified stations across 33 districts"
            role = "district_infrastructure_density"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "Measures existing charging supply and infrastructure desert intensity"

        elif col == "district_measured_discom_ev_kwh":
            desc = "DISCOM-Level Contextual Target: Official revenue-metered electricity consumed by EV charging (kWh)"
            unit = "Kilowatt-hours (kWh)"
            gran = "DISCOM / District contextual (FY2023-24)"
            src = "TGSPDCL / AIKosh / MeitY Official Ledger"
            deriv = "Annual sum of LT-IX monthly utility meter readings across section feeders"
            role = "discom_grid_load_target"
            tgt_pred = "TARGET"
            leak = "DISCOM_LEVEL_TARGET"
            notes = "Metered utility grid loading target; strictly preserved at district scale; NOT disaggregated to fake station kWh"

        elif col in ["district_measured_discom_connections", "discom_utility_data_available_flag", "district_discom_billed_revenue_inr_fy24", "district_discom_contracted_load_mean_kw", "district_discom_billed_peak_load_mean_kw"]:
            desc = f"District-level electricity grid and utility operational indicator ({col})"
            unit = "connections" if "connections" in col else "flag (0/1)" if "flag" in col else "INR" if "inr" in col else "kW"
            gran = "DISCOM / District contextual (FY2023-24)"
            src = "TGSPDCL Official LT-IX Ledger"
            deriv = "Aggregated from utility billing records across Discom sections"
            role = "district_grid_context"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "Real utility grid infrastructure metrics; NA for districts outside TGSPDCL service area"

        else:
            desc = f"Project feature ({col})"
            unit = "varies"
            gran = "station"
            src = "Project Pipeline"
            deriv = "Directly sourced or derived"
            role = "general_feature"
            tgt_pred = "PREDICTOR"
            leak = "CLEAN_PREDICTOR"
            notes = "Standard feature"

        dict_rows.append({
            "column_name": col,
            "column_index": idx,
            "description": desc,
            "datatype": col_type,
            "unit": unit,
            "granularity": gran,
            "source": src,
            "derivation_method": deriv,
            "analytical_role": role,
            "target_or_predictor": tgt_pred,
            "missing_percentage": missing_pct,
            "leakage_status": leak,
            "notes": notes
        })

    df_final_dict = pd.DataFrame(dict_rows)
    dict_csv = DATA_FINAL_DIR / "FINAL_PROJECT_DATA_DICTIONARY.csv"
    df_final_dict.to_csv(dict_csv, index=False)
    print(f"Saved Master Data Dictionary: {dict_csv} ({len(df_final_dict)} rows)")
    dict_sha = compute_sha256(dict_csv)
    print(f"Data Dictionary SHA-256: {dict_sha}")

    # 8. Final Comprehensive Validation
    print("\n--- RUNNING COMPREHENSIVE VALIDATION CHECKS ---")
    val_row_count = (len(m2) == 166)
    val_first25 = (list(m2.columns[:25]) == CORE_25_ATTRIBUTES)
    val_station_unique = m2["station_id"].is_unique
    val_coords_valid = (
        (m2["latitude"].between(15.8, 19.9)).all() and
        (m2["longitude"].between(77.2, 81.8)).all()
    )
    val_districts_complete = (m2["district"].nunique() == 33)
    val_distances_nonneg = (
        (m2["distance_to_highway_km"] >= 0.0).all() and
        (m2["distance_to_city_center_km"] >= 0.0).all()
    )
    val_no_dup_merges = (m2.groupby("station_id")["district"].count() == 1).all()

    print(f"Row count = 166:               {'PASS' if val_row_count else 'FAIL'} ({len(m2)})")
    print(f"First 25 columns match:        {'PASS' if val_first25 else 'FAIL'}")
    print(f"Station ID uniqueness:         {'PASS' if val_station_unique else 'FAIL'}")
    print(f"Coordinate bounding box:       {'PASS' if val_coords_valid else 'FAIL'}")
    print(f"33-District completeness:      {'PASS' if val_districts_complete else 'FAIL'} ({m2['district'].nunique()}/33)")
    print(f"Non-negative distances:        {'PASS' if val_distances_nonneg else 'FAIL'}")
    print(f"Zero Cartesian fan-out:        {'PASS' if val_no_dup_merges else 'FAIL'}")

    return {
        "final_sha": final_sha,
        "dict_sha": dict_sha,
        "shape": m2.shape
    }

if __name__ == "__main__":
    main()
