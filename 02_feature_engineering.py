#!/usr/bin/env python3
"""
=============================================================================
Step 2: Feature Engineering, Leakage Control & Certified Spatial Partition
=============================================================================
Project: Explainable AI Framework for Smart EV Charging Infrastructure Planning
Domain:  Telangana, India (33 Administrative Districts)

Key Functions:
- Enforces Hard-Coded Leakage Control Gate:
  * Prohibits direct charger specifications (power, type, capacity, connectors, count)
  * Prohibits operator and station provenance IDs
  * Prohibits target-derived district aggregates (fast_stations, fast_charger_ratio, etc.)
  * Programmatically validates zero leakage before matrix construction.
- Extracts the Certified 21 Exogenous Predictors:
  * 8 Spatial/GIS: latitude, longitude, distance_to_highway_km,
    distance_to_city_center_km, nearby_poi_count, highway_proximity_score,
    city_center_proximity_score, poi_diversity_index
  * 7 Road One-Hot Indicators: motorway, primary, residential, secondary, service, trunk, unclassified
  * 6 Macro District Context: pop_den11, urb_pct, proj_den24, nh_cross, ev_pen, discom_flag
- Enforces Frozen Certified Spatial Partition Gate:
  * Loads certified assignments from data/processed/phase10_final_split_assignments.csv
  * Strictly verifies:
      development_count == 119
      outer_test_count == 35
      development_district_count == 25
      outer_test_district_count == 8
      district_overlap == 0
      outer_class_0 == 9
      outer_class_1 == 26
  * Stops immediately if any condition is violated.
  * Persists results/metrics/spatial_split_verification.json.
- Keeps features in natural unscaled units (tree ensembles are scale-invariant).
- Persists:
  * data/processed/fast_charging_modeling_matrix.csv (154 x 24)
  * data/processed/phase10_development.csv (119 stations)
  * data/processed/phase10_outer_test.csv (35 stations)
  * models/feature_manifest.json
=============================================================================
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ========================= CONFIGURATION =========================
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"

for d in [DATA_PROCESSED_DIR, MODELS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 21 Certified Non-Leaky Predictors
SPATIAL_GIS_FEATURES = [
    "latitude",
    "longitude",
    "distance_to_highway_km",
    "distance_to_city_center_km",
    "nearby_poi_count",
    "highway_proximity_score",
    "city_center_proximity_score",
    "poi_diversity_index"
]

ROAD_ONE_HOT_FEATURES = [
    "road_type_motorway",
    "road_type_primary",
    "road_type_residential",
    "road_type_secondary",
    "road_type_service",
    "road_type_trunk",
    "road_type_unclassified"
]

DISTRICT_CONTEXT_FEATURES = [
    "district_population_density_2011",
    "district_urban_population_pct",
    "district_projected_density_fy24",
    "district_nh_crossing_count",
    "district_ev_penetration_rate",
    "discom_utility_data_available_flag"
]

CERTIFIED_21_PREDICTORS = SPATIAL_GIS_FEATURES + ROAD_ONE_HOT_FEATURES + DISTRICT_CONTEXT_FEATURES

# Strict Leakage Blocklist (exact terms or specific target-derived prefixes)
EXACT_LEAKAGE_BLOCKLIST = [
    "charger_type",
    "charging_power_kw",
    "station_capacity",
    "connector_type",
    "number_of_chargers",
    "operator",
    "district_fast_stations",
    "district_fast_charger_ratio",
    "district_total_stations",
    "district_total_capacity_kw",
    "district_station_density_per_1000_sqkm",
    "district_station_density_per_lakh_pop",
    "district_ev_to_charger_ratio",
    "_provenance",
    "data_source",
    "station_name",
    "address",
    "city",
    "state",
    "country",
    "fast_charging_flag",
    "traffic_density"
]


def load_and_validate_preprocessed():
    print("=" * 80)
    print("[INFO] STEP 2: STARTING FEATURE ENGINEERING & LEAKAGE CONTROL")
    print("=" * 80)

    preprocessed_path = DATA_PROCESSED_DIR / "telangana_ev_preprocessed.csv"
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"[ERROR] Preprocessed dataset missing at {preprocessed_path}. Run 01_data_preprocessing.py first.")

    df = pd.read_csv(preprocessed_path)
    print(f"[INFO] Loaded preprocessed records: {len(df)} stations x {len(df.columns)} columns")

    # Exclude the 12 unlabelled/held-out stations from the supervised matrix
    df_labeled = df[df["fast_charging_flag"].notna()].copy()
    df_labeled["fast_charging_flag"] = df_labeled["fast_charging_flag"].astype(int)

    n_labeled = len(df_labeled)
    c1 = (df_labeled["fast_charging_flag"] == 1).sum()
    c0 = (df_labeled["fast_charging_flag"] == 0).sum()

    print(f"[INFO] Supervised Labeled Stations: {n_labeled} ({c1} Class 1 DC Fast, {c0} Class 0 AC Slow)")
    if n_labeled != 154 or c1 != 131 or c0 != 23:
        raise ValueError(f"[ERROR] Expected exactly 154 labeled stations (131 Class 1, 23 Class 0). Found: {n_labeled} ({c1}/{c0}).")

    print("[OK] Supervised target universe confirmed: 154 records.")
    return df_labeled


def enforce_leakage_control_gate(candidate_features):
    print("\n" + "=" * 80)
    print("[INFO] ENFORCING PROGRAMMATIC LEAKAGE CONTROL GATE")
    print("=" * 80)

    leaks = []
    for feat in candidate_features:
        feat_clean = feat.lower().strip()
        # 1. Exact match on blocked feature
        if feat_clean in [b.lower() for b in EXACT_LEAKAGE_BLOCKLIST]:
            leaks.append((feat, f"Exact match on blocked feature '{feat_clean}'"))
        # 2. Contains direct charger/capacity/operator terms
        for term in ["charger_type", "charging_power", "connector_type", "number_of_chargers", "fast_charger_ratio", "fast_stations"]:
            if term in feat_clean:
                leaks.append((feat, f"Contains prohibited equipment term '{term}'"))

    if leaks:
        print("[ERROR] LEAKAGE DETECTED! The following features violate the leakage gate:")
        for feat, rule in leaks:
            print(f"  * Prohibited feature: '{feat}' triggered by: {rule}")
        raise RuntimeError("[ERROR] Leakage gate tripped. Supervised matrix construction halted.")

    print(f"[OK] Leakage Control Gate passed: 0 leakage features detected across {len(candidate_features)} candidates.")


def construct_feature_matrix(df):
    print("\n" + "=" * 80)
    print("[INFO] EXTRACTING 21 CERTIFIED EXOGENOUS PREDICTORS")
    print("=" * 80)

    # 1. Ensure Road-Type One-Hot Encodings are present
    if "road_type_primary" not in df.columns:
        print("[INFO] Generating one-hot indicators for road_type...")
        road_dummies = pd.get_dummies(df["road_type"].fillna("unclassified"), prefix="road_type", dtype=int)
        for cat in ["motorway", "primary", "residential", "secondary", "service", "trunk", "unclassified"]:
            col_name = f"road_type_{cat}"
            if col_name not in road_dummies.columns:
                road_dummies[col_name] = 0
            df[col_name] = road_dummies[col_name]
    else:
        for col in ROAD_ONE_HOT_FEATURES:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0).astype(int)

    # 2. Derive / verify proximity scores
    if "highway_proximity_score" not in df.columns or df["highway_proximity_score"].isna().any():
        df["highway_proximity_score"] = 1.0 / (1.0 + df["distance_to_highway_km"])
    if "city_center_proximity_score" not in df.columns or df["city_center_proximity_score"].isna().any():
        df["city_center_proximity_score"] = 1.0 / (1.0 + df["distance_to_city_center_km"])

    # 3. Verify all 21 certified predictors exist
    missing_preds = [f for f in CERTIFIED_21_PREDICTORS if f not in df.columns]
    if missing_preds:
        raise ValueError(f"[ERROR] Missing certified predictors in dataset: {missing_preds}")

    # 4. Check for missing values in predictor columns
    null_counts = df[CERTIFIED_21_PREDICTORS].isnull().sum()
    null_features = null_counts[null_counts > 0]
    if len(null_features) > 0:
        raise ValueError(f"[ERROR] Null values found in predictors:\n{null_features}")

    # Enforce Leakage Gate
    enforce_leakage_control_gate(CERTIFIED_21_PREDICTORS)

    # Assemble modeling matrix
    cols_order = ["station_id", "district", "fast_charging_flag"] + CERTIFIED_21_PREDICTORS
    df_matrix = df[cols_order].copy()

    out_matrix_path = DATA_PROCESSED_DIR / "fast_charging_modeling_matrix.csv"
    df_matrix.to_csv(out_matrix_path, index=False)
    print(f"[OK] Modeling matrix saved: {out_matrix_path} ({df_matrix.shape[0]} rows x {df_matrix.shape[1]} columns)")

    return df_matrix


def enforce_frozen_spatial_split_gate(df_matrix):
    print("\n" + "=" * 80)
    print("[INFO] ENFORCING FROZEN CERTIFIED SPATIAL SPLIT GATE")
    print("=" * 80)

    split_assign_path = DATA_PROCESSED_DIR / "phase10_final_split_assignments.csv"
    if not split_assign_path.exists():
        raise FileNotFoundError(f"[ERROR] Certified split artifact missing at {split_assign_path}!")

    df_assign = pd.read_csv(split_assign_path)
    print(f"[INFO] Loaded certified split assignments: {len(df_assign)} stations")

    # Merge certified partition onto current modeling matrix by station_id
    df_merged = pd.merge(
        df_matrix,
        df_assign[["station_id", "split_role", "cv_fold"]],
        on="station_id",
        how="inner"
    )

    if len(df_merged) != 154:
        raise ValueError(f"[ERROR] Station ID mismatch: expected 154 matches, got {len(df_merged)}")

    # Standardize split_role
    df_merged["split_role"] = df_merged["split_role"].str.upper()

    df_dev = df_merged[df_merged["split_role"] == "DEVELOPMENT"].copy()
    df_outer = df_merged[df_merged["split_role"] == "OUTER_TEST"].copy()

    # Verification checks
    dev_count = len(df_dev)
    outer_count = len(df_outer)
    dev_districts = sorted(df_dev["district"].unique())
    outer_districts = sorted(df_outer["district"].unique())
    dev_dist_count = len(dev_districts)
    outer_dist_count = len(outer_districts)
    overlap = set(dev_districts).intersection(set(outer_districts))
    district_overlap = len(overlap)

    outer_c0 = int((df_outer["fast_charging_flag"] == 0).sum())
    outer_c1 = int((df_outer["fast_charging_flag"] == 1).sum())
    dev_c0 = int((df_dev["fast_charging_flag"] == 0).sum())
    dev_c1 = int((df_dev["fast_charging_flag"] == 1).sum())

    print(f"[INFO] Spatial Partition Summary:")
    print(f"       Development Set: {dev_count} stations across {dev_dist_count} districts (Class 0: {dev_c0}, Class 1: {dev_c1})")
    print(f"       Outer Test Set:  {outer_count} stations across {outer_dist_count} districts (Class 0: {outer_c0}, Class 1: {outer_c1})")
    print(f"       District Overlap: {district_overlap}")

    # HARD VALIDATION GATE:
    failures = []
    if dev_count != 119:
        failures.append(f"development_count ({dev_count}) != 119")
    if outer_count != 35:
        failures.append(f"outer_test_count ({outer_count}) != 35")
    if dev_dist_count != 25:
        failures.append(f"development_district_count ({dev_dist_count}) != 25")
    if outer_dist_count != 8:
        failures.append(f"outer_test_district_count ({outer_dist_count}) != 8")
    if district_overlap != 0:
        failures.append(f"district_overlap ({district_overlap}) != 0; overlapping: {overlap}")
    if outer_c0 != 9:
        failures.append(f"outer_class_0 ({outer_c0}) != 9")
    if outer_c1 != 26:
        failures.append(f"outer_class_1 ({outer_c1}) != 26")

    if failures:
        print("\n" + "!" * 80)
        print("[ERROR] Certified spatial split integrity check failed.")
        for fail in failures:
            print(f"  * VIOLATION: {fail}")
        print("!" * 80 + "\n")
        raise RuntimeError("[ERROR] Certified spatial split integrity check failed. Halting pipeline.")

    print("\n[OK] ALL SPATIAL SPLIT INTEGRITY CONDITIONS VERIFIED SUCCESSFULLY:")
    print("  * development_count == 119 [VERIFIED]")
    print("  * outer_test_count == 35 [VERIFIED]")
    print("  * development_district_count == 25 [VERIFIED]")
    print("  * outer_test_district_count == 8 [VERIFIED]")
    print("  * district_overlap == 0 [VERIFIED]")
    print("  * outer_class_0 == 9 [VERIFIED]")
    print("  * outer_class_1 == 26 [VERIFIED]")

    # Save verification JSON artifact
    verification_data = {
        "status": "PASSED",
        "development_count": dev_count,
        "outer_test_count": outer_count,
        "development_district_count": dev_dist_count,
        "outer_test_district_count": outer_dist_count,
        "district_overlap": district_overlap,
        "outer_class_0": outer_c0,
        "outer_class_1": outer_c1,
        "dev_class_0": dev_c0,
        "dev_class_1": dev_c1,
        "outer_districts": outer_districts,
        "development_districts": dev_districts
    }
    split_verif_path = METRICS_DIR / "spatial_split_verification.json"
    with open(split_verif_path, "w", encoding="utf-8") as f:
        json.dump(verification_data, f, indent=2)
    print(f"[OK] Spatial split verification saved: {split_verif_path}")

    # Persist development and outer test sets
    dev_path = DATA_PROCESSED_DIR / "phase10_development.csv"
    outer_path = DATA_PROCESSED_DIR / "phase10_outer_test.csv"
    df_dev.to_csv(dev_path, index=False)
    df_outer.to_csv(outer_path, index=False)
    print(f"[OK] Saved development set: {dev_path} ({len(df_dev)} rows)")
    print(f"[OK] Saved outer test set:  {outer_path} ({len(df_outer)} rows)")

    # Save Feature Manifest (No Scaler required for Tree Ensembles)
    manifest = {
        "n_features": len(CERTIFIED_21_PREDICTORS),
        "target": "fast_charging_flag",
        "target_mapping": {"1": "DC Fast (>= 50kW)", "0": "AC Slow (< 50kW)"},
        "spatial_gis_features_8": SPATIAL_GIS_FEATURES,
        "road_one_hot_features_7": ROAD_ONE_HOT_FEATURES,
        "district_context_features_6": DISTRICT_CONTEXT_FEATURES,
        "certified_21_predictors": CERTIFIED_21_PREDICTORS,
        "scaling_applied": False,
        "scaling_note": "Tree ensembles (Random Forest, XGBoost, LightGBM) are scale-invariant. Features kept in natural units.",
        "outer_test_districts": outer_districts,
        "development_districts": dev_districts,
        "outer_test_size": outer_count,
        "development_size": dev_count,
        "total_class_0": dev_c0 + outer_c0,
        "total_class_1": dev_c1 + outer_c1,
        "class_imbalance_ratio": round((dev_c1 + outer_c1) / (dev_c0 + outer_c0), 2)
    }

    manifest_path = MODELS_DIR / "feature_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Feature manifest saved: {manifest_path}")

    # Remove obsolete scaler if it exists from previous run
    old_scaler = MODELS_DIR / "scaler.joblib"
    if old_scaler.exists():
        os.remove(old_scaler)
        print(f"[INFO] Removed obsolete {old_scaler} (Tree models use raw unscaled features)")

    return df_dev, df_outer


def main():
    df_labeled = load_and_validate_preprocessed()
    df_matrix = construct_feature_matrix(df_labeled)
    df_dev, df_outer = enforce_frozen_spatial_split_gate(df_matrix)

    print("\n" + "=" * 80)
    print("[OK] STEP 2 FEATURE ENGINEERING & SPATIAL PARTITION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
