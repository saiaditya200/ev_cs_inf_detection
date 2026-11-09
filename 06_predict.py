#!/usr/bin/env python3
"""
=============================================================================
06_predict.py
=============================================================================
Project: Explainable AI Framework for Smart EV Charging Infrastructure Planning
Purpose: Real-time inference tool for evaluating greenfield candidate charging
         station locations with calibrated threshold tau* = 0.63 and XAI guidance.
=============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / "india_wide"

def load_system():
    manifest_path = MODELS_DIR / "feature_manifest.json"
    thresh_path = MODELS_DIR / "selected_threshold.json"
    model_path = MODELS_DIR / "champion_model.joblib"

    with open(manifest_path, "r", encoding="utf-8") as f:
        predictors = json.load(f)["predictors"]

    with open(thresh_path, "r", encoding="utf-8") as f:
        tau = float(json.load(f)["calibrated_threshold"])

    model = joblib.load(model_path)
    return model, predictors, tau

def predict_candidate_stations():
    print("=" * 80)
    print("STAGE 6: REAL-TIME INFERENCE & CANDIDATE SITE SUITABILITY PREDICTOR")
    print("=" * 80)

    model, predictors, tau = load_system()
    print(f"[OK] Loaded Champion Model: {type(model).__name__}")
    print(f"[OK] Calibrated Decision Threshold (tau*): {tau:.4f} ({tau*100:.1f}%)")
    print(f"[OK] Input Feature Dimensions: {len(predictors)}")

    # Sample candidate locations representing diverse real-world Indian topologies
    candidates = [
        {
            "site_id": "CAND-DL-001",
            "name": "Delhi Outer Ring Road Arterial Plaza",
            "state": "Delhi",
            "latitude": 28.6139,
            "longitude": 77.2090,
            "distance_to_highway_km": 0.45,
            "distance_to_city_center_km": 5.2,
            "nearby_poi_count": 48,
            "highway_proximity_score": 0.95,
            "city_center_proximity_score": 0.72,
            "poi_diversity_index": 0.88,
            "road_type_motorway": 1,
            "road_type_primary": 0,
            "road_type_residential": 0,
            "road_type_secondary": 0,
            "road_type_service": 0,
            "road_type_trunk": 0,
            "road_type_unclassified": 0,
            "district_population_density_2011": 11320.0,
            "district_urban_population_pct": 97.5,
            "district_projected_density_fy24": 13500.0,
            "district_nh_crossing_count": 7,
            "district_ev_penetration_rate": 8.4,
            "grid_capacity_33kv_indicator": 1,
            "traffic_corridor_high_density": 1,
            "urban_tier_metro_or_tier1": 1
        },
        {
            "site_id": "CAND-KA-002",
            "name": "Bengaluru Electronic City NH-44 Hub",
            "state": "Karnataka",
            "latitude": 12.8399,
            "longitude": 77.6770,
            "distance_to_highway_km": 0.32,
            "distance_to_city_center_km": 14.5,
            "nearby_poi_count": 56,
            "highway_proximity_score": 0.98,
            "city_center_proximity_score": 0.45,
            "poi_diversity_index": 0.92,
            "road_type_motorway": 0,
            "road_type_primary": 1,
            "road_type_residential": 0,
            "road_type_secondary": 0,
            "road_type_service": 0,
            "road_type_trunk": 0,
            "road_type_unclassified": 0,
            "district_population_density_2011": 4381.0,
            "district_urban_population_pct": 90.9,
            "district_projected_density_fy24": 5200.0,
            "district_nh_crossing_count": 6,
            "district_ev_penetration_rate": 7.1,
            "grid_capacity_33kv_indicator": 1,
            "traffic_corridor_high_density": 1,
            "urban_tier_metro_or_tier1": 1
        },
        {
            "site_id": "CAND-MH-003",
            "name": "Mumbai-Pune Expressway Khalapur Plaza",
            "state": "Maharashtra",
            "latitude": 18.8315,
            "longitude": 73.2845,
            "distance_to_highway_km": 0.15,
            "distance_to_city_center_km": 42.0,
            "nearby_poi_count": 22,
            "highway_proximity_score": 0.99,
            "city_center_proximity_score": 0.20,
            "poi_diversity_index": 0.65,
            "road_type_motorway": 1,
            "road_type_primary": 0,
            "road_type_residential": 0,
            "road_type_secondary": 0,
            "road_type_service": 0,
            "road_type_trunk": 0,
            "road_type_unclassified": 0,
            "district_population_density_2011": 314.0,
            "district_urban_population_pct": 36.9,
            "district_projected_density_fy24": 380.0,
            "district_nh_crossing_count": 4,
            "district_ev_penetration_rate": 4.8,
            "grid_capacity_33kv_indicator": 1,
            "traffic_corridor_high_density": 1,
            "urban_tier_metro_or_tier1": 0
        },
        {
            "site_id": "CAND-UP-004",
            "name": "Rural Feeder Sub-Station Site",
            "state": "Uttar Pradesh",
            "latitude": 27.1767,
            "longitude": 78.0081,
            "distance_to_highway_km": 14.2,
            "distance_to_city_center_km": 18.0,
            "nearby_poi_count": 3,
            "highway_proximity_score": 0.12,
            "city_center_proximity_score": 0.25,
            "poi_diversity_index": 0.20,
            "road_type_motorway": 0,
            "road_type_primary": 0,
            "road_type_residential": 1,
            "road_type_secondary": 0,
            "road_type_service": 0,
            "road_type_trunk": 0,
            "road_type_unclassified": 0,
            "district_population_density_2011": 1084.0,
            "district_urban_population_pct": 22.3,
            "district_projected_density_fy24": 1250.0,
            "district_nh_crossing_count": 1,
            "district_ev_penetration_rate": 1.2,
            "grid_capacity_33kv_indicator": 0,
            "traffic_corridor_high_density": 0,
            "urban_tier_metro_or_tier1": 0
        }
    ]

    print(f"\nEvaluating {len(candidates)} real-world candidate site topologies across India:\n")

    for c in candidates:
        features_vec = [c[p] for p in predictors]
        X = pd.DataFrame([features_vec], columns=predictors)
        prob = model.predict_proba(X)[0, 1]
        is_fast = (prob >= tau)

        decision = "DC FAST CHARGING HUB (>= 50 kW)" if is_fast else "AC SLOW COMMUNITY CHARGER (< 50 kW)"
        badge = "[RECOMMENDED: DC FAST]" if is_fast else "[RECOMMENDED: AC SLOW]"

        print(f"┌─ Site ID: {c['site_id']} — {c['name']} ({c['state']})")
        print(f"│  Classification Decision : {badge} {decision}")
        print(f"│  Fast Siting Probability : {prob * 100:.2f}% (Threshold: {tau * 100:.1f}%)")
        print(f"│  Key Spatial Drivers    : Highway Dist = {c['distance_to_highway_km']} km, POIs = {c['nearby_poi_count']}, EV Pen = {c['district_ev_penetration_rate']}%")

        if is_fast and prob < 0.80:
            print(f"│  Actionable Guidance    : Marginal viability. Add secondary 33kV feed & expand POI amenities to achieve >= 80% saturation.")
        elif not is_fast:
            print(f"│  Actionable Guidance    : Siting unsuitable for high-power DC Fast. Candidate best serves destination AC Slow overnight charging.")
        else:
            print(f"│  Actionable Guidance    : Optimal high-capacity commercial corridor. Ready for immediate Phase-I deployment.")
        print(f"└" + "─" * 78)

    print("\n" + "=" * 80)
    print("[SUCCESS] Candidate location inference evaluation complete.")
    print("=" * 80)

if __name__ == "__main__":
    predict_candidate_stations()
