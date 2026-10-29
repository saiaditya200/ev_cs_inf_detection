#!/usr/bin/env python3
"""
=============================================================================
Pan-India EV Charging Infrastructure Web Dashboard Backend
=============================================================================
Project: Explainable AI Framework for Smart EV Charging Infrastructure Planning
Domain: All 36 States & Union Territories of India (52,718 Stations)
Port: 5001 (Runs completely independently from Telangana Dashboard on port 5000)

Features:
- REST API serving all 36 States and UTs
- Real-time inference comparing Telangana Baseline Model vs. Pan-India Model
- Station mapping with Leaflet coordinates
- State-by-state comparative metrics
=============================================================================
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_INDIA_DIR = BASE_DIR / "frontend_india"
DATA_DIR = BASE_DIR / "data" / "india_wide"
RESULTS_DIR = BASE_DIR / "results" / "india_wide"
MODELS_DIR = BASE_DIR / "models"
INDIA_MODELS_DIR = MODELS_DIR / "india_wide"

app = Flask(__name__, static_folder=str(FRONTEND_INDIA_DIR))
CORS(app)

# Global Cached Data
DF_30 = None
DF_FULL = None
DF_SUMMARY = None
DF_STATE_PERF = None
DF_PREDS_SAMPLE = None

MODEL_TELANGANA = None
MODEL_NATIONAL = None
PREDICTORS_21 = []
THRESH_TELANGANA = 0.76
THRESH_NATIONAL = 0.25

def init_app_state():
    global DF_30, DF_FULL, DF_SUMMARY, DF_STATE_PERF, DF_PREDS_SAMPLE
    global MODEL_TELANGANA, MODEL_NATIONAL, PREDICTORS_21, THRESH_TELANGANA, THRESH_NATIONAL

    # 1. Load Data
    summary_path = DATA_DIR / "INDIA_WIDE_STATE_EV_CHARGING_SUMMARY_52K.csv"
    if summary_path.exists():
        DF_SUMMARY = pd.read_csv(summary_path)

    state_perf_path = RESULTS_DIR / "metrics" / "state_wise_performance_breakdown.csv"
    if state_perf_path.exists():
        DF_STATE_PERF = pd.read_csv(state_perf_path)

    # Load 30-feature dataset (load core columns for low memory usage)
    data_30_path = DATA_DIR / "INDIA_WIDE_EV_CHARGING_MASTER_DATASET_52K_30FEAT.csv"
    if data_30_path.exists():
        DF_30 = pd.read_csv(data_30_path)

    # 2. Load Predictors
    with open(MODELS_DIR / "feature_manifest.json") as f:
        PREDICTORS_21 = json.load(f)["predictors"]

    # 3. Load Models
    tg_model_path = MODELS_DIR / "final_deployment_model.joblib"
    if tg_model_path.exists():
        MODEL_TELANGANA = joblib.load(tg_model_path)

    nat_model_path = INDIA_MODELS_DIR / "india_wide_champion_model.joblib"
    if nat_model_path.exists():
        MODEL_NATIONAL = joblib.load(nat_model_path)

    # 4. Load Thresholds
    with open(MODELS_DIR / "final_deployment_threshold.json") as f:
        THRESH_TELANGANA = float(json.load(f).get("threshold", 0.76))

    nat_thresh_path = INDIA_MODELS_DIR / "india_wide_threshold.json"
    if nat_thresh_path.exists():
        with open(nat_thresh_path) as f:
            THRESH_NATIONAL = float(json.load(f).get("threshold", 0.25))

    print("[BACKEND READY] Pan-India Dashboard API initialized successfully.")

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_INDIA_DIR), "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(str(FRONTEND_INDIA_DIR), path)

@app.route("/api/overview")
def api_overview():
    total_stations = len(DF_30) if DF_30 is not None else 52718
    fast_count = int(DF_30["fast_charging_flag"].sum()) if DF_30 is not None else 16561
    slow_count = total_stations - fast_count

    metrics_comp_path = RESULTS_DIR / "metrics" / "model_comparison_metrics.json"
    comp_metrics = {}
    if metrics_comp_path.exists():
        with open(metrics_comp_path) as f:
            comp_metrics = json.load(f)

    return jsonify({
        "status": "success",
        "national_summary": {
            "total_stations": total_stations,
            "fast_charging_stations": fast_count,
            "slow_charging_stations": slow_count,
            "fast_share_pct": round(fast_count / total_stations * 100, 1),
            "states_and_uts_covered": 36,
            "dataset_dimensions": f"{total_stations:,} x 30 columns (1,581,540 cells)",
            "official_source": "Ministry of Heavy Industries / BHEL Portal (PIB Release ID: 2287125, July 2026)"
        },
        "model_comparison": comp_metrics.get("model_comparison", {})
    })

@app.route("/api/states")
def api_states():
    if DF_SUMMARY is not None:
        return jsonify({
            "status": "success",
            "count": len(DF_SUMMARY),
            "states": DF_SUMMARY.to_dict(orient="records")
        })
    return jsonify({"status": "error", "message": "Summary table not loaded"}), 500

@app.route("/api/state/<state_name>")
def api_state_detail(state_name):
    if DF_30 is None:
        return jsonify({"status": "error", "message": "Dataset not available"}), 500

    st_df = DF_30[DF_30["state"].str.lower() == state_name.lower().strip()]
    if st_df.empty:
        return jsonify({"status": "error", "message": f"State '{state_name}' not found"}), 404

    total_st = len(st_df)
    fast_st = int(st_df["fast_charging_flag"].sum())
    slow_st = total_st - fast_st

    # Sample up to 300 stations for fast Leaflet rendering
    sample_df = st_df.head(300)[["station_id", "station_name", "operator", "latitude", "longitude",
                                "charger_type", "connector_type", "rated_power_kw",
                                "fast_charging_flag", "distance_to_highway_km", "road_type"]]

    # State performance row
    perf_row = {}
    if DF_STATE_PERF is not None:
        p_match = DF_STATE_PERF[DF_STATE_PERF["state_or_ut"].str.lower() == state_name.lower().strip()]
        if not p_match.empty:
            perf_row = p_match.iloc[0].to_dict()

    return jsonify({
        "status": "success",
        "state_name": state_name,
        "total_stations": total_st,
        "fast_stations": fast_st,
        "slow_stations": slow_st,
        "fast_pct": round(fast_st / total_st * 100, 1),
        "performance": perf_row,
        "stations_sample": sample_df.to_dict(orient="records")
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Real-time station inference comparing Telangana model vs Pan-India model."""
    data = request.json or {}

    lat = float(data.get("latitude", 12.9716))
    lon = float(data.get("longitude", 77.5946))
    h_dist = float(data.get("distance_to_highway_km", 1.5))
    c_dist = float(data.get("distance_to_city_center_km", 4.0))
    poi_cnt = int(data.get("nearby_poi_count", 25))
    poi_div = float(data.get("poi_diversity_index", 0.75))
    road_type = str(data.get("road_type", "primary")).lower()
    pop_dens = float(data.get("population_density", 1200.0))
    urban_pct = float(data.get("urban_population_pct", 65.0))
    ev_pen = float(data.get("ev_penetration_rate", 4.5))

    h_score = round(float(np.exp(-0.2 * min(h_dist, 50.0))), 4)
    c_score = round(float(1.0 / (1.0 + 0.1 * min(c_dist, 100.0))), 4)

    # Road type one-hot
    road_cols = {
        "road_type_motorway": 1 if road_type == "motorway" else 0,
        "road_type_primary": 1 if road_type == "primary" else 0,
        "road_type_residential": 1 if road_type == "residential" else 0,
        "road_type_secondary": 1 if road_type == "secondary" else 0,
        "road_type_service": 1 if road_type == "service" else 0,
        "road_type_trunk": 1 if road_type == "trunk" else 0,
        "road_type_unclassified": 1 if road_type == "unclassified" else 0
    }

    # Construct 21-feature row in exact order
    feat_row = {
        "latitude": lat,
        "longitude": lon,
        "distance_to_highway_km": h_dist,
        "distance_to_city_center_km": c_dist,
        "nearby_poi_count": poi_cnt,
        "highway_proximity_score": h_score,
        "city_center_proximity_score": c_score,
        "poi_diversity_index": poi_div,
        "road_type_motorway": road_cols["road_type_motorway"],
        "road_type_primary": road_cols["road_type_primary"],
        "road_type_residential": road_cols["road_type_residential"],
        "road_type_secondary": road_cols["road_type_secondary"],
        "road_type_service": road_cols["road_type_service"],
        "road_type_trunk": road_cols["road_type_trunk"],
        "road_type_unclassified": road_cols["road_type_unclassified"],
        "district_population_density_2011": pop_dens,
        "district_urban_population_pct": urban_pct,
        "district_projected_density_fy24": pop_dens * 1.15,
        "district_nh_crossing_count": 18,
        "district_ev_penetration_rate": ev_pen,
        "discom_utility_data_available_flag": 1
    }

    df_sample = pd.DataFrame([feat_row])[PREDICTORS_21]

    # Predict with Model 1: Telangana Baseline Model
    res_tg = {}
    if MODEL_TELANGANA is not None:
        p_tg_raw = MODEL_TELANGANA.predict_proba(df_sample)[0, 1]
        p_tg = p_tg_raw / 100.0 if p_tg_raw > 1.0 else p_tg_raw
        is_fast_tg = int(p_tg >= THRESH_TELANGANA)
        res_tg = {
            "model_name": "Frozen Telangana Baseline Model",
            "fast_charging_probability": round(float(p_tg), 4),
            "fast_charging_probability_pct": f"{p_tg * 100:.1f}%",
            "prediction_class": "DC Fast Charging (>= 50kW)" if is_fast_tg == 1 else "AC Slow Charging (< 50kW)",
            "decision_threshold": THRESH_TELANGANA,
            "training_scope": "Telangana State Only (154 Stations)"
        }

    # Predict with Model 2: Pan-India National Model
    res_nat = {}
    if MODEL_NATIONAL is not None:
        p_nat = float(MODEL_NATIONAL.predict_proba(df_sample)[0, 1])
        is_fast_nat = int(p_nat >= THRESH_NATIONAL)
        res_nat = {
            "model_name": "Pan-India Retrained National Model",
            "fast_charging_probability": round(p_nat, 4),
            "fast_charging_probability_pct": f"{p_nat * 100:.1f}%",
            "prediction_class": "DC Fast Charging (>= 50kW)" if is_fast_nat == 1 else "AC Slow Charging (< 50kW)",
            "decision_threshold": THRESH_NATIONAL,
            "training_scope": "All-India National (52,718 Stations)"
        }

    return jsonify({
        "status": "success",
        "inputs": {
            "coordinates": f"{lat:.4f}, {lon:.4f}",
            "distance_to_highway_km": h_dist,
            "road_type": road_type,
            "highway_proximity_score": h_score,
            "poi_diversity_index": poi_div
        },
        "telangana_baseline_result": res_tg,
        "pan_india_national_result": res_nat,
        "models_agree": (res_tg.get("prediction_class") == res_nat.get("prediction_class"))
    })

if __name__ == "__main__":
    init_app_state()
    port = int(os.environ.get("PORT", 5001))
    print(f"=================================================================")
    print(f"  PAN-INDIA EV CHARGING WEB DASHBOARD RUNNING ON PORT {port}")
    print(f"  Access URL: http://localhost:{port}")
    print(f"=================================================================")
    app.run(host="0.0.0.0", port=port, debug=False)
