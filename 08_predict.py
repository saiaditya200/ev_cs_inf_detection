#!/usr/bin/env python3
"""
=============================================================================
Step 8: Station-Level Prediction Pipeline & Planning Decision Inference
=============================================================================
Project: Explainable AI Framework for Smart EV Charging Infrastructure Planning
Domain: Telangana, India (33 Administrative Districts)

Purpose:
- Run a strict station-level inference flow using the frozen project schema.
- Reuse the selected model and frozen decision threshold from Step 3.
- Generate local TreeSHAP explanations.
- Generate model-based counterfactual-style site scenarios.
- Attach certified district EIPS planning context.
- Persist a machine-readable prediction result and run metadata.

Important methodological boundaries:
- The classification target is the observed fast-charging infrastructure class,
  not station-level demand or utilization.
- District EV penetration is contextual planning information, not a station
  demand measurement.
- Counterfactual scenarios are model-based planning simulations, not causal
  guarantees.
- No leakage field is allowed in user-provided station input.
- No silent fallback/default is used for required site inputs.
=============================================================================
"""

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ========================= CONFIGURATION =========================

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FINAL_DATA_PATH = BASE_DIR / "data" / "final" / "FINAL_PROJECT_DATASET.csv"
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLANNING_DIR = RESULTS_DIR / "planning"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

TELANGANA_BOUNDS = {
    "lat_min": 15.8,
    "lat_max": 19.9,
    "lon_min": 77.2,
    "lon_max": 81.8,
}

VALID_ROAD_CATEGORIES = [
    "motorway",
    "primary",
    "residential",
    "secondary",
    "service",
    "trunk",
    "unclassified",
]

ROAD_ALIASES = {
    "motorway_link": "motorway",
    "trunk_link": "trunk",
    "primary_link": "primary",
    "secondary_link": "secondary",
    "residential": "residential",
    "service": "service",
    "unclassified": "unclassified",
}

CERTIFIED_21_PREDICTORS = [
    "latitude",
    "longitude",
    "distance_to_highway_km",
    "distance_to_city_center_km",
    "nearby_poi_count",
    "highway_proximity_score",
    "city_center_proximity_score",
    "poi_diversity_index",
    "road_type_motorway",
    "road_type_primary",
    "road_type_residential",
    "road_type_secondary",
    "road_type_service",
    "road_type_trunk",
    "road_type_unclassified",
    "district_population_density_2011",
    "district_urban_population_pct",
    "district_projected_density_fy24",
    "district_nh_crossing_count",
    "district_ev_penetration_rate",
    "discom_utility_data_available_flag",
]

LEAKAGE_FIELDS = {
    "charger_type",
    "charging_power_kw",
    "station_capacity",
    "connector_type",
    "number_of_chargers",
    "operator",
    "district_fast_stations",
    "district_fast_charger_ratio",
    "fast_charging_flag",
    "target",
    "label",
    "y",
    "current_station_fast_flag",
}

REQUIRED_DISTRICT_CONTEXT = [
    "district_population_density_2011",
    "district_urban_population_pct",
    "district_projected_density_fy24",
    "district_nh_crossing_count",
    "district_ev_penetration_rate",
    "discom_utility_data_available_flag",
]

ACTIONABLE_LEVERS = [
    "distance_to_highway_km",
    "distance_to_city_center_km",
    "nearby_poi_count",
    "poi_diversity_index",
]


# Telangana district centroids used only for planning visualization/context.
# These are retained here for consistency with the GIS stage; Step 8 itself
# does not use them for ML inference.
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


def _first_present(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _safe_float(value, field_name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"[VALIDATION ERROR] '{field_name}' must be numeric. Received: {value!r}"
        )
    if not np.isfinite(result):
        raise ValueError(
            f"[VALIDATION ERROR] '{field_name}' must be finite. Received: {value!r}"
        )
    return result


def _normalise_district_name(value):
    return str(value).strip().casefold()


def _normalise_road_type(value):
    raw = str(value).strip().lower()
    if not raw:
        raise ValueError("[VALIDATION ERROR] 'road_type' is required.")

    if raw in VALID_ROAD_CATEGORIES:
        return raw

    if raw in ROAD_ALIASES:
        return ROAD_ALIASES[raw]

    # Handle common OSM prefixes without silently assigning an arbitrary class.
    for category in VALID_ROAD_CATEGORIES:
        if raw == f"{category}_link":
            return category

    raise ValueError(
        f"[VALIDATION ERROR] Unsupported road_type '{value}'. "
        f"Allowed categories: {VALID_ROAD_CATEGORIES} and supported link aliases."
    )


def _sanitize_filename(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return cleaned.strip("._") or "prediction"


def load_artifacts():
    # Primary deployment artifacts (final model after research-grade tuning)
    final_model_path = MODELS_DIR / "final_deployment_model.joblib"
    fallback_model_path = MODELS_DIR / "selected_model.joblib"
    model_path = final_model_path if final_model_path.exists() else fallback_model_path

    manifest_path = MODELS_DIR / "feature_manifest.json"

    final_metadata_path = MODELS_DIR / "final_deployment_metadata.json"
    fallback_metadata_path = MODELS_DIR / "model_metadata.json"
    metadata_path = final_metadata_path if final_metadata_path.exists() else fallback_metadata_path

    final_threshold_path = MODELS_DIR / "final_deployment_threshold.json"
    fallback_threshold_path = MODELS_DIR / "selected_threshold.json"
    threshold_path = final_threshold_path if final_threshold_path.exists() else fallback_threshold_path

    lookup_path = CONFIG_DIR / "district_context_lookup.json"
    eips_path = PLANNING_DIR / "district_eips_rankings.csv"

    required = [
        (model_path, "Deployment model"),
        (manifest_path, "Feature manifest"),
        (metadata_path, "Model metadata"),
        (threshold_path, "Decision threshold"),
        (lookup_path, "District context lookup"),
        (eips_path, "District EIPS rankings"),
        (FINAL_DATA_PATH, "Frozen master dataset"),
    ]

    for path, name in required:
        if not path.exists():
            raise FileNotFoundError(
                f"[ERROR] Missing artifact: {name} at {path}. "
                "Complete the earlier project steps first."
            )

    model = joblib.load(model_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_payload = json.load(f)

    with open(lookup_path, "r", encoding="utf-8") as f:
        district_lookup = json.load(f)

    df_eips = pd.read_csv(eips_path)
    df_master = pd.read_csv(FINAL_DATA_PATH)

    predictors = manifest.get("predictors") or manifest.get("certified_21_predictors")
    if not predictors:
        raise ValueError("[ERROR] Feature manifest does not contain a predictor list.")

    if list(predictors) != CERTIFIED_21_PREDICTORS:
        raise ValueError(
            "[ERROR] Feature manifest predictor order/content does not match the "
            "certified 21-predictor schema."
        )

    if int(manifest.get("n_features", len(predictors))) != 21:
        raise ValueError("[ERROR] Feature manifest n_features is not 21.")

    if bool(manifest.get("scaling", False)):
        raise ValueError(
            "[ERROR] Manifest says scaling=True, but the certified pipeline uses raw features."
        )

    selected_model_name = str(
        metadata.get("selected_champion_model") or metadata.get("selected_model", "")
    ).strip()
    threshold_model = str(
        threshold_payload.get("model_family") or threshold_payload.get("model", "")
    ).strip()

    if selected_model_name and threshold_model and selected_model_name != threshold_model:
        raise ValueError(
            "[ERROR] Selected model mismatch between model metadata and threshold payload."
        )

    threshold = _first_present(
        threshold_payload, ["threshold", "decision_threshold", "selected_threshold", "optimal_threshold"]
    )
    if threshold is None:
        threshold = metadata.get("selected_threshold") or metadata.get("decision_threshold")

    threshold = _safe_float(threshold, "selected decision threshold")
    if not (0.0 < threshold < 1.0):
        raise ValueError(
            f"[ERROR] Selected decision threshold must be between 0 and 1. Got {threshold}."
        )

    if len(district_lookup) != 33:
        raise ValueError(
            f"[ERROR] District lookup contains {len(district_lookup)} districts; expected 33."
        )

    if len(df_eips) != 33:
        raise ValueError(
            f"[ERROR] EIPS table contains {len(df_eips)} districts; expected 33."
        )

    required_eips_cols = {
        "district",
        "rank",
        "EIPS",
        "priority_tier",
        "infrastructure_gap_index",
        "district_ev_penetration_rate",
        "fast_stations",
        "eips_explanation",
    }
    missing_eips = sorted(required_eips_cols - set(df_eips.columns))
    if missing_eips:
        raise ValueError(
            f"[ERROR] EIPS table is missing required columns: {missing_eips}"
        )

    if len(df_master) != 166:
        raise ValueError(
            f"[ERROR] Frozen master dataset contains {len(df_master)} rows; expected 166."
        )

    if len(df_master.columns) != 78:
        raise ValueError(
            f"[ERROR] Frozen master dataset contains {len(df_master.columns)} columns; expected 78."
        )

    required_master = {"station_id", "district", "latitude", "longitude", "fast_charging_flag"}
    missing_master = sorted(required_master - set(df_master.columns))
    if missing_master:
        raise ValueError(
            f"[ERROR] Frozen master dataset is missing required columns: {missing_master}"
        )

    return {
        "model": model,
        "predictors": predictors,
        "metadata": metadata,
        "threshold_payload": threshold_payload,
        "threshold": threshold,
        "district_lookup": district_lookup,
        "df_eips": df_eips,
        "df_master": df_master,
        "selected_model_name": selected_model_name or type(model).__name__,
    }


def validate_and_construct_features(station_data, predictors, district_lookup):
    if not isinstance(station_data, dict):
        raise TypeError("[VALIDATION ERROR] Station input must be a JSON object/dictionary.")

    supplied_keys = {str(k).strip() for k in station_data.keys()}
    leaked = sorted(
        key for key in supplied_keys if key.casefold() in {x.casefold() for x in LEAKAGE_FIELDS}
    )
    if leaked:
        raise ValueError(
            "[VALIDATION ERROR] Leakage fields are not permitted in station input: "
            + ", ".join(leaked)
        )

    district_raw = station_data.get("district")
    if district_raw is None or not str(district_raw).strip():
        raise ValueError(
            "[VALIDATION ERROR] 'district' is required and must be one of the 33 Telangana districts."
        )

    matched_district = None
    norm = _normalise_district_name(district_raw)
    for district_name in district_lookup:
        if _normalise_district_name(district_name) == norm:
            matched_district = district_name
            break

    if matched_district is None:
        raise ValueError(
            f"[VALIDATION ERROR] Unknown district '{district_raw}'. "
            f"Expected one of the 33 districts in the certified district lookup."
        )

    dist_ctx = district_lookup[matched_district]
    missing_ctx = [
        key
        for key in REQUIRED_DISTRICT_CONTEXT
        if key not in dist_ctx or dist_ctx[key] in (None, "")
    ]
    if missing_ctx:
        raise ValueError(
            f"[VALIDATION ERROR] District context is incomplete for '{matched_district}': {missing_ctx}"
        )

    lat_raw = station_data.get("latitude")
    lon_raw = station_data.get("longitude")
    if lat_raw is None or lon_raw is None:
        raise ValueError(
            "[VALIDATION ERROR] 'latitude' and 'longitude' are required."
        )

    lat = _safe_float(lat_raw, "latitude")
    lon = _safe_float(lon_raw, "longitude")

    if not (TELANGANA_BOUNDS["lat_min"] <= lat <= TELANGANA_BOUNDS["lat_max"]):
        raise ValueError(
            f"[VALIDATION ERROR] Latitude {lat:.6f} is outside the certified Telangana project bounds."
        )

    if not (TELANGANA_BOUNDS["lon_min"] <= lon <= TELANGANA_BOUNDS["lon_max"]):
        raise ValueError(
            f"[VALIDATION ERROR] Longitude {lon:.6f} is outside the certified Telangana project bounds."
        )

    road_raw = station_data.get("road_type")
    if road_raw is None or not str(road_raw).strip():
        raise ValueError("[VALIDATION ERROR] 'road_type' is required.")
    road_type = _normalise_road_type(road_raw)

    required_site_fields = [
        "distance_to_highway_km",
        "distance_to_city_center_km",
        "nearby_poi_count",
        "poi_diversity_index",
    ]
    missing_site = [
        key
        for key in required_site_fields
        if station_data.get(key) is None
    ]
    if missing_site:
        raise ValueError(
            f"[VALIDATION ERROR] Missing required site attributes: {missing_site}"
        )

    dist_hwy = _safe_float(
        station_data["distance_to_highway_km"], "distance_to_highway_km"
    )
    dist_city = _safe_float(
        station_data["distance_to_city_center_km"], "distance_to_city_center_km"
    )
    poi_count = _safe_float(
        station_data["nearby_poi_count"], "nearby_poi_count"
    )
    poi_div = _safe_float(
        station_data["poi_diversity_index"], "poi_diversity_index"
    )

    if dist_hwy < 0 or dist_city < 0 or poi_count < 0:
        raise ValueError(
            "[VALIDATION ERROR] Distances and POI count must be non-negative."
        )

    if not (0.0 <= poi_div <= 1.0):
        raise ValueError(
            f"[VALIDATION ERROR] poi_diversity_index must be between 0.0 and 1.0. Got {poi_div}."
        )

    road_one_hot = {
        f"road_type_{category}": float(category == road_type)
        for category in VALID_ROAD_CATEGORIES
    }

    feature_dict = {
        "latitude": lat,
        "longitude": lon,
        "distance_to_highway_km": dist_hwy,
        "distance_to_city_center_km": dist_city,
        "nearby_poi_count": poi_count,
        "highway_proximity_score": 1.0 / (1.0 + dist_hwy),
        "city_center_proximity_score": 1.0 / (1.0 + dist_city),
        "poi_diversity_index": poi_div,
        **road_one_hot,
        "district_population_density_2011": _safe_float(
            dist_ctx["district_population_density_2011"],
            "district_population_density_2011",
        ),
        "district_urban_population_pct": _safe_float(
            dist_ctx["district_urban_population_pct"],
            "district_urban_population_pct",
        ),
        "district_projected_density_fy24": _safe_float(
            dist_ctx["district_projected_density_fy24"],
            "district_projected_density_fy24",
        ),
        "district_nh_crossing_count": _safe_float(
            dist_ctx["district_nh_crossing_count"],
            "district_nh_crossing_count",
        ),
        "district_ev_penetration_rate": _safe_float(
            dist_ctx["district_ev_penetration_rate"],
            "district_ev_penetration_rate",
        ),
        "discom_utility_data_available_flag": _safe_float(
            dist_ctx["discom_utility_data_available_flag"],
            "discom_utility_data_available_flag",
        ),
    }

    missing_predictors = [p for p in predictors if p not in feature_dict]
    if missing_predictors:
        raise ValueError(
            f"[VALIDATION ERROR] Predictor assembly is incomplete: {missing_predictors}"
        )

    extra_feature_keys = sorted(set(feature_dict) - set(predictors))
    if extra_feature_keys:
        raise ValueError(
            f"[VALIDATION ERROR] Feature assembly contains non-certified predictors: {extra_feature_keys}"
        )

    X_vector = np.asarray(
        [feature_dict[p] for p in predictors],
        dtype=float,
    ).reshape(1, -1)

    if X_vector.shape != (1, 21):
        raise ValueError(
            f"[VALIDATION ERROR] Expected feature matrix shape (1, 21), got {X_vector.shape}."
        )

    if not np.isfinite(X_vector).all():
        raise ValueError("[VALIDATION ERROR] Feature vector contains non-finite values.")

    return X_vector, feature_dict, matched_district, lat, lon, road_type


def _extract_class1_shap_values(explainer, X):
    shap_vals = explainer.shap_values(X)

    if isinstance(shap_vals, list):
        return np.asarray(shap_vals[1][0], dtype=float)

    shap_arr = np.asarray(shap_vals)

    if shap_arr.ndim == 3:
        # SHAP versions differ on whether class dimension is the final axis
        # or the middle axis. For a single binary sample, handle both forms.
        if shap_arr.shape[0] == 1 and shap_arr.shape[2] == 2:
            return np.asarray(shap_arr[0, :, 1], dtype=float)
        if shap_arr.shape[0] == 1 and shap_arr.shape[1] == 2:
            return np.asarray(shap_arr[0, 1, :], dtype=float)

    if shap_arr.ndim == 2:
        return np.asarray(shap_arr[0], dtype=float)

    if shap_arr.ndim == 1:
        return shap_arr.astype(float)

    raise ValueError(
        f"[XAI ERROR] Unsupported SHAP value shape: {shap_arr.shape}"
    )


def _build_local_shap_table(model, X, predictors, feature_dict):
    explainer = shap.TreeExplainer(model)
    local_shap = _extract_class1_shap_values(explainer, X)

    if len(local_shap) != len(predictors):
        raise ValueError(
            "[XAI ERROR] SHAP attribution count does not match the 21-predictor schema."
        )

    shap_df = pd.DataFrame(
        {
            "feature": predictors,
            "value": [feature_dict[p] for p in predictors],
            "shap_attribution": local_shap,
        }
    )
    return shap_df


def _candidate_values(current):
    return sorted(
        set(
            [
                max(0.0, min(current, 8.0)),
                0.0,
                0.5,
                1.0,
                2.0,
                3.0,
                5.0,
                8.0,
            ]
        )
    )


def _poi_candidates(current):
    upper = max(current, 20.0)
    return sorted(
        set(
            [
                max(0.0, current),
                5.0,
                8.0,
                12.0,
                16.0,
                upper,
            ]
        )
    )


def _diversity_candidates(current):
    return sorted(
        set(
            [
                max(0.0, min(current, 1.0)),
                0.25,
                0.50,
                0.65,
                0.78,
                0.90,
                1.0,
            ]
        )
    )


def _rebuild_feature_dict(base_features, updates):
    features = dict(base_features)
    features.update(updates)
    features["highway_proximity_score"] = 1.0 / (
        1.0 + features["distance_to_highway_km"]
    )
    features["city_center_proximity_score"] = 1.0 / (
        1.0 + features["distance_to_city_center_km"]
    )
    return features


def _vector_from_features(features, predictors):
    return np.asarray(
        [features[p] for p in predictors],
        dtype=float,
    ).reshape(1, -1)


def generate_counterfactual_scenarios(
    model,
    base_features,
    predictors,
    current_probability,
    decision_threshold,
):
    """
    Generate deterministic model-based scenarios by varying only the four
    approved actionable levers. These are planning simulations, not causal
    guarantees.
    """
    if current_probability >= decision_threshold:
        return {
            "status": "already_above_threshold",
            "target_probability": round(float(decision_threshold), 4),
            "candidate_count": 0,
            "best_scenario": None,
            "scenarios": [],
            "interpretation": (
                "The current feature vector is already above the frozen "
                "classification threshold."
            ),
        }

    candidates = []

    lever_values = {
        "distance_to_highway_km": _candidate_values(
            base_features["distance_to_highway_km"]
        ),
        "distance_to_city_center_km": _candidate_values(
            base_features["distance_to_city_center_km"]
        ),
        "nearby_poi_count": _poi_candidates(
            base_features["nearby_poi_count"]
        ),
        "poi_diversity_index": _diversity_candidates(
            base_features["poi_diversity_index"]
        ),
    }

    current = {
        lever: float(base_features[lever])
        for lever in ACTIONABLE_LEVERS
    }

    for lever in ACTIONABLE_LEVERS:
        for value in lever_values[lever]:
            changed = {lever: float(value)}
            candidate_features = _rebuild_feature_dict(base_features, changed)
            X_candidate = _vector_from_features(candidate_features, predictors)
            probability = float(model.predict_proba(X_candidate)[0, 1])

            delta = abs(value - current[lever])
            candidates.append(
                {
                    "changed_lever": lever,
                    "new_value": float(value),
                    "change_magnitude": float(delta),
                    "probability_dc_fast": probability,
                    "crosses_threshold": bool(probability >= decision_threshold),
                }
            )

    # Also test a compact two-lever scenario built from the most practical
    # endpoint values. This does not search arbitrary combinations.
    two_lever_pairs = [
        (
            "distance_to_highway_km",
            min(2.0, current["distance_to_highway_km"]),
            "nearby_poi_count",
            max(8.0, current["nearby_poi_count"]),
        ),
        (
            "distance_to_city_center_km",
            min(3.0, current["distance_to_city_center_km"]),
            "poi_diversity_index",
            max(0.78, current["poi_diversity_index"]),
        ),
        (
            "distance_to_highway_km",
            min(2.0, current["distance_to_highway_km"]),
            "poi_diversity_index",
            max(0.78, current["poi_diversity_index"]),
        ),
    ]

    for a, av, b, bv in two_lever_pairs:
        changed = {a: float(av), b: float(bv)}
        candidate_features = _rebuild_feature_dict(base_features, changed)
        X_candidate = _vector_from_features(candidate_features, predictors)
        probability = float(model.predict_proba(X_candidate)[0, 1])

        change_magnitude = abs(av - current[a]) + abs(bv - current[b])
        candidates.append(
            {
                "changed_lever": f"{a} + {b}",
                "new_value": {
                    a: float(av),
                    b: float(bv),
                },
                "change_magnitude": float(change_magnitude),
                "probability_dc_fast": probability,
                "crosses_threshold": bool(probability >= decision_threshold),
            }
        )

    candidates.sort(
        key=lambda row: (
            not row["crosses_threshold"],
            -row["probability_dc_fast"],
            row["change_magnitude"],
        )
    )

    best = candidates[0] if candidates else None

    return {
        "status": "scenarios_generated",
        "target_probability": round(float(decision_threshold), 4),
        "candidate_count": len(candidates),
        "best_scenario": best,
        "scenarios": candidates[:8],
        "interpretation": (
            "These scenarios show how the trained model responds to feasible "
            "changes in approved site-selection levers. They are model-based "
            "counterfactual simulations, not causal guarantees."
        ),
    }


def get_regional_planning_context(df_eips, district):
    norm = _normalise_district_name(district)
    matches = df_eips[
        df_eips["district"].astype(str).map(_normalise_district_name) == norm
    ]

    if len(matches) != 1:
        raise ValueError(
            f"[PLANNING ERROR] Expected exactly one EIPS row for district '{district}', found {len(matches)}."
        )

    row = matches.iloc[0]
    tier_raw = str(row["priority_tier"])
    if tier_raw == "Tier 1: High Priority (Immediate Fast-Charging Expansion)":
        tier_display = "Tier 1 — High Relative Planning Priority"
    else:
        tier_display = tier_raw

    return {
        "district": str(row["district"]),
        "eips_rank": int(row["rank"]),
        "eips_score": float(row["EIPS"]),
        "priority_tier": tier_display,
        "infrastructure_gap_index": float(row["infrastructure_gap_index"]),
        "ev_penetration_rate": float(row["district_ev_penetration_rate"]),
        "existing_fast_stations": int(row["fast_stations"]),
        "planning_narrative": str(row["eips_explanation"]),
        "eips_explanatory_statement": (
            "EIPS identifies districts with higher relative planning priority for "
            "investigation and potential infrastructure expansion. It does not prescribe "
            "an exact station location or exact station count."
        ),
    }


def build_result(
    station_data,
    model,
    selected_model_name,
    predictors,
    threshold,
    df_eips,
    X,
    feature_dict,
    district,
    lat,
    lon,
    road_type,
):
    prob_class1 = float(model.predict_proba(X)[0, 1])
    prob_class0 = float(1.0 - prob_class1)

    pred_label = int(prob_class1 >= threshold)

    if pred_label == 1:
        classification = "DC Fast-Charging Infrastructure (>= 50kW)"
        predicted_class_probability = prob_class1
    else:
        classification = "AC Slow/Medium Charging (< 50kW)"
        predicted_class_probability = prob_class0

    shap_df = _build_local_shap_table(
        model, X, predictors, feature_dict
    )

    top_positive = (
        shap_df.sort_values("shap_attribution", ascending=False)
        .head(3)
        .to_dict(orient="records")
    )
    top_negative = (
        shap_df.sort_values("shap_attribution", ascending=True)
        .head(3)
        .to_dict(orient="records")
    )

    cf = generate_counterfactual_scenarios(
        model=model,
        base_features=feature_dict,
        predictors=predictors,
        current_probability=prob_class1,
        decision_threshold=threshold,
    )

    regional = get_regional_planning_context(df_eips, district)

    input_mode = station_data.get("_input_mode", "custom")

    result = {
        "status": "success",
        "project": "Explainable AI Framework for Smart EV Charging Infrastructure Planning",
        "step": 8,
        "station_id": station_data.get("station_id", "STN_PRED_CUSTOM"),
        "station_name": station_data.get(
            "station_name", "Prospective EV Charging Site"
        ),
        "input_mode": input_mode,
        "coordinates": {
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
        },
        "district": district,
        "road_type": road_type,
        "model": {
            "name": selected_model_name,
            "runtime_class": type(model).__name__,
            "feature_count": len(predictors),
            "decision_threshold": round(float(threshold), 4),
            "scaling": False,
        },
        "prediction": {
            "class_label": pred_label,
            "classification": classification,
            "probability_dc_fast": round(prob_class1, 4),
            "probability_ac_slow": round(prob_class0, 4),
            "predicted_class_probability": round(
                predicted_class_probability, 4
            ),
            "threshold_applied": round(float(threshold), 4),
        },
        "feature_vector": {
            p: round(float(feature_dict[p]), 6)
            for p in predictors
        },
        "explainable_ai": {
            "target_interpretation": (
                "Observed fast-charging infrastructure classification; "
                "not station-level charging demand/utilization."
            ),
            "top_positive_drivers": [
                {
                    "feature": r["feature"],
                    "value": round(float(r["value"]), 4),
                    "shap_impact": round(float(r["shap_attribution"]), 6),
                }
                for r in top_positive
            ],
            "top_negative_drivers": [
                {
                    "feature": r["feature"],
                    "value": round(float(r["value"]), 4),
                    "shap_impact": round(float(r["shap_attribution"]), 6),
                }
                for r in top_negative
            ],
            "all_local_attributions": [
                {
                    "feature": r["feature"],
                    "value": round(float(r["value"]), 4),
                    "shap_impact": round(float(r["shap_attribution"]), 6),
                }
                for r in shap_df.to_dict(orient="records")
            ],
        },
        "actionable_counterfactual": cf,
        "regional_planning_context": regional,
        "methodological_note": (
            "EIPS is a district-level planning score. District EV penetration "
            "and grid context are used as contextual planning information and "
            "are not converted into station-level demand."
        ),
    }

    return result


def print_prediction_report(result):
    print("\n" + "=" * 80)
    print("STATION PREDICTION & PLANNING INFERENCE REPORT")
    print("=" * 80)
    print(f"Station ID:       {result['station_id']}")
    print(f"Station Name:     {result['station_name']}")
    print(
        "Coordinates:      "
        f"Lat {result['coordinates']['latitude']:.4f}°N, "
        f"Lon {result['coordinates']['longitude']:.4f}°E"
    )
    print(f"District:         {result['district']}")
    print(f"Road Type:        {result['road_type']}")
    print("-" * 80)
    print(
        f"DECISION THRESHOLD: {result['prediction']['threshold_applied']:.2f}"
    )
    print(
        f"PREDICTED CLASS:    {result['prediction']['classification']}"
    )
    print(
        "P(DC Fast >=50kW):  "
        f"{result['prediction']['probability_dc_fast']:.1%} | "
        f"P(AC Slow): {result['prediction']['probability_ac_slow']:.1%}"
    )
    print(
        "Predicted-class probability: "
        f"{result['prediction']['predicted_class_probability']:.1%}"
    )
    print("-" * 80)

    print("TOP POSITIVE SHAP DRIVERS:")
    for row in result["explainable_ai"]["top_positive_drivers"]:
        print(
            f"  + {row['feature']:<32} "
            f"value={row['value']:<10} "
            f"SHAP={row['shap_impact']:+.6f}"
        )

    print("TOP NEGATIVE SHAP DRIVERS:")
    for row in result["explainable_ai"]["top_negative_drivers"]:
        print(
            f"  - {row['feature']:<32} "
            f"value={row['value']:<10} "
            f"SHAP={row['shap_impact']:+.6f}"
        )

    print("-" * 80)
    print("COUNTERFACTUAL-STYLE PLANNING SCENARIOS:")
    cf = result["actionable_counterfactual"]
    if cf["best_scenario"] is None:
        print(f"  {cf['interpretation']}")
    else:
        best = cf["best_scenario"]
        print(f"  Best scenario: {best}")
        print(f"  Target threshold: {cf['target_probability']:.2f}")

    print("-" * 80)
    rc = result["regional_planning_context"]
    print("REGIONAL PLANNING (EIPS) CONTEXT:")
    print(f"  District Rank:        #{rc['eips_rank']}")
    print(f"  EIPS Score:           {rc['eips_score']:.2f}/100")
    print(f"  Priority Tier:        {rc['priority_tier']}")
    if "eips_explanatory_statement" in rc:
        print(f"  Planning Guidance:    {rc['eips_explanatory_statement']}")
    print(
        "  Infrastructure Gap:   "
        f"{rc['infrastructure_gap_index']:.4f}"
    )
    print(
        "  EV Penetration:       "
        f"{rc['ev_penetration_rate']:.4f} EV/k-pop"
    )
    print(
        "  Existing Fast:        "
        f"{rc['existing_fast_stations']}"
    )
    print("=" * 80 + "\n")


def get_sample_station():
    """
    Illustrative prospective-site input used for a smoke test/demo.
    It is not presented as a verified existing station.
    """
    return {
        "_input_mode": "illustrative_sample",
        "station_id": "TS_SAMPLE_PROSPECTIVE_01",
        "station_name": "Illustrative Medchal Commercial Corridor Site",
        "latitude": 17.6320,
        "longitude": 78.5610,
        "district": "Medchal-Malkajgiri",
        "road_type": "primary",
        "distance_to_highway_km": 0.85,
        "distance_to_city_center_km": 6.2,
        "nearby_poi_count": 9.0,
        "poi_diversity_index": 0.78,
    }


def load_input_json(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("[ERROR] Input JSON must contain one station object.")

    payload["_input_mode"] = "custom_json"
    return payload


def load_existing_station_by_id(df_master, station_id):
    matches = df_master[
        df_master["station_id"].astype(str).str.casefold()
        == str(station_id).strip().casefold()
    ]

    if len(matches) == 0:
        raise ValueError(
            f"[ERROR] Station ID '{station_id}' was not found in the frozen master dataset."
        )

    if len(matches) > 1:
        raise ValueError(
            f"[ERROR] Station ID '{station_id}' is not unique in the frozen master dataset."
        )

    row = matches.iloc[0]

    # Existing rows in the frozen master intentionally contain target/current-
    # infrastructure fields such as fast_charging_flag and charger_type.
    # Those fields must never reach model feature construction. For the
    # station-id inference path, keep only the certified pre-prediction inputs.
    allowed_fields = [
        "station_id",
        "station_name",
        "latitude",
        "longitude",
        "district",
        "road_type",
        "distance_to_highway_km",
        "distance_to_city_center_km",
        "nearby_poi_count",
        "poi_diversity_index",
    ]

    missing_allowed = [field for field in allowed_fields if field not in df_master.columns]
    if missing_allowed:
        raise ValueError(
            f"[ERROR] Frozen master is missing station input fields required for "
            f"station-id inference: {missing_allowed}"
        )

    sanitized = {}
    for field in allowed_fields:
        value = row[field]
        if pd.isna(value):
            sanitized[field] = None
        else:
            sanitized[field] = value.item() if hasattr(value, "item") else value

    sanitized["_input_mode"] = "existing_frozen_station"
    return sanitized


def persist_result(result):
    station_id = result["station_id"]
    filename = f"prediction_{_sanitize_filename(station_id)}.json"
    out_path = PREDICTIONS_DIR / filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Preserve the project demo artifact name for the default sample flow.
    if result["input_mode"] == "illustrative_sample":
        sample_path = PREDICTIONS_DIR / "sample_prediction.json"
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    run_metadata = {
        "step": 8,
        "status": "success",
        "station_id": station_id,
        "input_mode": result["input_mode"],
        "model_name": result["model"]["name"],
        "runtime_model_class": result["model"]["runtime_class"],
        "n_features": result["model"]["feature_count"],
        "decision_threshold": result["model"]["decision_threshold"],
        "prediction_class": result["prediction"]["class_label"],
        "probability_dc_fast": result["prediction"]["probability_dc_fast"],
        "district": result["district"],
    }

    metadata_path = PREDICTIONS_DIR / "prediction_run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="EV Charging Station Classification & Planning Inference"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--sample",
        action="store_true",
        help="Run the illustrative prospective-site smoke test.",
    )
    group.add_argument(
        "--input",
        type=str,
        help="Path to an input JSON object containing certified station-site attributes.",
    )
    group.add_argument(
        "--station-id",
        type=str,
        help="Evaluate an existing station by ID from the frozen master dataset.",
    )
    args = parser.parse_args()

    artifacts = load_artifacts()

    if args.input:
        station_payload = load_input_json(Path(args.input))
    elif args.station_id:
        station_payload = load_existing_station_by_id(
            artifacts["df_master"],
            args.station_id,
        )
        print(
            "[INFO] Existing-station inference: target/current-infrastructure "
            "fields were stripped before feature validation."
        )
    else:
        station_payload = get_sample_station()

    X, feature_dict, district, lat, lon, road_type = (
        validate_and_construct_features(
            station_payload,
            artifacts["predictors"],
            artifacts["district_lookup"],
        )
    )

    result = build_result(
        station_data=station_payload,
        model=artifacts["model"],
        selected_model_name=artifacts["selected_model_name"],
        predictors=artifacts["predictors"],
        threshold=artifacts["threshold"],
        df_eips=artifacts["df_eips"],
        X=X,
        feature_dict=feature_dict,
        district=district,
        lat=lat,
        lon=lon,
        road_type=road_type,
    )

    print_prediction_report(result)

    out_path = persist_result(result)
    print(f"[OK] Prediction result persisted: {out_path}")
    print(
        f"[OK] Frozen threshold used: {artifacts['threshold']:.2f} "
        f"({artifacts['threshold']:.0%})"
    )
    print(
        "[OK] Inference uses the frozen 21-feature order with no scaling "
        "and no leakage fields."
    )


if __name__ == "__main__":
    main()
