#!/usr/bin/env python3
"""
=============================================================================
Final Application Backend: Flask REST API & Web Dashboard
=============================================================================
Project: Explainable AI Framework for Smart EV Charging Infrastructure Planning
Domain: Telangana, India (33 Administrative Districts)

The web application is an integration layer over the completed numbered pipeline:
01 preprocessing -> 02 feature engineering -> 03 training -> 04 evaluation ->
05 XAI -> 06 EIPS planning -> 07 GIS -> 08 station inference.

Important boundaries:
- The classification target is observed fast-charging infrastructure class.
- District EV penetration is contextual adoption information, not station demand.
- TGSPDCL electricity data is contextual grid evidence; it is not disaggregated
  to individual stations.
- Counterfactuals are model-based planning simulations, not causal guarantees.
- The application never invents required station inputs or leakage fields.
=============================================================================
"""

import json
import os
import sys
import importlib.util
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
FINAL_DATA_PATH = DATA_DIR / "final" / "FINAL_PROJECT_DATASET.csv"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
PLANNING_DIR = RESULTS_DIR / "planning"
GIS_DIR = RESULTS_DIR / "gis"
XAI_DIR = RESULTS_DIR / "xai"
COUNTERFACTUALS_DIR = RESULTS_DIR / "counterfactuals"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

PROJECT_TITLE = "Explainable AI Framework for Smart EV Charging Infrastructure Planning"
MODEL_RUNTIME_VERSION = "1.3.0"

app = Flask(__name__, static_folder=str(FRONTEND_DIR))
CORS(app)

MODEL = None
PREDICTORS = []
FEATURE_MANIFEST = {}
MODEL_METADATA = {}
DISTRICT_LOOKUP = {}
DF_EIPS = None
DF_STATIONS = None
DF_METRICS = None
DF_CV = None
DF_OUTER = None
DF_SHAP_IMPORTANCE = None
DF_CF = None
PREDICT_MODULE = None
BACKEND_ERROR = None
THRESHOLD = None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _read_csv_candidates(paths):
    path = _first_existing(paths)
    if path is None:
        return None, None
    return pd.read_csv(path), path


def _load_threshold(threshold_override_path=None):
    global THRESHOLD
    if threshold_override_path and Path(threshold_override_path).exists():
        payload = _read_json(Path(threshold_override_path))
        value = payload.get("threshold") or payload.get("decision_threshold") or payload.get("selected_threshold")
        if value is not None:
            THRESHOLD = float(value)
            return

    final_threshold_path = MODELS_DIR / "final_deployment_threshold.json"
    fallback_threshold_path = MODELS_DIR / "selected_threshold.json"
    threshold_path = final_threshold_path if final_threshold_path.exists() else fallback_threshold_path

    if threshold_path.exists():
        payload = _read_json(threshold_path)
        value = payload.get("threshold")
        if value is None:
            value = payload.get("decision_threshold")
        if value is None:
            value = payload.get("selected_threshold")
        if value is not None:
            THRESHOLD = float(value)
            return
    value = MODEL_METADATA.get("selected_threshold") or MODEL_METADATA.get("decision_threshold")
    if value is not None:
        THRESHOLD = float(value)
        return
    raise FileNotFoundError(
        "Missing frozen decision threshold. Expected models/final_deployment_threshold.json "
        "(or fallback models/selected_threshold.json)."
    )


def _load_predict_module():
    global PREDICT_MODULE
    module_path = BASE_DIR / "08_predict.py"
    if not module_path.exists():
        raise FileNotFoundError("08_predict.py is required for the /api/predict endpoint.")
    spec = importlib.util.spec_from_file_location("ev_predict_module", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError("Could not create import specification for 08_predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PREDICT_MODULE = module


def _normalise_bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])



def _normalise_model_key(value):
    """Map model naming variants to one stable key for API joins."""
    s = str(value).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    aliases = {
        "randomforest": "Random_Forest",
        "randomforestclassifier": "Random_Forest",
        "rf": "Random_Forest",
        "xgboost": "XGBoost",
        "xgb": "XGBoost",
        "xgbclassifier": "XGBoost",
        "lightgbm": "LightGBM",
        "lgbm": "LightGBM",
        "lgbmclassifier": "LightGBM",
    }
    return aliases.get(s, str(value).strip())


def _model_name_column(df):
    if df is None:
        return None
    for col in ["model_name", "model", "classifier", "algorithm", "model_type", "Model"]:
        if col in df.columns:
            return col
    return None


def _metric_value(row, candidates):
    for key in candidates:
        if key in row:
            value = row[key]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _collect_model_names():
    names = set()
    for df in [DF_METRICS, DF_CV, DF_OUTER]:
        col = _model_name_column(df)
        if col is None:
            continue
        for value in df[col].dropna().tolist():
            names.add(_normalise_model_key(value))
    names.update(["Random_Forest", "XGBoost", "LightGBM"])
    return sorted(names, key=lambda x: ["Random_Forest", "XGBoost", "LightGBM"].index(x) if x in ["Random_Forest", "XGBoost", "LightGBM"] else 99)


def _find_model_row(df, model_name):
    if df is None or len(df) == 0:
        return None
    col = _model_name_column(df)
    if col is None:
        return None
    mapped = df[col].map(_normalise_model_key)
    rows = df[mapped == model_name]
    return rows.iloc[0].to_dict() if len(rows) else None


def _metric_record(model_name):
    cv_row = _find_model_row(DF_CV, model_name)
    outer_row = _find_model_row(DF_OUTER, model_name)
    combined_row = _find_model_row(DF_METRICS, model_name)

    record = {
        "model_name": model_name,
        "is_champion": model_name == _normalise_model_key(MODEL_METADATA.get("selected_champion_model", "Random_Forest")),
        "cv_pr_auc": None,
        "cv_pr_auc_mean": None,
        "cv_pr_auc_std": None,
        "cv_balanced_accuracy": None,
        "outer_test_pr_auc": None,
        "outer_test_roc_auc": None,
        "outer_test_accuracy": None,
        "outer_test_balanced_acc": None,
        "outer_test_macro_f1": None,
        "outer_test_minority_f1": None,
    }

    # Prefer dedicated CV artifacts for development values.
    if cv_row:
        cv = _metric_value(cv_row, [
            "cv_pr_auc_mean", "cv_pr_auc", "pr_auc_mean", "pr_auc",
            "PR-AUC", "PR_AUC", "mean_pr_auc", "pr_auc_cv"
        ])
        cv_std = _metric_value(cv_row, ["cv_pr_auc_std", "pr_auc_std", "std_pr_auc"])
        cv_bal = _metric_value(cv_row, [
            "cv_balanced_accuracy", "cv_balanced_acc", "balanced_accuracy_mean",
            "balanced_accuracy", "balanced_acc_mean", "balanced_acc"
        ])
        record["cv_pr_auc"] = cv
        record["cv_pr_auc_mean"] = cv
        record["cv_pr_auc_std"] = cv_std
        record["cv_balanced_accuracy"] = cv_bal

    # Dedicated outer-test artifact first, then comparison table for compatibility.
    for target, keys in {
        "outer_test_pr_auc": ["outer_test_pr_auc", "pr_auc", "PR-AUC", "PR_AUC"],
        "outer_test_roc_auc": ["outer_test_roc_auc", "roc_auc", "ROC-AUC", "ROC_AUC"],
        "outer_test_accuracy": ["outer_test_accuracy", "accuracy", "Accuracy"],
        "outer_test_balanced_acc": ["outer_test_balanced_acc", "balanced_accuracy", "balanced_acc", "BalancedAccuracy"],
        "outer_test_macro_f1": ["outer_test_macro_f1", "macro_f1", "MacroF1", "f1_macro"],
        "outer_test_minority_f1": ["outer_test_minority_f1", "minority_f1", "minority_class_f1", "MinorityClassF1"],
    }.items():
        value = None
        if outer_row:
            value = _metric_value(outer_row, keys)
        if value is None and combined_row:
            value = _metric_value(combined_row, keys)
        record[target] = value

    # If the comparison artifact carries a CV field, use it only when the dedicated CV file did not.
    if record["cv_pr_auc"] is None and combined_row:
        value = _metric_value(combined_row, [
            "cv_pr_auc_mean", "cv_pr_auc", "development_cv_pr_auc",
            "development_pr_auc", "pr_auc_cv"
        ])
        if value is not None:
            record["cv_pr_auc"] = value
            record["cv_pr_auc_mean"] = value

    return record

def _select_champion_record():
    if DF_METRICS is None or len(DF_METRICS) == 0:
        return {}
    selected = str(MODEL_METADATA.get("selected_champion_model", "Random_Forest"))
    df = DF_METRICS.copy()
    if "is_champion" in df.columns:
        champ_mask = _normalise_bool_series(df["is_champion"])
        rows = df[champ_mask]
        if len(rows):
            return rows.iloc[0].to_dict()
    if "model_name" in df.columns:
        rows = df[df["model_name"].astype(str) == selected]
        if len(rows):
            return rows.iloc[0].to_dict()
    return df.iloc[0].to_dict()


def _numeric_from_row(row, *keys):
    for key in keys:
        if key in row and row[key] is not None and pd.notna(row[key]):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return None


def _grid_context_summary():
    summary_file = PLANNING_DIR / "planning_summary_report.txt"
    if summary_file.exists():
        text = summary_file.read_text(encoding="utf-8", errors="ignore")
        import re
        patterns = [
            r"(?:55\.73|total).*?GWh",
            r"(?:grid|TGSPDCL).*?([0-9]+(?:\.[0-9]+)?)\s*GWh",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                nums = re.findall(r"[0-9]+(?:\.[0-9]+)?", match.group(0))
                if nums:
                    return float(nums[-1])
    candidates = [
        BASE_DIR / "data" / "processed" / "district_year_panel.csv",
        BASE_DIR / "data" / "processed" / "district_grid_context.csv",
        BASE_DIR / "data" / "processed" / "telangana_grid_context.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            grid_cols = [
                c for c in df.columns
                if "discom" in c.lower() and "kwh" in c.lower()
            ]
            if grid_cols:
                vals = pd.to_numeric(df[grid_cols[0]], errors="coerce").dropna()
                if len(vals):
                    return float(vals.sum() / 1e6)
        except Exception:
            continue
    return None


def initialize_backend():
    global MODEL, PREDICTORS, FEATURE_MANIFEST, MODEL_METADATA, DISTRICT_LOOKUP
    global DF_EIPS, DF_STATIONS, DF_METRICS, DF_CV, DF_OUTER, DF_SHAP_IMPORTANCE, DF_CF
    global BACKEND_ERROR

    print("=" * 80)
    print("[INFO] INITIALIZING FINAL FLASK APPLICATION BACKEND")
    print("=" * 80)
    print("[INFO] Python: {}".format(sys.version.split()[0]))
    print("[INFO] scikit-learn: {}".format(sklearn.__version__))

    try:
        # Reversible runtime configuration switch (Stage 3)
        runtime_config_env = os.environ.get("MODEL_RUNTIME_CONFIG")
        active_role_env = os.environ.get("ACTIVE_MODEL_ROLE")
        runtime_config_path = None
        if runtime_config_env and (BASE_DIR / runtime_config_env).exists():
            runtime_config_path = BASE_DIR / runtime_config_env
        elif active_role_env == "candidate":
            runtime_config_path = CONFIG_DIR / "model_runtime_candidate.json"

        threshold_override = None
        if runtime_config_path and runtime_config_path.exists():
            cfg = _read_json(runtime_config_path)
            model_path = BASE_DIR / cfg["model_path"]
            meta_path = BASE_DIR / cfg.get("metadata_path", "models/joint_rf_0257_deployment_metadata.json")
            manifest_path = BASE_DIR / cfg.get("feature_manifest", "models/feature_manifest.json")
            threshold_override = BASE_DIR / cfg["threshold_path"]
            active_model_role = cfg.get("active_model_role", "candidate")
            print("[INFO] REVERSIBLE RUNTIME: Loaded {} model ({}) from {}".format(
                active_model_role, cfg.get("model_id"), cfg["model_path"]
            ))
        else:
            final_model_path = MODELS_DIR / "final_deployment_model.joblib"
            fallback_model_path = MODELS_DIR / "selected_model.joblib"
            model_path = final_model_path if final_model_path.exists() else fallback_model_path

            manifest_path = MODELS_DIR / "feature_manifest.json"

            final_meta_path = MODELS_DIR / "final_deployment_metadata.json"
            fallback_meta_path = MODELS_DIR / "model_metadata.json"
            meta_path = final_meta_path if final_meta_path.exists() else fallback_meta_path
            print("[INFO] RUNTIME: Defaulting to frozen production deployment model.")

        lookup_path = CONFIG_DIR / "district_context_lookup.json"

        required = [model_path, manifest_path, meta_path, lookup_path, FINAL_DATA_PATH]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing required application artifacts: {}".format(missing))

        if sklearn.__version__ != MODEL_RUNTIME_VERSION:
            print(
                "[WARNING] Model runtime expects scikit-learn {} but current runtime is {}. "
                "Run the application with: py -3.8 app.py".format(
                    MODEL_RUNTIME_VERSION, sklearn.__version__
                )
            )

        MODEL = joblib.load(model_path)
        FEATURE_MANIFEST = _read_json(manifest_path)
        MODEL_METADATA = _read_json(meta_path)
        DISTRICT_LOOKUP = _read_json(lookup_path)
        PREDICTORS = FEATURE_MANIFEST.get("certified_21_predictors") or FEATURE_MANIFEST.get("predictors") or []
        _load_threshold(threshold_override)
        _load_predict_module()

        DF_EIPS = pd.read_csv(PLANNING_DIR / "district_eips_rankings.csv")
        DF_STATIONS = pd.read_csv(FINAL_DATA_PATH)

        DF_METRICS, _ = _read_csv_candidates([
            RESULTS_DIR / "model_comparison.csv",
            RESULTS_DIR / "metrics" / "model_comparison.csv",
        ])
        DF_CV, _ = _read_csv_candidates([
            METRICS_DIR / "cv_metrics.csv",
            RESULTS_DIR / "cv_metrics.csv",
        ])
        DF_OUTER, _ = _read_csv_candidates([
            RESULTS_DIR / "outer_test_metrics.csv",
            RESULTS_DIR / "metrics" / "outer_test_metrics.csv",
        ])
        DF_SHAP_IMPORTANCE, _ = _read_csv_candidates([
            XAI_DIR / "shap_feature_importance.csv",
            XAI_DIR / "feature_importance.csv",
        ])
        DF_CF, _ = _read_csv_candidates([
            XAI_DIR / "counterfactual_analysis.csv",
            COUNTERFACTUALS_DIR / "counterfactual_analysis.csv",
        ])

        print("[OK] Frozen master station dataset loaded: {} rows x {} columns".format(
            len(DF_STATIONS), len(DF_STATIONS.columns)
        ))
        print("[OK] EIPS district rankings loaded: {} districts".format(len(DF_EIPS)))
        print("[OK] Selected model loaded: {}".format(type(MODEL).__name__))
        print("[OK] Certified predictors: {}".format(len(PREDICTORS)))
        print("[OK] Frozen decision threshold: {:.2f}".format(THRESHOLD))
        print("[OK] Prediction module loaded from 08_predict.py")
        print("[OK] Final application backend initialization complete.")
    except Exception as exc:
        BACKEND_ERROR = str(exc)
        print("[ERROR] Backend initialization failed: {}".format(exc))

    print("=" * 80)


@app.route("/")
def serve_index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:path>")
def serve_static(path):
    target = FRONTEND_DIR / path
    if target.exists() and target.is_file():
        return send_from_directory(str(FRONTEND_DIR), path)
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/results/figures/<path:filename>")
def serve_figures(filename):
    return send_from_directory(str(FIGURES_DIR), filename)


@app.route("/results/gis/<path:filename>")
def serve_gis(filename):
    return send_from_directory(str(GIS_DIR), filename)


@app.route("/api/health", methods=["GET"])
def api_health():
    champion = _select_champion_record()
    tier1_count = int((DF_EIPS["rank"] <= 10).sum()) if DF_EIPS is not None and "rank" in DF_EIPS.columns else 0
    fast_count = 0
    slow_count = 0
    unknown_count = 0
    if DF_STATIONS is not None and "fast_charging_flag" in DF_STATIONS.columns:
        flags = pd.to_numeric(DF_STATIONS["fast_charging_flag"], errors="coerce")
        fast_count = int((flags == 1).sum())
        slow_count = int((flags == 0).sum())
        unknown_count = int(flags.isna().sum())

    grid_gwh = _grid_context_summary()
    return jsonify(_json_safe({
        "status": "healthy" if BACKEND_ERROR is None and MODEL is not None else "degraded",
        "project_title": PROJECT_TITLE,
        "domain": "Telangana, India",
        "district_count": int(len(DF_EIPS)) if DF_EIPS is not None else 0,
        "station_count": int(len(DF_STATIONS)) if DF_STATIONS is not None else 0,
        "station_class_counts": {
            "dc_fast": fast_count,
            "ac_slow": slow_count,
            "unknown": unknown_count,
        },
        "observed_tgspdcl_districts": int(sum(
            1 for _, r in DF_EIPS.iterrows()
            if _numeric_from_row(r, "grid_observed_gwh", "district_measured_discom_ev_kwh") not in (None, 0.0)
        )) if DF_EIPS is not None else 0,
        "grid_throughput_gwh": grid_gwh,
        "model_loaded": MODEL is not None,
        "model_name": type(MODEL).__name__ if MODEL is not None else None,
        "sklearn_version": sklearn.__version__,
        "expected_model_runtime": MODEL_RUNTIME_VERSION,
        "n_predictors": len(PREDICTORS),
        "decision_threshold": THRESHOLD,
        "selected_model": MODEL_METADATA.get("selected_champion_model"),
        "cv_pr_auc": _metric_record(_normalise_model_key(MODEL_METADATA.get("selected_champion_model", "Random_Forest"))).get("cv_pr_auc"),
        "outer_test_pr_auc": champion.get("outer_test_pr_auc") or _metric_record(_normalise_model_key(MODEL_METADATA.get("selected_champion_model", "Random_Forest"))).get("outer_test_pr_auc"),
        "outer_test_accuracy": champion.get("outer_test_accuracy"),
        "tier1_districts_count": tier1_count,
        "backend_error": BACKEND_ERROR,
    }))


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    model_records = [_metric_record(name) for name in _collect_model_names()]
    selected_model = _normalise_model_key(
        MODEL_METADATA.get("selected_champion_model", "Random_Forest")
    )
    champion = next((r for r in model_records if r["model_name"] == selected_model), {})

    outer_station_count = None
    outer_district_count = None
    development_size = MODEL_METADATA.get("development_stations")
    development_districts = MODEL_METADATA.get("development_districts")

    if DF_OUTER is not None:
        if "station_id" in DF_OUTER.columns:
            outer_station_count = int(DF_OUTER["station_id"].nunique())
        elif len(DF_OUTER) == 1:
            outer_station_count = _numeric_from_row(
                DF_OUTER.iloc[0], "outer_test_size", "test_size"
            )
        if "district" in DF_OUTER.columns:
            outer_district_count = int(DF_OUTER["district"].nunique())

    if outer_station_count is None:
        outer_station_count = MODEL_METADATA.get("outer_test_size")
    if outer_district_count is None:
        outer_districts = FEATURE_MANIFEST.get("outer_test_districts", [])
        outer_district_count = len(outer_districts) if isinstance(outer_districts, list) else None

    return jsonify(_json_safe({
        "status": "success",
        "selected_model": selected_model,
        "decision_threshold": THRESHOLD,
        "models": model_records,
        "cv_metrics": {
            "cv_pr_auc": champion.get("cv_pr_auc"),
            "cv_pr_auc_std": champion.get("cv_pr_auc_std"),
            "cv_balanced_accuracy": champion.get("cv_balanced_accuracy"),
            "cv_folds": 4,
            "grouping": "district-level GroupKFold",
            "development_stations": development_size,
            "development_districts": development_districts,
        },
        "outer_test_metrics": {
            "outer_test_pr_auc": champion.get("outer_test_pr_auc"),
            "outer_test_roc_auc": champion.get("outer_test_roc_auc"),
            "outer_test_accuracy": champion.get("outer_test_accuracy"),
            "outer_test_balanced_acc": champion.get("outer_test_balanced_acc"),
            "outer_test_macro_f1": champion.get("outer_test_macro_f1"),
            "outer_test_stations": outer_station_count,
            "outer_test_districts": outer_district_count,
        },
        "evaluation_strategy": "District-held-out outer test with 4-fold GroupKFold by district on the development set.",
    }))


def _canonicalize_counterfactual_records(df):
    if df is None or len(df) == 0:
        return []
    aliases = {
        "station_id": ["station_id", "Station ID", "id"],
        "district": ["district", "District"],
        "original_prob_fast": ["original_prob_fast", "original_probability", "original_p_fast", "original_p"],
        "counterfactual_prob_fast": ["counterfactual_prob_fast", "counterfactual_probability", "counterfactual_p_fast", "target_prob_fast", "target_p"],
        "actionable_modifications": ["actionable_modifications", "actionable_guidance", "guidance", "recommendation", "changed_lever"],
    }
    records = []
    for _, row in df.head(15).iterrows():
        rec = {}
        for canonical, keys in aliases.items():
            value = None
            for key in keys:
                if key in row and pd.notna(row[key]):
                    value = row[key]
                    break
            rec[canonical] = value
        records.append(rec)
    return records


@app.route("/api/xai", methods=["GET"])
def api_xai():
    shap_list = DF_SHAP_IMPORTANCE.to_dict(orient="records") if DF_SHAP_IMPORTANCE is not None else []
    cf_list = _canonicalize_counterfactual_records(DF_CF)
    return jsonify(_json_safe({
        "status": "success",
        "decision_threshold": THRESHOLD,
        "shap_importance": shap_list,
        "counterfactual_analysis": cf_list,
    }))


@app.route("/api/planning", methods=["GET"])
def api_planning():
    eips_list = DF_EIPS.to_dict(orient="records") if DF_EIPS is not None else []
    ahp_path = PLANNING_DIR / "ahp_weights.csv"
    ahp_weights = pd.read_csv(ahp_path).to_dict(orient="records") if ahp_path.exists() else []
    return jsonify(_json_safe({
        "status": "success",
        "total_districts": len(eips_list),
        "ahp_weights": ahp_weights,
        "district_rankings": eips_list,
        "eips_rankings": eips_list,
    }))


@app.route("/api/stations", methods=["GET"])
def api_stations():
    if DF_STATIONS is None:
        return jsonify({"status": "error", "message": "Station dataset not loaded"}), 500

    safe_cols = [
        "station_id", "station_name", "district", "latitude", "longitude",
        "distance_to_highway_km", "distance_to_city_center_km", "nearby_poi_count",
        "poi_diversity_index", "road_type", "fast_charging_flag",
    ]
    cols = [c for c in safe_cols if c in DF_STATIONS.columns]
    stations = DF_STATIONS[cols].replace({np.nan: None}).to_dict(orient="records")
    for row in stations:
        flag = row.get("fast_charging_flag")
        if flag in (1, 1.0, "1"):
            row["fast_charging_desc"] = "DC Fast (>= 50 kW)"
        elif flag in (0, 0.0, "0"):
            row["fast_charging_desc"] = "AC / Non-Fast (observed)"
        else:
            row["fast_charging_desc"] = "Unlabeled / Unknown"

    return jsonify(_json_safe({
        "status": "success",
        "count": len(stations),
        "coverage_note": "Collected/verified station records; not a claim of a complete statewide census.",
        "stations": stations,
    }))


@app.route("/api/districts", methods=["GET"])
def api_districts():
    districts = sorted(list(DISTRICT_LOOKUP.keys()))
    return jsonify(_json_safe({"status": "success", "districts": districts}))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if MODEL is None or PREDICT_MODULE is None:
        return jsonify({"status": "error", "message": "Inference engine is not initialized."}), 500

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not data:
        return jsonify({"status": "error", "message": "A non-empty JSON object is required."}), 400

    try:
        X, feature_dict, district, lat, lon, road_type = PREDICT_MODULE.validate_and_construct_features(
            data,
            PREDICTORS,
            DISTRICT_LOOKUP,
        )
        result = PREDICT_MODULE.build_result(
            station_data=data,
            model=MODEL,
            selected_model_name=_normalise_model_key(
                MODEL_METADATA.get("selected_champion_model", "Random_Forest")
            ),
            predictors=PREDICTORS,
            threshold=THRESHOLD,
            df_eips=DF_EIPS,
            X=X,
            feature_dict=feature_dict,
            district=district,
            lat=lat,
            lon=lon,
            road_type=road_type,
        )
        return jsonify(_json_safe(result))
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


initialize_backend()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EV Infrastructure Planning Web Application")
    parser.add_argument("--dataset", choices=["telangana", "india"], default="telangana",
                        help="Dataset/Model scope: 'telangana' (Port 5000) or 'india' (Port 5001)")
    parser.add_argument("--india", action="store_true", help="Launch Pan-India 52K dataset dashboard on Port 5001")
    parser.add_argument("--port", type=int, default=None, help="Port to run server on")
    args, _ = parser.parse_known_args()

    if args.india or args.dataset == "india" or os.environ.get("DATASET", "").lower() == "india":
        port = args.port or int(os.environ.get("PORT", "5001"))
        print("\n" + "=" * 80)
        print(f"[INFO] LAUNCHING PAN-INDIA 52,718 STATIONS WEB APPLICATION")
        print(f"[INFO] Domain: All 36 States & Union Territories of India")
        print(f"[INFO] Open: http://127.0.0.1:{port}")
        print("=" * 80)
        import app_india_wide
        app_india_wide.app.run(host="0.0.0.0", port=port, debug=False)
    else:
        port = args.port or int(os.environ.get("PORT", "5000"))
        print("\n" + "=" * 80)
        print(f"[INFO] LAUNCHING TELANGANA 33 DISTRICTS BASELINE WEB APPLICATION")
        print(f"[INFO] Domain: Telangana State (166 Verified Public Stations)")
        print(f"[INFO] Open: http://127.0.0.1:{port}")
        print("=" * 80)
        app.run(host="0.0.0.0", port=port, debug=False)
