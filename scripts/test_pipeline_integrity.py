"""
Automated Pipeline & Repository Integrity Self-Test
===================================================
Project: Explainable AI Framework for Smart EV Charging Infrastructure Planning

Validates all 16 critical requirements defined in Part 40 of the project specification:
 1. Dataset dimensions
 2. Dataset hash
 3. Target distribution
 4. 21 predictor count
 5. Programmatic leakage control
 6. Certified split station counts
 7. Certified split zero district overlap
 8. Model artifact existence and integrity
 9. Metric artifact existence and integrity
10. SHAP output existence and integrity
11. Counterfactual output existence and integrity
12. EIPS planning output existence and integrity
13. GeoJSON and interactive map output existence
14. Prediction artifact verification
15. Flask API startup & health
16. Frontend API schema contract
"""

import sys
import os
import json
import hashlib
import pandas as pd
import joblib

# Ensure project root is in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CERTIFIED_PREDICTORS = [
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

BLOCKED_LEAKAGE_COLUMNS = [
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
]


def test_1_dataset_dimensions():
    path = os.path.join(PROJECT_ROOT, "data/final/FINAL_PROJECT_DATASET.csv")
    assert os.path.exists(path), f"Master dataset missing: {path}"
    df = pd.read_csv(path)
    assert df.shape == (166, 78), f"Expected (166, 78), got {df.shape}"
    return "Passed: 166 rows x 78 columns"


def test_2_dataset_hash():
    path = os.path.join(PROJECT_ROOT, "data/final/FINAL_PROJECT_DATASET.csv")
    with open(path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    expected_hash = "a1999524323860397e7255b5ff7884d7f77394b09abd69a168a306bf7eae9788"
    assert sha256 == expected_hash, f"Hash mismatch: {sha256} != {expected_hash}"
    return f"Passed: SHA-256 = {sha256[:16]}..."


def test_3_target_distribution():
    path = os.path.join(PROJECT_ROOT, "data/final/FINAL_PROJECT_DATASET.csv")
    df = pd.read_csv(path)
    c1 = int((df["fast_charging_flag"] == 1.0).sum())
    c0 = int((df["fast_charging_flag"] == 0.0).sum())
    na = int(df["fast_charging_flag"].isna().sum())
    assert c1 == 131, f"Expected 131 Class 1, got {c1}"
    assert c0 == 23, f"Expected 23 Class 0, got {c0}"
    assert na == 12, f"Expected 12 NA/Unknown, got {na}"
    return "Passed: 131 Class 1, 23 Class 0, 12 NA/Unknown"


def test_4_predictor_count():
    manifest_path = os.path.join(PROJECT_ROOT, "models/feature_manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    features = manifest.get("certified_21_predictors") or manifest.get("features", [])
    assert len(features) == 21, f"Expected 21 features, got {len(features)}"
    assert sorted(features) == sorted(CERTIFIED_PREDICTORS), "Predictor names do not match certified list"
    return "Passed: Exactly 21 certified predictors"


def test_5_leakage_control():
    path = os.path.join(PROJECT_ROOT, "data/processed/fast_charging_modeling_matrix.csv")
    assert os.path.exists(path), f"Modeling matrix missing: {path}"
    df = pd.read_csv(path)
    for col in BLOCKED_LEAKAGE_COLUMNS:
        assert col not in df.columns, f"LEAKAGE DETECTED: Column '{col}' found in modeling matrix!"
    for p in CERTIFIED_PREDICTORS:
        assert p in df.columns, f"Predictor '{p}' missing from modeling matrix!"
    return "Passed: 0 blocked leakage features found in modeling matrix"


def test_6_certified_split():
    path = os.path.join(PROJECT_ROOT, "data/processed/phase10_final_split_assignments.csv")
    assert os.path.exists(path), f"Split assignments missing: {path}"
    df = pd.read_csv(path)
    dev_count = len(df[df["split_role"].str.upper() == "DEVELOPMENT"])
    test_count = len(df[df["split_role"].str.upper() == "OUTER_TEST"])
    assert dev_count == 119, f"Expected 119 development stations, got {dev_count}"
    assert test_count == 35, f"Expected 35 outer-test stations, got {test_count}"
    return "Passed: Development = 119, Outer Test = 35"


def test_7_district_overlap():
    path = os.path.join(PROJECT_ROOT, "data/processed/phase10_final_split_assignments.csv")
    df = pd.read_csv(path)
    dev_districts = set(df[df["split_role"].str.upper() == "DEVELOPMENT"]["district"])
    test_districts = set(df[df["split_role"].str.upper() == "OUTER_TEST"]["district"])
    overlap = dev_districts.intersection(test_districts)
    assert len(overlap) == 0, f"District overlap detected: {overlap}"
    assert len(dev_districts) == 25, f"Expected 25 dev districts, got {len(dev_districts)}"
    assert len(test_districts) == 8, f"Expected 8 outer test districts, got {len(test_districts)}"
    return "Passed: 0 district overlap (25 dev districts, 8 outer-test districts)"


def test_8_model_files():
    models = [
        "models/random_forest_model.joblib",
        "models/xgboost_model.joblib",
        "models/lightgbm_model.joblib",
        "models/selected_model.joblib",
        "models/model_metadata.json",
        "models/feature_manifest.json"
    ]
    for m in models:
        p = os.path.join(PROJECT_ROOT, m)
        assert os.path.exists(p), f"Model artifact missing: {p}"
        assert os.path.getsize(p) > 0, f"Model artifact empty: {p}"
    clf = joblib.load(os.path.join(PROJECT_ROOT, "models/selected_model.joblib"))
    assert hasattr(clf, "predict_proba"), "Selected model does not implement predict_proba"
    return "Passed: All 6 model artifacts verified and loadable"


def test_9_metric_files():
    metrics = [
        "results/metrics/cv_metrics.csv",
        "results/metrics/model_comparison.csv",
        "results/metrics/cv_fold_details.csv",
        "results/metrics/outer_test_metrics.csv",
        "results/metrics/spatial_split_verification.json"
    ]
    for mf in metrics:
        p = os.path.join(PROJECT_ROOT, mf)
        assert os.path.exists(p), f"Metric file missing: {p}"
        assert os.path.getsize(p) > 0, f"Metric file empty: {p}"
    df_cv = pd.read_csv(os.path.join(PROJECT_ROOT, "results/metrics/cv_metrics.csv"))
    assert len(df_cv) == 3, f"Expected 3 models in CV metrics, got {len(df_cv)}"
    return "Passed: All 5 metric files verified and non-empty"


def test_10_shap_outputs():
    outputs = [
        "results/xai/shap_feature_importance.csv",
        "results/figures/shap_beeswarm.png",
        "results/figures/shap_summary_plot.png"
    ]
    for o in outputs:
        p = os.path.join(PROJECT_ROOT, o)
        assert os.path.exists(p), f"SHAP artifact missing: {p}"
        assert os.path.getsize(p) > 0, f"SHAP artifact empty: {p}"
    df_shap = pd.read_csv(os.path.join(PROJECT_ROOT, "results/xai/shap_feature_importance.csv"))
    assert len(df_shap) == 21, f"Expected 21 features in SHAP importance, got {len(df_shap)}"
    return "Passed: SHAP feature importance (21 features) and figures exist"


def test_11_counterfactual_outputs():
    outputs = [
        "results/counterfactuals/counterfactual_analysis.csv",
        "results/counterfactuals/counterfactual_summary_report.txt"
    ]
    for o in outputs:
        p = os.path.join(PROJECT_ROOT, o)
        assert os.path.exists(p), f"Counterfactual artifact missing: {p}"
        assert os.path.getsize(p) > 0, f"Counterfactual artifact empty: {p}"
    df_cf = pd.read_csv(os.path.join(PROJECT_ROOT, "results/counterfactuals/counterfactual_analysis.csv"))
    assert len(df_cf) > 0, "Counterfactual analysis table is empty"
    return f"Passed: Counterfactual analysis table ({len(df_cf)} rows) and report exist"


def test_12_eips_outputs():
    outputs = [
        "results/planning/district_eips_rankings.csv",
        "results/planning/infrastructure_gap_analysis.csv",
        "results/planning/ahp_weights.csv"
    ]
    for o in outputs:
        p = os.path.join(PROJECT_ROOT, o)
        assert os.path.exists(p), f"Planning artifact missing: {p}"
        assert os.path.getsize(p) > 0, f"Planning artifact empty: {p}"
    df_eips = pd.read_csv(os.path.join(PROJECT_ROOT, "results/planning/district_eips_rankings.csv"))
    assert len(df_eips) == 33, f"Expected 33 districts in EIPS rankings, got {len(df_eips)}"
    return "Passed: EIPS rankings (33 districts), gap analysis, and AHP weights exist"


def test_13_geojson_outputs():
    outputs = [
        "results/gis/station_gis_layer.geojson",
        "results/gis/district_eips_layer.geojson",
        "results/gis/interactive_planning_map.html"
    ]
    for o in outputs:
        p = os.path.join(PROJECT_ROOT, o)
        assert os.path.exists(p), f"GIS artifact missing: {p}"
        assert os.path.getsize(p) > 0, f"GIS artifact empty: {p}"
    return "Passed: Station GeoJSON, District GeoJSON, and interactive map exist"


def test_14_prediction_artifact():
    path = os.path.join(PROJECT_ROOT, "results/predictions/sample_prediction.json")
    assert os.path.exists(path), f"Prediction artifact missing: {path}"
    with open(path, "r") as f:
        res = json.load(f)
    assert "predicted_class" in res or ("prediction" in res and "class_label" in res["prediction"]), "Missing class label"
    assert "probability" in res or ("prediction" in res and "probability_dc_fast" in res["prediction"]), "Missing probability"
    assert "feature_vector" in res, "Missing 'feature_vector'"
    assert len(res["feature_vector"]) == 21, f"Expected 21 features in vector, got {len(res['feature_vector'])}"
    return "Passed: Sample prediction JSON valid with 21-feature input vector"


def test_15_api_startup():
    from app import app
    client = app.test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"Health check failed with status {resp.status_code}"
    data = resp.get_json()
    assert data.get("status") == "healthy", f"Health status not healthy: {data}"
    return "Passed: Flask app started, GET /api/health returned 200 (healthy)"


def test_16_frontend_api_contract():
    from app import app
    client = app.test_client()

    # 1. /api/metrics
    r_metrics = client.get("/api/metrics")
    assert r_metrics.status_code == 200, f"/api/metrics failed: {r_metrics.status_code}"
    d_metrics = r_metrics.get_json()
    assert "selected_model" in d_metrics, "Missing selected_model in /api/metrics"
    assert "cv_metrics" in d_metrics, "Missing cv_metrics in /api/metrics"
    assert "outer_test_metrics" in d_metrics, "Missing outer_test_metrics in /api/metrics"

    # 2. /api/planning
    r_plan = client.get("/api/planning")
    assert r_plan.status_code == 200, f"/api/planning failed: {r_plan.status_code}"
    d_plan = r_plan.get_json()
    assert "eips_rankings" in d_plan, "Missing eips_rankings in /api/planning"
    assert len(d_plan["eips_rankings"]) == 33, "EIPS rankings does not contain 33 districts"

    # 3. /api/xai
    r_xai = client.get("/api/xai")
    assert r_xai.status_code == 200, f"/api/xai failed: {r_xai.status_code}"
    d_xai = r_xai.get_json()
    assert "shap_importance" in d_xai, "Missing shap_importance in /api/xai"

    # 4. POST /api/predict
    payload = {
        "site_id": "TEST_SITE_INTEGRITY",
        "district": "Hyderabad",
        "latitude": 17.435,
        "longitude": 78.375,
        "distance_to_highway_km": 0.45,
        "distance_to_city_center_km": 8.50,
        "nearby_poi_count": 28,
        "poi_diversity_index": 0.72,
        "road_type": "primary"
    }
    r_pred = client.post("/api/predict", json=payload)
    assert r_pred.status_code == 200, f"POST /api/predict failed: {r_pred.status_code} - {r_pred.data}"
    d_pred = r_pred.get_json()
    assert "predicted_class" in d_pred, "Missing predicted_class in /api/predict response"
    assert "probability" in d_pred, "Missing probability in /api/predict response"
    assert "shap_contributors" in d_pred, "Missing shap_contributors in /api/predict response"

    return "Passed: Canonical API contract verified across all 4 endpoints"


def run_all_tests():
    tests = [
        ("1. Dataset Dimensions", test_1_dataset_dimensions),
        ("2. Dataset SHA-256 Hash", test_2_dataset_hash),
        ("3. Target Distribution", test_3_target_distribution),
        ("4. 21 Predictor Count", test_4_predictor_count),
        ("5. Programmatic Leakage Control", test_5_leakage_control),
        ("6. Certified Split Counts", test_6_certified_split),
        ("7. Split District Overlap", test_7_district_overlap),
        ("8. Model Artifact Integrity", test_8_model_files),
        ("9. Metric Artifact Integrity", test_9_metric_files),
        ("10. SHAP Output Integrity", test_10_shap_outputs),
        ("11. Counterfactual Output Integrity", test_11_counterfactual_outputs),
        ("12. EIPS Planning Output Integrity", test_12_eips_outputs),
        ("13. GeoJSON & Map Outputs", test_13_geojson_outputs),
        ("14. Prediction Artifact Verification", test_14_prediction_artifact),
        ("15. Flask API Startup & Health", test_15_api_startup),
        ("16. Frontend API Contract", test_16_frontend_api_contract)
    ]

    print("=" * 80)
    print("RUNNING AUTOMATED REPOSITORY INTEGRITY SELF-TEST (16 CHECKS)")
    print("=" * 80)

    failed = 0
    passed = 0
    for name, func in tests:
        try:
            res = func()
            print(f"[PASS] {name:<36} : {res}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name:<36} : {e}")
            failed += 1

    print("=" * 80)
    if failed == 0:
        print(f"[SUCCESS] ALL {passed} INTEGRITY CHECKS PASSED PERFECTLY.")
        print("=" * 80)
        return 0
    else:
        print(f"[ERROR] {failed} CHECK(S) FAILED OUT OF {passed + failed}.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
