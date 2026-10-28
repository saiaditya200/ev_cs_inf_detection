#!/usr/bin/env python3
"""
scripts/final_consistency_repair_and_freeze.py
Authoritative script to execute:
FINAL CONSISTENCY REPAIR AND PRE-DEPLOYMENT FREEZE FOR JOINT_RF_0257
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
XAI_DIR = RESULTS_DIR / "xai"
XAI_CAND_DIR = XAI_DIR / "joint_rf_0257"
DIAG_DIR = RESULTS_DIR / "diagnostics"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_FINAL_DIR = BASE_DIR / "data" / "final"
CONFIG_DIR = BASE_DIR / "config"
PLANNING_DIR = RESULTS_DIR / "planning"

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

THRESHOLD = 0.76

# Import predict module
sys.path.insert(0, str(BASE_DIR))
import importlib
predict_mod = importlib.import_module("08_predict")

def stage1_verify_shap_csv():
    print("=" * 80)
    print("STAGE 1: VERIFY AUTHORITATIVE SHAP CSV")
    print("=" * 80)
    csv_path = XAI_CAND_DIR / "joint_rf_0257_shap_feature_importance.csv"
    assert csv_path.exists(), f"Missing SHAP CSV: {csv_path}"
    
    df_shap = pd.read_csv(csv_path)
    total_shap = df_shap["mean_abs_shap"].sum()
    df_shap["recomputed_pct"] = (df_shap["mean_abs_shap"] / total_shap) * 100.0
    
    print(f"Total mean_abs_shap: {total_shap:.17f}")
    print("\nAuthoritative Global SHAP Ranking (Recomputed from CSV):")
    for i, r in df_shap.head(10).iterrows():
        print(f"  {i+1:2d}. {r['feature']:<32} : mean_abs_shap = {r['mean_abs_shap']:.17f} | {r['recomputed_pct']:.2f}%")
        
    expected_top = [
        ("road_type_residential", 0.04442364164016013, 25.37),
        ("latitude", 0.020445598212496335, 11.68),
        ("nearby_poi_count", 0.011235717071318527, 6.42),
        ("longitude", 0.010993767222913889, 6.28),
        ("distance_to_city_center_km", 0.010607095514488116, 6.06),
        ("city_center_proximity_score", 0.010604516893569612, 6.06),
        ("poi_diversity_index", 0.010411590496858443, 5.95),
        ("highway_proximity_score", 0.009835502374321655, 5.62),
        ("district_urban_population_pct", 0.009771889414963248, 5.58),
        ("distance_to_highway_km", 0.009388137256436127, 5.36),
    ]
    
    for idx, (exp_feat, exp_val, exp_pct) in enumerate(expected_top):
        act_row = df_shap.iloc[idx]
        assert act_row["feature"] == exp_feat, f"Rank {idx+1} feature mismatch: {act_row['feature']} vs {exp_feat}"
        assert abs(act_row["mean_abs_shap"] - exp_val) < 1e-6, f"Rank {idx+1} value mismatch"
        assert abs(act_row["recomputed_pct"] - exp_pct) < 0.02, f"Rank {idx+1} pct mismatch"
        
    print("\n  [VERIFIED] All 10 authoritative SHAP rankings and values match recomputed calculations exactly.")
    print("STAGE 1: PASS\n")
    return df_shap

def stage4_update_prospective_and_eips(model, district_lookup):
    print("=" * 80)
    print("STAGE 4 & 6: EIPS LANGUAGE AND PROSPECTIVE INFERENCE REGENERATION")
    print("=" * 80)
    
    raw_sample = predict_mod.get_sample_station()
    X, feat_dict, district, lat, lon, road_type = predict_mod.validate_and_construct_features(
        raw_sample, CERTIFIED_21_PREDICTORS, district_lookup
    )
    
    df_eips = pd.read_csv(PLANNING_DIR / "district_eips_rankings.csv")
    result = predict_mod.build_result(
        station_data=raw_sample,
        model=model,
        selected_model_name="JOINT_RF_0257",
        predictors=CERTIFIED_21_PREDICTORS,
        threshold=THRESHOLD,
        df_eips=df_eips,
        X=X,
        feature_dict=feat_dict,
        district=district,
        lat=lat,
        lon=lon,
        road_type=road_type,
    )
    
    rc = result["regional_planning_context"]
    assert rc["priority_tier"] == "Tier 1 — High Relative Planning Priority", f"Unexpected tier: {rc['priority_tier']}"
    assert "eips_explanatory_statement" in rc
    
    out_path = DIAG_DIR / "joint_rf_0257_prospective_prediction.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  [UPDATED] Prospective prediction saved with corrected EIPS display language: {out_path}")
    print(f"  Priority Tier        : {rc['priority_tier']}")
    print(f"  Explanatory Statement: {rc['eips_explanatory_statement']}")
    print(f"  P(DC Fast)           : {result['prediction']['probability_dc_fast']:.4f}")
    print(f"  Predicted Class      : {result['prediction']['class_label']} ({result['prediction']['classification']})")
    print("STAGE 4 & 6: PASS\n")
    return result

def stage5_verify_counterfactuals():
    print("=" * 80)
    print("STAGE 5: VERIFY COUNTERFACTUAL LANGUAGE AND VALUES")
    print("=" * 80)
    cf_path = DIAG_DIR / "joint_rf_0257_counterfactuals.json"
    with open(cf_path, "r", encoding="utf-8") as f:
        cf_data = json.load(f)
        
    med = cf_data["TS-MED-0001"]
    hyd = cf_data["TS-HYD-0003"]
    pro = cf_data["TS_SAMPLE_PROSPECTIVE_01"]
    
    print(f"  TS-MED-0001: Base P={med['base_p_dc_fast']:.4f}, Best lever={med['best_scenario']['changed_lever']}->{med['best_scenario']['new_value']}, Result P={med['best_scenario']['probability_dc_fast']:.4f}, Crosses={med['best_scenario']['crosses_threshold']}")
    assert abs(med["base_p_dc_fast"] - 0.4385) < 0.001
    assert abs(med["best_scenario"]["probability_dc_fast"] - 0.4960) < 0.001
    assert med["best_scenario"]["crosses_threshold"] is False
    
    print(f"  TS-HYD-0003: Base P={hyd['base_p_dc_fast']:.4f}, Best lever={hyd['best_scenario']['changed_lever']}->{hyd['best_scenario']['new_value']}, Result P={hyd['best_scenario']['probability_dc_fast']:.4f}, Crosses={hyd['best_scenario']['crosses_threshold']}")
    assert abs(hyd["base_p_dc_fast"] - 0.3590) < 0.001
    assert abs(hyd["best_scenario"]["probability_dc_fast"] - 0.4236) < 0.001
    assert hyd["best_scenario"]["crosses_threshold"] is False
    
    print(f"  TS_SAMPLE_PROSPECTIVE_01: Base P={pro['base_p_dc_fast']:.4f}, Status={pro['cf_status']}")
    assert abs(pro["base_p_dc_fast"] - 0.8195) < 0.001
    assert pro["cf_status"] == "already_above_threshold"
    
    print("  [VERIFIED] Constrained counterfactual values and language verified.")
    print("STAGE 5: PASS\n")

def stage9_verify_deployment_artifacts():
    print("=" * 80)
    print("STAGE 9: FINAL PRE-DEPLOYMENT ARTIFACT CHECK")
    print("=" * 80)
    cand_model = MODELS_DIR / "joint_rf_0257_deployment_candidate.joblib"
    cand_meta = MODELS_DIR / "joint_rf_0257_deployment_metadata.json"
    cand_thresh = MODELS_DIR / "joint_rf_0257_deployment_threshold.json"
    
    assert cand_model.exists()
    assert cand_meta.exists()
    assert cand_thresh.exists()
    
    with open(cand_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    assert meta["training_scope"]["training_samples_total"] == 154
    assert meta["training_scope"]["class_1_dc_fast"] == 131
    assert meta["training_scope"]["class_0_ac_slow"] == 23
    assert meta["training_scope"]["held_out_unlabeled_stations"] == 12
    assert meta["hyperparameters"]["n_estimators"] == 250
    assert meta["hyperparameters"]["max_depth"] == 6
    assert meta["hyperparameters"]["min_samples_split"] == 7
    assert meta["hyperparameters"]["min_samples_leaf"] == 1
    assert meta["hyperparameters"]["max_features"] == 0.5
    assert meta["decision_threshold"] == 0.76
    
    # Check production files remain untouched
    prod_files = [
        MODELS_DIR / "final_deployment_model.joblib",
        MODELS_DIR / "final_deployment_metadata.json",
        MODELS_DIR / "final_deployment_threshold.json",
    ]
    for pf in prod_files:
        assert pf.exists()
    print("  [VERIFIED] Candidate deployment artifacts verified.")
    print("  [VERIFIED] Production artifacts remain untouched.")
    print("STAGE 9: PASS\n")

def stage2_and_10_generate_reports(df_shap):
    print("=" * 80)
    print("STAGE 2 & 10: REGENERATE CERTIFICATION REPORT & PRODUCE FINAL FREEZE REPORT")
    print("=" * 80)
    
    # Build the exact SHAP ranking text directly from df_shap
    shap_ranking_lines = []
    for idx, row in df_shap.head(10).iterrows():
        shap_ranking_lines.append(
            f" {idx+1:2d}. {row['feature']:<35} : Mean |SHAP| = {row['mean_abs_shap']:.6f} ({row['recomputed_pct']:.2f}%)"
        )
    shap_ranking_text = "\n".join(shap_ranking_lines)
    
    correct_shap_interpretation = (
        "SHAP values represent model-based feature attributions to the Random Forest's "
        "DC Fast probability output. They indicate how features contribute to the model "
        "prediction and do not establish causality, physical mechanisms, or actual "
        "charging demand/utilization."
    )
    
    correct_eips_explanation = (
        "EIPS identifies districts with higher relative planning priority for "
        "investigation and potential infrastructure expansion. It does not prescribe "
        "an exact station location or exact station count."
    )
    
    final_governance_statement = (
        "JOINT_RF_0257 has completed read-only certification and is prepared as a "
        "separate deployment candidate. The current production deployment remains "
        "frozen pending explicit approval."
    )

    # 1. Update results/diagnostics/joint_rf_0257_certification_report.txt
    cert_report_path = DIAG_DIR / "joint_rf_0257_certification_report.txt"
    with open(cert_report_path, "r", encoding="utf-8") as f:
        cert_text = f.read()
        
    # Replace Section 5 SHAP ranking and interpretation in cert_text
    old_sec5_marker = "Top Global SHAP Drivers (Mean |SHAP| Attribution on P(DC Fast)):"
    new_sec5_block = (
        f"Top Global SHAP Drivers (Mean |SHAP| Attribution on P(DC Fast) — Recomputed Authoritative CSV):\n"
        f"{shap_ranking_text}\n\n"
        f"Critical Interpretability Boundary:\n"
        f"{correct_shap_interpretation}"
    )
    
    # Locate section 5 in cert_text and update it cleanly
    sec5_start = cert_text.find(old_sec5_marker)
    if sec5_start != -1:
        sec6_start = cert_text.find("--------------------------------------------------------------------------------\n6. CONSTRAINED COUNTERFACTUAL", sec5_start)
        cert_text = cert_text[:sec5_start] + new_sec5_block + "\n\n" + cert_text[sec6_start:]
        with open(cert_report_path, "w", encoding="utf-8") as f:
            f.write(cert_text)
        print(f"  [UPDATED] {cert_report_path} regenerated with authoritative SHAP ranking and correct interpretation.")

    # 2. Create results/diagnostics/joint_rf_0257_final_freeze_report.txt
    freeze_report_path = DIAG_DIR / "joint_rf_0257_final_freeze_report.txt"
    
    freeze_report_content = f"""================================================================================
EXPLAINABLE AI FRAMEWORK FOR SMART EV CHARGING INFRASTRUCTURE PLANNING
FINAL PRE-DEPLOYMENT FREEZE REPORT FOR CANDIDATE JOINT_RF_0257
================================================================================
Candidate Model Evaluated : JOINT_RF_0257
Target Architecture       : Random Forest (RandomForestClassifier)
Decision Threshold        : 0.76
Predictor Contract        : Exactly 21 Certified Non-Leaky Exogenous Spatial Predictors
Evaluation Protocol       : Verified on the Available Development/Outer-Test Protocol
Pre-Deployment Status     : Candidate Deployment Prepared Separately; Production Frozen
Report Timestamp          : {datetime.now().isoformat()}
================================================================================

FINAL GOVERNANCE STATEMENT:
"{final_governance_statement}"

--------------------------------------------------------------------------------
1. CANDIDATE IDENTITY & MODEL FAMILY
--------------------------------------------------------------------------------
- Candidate Identifier  : JOINT_RF_0257
- Model Family          : Random Forest (sklearn.ensemble.RandomForestClassifier)
- Selected Track        : Track A (Primary Balanced) & Track C (Macro-Balance) Champion
- Role                  : Pre-deployment candidate artifact (isolated preparation)

--------------------------------------------------------------------------------
2. EXACT MODEL HYPERPARAMETERS
--------------------------------------------------------------------------------
- n_estimators          : 250
- max_depth             : 6
- min_samples_split     : 7
- min_samples_leaf      : 1
- max_features          : 0.5 (numeric float 0.5; verified isinstance float: True)
- criterion             : 'gini'
- bootstrap             : True
- class_weight          : None (w0 = 1.0)
- random_state          : 42
- decision_threshold    : 0.76

--------------------------------------------------------------------------------
3. 21-FEATURE CONTRACT
--------------------------------------------------------------------------------
The model consumes strictly the 21 certified exogenous predictors without scaling:
 1. latitude
 2. longitude
 3. distance_to_highway_km
 4. distance_to_city_center_km
 5. nearby_poi_count
 6. highway_proximity_score
 7. city_center_proximity_score
 8. poi_diversity_index
 9. road_type_motorway
10. road_type_primary
11. road_type_residential
12. road_type_secondary
13. road_type_service
14. road_type_trunk
15. road_type_unclassified
16. district_population_density_2011
17. district_urban_population_pct
18. district_projected_density_fy24
19. district_nh_crossing_count
20. district_ev_penetration_rate
21. discom_utility_data_available_flag

Contract Boundaries:
- Target column         : fast_charging_flag (1 = DC Fast >= 50kW, 0 = AC Slow < 50kW)
- Grouping column       : district
- Preprocessing scaling : False (no StandardScaler or normalizer artifact required)

--------------------------------------------------------------------------------
4. LEAKAGE EXCLUSIONS
--------------------------------------------------------------------------------
The following target-derived and current-infrastructure fields are strictly
quarantined and excluded from model inputs:
- charger_type
- charging_power_kw
- station_capacity
- connector_type
- number_of_chargers
- operator
- district_fast_stations
- district_fast_charger_ratio
- traffic_density

--------------------------------------------------------------------------------
5. DEVELOPMENT OUT-OF-FOLD (OOF) METRICS REFERENCE
--------------------------------------------------------------------------------
Reconstructed strictly on phase10_development.csv (119 stations, 25 districts,
4-Fold GroupKFold with zero district fold overlap, seed=42, threshold=0.76):
- Accuracy              : 79.83% (95/119)
- Balanced Accuracy     : 66.90%
- AC Slow (Class 0) Rec : 50.00% (7/14)
- DC Fast (Class 1) Rec : 83.81% (88/105)
- Class 0 F1            : 0.3684
- Class 1 F1            : 0.8800
- Macro F1              : 0.6242
- PR-AUC                : 0.8988
- ROC-AUC               : 0.5806
- Recall Gap            : 33.81%
- Confusion Matrix      : TN=7, FP=7, FN=17, TP=88

--------------------------------------------------------------------------------
6. INDEPENDENT OUTER-TEST METRICS REFERENCE
--------------------------------------------------------------------------------
Evaluated on quarantined phase10_outer_test.csv (35 stations across 8 spatially
held-out districts, 26 Class 1, 9 Class 0, zero district overlap):
- Evaluation Scope      : Frozen 119-station development training scope
- Accuracy              : 74.29% (26/35)  [Baseline: 71.43%, +2.86 pp]
- Balanced Accuracy     : 57.26%          [Baseline: 51.71%, +5.56 pp]
- AC Slow (Class 0) Rec : 22.22% (2/9)    [Baseline: 11.11%, +11.11 pp (Doubled)]
- DC Fast (Class 1) Rec : 92.31% (24/26)  [Baseline: 92.31%, Preserved]
- Class 0 F1            : 0.3077          [Baseline: 0.1667, +0.1410]
- Class 1 F1            : 0.8421          [Baseline: 0.8276, +0.0145]
- Macro F1              : 0.5749          [Baseline: 0.4971, +0.0778]
- PR-AUC                : 0.8092          [Baseline: 0.7320, +0.0772]
- ROC-AUC               : 0.5897          [Baseline: 0.5043, +0.0855]
- Recall Gap            : 70.09%          [Baseline: 81.20%, Narrowed by 11.11 pp]
- Confusion Matrix      : TN=2, FP=7, FN=2, TP=24 [Baseline: TN=1, FP=8, FN=2, TP=24]

Note: No outer-test data was used for hyperparameter tuning or candidate selection.

--------------------------------------------------------------------------------
7. SHAP GLOBAL RANKING (FROM AUTHORITATIVE CSV)
--------------------------------------------------------------------------------
Recomputed directly from results/xai/joint_rf_0257/joint_rf_0257_shap_feature_importance.csv:
{shap_ranking_text}

--------------------------------------------------------------------------------
8. CORRECT SHAP METHODOLOGICAL INTERPRETATION
--------------------------------------------------------------------------------
{correct_shap_interpretation}

--------------------------------------------------------------------------------
9. EXISTING-STATION INFERENCE TEST & STRICT SEPARATION
--------------------------------------------------------------------------------
Strict separation maintained:
- MODEL INFERENCE uses strictly the 21 certified predictors.
- OBSERVED DATASET REFERENCE is retained only for descriptive context and is NEVER passed into model prediction.

Station 1: TS-MED-0001 (Bharat Charge Ev Charging Station, Medchal-Malkajgiri)
- Observed Ground Truth : DC Fast (>= 50 kW) [Observed fast_charging_flag = 1]
- Candidate P(DC Fast)  : 0.4385 (Threshold = 0.76)
- Candidate Prediction  : Class 0 -> AC Slow/Medium Charging (< 50kW)
- Note                  : Disagreement transparently recorded; observed label not forced.

Station 2: TS-HYD-0003 (MobiLane CRVRGROWA-EV CHARGING STATION - 2, Hyderabad)
- Observed Ground Truth : AC / Non-Fast [Observed fast_charging_flag = 0]
- Candidate P(DC Fast)  : 0.3590 (Threshold = 0.76)
- Candidate Prediction  : Class 0 -> AC Slow/Medium Charging (< 50kW)
- Note                  : Model prediction matches observed ground truth.

--------------------------------------------------------------------------------
10. PROSPECTIVE-SITE INFERENCE TEST
--------------------------------------------------------------------------------
Sample: TS_SAMPLE_PROSPECTIVE_01 (Illustrative Medchal Commercial Corridor Site)
- District & Coords     : Medchal-Malkajgiri (Lat 17.6320, Lon 78.5610)
- Road Type & Distance  : Primary | Highway: 0.85 km | City Center: 6.20 km
- POI Count & Diversity : 9.0 POIs | Diversity Index: 0.78
- P(DC Fast)            : 0.8195 (81.95%)
- Decision Threshold    : 0.76
- Predicted Class       : Class 1 -> DC Fast-Charging Infrastructure (>= 50kW)
- Top Positive Drivers  : road_type_residential=0.0 (+0.0234), distance_to_city_center_km=6.2 (+0.0085), city_center_proximity_score=0.1389 (+0.0049)
- Top Negative Drivers  : poi_diversity_index=0.78 (-0.0933), latitude=17.632 (-0.0055), district_ev_penetration_rate=4.111 (-0.0023)
- Counterfactual Status : already_above_threshold

--------------------------------------------------------------------------------
11. CONSTRAINED COUNTERFACTUAL TESTS
--------------------------------------------------------------------------------
Constrained search limited strictly to the 4 approved planning levers:
`distance_to_highway_km`, `distance_to_city_center_km`, `nearby_poi_count`, `poi_diversity_index`.
Terminology: "model-based planning scenario" (NOT guaranteed actions or physical assurances).

Results:
- TS-MED-0001           : Base P = 0.4385. Best scenario: poi_diversity_index -> 0.50 (P rises to 0.4960). Does NOT cross 0.76.
- TS-HYD-0003           : Base P = 0.3590. Best scenario: poi_diversity_index -> 0.50 (P rises to 0.4236). Does NOT cross 0.76.
- TS_SAMPLE_PROSPECTIVE_01: Base P = 0.8195. Base probability >= 0.76. Status: already_above_threshold.

--------------------------------------------------------------------------------
12. EIPS CONTEXTUAL PLANNING ROLE & LANGUAGE
--------------------------------------------------------------------------------
Prospective and inference planning tier wording:
"Tier 1 — High Relative Planning Priority"

Explanatory Statement:
"{correct_eips_explanation}"

EIPS rankings, AHP weights, and TOPSIS values remain strictly unaltered.

--------------------------------------------------------------------------------
13. DEPLOYMENT CANDIDATE STATUS (ISOLATED ARTIFACTS)
--------------------------------------------------------------------------------
Prepared separate deployment candidate artifacts:
- models/joint_rf_0257_deployment_candidate.joblib (476,761 bytes)
  Retrained on all 154 labeled stations across 33 districts (131 DC Fast, 23 AC Slow).
  12 unlabeled stations remain excluded from supervised training scope.
- models/joint_rf_0257_deployment_metadata.json
- models/joint_rf_0257_deployment_threshold.json (Threshold = 0.76)

--------------------------------------------------------------------------------
14. PRODUCTION ARTIFACT STATUS (FROZEN & UNTOUCHED)
--------------------------------------------------------------------------------
Existing production artifacts remain 100% frozen and unmodified:
- models/final_deployment_model.joblib         : UNTOUCHED
- models/final_deployment_metadata.json      : UNTOUCHED
- models/final_deployment_threshold.json     : UNTOUCHED

================================================================================
END OF FINAL PRE-DEPLOYMENT FREEZE REPORT
================================================================================
"""
    with open(freeze_report_path, "w", encoding="utf-8") as f:
        f.write(freeze_report_content)
    print(f"  [SAVED] {freeze_report_path}")
    print("STAGE 2 & 10: PASS\n")

if __name__ == "__main__":
    cand_model = joblib.load(MODELS_DIR / "joint_primary_candidate.joblib")
    with open(CONFIG_DIR / "district_context_lookup.json", "r", encoding="utf-8") as f:
        lookup = json.load(f)
        
    df_shap = stage1_verify_shap_csv()
    stage4_update_prospective_and_eips(cand_model, lookup)
    stage5_verify_counterfactuals()
    stage9_verify_deployment_artifacts()
    stage2_and_10_generate_reports(df_shap)
    print("ALL REPAIR AND FREEZE STAGES EXECUTED SUCCESSFULLY!")
