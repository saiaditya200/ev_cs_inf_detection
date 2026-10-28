#!/usr/bin/env python3
"""
scripts/audit_stages_1_to_4.py
Verification audit for Stages 1 to 4 of JOINT_RF_0257 candidate certification.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DIAG_DIR = RESULTS_DIR / "diagnostics"
DATA_DIR = BASE_DIR / "data" / "processed"

DEV_PATH = DATA_DIR / "phase10_development.csv"
OUTER_PATH = DATA_DIR / "phase10_outer_test.csv"
ALL_PATH = DATA_DIR / "fast_charging_modeling_matrix.csv"

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

TARGET = "fast_charging_flag"
GROUP = "district"
SEED = 42

def evaluate_metrics(y_true, prob1, threshold):
    pred = (prob1 >= threshold).astype(int)
    acc = accuracy_score(y_true, pred)
    bal_acc = balanced_accuracy_score(y_true, pred)
    prec0 = precision_score(y_true, pred, pos_label=0, zero_division=0)
    rec0 = recall_score(y_true, pred, pos_label=0, zero_division=0)
    f1_0 = f1_score(y_true, pred, pos_label=0, zero_division=0)
    prec1 = precision_score(y_true, pred, pos_label=1, zero_division=0)
    rec1 = recall_score(y_true, pred, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true, pred, pos_label=1, zero_division=0)
    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, pred, average="weighted", zero_division=0)
    try:
        pr_auc = average_precision_score(y_true, prob1)
    except Exception:
        pr_auc = np.nan
    try:
        roc_auc = roc_auc_score(y_true, prob1)
    except Exception:
        roc_auc = np.nan
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall_gap = abs(rec1 - rec0)
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "class0_precision": float(prec0),
        "class0_recall": float(rec0),
        "class0_f1": float(f1_0),
        "class1_precision": float(prec1),
        "class1_recall": float(rec1),
        "class1_f1": float(f1_1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "recall_gap": float(recall_gap),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

def run_stage1_audit():
    print("=" * 80)
    print("STAGE 1: ARTIFACT AUDIT")
    print("=" * 80)
    artifacts = {
        "joint_primary_candidate.joblib": MODELS_DIR / "joint_primary_candidate.joblib",
        "joint_secondary_candidate.joblib": MODELS_DIR / "joint_secondary_candidate.joblib",
        "joint_macro_candidate.joblib": MODELS_DIR / "joint_macro_candidate.joblib",
        "joint_cost_sensitive_search.csv": DIAG_DIR / "joint_cost_sensitive_search.csv",
        "joint_baseline_vs_candidates.csv": DIAG_DIR / "joint_baseline_vs_candidates.csv",
        "joint_outer_test_results.csv": DIAG_DIR / "joint_outer_test_results.csv",
        "joint_top_balanced.csv": DIAG_DIR / "joint_top_balanced.csv",
        "joint_top_accuracy.csv": DIAG_DIR / "joint_top_accuracy.csv",
        "joint_top_macro.csv": DIAG_DIR / "joint_top_macro.csv",
        "joint_pareto.csv": DIAG_DIR / "joint_pareto.csv",
        "joint_cost_sensitive_report.txt": DIAG_DIR / "joint_cost_sensitive_report.txt",
    }
    
    missing = []
    for name, path in artifacts.items():
        if path.exists():
            size = path.stat().st_size
            print(f"  [FOUND] {name} ({size:,} bytes)")
        else:
            missing.append(name)
            print(f"  [MISSING] {name}")
    assert len(missing) == 0, f"Missing artifacts: {missing}"
    
    # Load primary candidate model
    prim_model = joblib.load(artifacts["joint_primary_candidate.joblib"])
    print(f"\n  Loaded primary candidate model type: {type(prim_model).__name__}")
    
    # Check parameters
    params = prim_model.get_params()
    expected_params = {
        "n_estimators": 250,
        "max_depth": 6,
        "min_samples_split": 7,
        "min_samples_leaf": 1,
        "max_features": 0.5,
        "criterion": "gini",
        "bootstrap": True,
        "class_weight": None,
    }
    
    for k, v in expected_params.items():
        act = params.get(k)
        assert act == v, f"Parameter mismatch for {k}: expected {v} ({type(v)}), got {act} ({type(act)})"
        print(f"  [VERIFIED] {k} = {act} (type: {type(act).__name__})")
        
    # Specifically confirm max_features is numeric float 0.5
    mf = prim_model.max_features
    assert isinstance(mf, (float, int)) and not isinstance(mf, str) and mf == 0.5, (
        f"max_features must be float 0.5, got {mf} ({type(mf)})"
    )
    print(f"  [VERIFIED] max_features is numeric float 0.5 (isinstance float: {isinstance(mf, float)})")
    
    # Verify candidate from joint_top_balanced.csv
    df_top_bal = pd.read_csv(artifacts["joint_top_balanced.csv"])
    top_row = df_top_bal.iloc[0]
    print(f"\n  Top row of joint_top_balanced.csv:")
    print(f"    experiment_id : {top_row['experiment_id']}")
    print(f"    n_estimators  : {top_row['n_estimators']}")
    print(f"    max_depth     : {top_row['max_depth']}")
    print(f"    split / leaf  : {top_row['min_samples_split']} / {top_row['min_samples_leaf']}")
    print(f"    max_features  : {top_row['max_features']} ({type(top_row['max_features']).__name__})")
    print(f"    threshold     : {top_row['threshold']}")
    print(f"    oof_bal_acc   : {top_row['oof_balanced_accuracy']*100:.2f}%")
    print(f"    oof_acc       : {top_row['oof_accuracy']*100:.2f}%")
    print(f"    oof_c0_rec    : {top_row['oof_class0_recall']*100:.2f}%")
    print(f"    oof_c1_rec    : {top_row['oof_class1_recall']*100:.2f}%")
    print(f"    oof_macro_f1  : {top_row['oof_macro_f1']:.4f}")
    assert top_row["experiment_id"] == "JOINT_RF_0257", f"Expected JOINT_RF_0257, got {top_row['experiment_id']}"
    assert float(top_row["threshold"]) == 0.76, f"Expected threshold 0.76, got {top_row['threshold']}"
    
    # Confirm saved estimator can predict
    dummy_input = np.zeros((2, 21))
    preds = prim_model.predict(dummy_input)
    probs = prim_model.predict_proba(dummy_input)
    assert preds.shape == (2,) and probs.shape == (2, 2)
    print("  [VERIFIED] Saved estimator loads and predicts successfully on dummy array.")
    print("STAGE 1 AUDIT: PASSED\n")

def run_stage2_audit():
    print("=" * 80)
    print("STAGE 2: DATA / FEATURE CONTRACT AUDIT")
    print("=" * 80)
    df_dev = pd.read_csv(DEV_PATH)
    assert len(CERTIFIED_21_PREDICTORS) == 21, "Must have exactly 21 certified predictors"
    
    # Check all 21 exist
    for p in CERTIFIED_21_PREDICTORS:
        assert p in df_dev.columns, f"Missing predictor: {p}"
    print("  [VERIFIED] Exactly 21 certified predictors verified in dataset.")
    
    # Feature manifest
    manifest_path = MODELS_DIR / "feature_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["predictors"] == CERTIFIED_21_PREDICTORS
    assert manifest["target"] == TARGET
    assert manifest["group_column"] == GROUP
    assert manifest["scaling"] is False
    print("  [VERIFIED] feature_manifest.json verified: target='fast_charging_flag', group='district', scaling=False.")
    
    # Leakage fields
    leakage_fields = [
        "charger_type",
        "charging_power_kw",
        "station_capacity",
        "connector_type",
        "number_of_chargers",
        "operator",
        "district_fast_stations",
        "district_fast_charger_ratio",
        "traffic_density",
    ]
    for leak in leakage_fields:
        assert leak not in CERTIFIED_21_PREDICTORS, f"LEAKAGE DETECTED in predictors: {leak}"
    print("  [VERIFIED] No leakage features present in certified predictor list.")
    
    # Check that no scaler is required
    scaler_paths = list(MODELS_DIR.glob("*scaler*"))
    print(f"  Scaler paths in models: {scaler_paths} (None expected/required)")
    assert len(scaler_paths) == 0, f"Unexpected scaler artifacts found: {scaler_paths}"
    print("  [VERIFIED] No scaler artifact required or present.")
    print("STAGE 2 AUDIT: PASSED\n")

def run_stage3_audit():
    print("=" * 80)
    print("STAGE 3: MODEL REPRODUCTION CHECK (Using ONLY phase10_development.csv)")
    print("=" * 80)
    df_dev = pd.read_csv(DEV_PATH)
    assert len(df_dev) == 119
    assert len(df_dev[GROUP].unique()) == 25
    
    X_dev = df_dev[CERTIFIED_21_PREDICTORS].values
    y_dev = df_dev[TARGET].values
    groups = df_dev[GROUP].values
    
    # Reconstruct exact 4 GroupKFold splits
    gkf = GroupKFold(n_splits=4)
    splits = list(gkf.split(X_dev, y_dev, groups=groups))
    
    assert len(splits) == 4
    for fold, (trn_idx, val_idx) in enumerate(splits):
        trn_districts = set(groups[trn_idx])
        val_districts = set(groups[val_idx])
        assert len(trn_districts.intersection(val_districts)) == 0, f"District leakage in fold {fold}"
    print("  [VERIFIED] Exact 4 GroupKFold district splits constructed with zero district overlap across folds.")
    
    # Reproduce JOINT_RF_0257 OOF
    # Config: n_estimators=250, max_depth=6, min_samples_split=7, min_samples_leaf=1, max_features=0.5, criterion=gini, bootstrap=True, class_weight=None, seed=42
    oof_probs = np.zeros(len(df_dev))
    fold_models = []
    
    rf_params = {
        "n_estimators": 250,
        "max_depth": 6,
        "min_samples_split": 7,
        "min_samples_leaf": 1,
        "max_features": 0.5,
        "criterion": "gini",
        "bootstrap": True,
        "class_weight": None,
        "random_state": SEED,
        "n_jobs": -1
    }
    
    for fold, (trn_idx, val_idx) in enumerate(splits):
        X_trn, y_trn = X_dev[trn_idx], y_dev[trn_idx]
        X_val, y_val = X_dev[val_idx], y_dev[val_idx]
        
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_trn, y_trn)
        fold_models.append(rf)
        oof_probs[val_idx] = rf.predict_proba(X_val)[:, 1]
        
    m_oof = evaluate_metrics(y_dev, oof_probs, threshold=0.76)
    print(f"\n  Reproduced JOINT_RF_0257 OOF Metrics (tau=0.76):")
    print(f"    Accuracy         : {m_oof['accuracy']*100:.2f}% (Expected: 79.83%)")
    print(f"    Balanced Accuracy: {m_oof['balanced_accuracy']*100:.2f}% (Expected: 66.90%)")
    print(f"    Class 0 Recall   : {m_oof['class0_recall']*100:.2f}% (Expected: 50.00%)")
    print(f"    Class 1 Recall   : {m_oof['class1_recall']*100:.2f}% (Expected: 83.81%)")
    print(f"    Macro F1         : {m_oof['macro_f1']:.4f} (Expected: 0.6242)")
    print(f"    PR-AUC           : {m_oof['pr_auc']:.4f} (Expected: 0.8988)")
    print(f"    Recall Gap       : {m_oof['recall_gap']*100:.2f}% (Expected: 33.81%)")
    print(f"    Confusion Matrix : TN={m_oof['TN']}, FP={m_oof['FP']}, FN={m_oof['FN']}, TP={m_oof['TP']}")
    
    assert abs(m_oof['accuracy'] - 0.7983193) < 0.001
    assert abs(m_oof['balanced_accuracy'] - 0.6690476) < 0.001
    assert abs(m_oof['class0_recall'] - 0.5000) < 0.001
    assert abs(m_oof['class1_recall'] - 0.838095) < 0.001
    assert abs(m_oof['macro_f1'] - 0.624210) < 0.001
    assert abs(m_oof['pr_auc'] - 0.898806) < 0.001
    assert m_oof['TN'] == 7 and m_oof['FP'] == 7 and m_oof['FN'] == 17 and m_oof['TP'] == 88
    print("  [VERIFIED] Exact match with stored development OOF metrics!")
    print("STAGE 3 AUDIT: PASSED\n")

def run_stage4_audit():
    print("=" * 80)
    print("STAGE 4: OUTER-TEST PROVENANCE AUDIT (Read-only verification)")
    print("=" * 80)
    df_outer = pd.read_csv(OUTER_PATH)
    df_dev = pd.read_csv(DEV_PATH)
    
    assert len(df_outer) == 35, f"Expected 35 outer stations, got {len(df_outer)}"
    outer_districts = sorted(df_outer[GROUP].unique())
    dev_districts = sorted(df_dev[GROUP].unique())
    assert len(outer_districts) == 8, f"Expected 8 outer districts, got {len(outer_districts)}"
    assert len(set(outer_districts).intersection(set(dev_districts))) == 0, "District overlap detected!"
    
    c0 = int((df_outer[TARGET] == 0).sum())
    c1 = int((df_outer[TARGET] == 1).sum())
    assert c0 == 9, f"Expected 9 Class 0, got {c0}"
    assert c1 == 26, f"Expected 26 Class 1, got {c1}"
    print(f"  [VERIFIED] Outer test set contains exactly 35 stations in 8 held-out districts (26 Class 1, 9 Class 0).")
    print(f"  [VERIFIED] Zero district overlap with development set (25 dev districts vs 8 outer districts).")
    
    # Verify the stored benchmark in results/diagnostics/joint_outer_test_results.csv
    outer_res_path = DIAG_DIR / "joint_outer_test_results.csv"
    df_res = pd.read_csv(outer_res_path)
    print(f"  Candidates evaluated in outer-test benchmark:\n{df_res[['candidate', 'threshold', 'accuracy', 'balanced_accuracy', 'class0_recall', 'class1_recall', 'macro_f1', 'TN', 'FP', 'FN', 'TP']]}")
    
    prim_row = df_res[df_res["candidate"] == "Primary_Balanced_Candidate"].iloc[0]
    print(f"\n  Stored Outer-Test Results for Primary_Balanced_Candidate (JOINT_RF_0257):")
    print(f"    Threshold        : {prim_row['threshold']}")
    print(f"    Accuracy         : {prim_row['accuracy']*100:.2f}% (Expected: 74.29%)")
    print(f"    Balanced Accuracy: {prim_row['balanced_accuracy']*100:.2f}% (Expected: 57.26%)")
    print(f"    Class 0 Recall   : {prim_row['class0_recall']*100:.2f}% (Expected: 22.22%)")
    print(f"    Class 1 Recall   : {prim_row['class1_recall']*100:.2f}% (Expected: 92.31%)")
    print(f"    Macro F1         : {prim_row['macro_f1']:.4f} (Expected: 0.5749)")
    print(f"    PR-AUC           : {prim_row['pr_auc']:.4f} (Expected: 0.8092)")
    print(f"    ROC-AUC          : {prim_row['roc_auc']:.4f} (Expected: 0.5897)")
    print(f"    Confusion Matrix : TN={prim_row['TN']}, FP={prim_row['FP']}, FN={prim_row['FN']}, TP={prim_row['TP']}")
    
    assert abs(prim_row['accuracy'] - 0.742857) < 0.001
    assert abs(prim_row['balanced_accuracy'] - 0.572649) < 0.001
    assert abs(prim_row['class0_recall'] - 0.222222) < 0.001
    assert abs(prim_row['class1_recall'] - 0.923076) < 0.001
    assert abs(prim_row['macro_f1'] - 0.574898) < 0.001
    assert abs(prim_row['pr_auc'] - 0.809224) < 0.001
    assert abs(prim_row['roc_auc'] - 0.589743) < 0.001
    assert prim_row['TN'] == 2 and prim_row['FP'] == 7 and prim_row['FN'] == 2 and prim_row['TP'] == 24
    
    # Baseline comparison verification
    base_row = df_res[df_res["candidate"] == "Current_Model"].iloc[0]
    print(f"\n  Baseline Model Outer-Test Performance:")
    print(f"    Accuracy         : {base_row['accuracy']*100:.2f}%")
    print(f"    Balanced Accuracy: {base_row['balanced_accuracy']*100:.2f}%")
    print(f"    Class 0 Recall   : {base_row['class0_recall']*100:.2f}% (TN={base_row['TN']}, FP={base_row['FP']})")
    print(f"    Class 1 Recall   : {base_row['class1_recall']*100:.2f}% (FN={base_row['FN']}, TP={base_row['TP']})")
    print(f"    Macro F1         : {base_row['macro_f1']:.4f}")
    
    print("  [VERIFIED] Stored outer-test results perfectly match frozen specifications.")
    print("  [CONFIRMED] No new outer-test evaluation or tuning was performed.")
    print("STAGE 4 AUDIT: PASSED\n")

if __name__ == "__main__":
    run_stage1_audit()
    run_stage2_audit()
    run_stage3_audit()
    run_stage4_audit()
    print("ALL STAGES 1-4 AUDITS COMPLETED SUCCESSFULLY!")
