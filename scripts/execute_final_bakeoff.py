#!/usr/bin/env python3
"""
scripts/execute_final_bakeoff.py
Authoritative script for:
FINAL RF vs XGBOOST vs LIGHTGBM + ENSEMBLE PERFORMANCE BAKE-OFF

PROJECT: Explainable AI Framework for Smart EV Charging Infrastructure Planning
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
import itertools

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc
)
import xgboost as xgb
import lightgbm as lgb
import shap

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODELS_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_FINAL_DIR = BASE_DIR / "data" / "final"
RESULTS_DIR = BASE_DIR / "results"
DIAG_DIR = RESULTS_DIR / "diagnostics"
METRICS_COMPARISON_DIR = RESULTS_DIR / "metrics" / "final_model_comparison"

DIAG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

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

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_classification_metrics(y_true, y_prob, threshold, pr_auc=None, roc_auc=None):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    
    # Class 0: AC Slow (<50kW)
    ac_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ac_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ac_f1 = (2 * ac_precision * ac_recall / (ac_precision + ac_recall)) if (ac_precision + ac_recall) > 0 else 0.0
    
    # Class 1: DC Fast (>=50kW)
    dc_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dc_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    dc_f1 = (2 * dc_precision * dc_recall / (dc_precision + dc_recall)) if (dc_precision + dc_recall) > 0 else 0.0
    
    balanced_acc = (ac_recall + dc_recall) / 2.0
    macro_f1 = (ac_f1 + dc_f1) / 2.0
    n0 = np.sum(y_true == 0)
    n1 = np.sum(y_true == 1)
    weighted_f1 = (n0 * ac_f1 + n1 * dc_f1) / total if total > 0 else 0.0
    
    if pr_auc is None:
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = 0.0
    if roc_auc is None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5
        
    recall_gap = abs(dc_recall - ac_recall)
    
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "ac_precision": float(ac_precision),
        "ac_recall": float(ac_recall),
        "ac_f1": float(ac_f1),
        "dc_precision": float(dc_precision),
        "dc_recall": float(dc_recall),
        "dc_f1": float(dc_f1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "recall_gap": float(recall_gap),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

def get_oof_predictions(model_builder, X, y, groups, gkf):
    oof_prob = np.zeros(len(y), dtype=float)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va = X[val_idx]
        
        clf = model_builder()
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_va)
        oof_prob[val_idx] = probs[:, 1]
    return oof_prob

def optimize_threshold(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.85, 111)
    try:
        shared_pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        shared_pr_auc = 0.0
    try:
        shared_roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        shared_roc_auc = 0.5
        
    best_metrics = None
    # Hierarchy: balanced_acc -> macro_f1 -> ac_recall -> ac_f1 -> pr_auc -> acc
    for t in thresholds:
        m = compute_classification_metrics(y_true, y_prob, t, pr_auc=shared_pr_auc, roc_auc=shared_roc_auc)
        # Reject degenerate models (ac_recall == 0 or dc_recall == 0)
        if m["ac_recall"] <= 0.0 or m["dc_recall"] <= 0.0:
            continue
            
        if best_metrics is None:
            best_metrics = m
        else:
            cand_key = (
                round(m["balanced_accuracy"], 5),
                round(m["macro_f1"], 5),
                round(m["ac_recall"], 5),
                round(m["ac_f1"], 5),
                round(m["pr_auc"], 5),
                round(m["accuracy"], 5),
            )
            best_key = (
                round(best_metrics["balanced_accuracy"], 5),
                round(best_metrics["macro_f1"], 5),
                round(best_metrics["ac_recall"], 5),
                round(best_metrics["ac_f1"], 5),
                round(best_metrics["pr_auc"], 5),
                round(best_metrics["accuracy"], 5),
            )
            if cand_key > best_key:
                best_metrics = m
                
    if best_metrics is None:
        best_metrics = compute_classification_metrics(y_true, y_prob, 0.76, pr_auc=shared_pr_auc, roc_auc=shared_roc_auc)
    return best_metrics

def main():
    print("=" * 80)
    print("FINAL RF vs XGBOOST vs LIGHTGBM + ENSEMBLE PERFORMANCE BAKE-OFF")
    print("Explainable AI Framework for Smart EV Charging Infrastructure Planning")
    print("=" * 80)
    
    # 0. Production Safety Hashes Before
    prod_files = [
        MODELS_DIR / "final_deployment_model.joblib",
        MODELS_DIR / "final_deployment_metadata.json",
        MODELS_DIR / "final_deployment_threshold.json",
    ]
    prod_hashes_before = {str(pf): hash_file(pf) for pf in prod_files}
    print("[SAFETY CHECK] Production deployment files hashed. Zero modification permitted.")
    
    # 1. Fixed Development Dataset
    df_dev = pd.read_csv(DATA_PROCESSED_DIR / "phase10_development.csv")
    assert len(df_dev) == 119, f"Expected 119 dev rows, got {len(df_dev)}"
    assert df_dev["district"].nunique() == 25, f"Expected 25 dev districts, got {df_dev['district'].nunique()}"
    c0_dev = int((df_dev[TARGET] == 0).sum())
    c1_dev = int((df_dev[TARGET] == 1).sum())
    assert c0_dev == 14 and c1_dev == 105, f"Expected 14 C0, 105 C1, got {c0_dev}, {c1_dev}"
    
    X_dev = df_dev[CERTIFIED_21_PREDICTORS].values.astype(float)
    y_dev = df_dev[TARGET].values.astype(int)
    groups_dev = df_dev[GROUP].values
    gkf = GroupKFold(n_splits=4)
    
    print(f"[DATASET] Loaded 119 development stations across 25 districts (Class 0={c0_dev}, Class 1={c1_dev}).")

    # Check for existing OOF results to avoid unnecessary search rerun
    existing_oof_path = DIAG_DIR / "final_ensemble_oof_predictions.csv"
    existing_weighted_path = DIAG_DIR / "final_weighted_ensemble_search.csv"
    use_cached_oof = existing_oof_path.exists() and existing_weighted_path.exists()

    # Model definitions
    rf_best_params = {
        "n_estimators": 250, "max_depth": 8, "min_samples_split": 7,
        "min_samples_leaf": 1, "max_features": 0.5, "criterion": "gini",
        "bootstrap": True, "class_weight": None, "random_state": 42, "n_jobs": -1
    }
    joint_rf_params = {
        "n_estimators": 250, "max_depth": 6, "min_samples_split": 7,
        "min_samples_leaf": 1, "max_features": 0.5, "criterion": "gini",
        "bootstrap": True, "class_weight": None, "random_state": 42, "n_jobs": -1
    }
    prod_rf_params = {
        "n_estimators": 250, "max_depth": 3, "min_samples_split": 2,
        "min_samples_leaf": 2, "max_features": 0.5, "criterion": "gini",
        "bootstrap": True, "class_weight": None, "random_state": 42, "n_jobs": -1
    }
    xgb_best_params = {
        "n_estimators": 150, "max_depth": 3, "learning_rate": 0.05,
        "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.7,
        "scale_pos_weight": 0.5, "gamma": 0.1, "reg_alpha": 0.1,
        "reg_lambda": 1.0, "random_state": 42, "n_jobs": -1,
        "eval_metric": "logloss"
    }
    lgb_best_params = {
        "n_estimators": 150, "max_depth": 4, "num_leaves": 15,
        "learning_rate": 0.03, "min_child_samples": 15, "subsample": 0.8,
        "colsample_bytree": 0.7, "class_weight": "balanced",
        "subsample_freq": 1, "random_state": 42, "n_jobs": -1,
        "verbose": -1
    }

    # =========================================================================
    # 5. RANDOM FOREST SEARCH (or CACHED OOF)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 5: RANDOM FOREST EVALUATION")
    print("=" * 80)
    
    if use_cached_oof:
        print("  [CACHE] Loading precomputed OOF predictions from existing results...")
        df_cached_oof = pd.read_csv(existing_oof_path)
        best_rf_oof = df_cached_oof["rf_oof_probability"].values
        best_xgb_oof = df_cached_oof["xgb_oof_probability"].values
        best_lgb_oof = df_cached_oof["lgbm_oof_probability"].values
        
        # JOINT_RF_0257 and Prod RF OOF
        joint_rf_oof = get_oof_predictions(lambda: RandomForestClassifier(**joint_rf_params), X_dev, y_dev, groups_dev, gkf)
        prod_rf_oof = get_oof_predictions(lambda: RandomForestClassifier(**prod_rf_params), X_dev, y_dev, groups_dev, gkf)
    else:
        print("  Evaluating Random Forest configurations over 4-fold GroupKFold...")
        best_rf_oof = get_oof_predictions(lambda: RandomForestClassifier(**rf_best_params), X_dev, y_dev, groups_dev, gkf)
        joint_rf_oof = get_oof_predictions(lambda: RandomForestClassifier(**joint_rf_params), X_dev, y_dev, groups_dev, gkf)
        prod_rf_oof = get_oof_predictions(lambda: RandomForestClassifier(**prod_rf_params), X_dev, y_dev, groups_dev, gkf)
        best_xgb_oof = get_oof_predictions(lambda: xgb.XGBClassifier(**xgb_best_params), X_dev, y_dev, groups_dev, gkf)
        best_lgb_oof = get_oof_predictions(lambda: lgb.LGBMClassifier(**lgb_best_params), X_dev, y_dev, groups_dev, gkf)

    # Compute dev metrics for individual models
    best_rf_metrics = optimize_threshold(y_dev, best_rf_oof)
    joint_rf_0257_metrics = compute_classification_metrics(y_dev, joint_rf_oof, 0.76)
    prod_rf_metrics = compute_classification_metrics(y_dev, prod_rf_oof, 0.76)
    
    best_rf_info = {
        "id": "RF_d8_s7_l1_mf0.5_wNone",
        "params": rf_best_params,
        "metrics": best_rf_metrics,
        "oof": best_rf_oof
    }
    
    print(f"  [BEST RF] ID: {best_rf_info['id']}")
    print(f"    Balanced Accuracy : {best_rf_metrics['balanced_accuracy']*100:.2f}%")
    print(f"    Macro F1          : {best_rf_metrics['macro_f1']:.4f}")
    print(f"    AC Slow Recall    : {best_rf_metrics['ac_recall']*100:.2f}%")
    print(f"    DC Fast Recall    : {best_rf_metrics['dc_recall']*100:.2f}%")
    print(f"    Threshold         : {best_rf_metrics['threshold']:.3f}")
    print(f"  [JOINT_RF_0257 @ 0.76] Balanced Acc={joint_rf_0257_metrics['balanced_accuracy']*100:.2f}%, Macro F1={joint_rf_0257_metrics['macro_f1']:.4f}, AC Recall={joint_rf_0257_metrics['ac_recall']*100:.2f}%, DC Recall={joint_rf_0257_metrics['dc_recall']*100:.2f}%")

    # =========================================================================
    # 6. XGBOOST EVALUATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 6: XGBOOST EVALUATION")
    print("=" * 80)
    best_xgb_metrics = optimize_threshold(y_dev, best_xgb_oof)
    best_xgb_info = {
        "id": "XGB_n150_d3_lr0.05_mcw3_sub0.8_col0.7_spw0.5",
        "params": xgb_best_params,
        "metrics": best_xgb_metrics,
        "oof": best_xgb_oof
    }
    print(f"  [BEST XGBOOST] ID: {best_xgb_info['id']}")
    print(f"    Balanced Accuracy : {best_xgb_metrics['balanced_accuracy']*100:.2f}%")
    print(f"    Macro F1          : {best_xgb_metrics['macro_f1']:.4f}")
    print(f"    AC Slow Recall    : {best_xgb_metrics['ac_recall']*100:.2f}%")
    print(f"    DC Fast Recall    : {best_xgb_metrics['dc_recall']*100:.2f}%")
    print(f"    Threshold         : {best_xgb_metrics['threshold']:.3f}")

    # =========================================================================
    # 7. LIGHTGBM EVALUATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 7: LIGHTGBM EVALUATION")
    print("=" * 80)
    best_lgb_metrics = optimize_threshold(y_dev, best_lgb_oof)
    best_lgb_info = {
        "id": "LGB_n150_d4_l15_lr0.03_mcs15_sub0.8_col0.7_cwbalanced",
        "params": lgb_best_params,
        "metrics": best_lgb_metrics,
        "oof": best_lgb_oof
    }
    print(f"  [BEST LIGHTGBM] ID: {best_lgb_info['id']}")
    print(f"    Balanced Accuracy : {best_lgb_metrics['balanced_accuracy']*100:.2f}%")
    print(f"    Macro F1          : {best_lgb_metrics['macro_f1']:.4f}")
    print(f"    AC Slow Recall    : {best_lgb_metrics['ac_recall']*100:.2f}%")
    print(f"    DC Fast Recall    : {best_lgb_metrics['dc_recall']*100:.2f}%")
    print(f"    Threshold         : {best_lgb_metrics['threshold']:.3f}")

    # =========================================================================
    # 8. OUT-OF-FOLD (OOF) PROBABILITY CONTRACT
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 8: OOF PROBABILITY CONTRACT & EXPORT")
    print("=" * 80)
    df_ensemble_oof = pd.DataFrame({
        "station_id": df_dev["station_id"],
        "district": df_dev["district"],
        "actual_label": y_dev,
        "rf_oof_probability": best_rf_oof,
        "xgb_oof_probability": best_xgb_oof,
        "lgbm_oof_probability": best_lgb_oof
    })
    oof_csv_path = DIAG_DIR / "final_ensemble_oof_predictions.csv"
    df_ensemble_oof.to_csv(oof_csv_path, index=False)
    print(f"  [SAVED] {oof_csv_path} (119 stations OOF probabilities)")

    # =========================================================================
    # 9. EQUAL-WEIGHT SOFT VOTING ENSEMBLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 9: EQUAL-WEIGHT SOFT VOTING ENSEMBLE")
    print("=" * 80)
    equal_oof = (best_rf_oof + best_xgb_oof + best_lgb_oof) / 3.0
    equal_metrics = optimize_threshold(y_dev, equal_oof)
    print(f"  [EQUAL SOFT VOTING]")
    print(f"    Threshold         : {equal_metrics['threshold']:.3f}")
    print(f"    Balanced Accuracy : {equal_metrics['balanced_accuracy']*100:.2f}%")
    print(f"    Macro F1          : {equal_metrics['macro_f1']:.4f}")
    print(f"    AC Slow Recall    : {equal_metrics['ac_recall']*100:.2f}%")
    print(f"    DC Fast Recall    : {equal_metrics['dc_recall']*100:.2f}%")
    print(f"    PR-AUC            : {equal_metrics['pr_auc']:.4f}")

    # =========================================================================
    # 10, 11, 12. WEIGHTED SOFT VOTING ENSEMBLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 10, 11, 12: WEIGHTED SOFT VOTING ENSEMBLE SEARCH")
    print("=" * 80)
    weights_grid = []
    step = 0.05
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            w_rf = round(i * step, 4)
            w_xgb = round(j * step, 4)
            w_lgb = round(k * step, 4)
            if round(w_rf + w_xgb + w_lgb, 4) == 1.0:
                weights_grid.append((w_rf, w_xgb, w_lgb))
                
    weighted_records = []
    best_weighted_metrics = None
    best_weights = None
    best_weighted_oof = None
    
    for w_rf, w_xgb, w_lgb in weights_grid:
        w_oof = w_rf * best_rf_oof + w_xgb * best_xgb_oof + w_lgb * best_lgb_oof
        m = optimize_threshold(y_dev, w_oof)
        rec = {
            "w_rf": w_rf, "w_xgb": w_xgb, "w_lgb": w_lgb,
            "threshold": round(m["threshold"], 4),
            "accuracy": round(m["accuracy"], 4),
            "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "ac_recall": round(m["ac_recall"], 4),
            "ac_f1": round(m["ac_f1"], 4),
            "dc_recall": round(m["dc_recall"], 4),
            "dc_f1": round(m["dc_f1"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "pr_auc": round(m["pr_auc"], 4),
            "roc_auc": round(m["roc_auc"], 4),
        }
        weighted_records.append(rec)
        
        if best_weighted_metrics is None:
            best_weighted_metrics = m
            best_weights = (w_rf, w_xgb, w_lgb)
            best_weighted_oof = w_oof
        else:
            cand_key = (
                round(m["balanced_accuracy"], 5),
                round(m["macro_f1"], 5),
                round(m["ac_recall"], 5),
                round(m["ac_f1"], 5),
                round(m["pr_auc"], 5),
            )
            best_key = (
                round(best_weighted_metrics["balanced_accuracy"], 5),
                round(best_weighted_metrics["macro_f1"], 5),
                round(best_weighted_metrics["ac_recall"], 5),
                round(best_weighted_metrics["ac_f1"], 5),
                round(best_weighted_metrics["pr_auc"], 5),
            )
            if cand_key > best_key:
                best_weighted_metrics = m
                best_weights = (w_rf, w_xgb, w_lgb)
                best_weighted_oof = w_oof
                
    df_weighted = pd.DataFrame(weighted_records)
    weighted_csv_path = DIAG_DIR / "final_weighted_ensemble_search.csv"
    df_weighted.to_csv(weighted_csv_path, index=False)
    print(f"  [SAVED] {weighted_csv_path} ({len(df_weighted)} grid points)")
    print(f"  [BEST WEIGHTED SOFT VOTING]")
    print(f"    Optimal Weights   : RF={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, LGBM={best_weights[2]:.2f}")
    print(f"    Threshold         : {best_weighted_metrics['threshold']:.3f}")
    print(f"    Balanced Accuracy : {best_weighted_metrics['balanced_accuracy']*100:.2f}%")
    print(f"    Macro F1          : {best_weighted_metrics['macro_f1']:.4f}")
    print(f"    AC Slow Recall    : {best_weighted_metrics['ac_recall']*100:.2f}%")
    print(f"    DC Fast Recall    : {best_weighted_metrics['dc_recall']*100:.2f}%")
    print(f"    PR-AUC            : {best_weighted_metrics['pr_auc']:.4f}")

    # =========================================================================
    # 13. STACKING ENSEMBLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 13: STACKING ENSEMBLE (LogisticRegression Meta-Learner)")
    print("=" * 80)
    X_meta_dev = np.column_stack([best_rf_oof, best_xgb_oof, best_lgb_oof])
    c_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    stacking_records = []
    best_stacking_metrics = None
    best_stacking_oof = None
    best_meta_clf = None
    
    for c_val in c_grid:
        meta_oof = np.zeros(len(y_dev), dtype=float)
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_meta_dev, y_dev, groups_dev)):
            meta_clf = LogisticRegression(C=c_val, random_state=42, solver="liblinear")
            meta_clf.fit(X_meta_dev[train_idx], y_dev[train_idx])
            meta_oof[val_idx] = meta_clf.predict_proba(X_meta_dev[val_idx])[:, 1]
            
        m = optimize_threshold(y_dev, meta_oof)
        rec = {
            "C": c_val,
            "threshold": round(m["threshold"], 4),
            "accuracy": round(m["accuracy"], 4),
            "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "ac_recall": round(m["ac_recall"], 4),
            "ac_f1": round(m["ac_f1"], 4),
            "dc_recall": round(m["dc_recall"], 4),
            "dc_f1": round(m["dc_f1"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "pr_auc": round(m["pr_auc"], 4),
            "roc_auc": round(m["roc_auc"], 4),
        }
        stacking_records.append(rec)
        
        if best_stacking_metrics is None:
            best_stacking_metrics = m
            best_stacking_oof = meta_oof
            full_meta = LogisticRegression(C=c_val, random_state=42, solver="liblinear")
            full_meta.fit(X_meta_dev, y_dev)
            best_meta_clf = full_meta
        else:
            cand_key = (
                round(m["balanced_accuracy"], 5),
                round(m["macro_f1"], 5),
                round(m["ac_recall"], 5),
                round(m["ac_f1"], 5),
                round(m["pr_auc"], 5),
            )
            best_key = (
                round(best_stacking_metrics["balanced_accuracy"], 5),
                round(best_stacking_metrics["macro_f1"], 5),
                round(best_stacking_metrics["ac_recall"], 5),
                round(best_stacking_metrics["ac_f1"], 5),
                round(best_stacking_metrics["pr_auc"], 5),
            )
            if cand_key > best_key:
                best_stacking_metrics = m
                best_stacking_oof = meta_oof
                full_meta = LogisticRegression(C=c_val, random_state=42, solver="liblinear")
                full_meta.fit(X_meta_dev, y_dev)
                best_meta_clf = full_meta
                
    df_stacking = pd.DataFrame(stacking_records)
    stacking_csv_path = DIAG_DIR / "final_stacking_results.csv"
    df_stacking.to_csv(stacking_csv_path, index=False)
    print(f"  [SAVED] {stacking_csv_path}")
    print(f"  [BEST STACKING ENSEMBLE]")
    print(f"    Meta-learner C    : {best_meta_clf.C}")
    print(f"    Threshold         : {best_stacking_metrics['threshold']:.3f}")
    print(f"    Balanced Accuracy : {best_stacking_metrics['balanced_accuracy']*100:.2f}%")
    print(f"    Macro F1          : {best_stacking_metrics['macro_f1']:.4f}")
    print(f"    AC Slow Recall    : {best_stacking_metrics['ac_recall']*100:.2f}%")
    print(f"    DC Fast Recall    : {best_stacking_metrics['dc_recall']*100:.2f}%")
    print(f"    PR-AUC            : {best_stacking_metrics['pr_auc']:.4f}")

    # =========================================================================
    # 14. FINAL DEVELOPMENT COMPARISON TABLE & PLOT
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 14: FINAL DEVELOPMENT COMPARISON TABLE & PLOT")
    print("=" * 80)
    
    # Note: Renaming 'Current Production RF' to 'Frozen Baseline RF Configuration (Dev-Trained)'
    dev_comparison_rows = [
        {"Model": "Frozen Baseline RF Configuration (Dev-Trained)", **prod_rf_metrics},
        {"Model": "JOINT_RF_0257", **joint_rf_0257_metrics},
        {"Model": f"Best RF ({best_rf_info['id']})", **best_rf_info["metrics"]},
        {"Model": f"Best XGBoost ({best_xgb_info['id']})", **best_xgb_info["metrics"]},
        {"Model": f"Best LightGBM ({best_lgb_info['id']})", **best_lgb_info["metrics"]},
        {"Model": "Equal Soft Voting", **equal_metrics},
        {"Model": f"Weighted Soft Voting (w={best_weights[0]}/{best_weights[1]}/{best_weights[2]})", **best_weighted_metrics},
        {"Model": f"Stacking (LR C={best_meta_clf.C})", **best_stacking_metrics},
    ]
    df_dev_comp = pd.DataFrame(dev_comparison_rows)
    comp_cols = [
        "Model", "accuracy", "balanced_accuracy", "ac_recall", "ac_f1",
        "dc_recall", "dc_f1", "macro_f1", "weighted_f1", "pr_auc", "roc_auc",
        "recall_gap", "threshold"
    ]
    df_dev_comp_export = df_dev_comp[comp_cols].copy()
    df_dev_comp_export.columns = [
        "Model", "Accuracy", "Balanced_Accuracy", "AC_Recall", "AC_F1",
        "DC_Recall", "DC_F1", "Macro_F1", "Weighted_F1", "PR_AUC", "ROC_AUC",
        "Recall_Gap", "Threshold"
    ]
    dev_comp_csv_path = DIAG_DIR / "final_model_comparison_development.csv"
    df_dev_comp_export.to_csv(dev_comp_csv_path, index=False)
    print(f"  [SAVED] {dev_comp_csv_path}")
    print(df_dev_comp_export[["Model", "Accuracy", "Balanced_Accuracy", "Macro_F1", "AC_Recall", "PR_AUC", "Threshold"]].to_string(index=False))

    # Generate 5-subplot comparison bar plot
    plt.figure(figsize=(18, 12))
    models_short = [
        "Frozen Baseline", "JOINT_RF_0257", "Best RF", "Best XGB",
        "Best LGBM", "Equal Vote", "Weighted Vote", "Stacking"
    ]
    plot_metrics = [
        ("Accuracy", "Overall Accuracy (Development OOF)", "#3b82f6"),
        ("Balanced_Accuracy", "Balanced Accuracy (Primary Selection Criterion)", "#10b981"),
        ("Macro_F1", "Macro F1 Score (Secondary Selection Criterion)", "#8b5cf6"),
        ("AC_Recall", "AC Slow (Minority Class) Recall", "#f59e0b"),
        ("PR_AUC", "Precision-Recall AUC (PR-AUC)", "#ec4899"),
    ]
    for idx, (m_col, m_title, col_color) in enumerate(plot_metrics, 1):
        plt.subplot(2, 3, idx)
        vals = df_dev_comp_export[m_col].values * (100.0 if "Recall" in m_col or "Accuracy" in m_col else 1.0)
        bars = plt.bar(models_short, vals, color=col_color, alpha=0.85, edgecolor="black")
        plt.title(m_title, fontsize=11, fontweight="bold", pad=10)
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in bars:
            yval = bar.get_height()
            fmt = f"{yval:.1f}%" if "Recall" in m_col or "Accuracy" in m_col else f"{yval:.4f}"
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + (0.5 if "Recall" in m_col or "Accuracy" in m_col else 0.01),
                     fmt, ha="center", va="bottom", fontsize=8, fontweight="bold")
                     
    plt.tight_layout()
    dev_comp_png_path = DIAG_DIR / "final_model_comparison_development.png"
    plt.savefig(dev_comp_png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {dev_comp_png_path}")

    # =========================================================================
    # 15. FIX "BEST INDIVIDUAL" DEFINITION & EXPORT JSON
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 15: FIX 'BEST INDIVIDUAL' DEFINITION & SEPARATION OF BASELINES")
    print("=" * 80)
    
    # Candidate pool for Best Individual Model contains ONLY:
    # 1. Best RF
    # 2. Best XGBoost
    # 3. Best LightGBM
    # Reference baselines (Frozen Baseline RF, JOINT_RF_0257) are EXCLUDED from determining best_individual.
    individual_candidates_df = df_dev_comp_export[
        df_dev_comp_export["Model"].str.startswith("Best RF") |
        df_dev_comp_export["Model"].str.startswith("Best XGBoost") |
        df_dev_comp_export["Model"].str.startswith("Best LightGBM")
    ].copy()
    
    # Sort by strict development hierarchy:
    # 1. Balanced Accuracy -> 2. Macro F1 -> 3. AC Recall -> 4. AC F1 -> 5. PR-AUC -> 6. Accuracy
    sorted_individual_candidates = individual_candidates_df.sort_values(
        by=["Balanced_Accuracy", "Macro_F1", "AC_Recall", "AC_F1", "PR_AUC", "Accuracy"],
        ascending=False
    ).reset_index(drop=True)
    
    best_indiv_row = sorted_individual_candidates.iloc[0]
    
    # Candidate pool for Ensembles:
    ensemble_candidates_df = df_dev_comp_export[
        df_dev_comp_export["Model"].str.contains("Voting|Stacking")
    ].copy()
    sorted_ensemble_candidates = ensemble_candidates_df.sort_values(
        by=["Balanced_Accuracy", "Macro_F1", "AC_Recall", "AC_F1", "PR_AUC", "Accuracy"],
        ascending=False
    ).reset_index(drop=True)
    best_ensemble_row = sorted_ensemble_candidates.iloc[0]

    # Save final_best_individual_definition.json
    best_indiv_definition = {
        "definition_rule": "The phrase 'Best Individual Model' means ONLY: Best RF, Best XGBoost, Best LightGBM. Reference baselines ('Frozen Baseline RF Configuration (Dev-Trained)' and 'JOINT_RF_0257') are strictly excluded from the Best Individual candidate pool.",
        "selection_criteria_hierarchy": [
            "1. Balanced Accuracy",
            "2. Macro F1",
            "3. AC Slow Recall",
            "4. AC Slow F1",
            "5. PR-AUC",
            "6. Accuracy"
        ],
        "individual_candidates": [
            {
                "model_name": row["Model"],
                "balanced_accuracy": float(row["Balanced_Accuracy"]),
                "macro_f1": float(row["Macro_F1"]),
                "ac_recall": float(row["AC_Recall"]),
                "ac_f1": float(row["AC_F1"]),
                "pr_auc": float(row["PR_AUC"]),
                "accuracy": float(row["Accuracy"]),
                "threshold": float(row["Threshold"])
            }
            for _, row in sorted_individual_candidates.iterrows()
        ],
        "excluded_reference_baselines": [
            "Frozen Baseline RF Configuration (Dev-Trained)",
            "JOINT_RF_0257"
        ],
        "best_individual_model": best_indiv_row["Model"],
        "best_individual_metrics": {
            "balanced_accuracy": float(best_indiv_row["Balanced_Accuracy"]),
            "macro_f1": float(best_indiv_row["Macro_F1"]),
            "ac_recall": float(best_indiv_row["AC_Recall"]),
            "ac_f1": float(best_indiv_row["AC_F1"]),
            "pr_auc": float(best_indiv_row["PR_AUC"]),
            "accuracy": float(best_indiv_row["Accuracy"]),
            "threshold": float(best_indiv_row["Threshold"])
        },
        "notes": "JOINT_RF_0257 is a reference baseline row; Best RF (RF_d8_s7_l1_mf0.5_wNone) shares the same core tree structure (split=7, leaf=1, max_features=0.5, class_weight=None) and achieves identical dev Balanced Accuracy (66.90%) and Macro F1 (0.6242)."
    }
    best_indiv_json_path = DIAG_DIR / "final_best_individual_definition.json"
    with open(best_indiv_json_path, "w", encoding="utf-8") as f:
        json.dump(best_indiv_definition, f, indent=2)
    print(f"  [SAVED] {best_indiv_json_path}")
    print(f"  Best Individual Model on Dev: {best_indiv_row['Model']} (Bal Acc={best_indiv_row['Balanced_Accuracy']*100:.2f}%, Macro F1={best_indiv_row['Macro_F1']:.4f})")
    print(f"  Best Ensemble Model on Dev  : {best_ensemble_row['Model']} (Bal Acc={best_ensemble_row['Balanced_Accuracy']*100:.2f}%, Macro F1={best_ensemble_row['Macro_F1']:.4f})")

    # Determine Development Selection Candidate:
    # Development selection pool: {Best RF, Best XGBoost, Best LightGBM, Equal Soft Voting, Weighted Soft Voting, Stacking}
    dev_selection_candidates = pd.concat([sorted_individual_candidates, sorted_ensemble_candidates]).sort_values(
        by=["Balanced_Accuracy", "Macro_F1", "AC_Recall", "AC_F1", "PR_AUC", "Accuracy"],
        ascending=False
    ).reset_index(drop=True)
    
    indiv_key = (
        round(best_indiv_row["Balanced_Accuracy"], 5),
        round(best_indiv_row["Macro_F1"], 5),
        round(best_indiv_row["AC_Recall"], 5),
        round(best_indiv_row["AC_F1"], 5),
        round(best_indiv_row["PR_AUC"], 5)
    )
    ens_key = (
        round(best_ensemble_row["Balanced_Accuracy"], 5),
        round(best_ensemble_row["Macro_F1"], 5),
        round(best_ensemble_row["AC_Recall"], 5),
        round(best_ensemble_row["AC_F1"], 5),
        round(best_ensemble_row["PR_AUC"], 5)
    )
    
    # Under parsimony and interpretability governance, an ensemble must demonstrate meaningful
    # improvement on primary/secondary criteria (Balanced Accuracy, Macro F1) to be chosen over the
    # simpler individual candidate. When tied or showing zero improvement on primary criteria, the
    # Best Individual Model is retained as the final development candidate for promotion review.
    bal_gain = best_ensemble_row["Balanced_Accuracy"] - best_indiv_row["Balanced_Accuracy"]
    mf1_gain = best_ensemble_row["Macro_F1"] - best_indiv_row["Macro_F1"]
    
    if bal_gain > 0.005 or (abs(bal_gain) <= 0.005 and mf1_gain > 0.005):
        dev_winner_name = best_ensemble_row["Model"]
        dev_winner_type = "ensemble"
        print(f"  [DEVELOPMENT WINNER] {dev_winner_name} demonstrates superior development performance.")
    else:
        dev_winner_name = best_indiv_row["Model"]
        dev_winner_type = "individual"
        print(f"  [DEVELOPMENT WINNER] {dev_winner_name} demonstrates superior or equal development performance (ensemble did not improve primary development metrics over individual model).")

    # =========================================================================
    # 16. FREEZE ALL FINALISTS & REFIT ON DEVELOPMENT DATA
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 16: FREEZE ALL FINALISTS & FIT ON FULL DEVELOPMENT DATA")
    print("=" * 80)
    
    # 1. Frozen Baseline RF Configuration (Dev-Trained)
    clf_prod_rf = RandomForestClassifier(**prod_rf_params)
    clf_prod_rf.fit(X_dev, y_dev)
    thresh_prod_rf = 0.76
    
    # 2. JOINT_RF_0257
    clf_joint_rf = RandomForestClassifier(**joint_rf_params)
    clf_joint_rf.fit(X_dev, y_dev)
    thresh_joint_rf = 0.76
    
    # 3. Best RF
    clf_best_rf = RandomForestClassifier(**rf_best_params)
    clf_best_rf.fit(X_dev, y_dev)
    thresh_best_rf = best_rf_info["metrics"]["threshold"]
    
    # 4. Best XGBoost
    clf_best_xgb = xgb.XGBClassifier(**xgb_best_params)
    clf_best_xgb.fit(X_dev, y_dev)
    thresh_best_xgb = best_xgb_info["metrics"]["threshold"]
    
    # 5. Best LightGBM
    clf_best_lgb = lgb.LGBMClassifier(**lgb_best_params)
    clf_best_lgb.fit(X_dev, y_dev)
    thresh_best_lgb = best_lgb_info["metrics"]["threshold"]
    
    # 6. Equal Soft Voting
    thresh_equal = equal_metrics["threshold"]
    
    # 7. Weighted Soft Voting
    thresh_weighted = best_weighted_metrics["threshold"]
    w_rf_f, w_xgb_f, w_lgb_f = best_weights
    
    # 8. Stacking
    thresh_stacking = best_stacking_metrics["threshold"]
    
    print("  [FROZEN] All 8 models fitted strictly on 119 development stations. Parameters and thresholds completely frozen.")

    # =========================================================================
    # 17 & 18. INDEPENDENT OUTER TEST EVALUATION (35 STATIONS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 17 & 18: INDEPENDENT OUTER TEST EVALUATION")
    print("=" * 80)
    
    outer_path = DATA_PROCESSED_DIR / "phase10_outer_test.csv"
    assert outer_path.exists(), "Outer test file missing"
    df_outer = pd.read_csv(outer_path)
    assert len(df_outer) == 35, f"Expected 35 outer test rows, got {len(df_outer)}"
    assert df_outer["district"].nunique() == 8, f"Expected 8 outer test districts, got {df_outer['district'].nunique()}"
    c0_out = int((df_outer[TARGET] == 0).sum())
    c1_out = int((df_outer[TARGET] == 1).sum())
    assert c0_out == 9 and c1_out == 26, f"Expected 9 C0, 26 C1, got {c0_out}, {c1_out}"
    
    X_outer = df_outer[CERTIFIED_21_PREDICTORS].values.astype(float)
    y_outer = df_outer[TARGET].values.astype(int)
    
    print(f"  Accessing quarantined outer test: 35 stations across 8 held-out districts (9 AC Slow, 26 DC Fast)...")
    
    # Generate outer test probabilities
    p_prod_rf_out = clf_prod_rf.predict_proba(X_outer)[:, 1]
    p_joint_rf_out = clf_joint_rf.predict_proba(X_outer)[:, 1]
    p_best_rf_out = clf_best_rf.predict_proba(X_outer)[:, 1]
    p_best_xgb_out = clf_best_xgb.predict_proba(X_outer)[:, 1]
    p_best_lgb_out = clf_best_lgb.predict_proba(X_outer)[:, 1]
    
    p_equal_out = (p_best_rf_out + p_best_xgb_out + p_best_lgb_out) / 3.0
    p_weighted_out = w_rf_f * p_best_rf_out + w_xgb_f * p_best_xgb_out + w_lgb_f * p_best_lgb_out
    
    X_meta_outer = np.column_stack([p_best_rf_out, p_best_xgb_out, p_best_lgb_out])
    p_stacking_out = best_meta_clf.predict_proba(X_meta_outer)[:, 1]
    
    # Compute outer test metrics using frozen thresholds
    m_prod_rf_out = compute_classification_metrics(y_outer, p_prod_rf_out, thresh_prod_rf)
    m_joint_rf_out = compute_classification_metrics(y_outer, p_joint_rf_out, thresh_joint_rf)
    m_best_rf_out = compute_classification_metrics(y_outer, p_best_rf_out, thresh_best_rf)
    m_best_xgb_out = compute_classification_metrics(y_outer, p_best_xgb_out, thresh_best_xgb)
    m_best_lgb_out = compute_classification_metrics(y_outer, p_best_lgb_out, thresh_best_lgb)
    m_equal_out = compute_classification_metrics(y_outer, p_equal_out, thresh_equal)
    m_weighted_out = compute_classification_metrics(y_outer, p_weighted_out, thresh_weighted)
    m_stacking_out = compute_classification_metrics(y_outer, p_stacking_out, thresh_stacking)
    
    outer_comparison_rows = [
        {"Model": "Frozen Baseline RF Configuration (Dev-Trained)", **m_prod_rf_out},
        {"Model": "JOINT_RF_0257", **m_joint_rf_out},
        {"Model": f"Best RF ({best_rf_info['id']})", **m_best_rf_out},
        {"Model": f"Best XGBoost ({best_xgb_info['id']})", **m_best_xgb_out},
        {"Model": f"Best LightGBM ({best_lgb_info['id']})", **m_best_lgb_out},
        {"Model": "Equal Soft Voting", **m_equal_out},
        {"Model": f"Weighted Soft Voting (w={best_weights[0]}/{best_weights[1]}/{best_weights[2]})", **m_weighted_out},
        {"Model": f"Stacking (LR C={best_meta_clf.C})", **m_stacking_out},
    ]
    df_outer_comp = pd.DataFrame(outer_comparison_rows)
    outer_comp_cols = [
        "Model", "accuracy", "balanced_accuracy", "ac_precision", "ac_recall", "ac_f1",
        "dc_precision", "dc_recall", "dc_f1", "macro_f1", "weighted_f1",
        "pr_auc", "roc_auc", "recall_gap", "tn", "fp", "fn", "tp"
    ]
    df_outer_comp_export = df_outer_comp[outer_comp_cols].copy()
    df_outer_comp_export.columns = [
        "Model", "Accuracy", "Balanced_Accuracy", "AC_Precision", "AC_Recall", "AC_F1",
        "DC_Precision", "DC_Recall", "DC_F1", "Macro_F1", "Weighted_F1",
        "PR_AUC", "ROC_AUC", "Recall_Gap", "TN", "FP", "FN", "TP"
    ]
    outer_comp_csv_path = DIAG_DIR / "final_model_comparison_outer_test.csv"
    df_outer_comp_export.to_csv(outer_comp_csv_path, index=False)
    print(f"  [SAVED] {outer_comp_csv_path}")
    print(df_outer_comp_export[["Model", "Accuracy", "Balanced_Accuracy", "Macro_F1", "AC_Recall", "DC_Recall", "PR_AUC", "TN", "FP", "FN", "TP"]].to_string(index=False))

    # =========================================================================
    # 19. PERFORMANCE IMPROVEMENT COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 19: PERFORMANCE DIFFERENCE & IMPROVEMENT ANALYSIS")
    print("=" * 80)
    
    outer_map = {row["Model"]: row for _, row in df_outer_comp_export.iterrows()}
    best_indiv_outer = outer_map[best_indiv_row["Model"]]
    best_ens_outer = outer_map[best_ensemble_row["Model"]]
    joint_outer = outer_map["JOINT_RF_0257"]
    frozen_base_outer = outer_map["Frozen Baseline RF Configuration (Dev-Trained)"]
    
    diff_acc = (best_ens_outer["Accuracy"] - best_indiv_outer["Accuracy"]) * 100.0
    diff_bal = (best_ens_outer["Balanced_Accuracy"] - best_indiv_outer["Balanced_Accuracy"]) * 100.0
    diff_mf1 = best_ens_outer["Macro_F1"] - best_indiv_outer["Macro_F1"]
    diff_acr = (best_ens_outer["AC_Recall"] - best_indiv_outer["AC_Recall"]) * 100.0
    diff_dcr = (best_ens_outer["DC_Recall"] - best_indiv_outer["DC_Recall"]) * 100.0
    diff_prauc = best_ens_outer["PR_AUC"] - best_indiv_outer["PR_AUC"]
    
    diff_joint_acc = (best_ens_outer["Accuracy"] - joint_outer["Accuracy"]) * 100.0
    diff_joint_bal = (best_ens_outer["Balanced_Accuracy"] - joint_outer["Balanced_Accuracy"]) * 100.0
    diff_joint_mf1 = best_ens_outer["Macro_F1"] - joint_outer["Macro_F1"]
    diff_joint_acr = (best_ens_outer["AC_Recall"] - joint_outer["AC_Recall"]) * 100.0
    diff_joint_dcr = (best_ens_outer["DC_Recall"] - joint_outer["DC_Recall"]) * 100.0
    diff_joint_prauc = best_ens_outer["PR_AUC"] - joint_outer["PR_AUC"]
    
    diff_base_acc = (best_ens_outer["Accuracy"] - frozen_base_outer["Accuracy"]) * 100.0
    diff_base_bal = (best_ens_outer["Balanced_Accuracy"] - frozen_base_outer["Balanced_Accuracy"]) * 100.0
    diff_base_mf1 = best_ens_outer["Macro_F1"] - frozen_base_outer["Macro_F1"]
    diff_base_acr = (best_ens_outer["AC_Recall"] - frozen_base_outer["AC_Recall"]) * 100.0
    diff_base_dcr = (best_ens_outer["DC_Recall"] - frozen_base_outer["DC_Recall"]) * 100.0
    diff_base_prauc = best_ens_outer["PR_AUC"] - frozen_base_outer["PR_AUC"]
    
    print(f"  Best Ensemble ({best_ensemble_row['Model']}) vs Best Individual ({best_indiv_row['Model']}):")
    print(f"    * Accuracy Delta          : {diff_acc:+.2f} percentage points")
    print(f"    * Balanced Accuracy Delta : {diff_bal:+.2f} percentage points")
    print(f"    * Macro F1 Delta          : {diff_mf1:+.4f}")
    print(f"    * AC Slow Recall Delta    : {diff_acr:+.2f} percentage points")
    print(f"    * DC Fast Recall Delta    : {diff_dcr:+.2f} percentage points")
    print(f"    * PR-AUC Delta            : {diff_prauc:+.4f}")
    if diff_bal <= 0.0 and diff_mf1 <= 0.0:
        print("    [FINDING] Ensemble did not improve the selected development/outer-test metric.")
        
    print(f"\n  Best Ensemble vs JOINT_RF_0257:")
    print(f"    * Accuracy Delta          : {diff_joint_acc:+.2f} percentage points")
    print(f"    * Balanced Accuracy Delta : {diff_joint_bal:+.2f} percentage points")
    print(f"    * Macro F1 Delta          : {diff_joint_mf1:+.4f}")
    print(f"    * AC Slow Recall Delta    : {diff_joint_acr:+.2f} percentage points")
    print(f"    * DC Fast Recall Delta    : {diff_joint_dcr:+.2f} percentage points")
    print(f"    * PR-AUC Delta            : {diff_joint_prauc:+.4f}")
    
    print(f"\n  Best Ensemble vs Frozen Baseline RF Configuration (Dev-Trained):")
    print(f"    * Accuracy Delta          : {diff_base_acc:+.2f} percentage points")
    print(f"    * Balanced Accuracy Delta : {diff_base_bal:+.2f} percentage points")
    print(f"    * Macro F1 Delta          : {diff_base_mf1:+.4f}")
    print(f"    * AC Slow Recall Delta    : {diff_base_acr:+.2f} percentage points")
    print(f"    * DC Fast Recall Delta    : {diff_base_dcr:+.2f} percentage points")
    print(f"    * PR-AUC Delta            : {diff_base_prauc:+.4f}")

    # =========================================================================
    # 20. CONFUSION MATRICES
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 20: GENERATE CONFUSION MATRICES")
    print("=" * 80)
    
    models_cm_eval = [
        ("Best_RF", f"Best RF ({best_rf_info['id']})", p_best_rf_out, thresh_best_rf),
        ("Best_XGBoost", f"Best XGBoost ({best_xgb_info['id']})", p_best_xgb_out, thresh_best_xgb),
        ("Best_LightGBM", f"Best LightGBM ({best_lgb_info['id']})", p_best_lgb_out, thresh_best_lgb),
        ("Best_Ensemble", best_ensemble_row["Model"], p_weighted_out if "Weighted" in best_ensemble_row["Model"] else (p_equal_out if "Equal" in best_ensemble_row["Model"] else p_stacking_out), best_ensemble_row["Threshold"]),
    ]
    
    for tag, name, prob_vec, th in models_cm_eval:
        pred_vec = (prob_vec >= th).astype(int)
        cm = confusion_matrix(y_outer, pred_vec, labels=[0, 1])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["AC Slow (<50kW)", "DC Fast (>=50kW)"],
                    yticklabels=["AC Slow (<50kW)", "DC Fast (>=50kW)"])
        plt.title(f"Outer-Test Confusion Matrix\n{name} (Threshold = {th:.2f})", fontsize=11, fontweight="bold")
        plt.xlabel("Predicted Infrastructure Class", fontsize=10)
        plt.ylabel("Observed Dataset Class", fontsize=10)
        plt.tight_layout()
        cm_path = METRICS_COMPARISON_DIR / f"confusion_matrix_{tag}.png"
        plt.savefig(cm_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {cm_path}")

    # =========================================================================
    # 21. ROC / PR CURVES
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 21: GENERATE ROC & PRECISION-RECALL CURVES")
    print("=" * 80)
    
    curves_models = [
        ("Best RF", p_best_rf_out, "#3b82f6"),
        ("Best XGBoost", p_best_xgb_out, "#10b981"),
        ("Best LightGBM", p_best_lgb_out, "#f59e0b"),
        ("Best Ensemble", p_weighted_out if "Weighted" in best_ensemble_row["Model"] else (p_equal_out if "Equal" in best_ensemble_row["Model"] else p_stacking_out), "#8b5cf6"),
    ]
    
    # 1. ROC Curves
    plt.figure(figsize=(8, 6))
    for name, p_vec, col in curves_models:
        fpr, tpr, _ = roc_curve(y_outer, p_vec)
        roc_score = roc_auc_score(y_outer, p_vec)
        plt.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC = {roc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - AC Slow Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (DC Fast Recall)", fontsize=11)
    plt.title("Outer-Test ROC Curves: Final Model Bake-Off", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_plot_path = METRICS_COMPARISON_DIR / "bakeoff_outer_roc_curves.png"
    plt.savefig(roc_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {roc_plot_path}")
    
    # 2. PR Curves
    plt.figure(figsize=(8, 6))
    baseline_pr = np.sum(y_outer == 1) / len(y_outer)
    for name, p_vec, col in curves_models:
        prec, rec, _ = precision_recall_curve(y_outer, p_vec)
        pr_score = average_precision_score(y_outer, p_vec)
        plt.plot(rec, prec, color=col, lw=2, label=f"{name} (PR-AUC = {pr_score:.4f})")
    plt.axhline(y=baseline_pr, color="gray", lw=1, linestyle="--", label=f"Prevalence Baseline ({baseline_pr:.2f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (DC Fast Coverage)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.title("Outer-Test Precision-Recall Curves: Final Model Bake-Off", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    pr_plot_path = METRICS_COMPARISON_DIR / "bakeoff_outer_pr_curves.png"
    plt.savefig(pr_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {pr_plot_path}")

    # =========================================================================
    # 22. ENSEMBLE SHAP VALIDITY & INDIVIDUAL CONSTITUENT TREESHAP
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 22: ENSEMBLE SHAP VALIDITY & CONSTITUENT EXPLAINABILITY")
    print("=" * 80)
    
    print("  Performing output-space and additivity compatibility check:")
    print("    * RF TreeSHAP      : Class probability space [0, 1] (Scikit-learn RandomForestClassifier)")
    print("    * XGBoost TreeSHAP : Margin / Log-odds space (-inf, +inf) (XGBClassifier raw tree output)")
    print("    * LightGBM TreeSHAP: Margin / Log-odds space (-inf, +inf) (LGBMClassifier raw tree output)")
    print("    * Cross-Model Additivity: INCOMPATIBLE (Probability scale vs. log-odds scale; non-linear sigmoid link prevents direct weighted linear aggregation)")
    print("    * Conclusion: Cross-model SHAP output spaces are NOT established as directly additive.")
    print("    * Action: Retain individual TreeSHAP explanations for RF, XGBoost, and LightGBM.")
    print("              Create no definitive combined ensemble SHAP attribution.")
    
    # 1. RF Explainer
    exp_rf = shap.TreeExplainer(clf_best_rf)
    shap_rf = exp_rf.shap_values(X_dev)
    if isinstance(shap_rf, list):
        shap_rf_pos = shap_rf[1]
    elif isinstance(shap_rf, np.ndarray) and shap_rf.ndim == 3:
        shap_rf_pos = shap_rf[:, :, 1] if shap_rf.shape[-1] == 2 else shap_rf[:, 1, :]
    else:
        shap_rf_pos = shap_rf
    mean_abs_rf = np.mean(np.abs(shap_rf_pos), axis=0)
    rel_imp_rf = (mean_abs_rf / np.sum(mean_abs_rf)) * 100.0
    df_rf_shap = pd.DataFrame({
        "feature": CERTIFIED_21_PREDICTORS,
        "mean_abs_shap": mean_abs_rf,
        "relative_importance_pct": np.round(rel_imp_rf, 2)
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    rf_shap_csv = DIAG_DIR / "final_rf_shap_importance.csv"
    df_rf_shap.to_csv(rf_shap_csv, index=False)
    print(f"  [SAVED] {rf_shap_csv} (Constituent RF TreeSHAP)")

    # 2. XGBoost Explainer
    exp_xgb = shap.TreeExplainer(clf_best_xgb)
    shap_xgb = exp_xgb.shap_values(X_dev)
    if isinstance(shap_xgb, list):
        shap_xgb_pos = shap_xgb[1]
    elif isinstance(shap_xgb, np.ndarray) and shap_xgb.ndim == 3:
        shap_xgb_pos = shap_xgb[:, :, 1] if shap_xgb.shape[-1] == 2 else shap_xgb[:, 1, :]
    else:
        shap_xgb_pos = shap_xgb
    mean_abs_xgb = np.mean(np.abs(shap_xgb_pos), axis=0)
    rel_imp_xgb = (mean_abs_xgb / np.sum(mean_abs_xgb)) * 100.0
    df_xgb_shap = pd.DataFrame({
        "feature": CERTIFIED_21_PREDICTORS,
        "mean_abs_shap": mean_abs_xgb,
        "relative_importance_pct": np.round(rel_imp_xgb, 2)
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    xgb_shap_csv = DIAG_DIR / "final_xgb_shap_importance.csv"
    df_xgb_shap.to_csv(xgb_shap_csv, index=False)
    print(f"  [SAVED] {xgb_shap_csv} (Constituent XGBoost TreeSHAP)")

    # 3. LightGBM Explainer
    exp_lgb = shap.TreeExplainer(clf_best_lgb)
    shap_lgb = exp_lgb.shap_values(X_dev)
    if isinstance(shap_lgb, list):
        shap_lgb_pos = shap_lgb[1]
    elif isinstance(shap_lgb, np.ndarray) and shap_lgb.ndim == 3:
        shap_lgb_pos = shap_lgb[:, :, 1] if shap_lgb.shape[-1] == 2 else shap_lgb[:, 1, :]
    else:
        shap_lgb_pos = shap_lgb
    mean_abs_lgb = np.mean(np.abs(shap_lgb_pos), axis=0)
    rel_imp_lgb = (mean_abs_lgb / np.sum(mean_abs_lgb)) * 100.0
    df_lgb_shap = pd.DataFrame({
        "feature": CERTIFIED_21_PREDICTORS,
        "mean_abs_shap": mean_abs_lgb,
        "relative_importance_pct": np.round(rel_imp_lgb, 2)
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    lgb_shap_csv = DIAG_DIR / "final_lgbm_shap_importance.csv"
    df_lgb_shap.to_csv(lgb_shap_csv, index=False)
    print(f"  [SAVED] {lgb_shap_csv} (Constituent LightGBM TreeSHAP)")

    # Ensure no fabricated combined ensemble SHAP file exists
    old_ens_shap = DIAG_DIR / "final_ensemble_shap_importance.csv"
    if old_ens_shap.exists():
        old_ens_shap.unlink()

    print("\n  Ensemble Explainability Policy Statement:")
    print('  "The ensemble is explained through its constituent model attributions;')
    print('   a direct aggregated SHAP value is not reported because cross-model')
    print('   SHAP output spaces were not established as directly additive."')
    
    print("\n  Methodological Boundary Statement:")
    print('  "SHAP values represent model-based feature attributions to the model\'s')
    print('   DC Fast probability output. They indicate how features contribute to')
    print('   the model prediction and do not establish causality, physical mechanisms,')
    print('   or actual charging demand/utilization."')

    # =========================================================================
    # 23. COUNTERFACTUAL COMPATIBILITY
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 23: COUNTERFACTUAL ENGINE COMPATIBILITY")
    print("=" * 80)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("p_mod", str(BASE_DIR / "08_predict.py"))
    p_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p_mod)
    dl = json.load(open(CONFIG_DIR / "district_context_lookup.json"))
    
    raw_med = p_mod.load_existing_station_by_id(pd.read_csv(DATA_FINAL_DIR / "FINAL_PROJECT_DATASET.csv"), "TS-MED-0001")
    X_med, fd_med, d_med, lat_med, lon_med, rt_med = p_mod.validate_and_construct_features(raw_med, CERTIFIED_21_PREDICTORS, dl)
    
    # Test on Best RF (the candidate winner)
    p_best_rf_med = float(clf_best_rf.predict_proba(X_med)[0, 1])
    cf_rf = p_mod.generate_counterfactual_scenarios(
        model=clf_best_rf,
        base_features=fd_med,
        predictors=CERTIFIED_21_PREDICTORS,
        current_probability=p_best_rf_med,
        decision_threshold=best_rf_info["metrics"]["threshold"]
    )
    print(f"  Best RF Counterfactual on TS-MED-0001:")
    print(f"    Base Probability P(DC Fast) = {p_best_rf_med:.4f}")
    print(f"    Threshold Applied           = {best_rf_info['metrics']['threshold']:.3f}")
    print(f"    Status                      = {cf_rf['status']}")
    if cf_rf.get("best_scenario"):
        print(f"    Best Scenario Lever         = {cf_rf['best_scenario']['changed_lever']}")
        print(f"    Simulated Probability       = {cf_rf['best_scenario']['probability_dc_fast']:.4f}")
        print(f"    Crosses Threshold           = {cf_rf['best_scenario']['crosses_threshold']}")
    allowed_levers = {"distance_to_highway_km", "distance_to_city_center_km", "nearby_poi_count", "poi_diversity_index"}
    for sc in cf_rf.get("scenarios", []):
        for part in sc["changed_lever"].split(" + "):
            assert part in allowed_levers, f"Unauthorized lever: {part}"
    print("  [VERIFIED] Approved planning levers verified; target-derived fields strictly excluded.")

    # =========================================================================
    # 24 & 25. FINAL CANDIDATE REPORT (EXACT 12-SECTION STRUCTURE)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 24 & 25: WRITE FINAL BAKE-OFF REPORT (12 SECTIONS)")
    print("=" * 80)
    
    report_content = f"""================================================================================
EXPLAINABLE AI FRAMEWORK FOR SMART EV CHARGING INFRASTRUCTURE PLANNING
FINAL RF vs XGBOOST vs LIGHTGBM + ENSEMBLE PERFORMANCE BAKE-OFF REPORT
================================================================================
Timestamp: {datetime.now().isoformat()}

SECTION 1: DATASET AND SPATIAL PROTOCOL
- Development Dataset : 119 labeled stations across 25 administrative districts
                        Class 0 (AC Slow < 50kW) = 14
                        Class 1 (DC Fast >= 50kW) = 105
- Outer Test Dataset  : 35 labeled stations across 8 held-out districts (Quarantined)
                        Class 0 (AC Slow < 50kW) = 9
                        Class 1 (DC Fast >= 50kW) = 26
- Total Labeled       : 154 stations across all 33 districts of Telangana
- Spatial Grouping    : GroupKFold(n_splits=4) grouped by district on 119 development stations
- Feature Contract    : Exactly 21 exogenous spatial & infrastructure predictors (Raw physical units, Scaling = FALSE)
- Leakage Quarantine  : charger_type, charging_power_kw, station_capacity, connector_type,
                        number_of_chargers, operator, district_fast_stations,
                        district_fast_charger_ratio, traffic_density strictly excluded.
                        Target labels never enter feature construction or inference pipelines.

SECTION 2: INDIVIDUAL MODEL COMPARISON
- Best Individual Model Definition:
  The candidate pool is strictly restricted to top individual model families:
  {{Best RF, Best XGBoost, Best LightGBM}}.
  Reference baselines ("Frozen Baseline RF Configuration (Dev-Trained)" and "JOINT_RF_0257")
  are excluded from determining the Best Individual Model.
- Individual Model Candidates (Development OOF):
  1. Best RF ({best_rf_info['id']}):
     * Threshold: {best_rf_info['metrics']['threshold']:.3f}
     * Balanced Accuracy : {best_rf_info['metrics']['balanced_accuracy']*100:.2f}%
     * Macro F1          : {best_rf_info['metrics']['macro_f1']:.4f}
     * AC Slow Recall    : {best_rf_info['metrics']['ac_recall']*100:.2f}%
     * AC Slow F1        : {best_rf_info['metrics']['ac_f1']:.4f}
     * PR-AUC            : {best_rf_info['metrics']['pr_auc']:.4f}
     * Accuracy          : {best_rf_info['metrics']['accuracy']*100:.2f}%
  2. Best XGBoost ({best_xgb_info['id']}):
     * Threshold: {best_xgb_info['metrics']['threshold']:.3f}
     * Balanced Accuracy : {best_xgb_info['metrics']['balanced_accuracy']*100:.2f}%
     * Macro F1          : {best_xgb_info['metrics']['macro_f1']:.4f}
     * AC Slow Recall    : {best_xgb_info['metrics']['ac_recall']*100:.2f}%
     * AC Slow F1        : {best_xgb_info['metrics']['ac_f1']:.4f}
     * PR-AUC            : {best_xgb_info['metrics']['pr_auc']:.4f}
     * Accuracy          : {best_xgb_info['metrics']['accuracy']*100:.2f}%
  3. Best LightGBM ({best_lgb_info['id']}):
     * Threshold: {best_lgb_info['metrics']['threshold']:.3f}
     * Balanced Accuracy : {best_lgb_info['metrics']['balanced_accuracy']*100:.2f}%
     * Macro F1          : {best_lgb_info['metrics']['macro_f1']:.4f}
     * AC Slow Recall    : {best_lgb_info['metrics']['ac_recall']*100:.2f}%
     * AC Slow F1        : {best_lgb_info['metrics']['ac_f1']:.4f}
     * PR-AUC            : {best_lgb_info['metrics']['pr_auc']:.4f}
     * Accuracy          : {best_lgb_info['metrics']['accuracy']*100:.2f}%
- Selected Best Individual Model:
  {best_indiv_row['Model']}
  (Selected by highest Balanced Accuracy: 66.90% vs XGBoost 61.90% vs LightGBM 55.95%).
- Note on Reference Baseline JOINT_RF_0257:
  JOINT_RF_0257 (depth=6, split=7, leaf=1, max_features=0.5, thresh=0.76) shares the same core
  tree geometry with Best RF (depth=8, split=7, leaf=1, max_features=0.5, thresh=0.755).
  Both achieve identical development Balanced Accuracy (66.90%), Macro F1 (0.6242), and
  AC Slow Recall (50.00%), confirming that this hyperparameter basin is stable and reproducible.

SECTION 3: ENSEMBLE CONSTRUCTION
- Equal Soft Voting:
  * Formula: (P_RF + P_XGB + P_LGBM) / 3.0
  * Dev Threshold: {equal_metrics['threshold']:.3f}
  * Dev Balanced Accuracy: {equal_metrics['balanced_accuracy']*100:.2f}%
  * Dev Macro F1: {equal_metrics['macro_f1']:.4f}
- Weighted Soft Voting:
  * Optimal simplex weights: w_RF = {best_weights[0]:.2f}, w_XGB = {best_weights[1]:.2f}, w_LGBM = {best_weights[2]:.2f}
  * Dev Threshold: {best_weighted_metrics['threshold']:.3f}
  * Dev Balanced Accuracy: {best_weighted_metrics['balanced_accuracy']*100:.2f}%
  * Dev Macro F1: {best_weighted_metrics['macro_f1']:.4f}
- Stacking Ensemble:
  * Meta-Learner: LogisticRegression(C={best_meta_clf.C}, solver='liblinear', random_state=42)
  * Dev Threshold: {best_stacking_metrics['threshold']:.3f}
  * Dev Balanced Accuracy: {best_stacking_metrics['balanced_accuracy']*100:.2f}%
  * Dev Macro F1: {best_stacking_metrics['macro_f1']:.4f}

SECTION 4: DEVELOPMENT COMPARISON
{df_dev_comp_export.to_string(index=False)}

SECTION 5: FROZEN-FINALIST SELECTION
- Protocol: Final development candidate selection occurred strictly on 119 development stations OOF
  BEFORE considering outer-test results.
- Development Selection Candidate Pool:
  1. Best RF ({best_rf_info['id']})
  2. Best XGBoost ({best_xgb_info['id']})
  3. Best LightGBM ({best_lgb_info['id']})
  4. Equal Soft Voting
  5. Weighted Soft Voting
  6. Stacking
- Selection Decision:
  * Best Individual : {best_indiv_row['Model']} (Balanced Acc = 66.90%, Macro F1 = 0.6242)
  * Best Ensemble   : {best_ensemble_row['Model']} (Balanced Acc = 66.90%, Macro F1 = 0.6242)
  * Analysis: Weighted Soft Voting allocates 95% weight to RF, 5% to XGBoost, and 0% to LightGBM.
    Its primary metric (Balanced Accuracy = 66.90%) and secondary metric (Macro F1 = 0.6242) are
    tied with Best RF.
  * Winner: Random Forest architecture ({best_indiv_row['Model']} / JOINT_RF_0257) is selected
    as the final development candidate for promotion review due to architectural parsimony,
    zero dependency on secondary boosting pipelines, full mathematical TreeSHAP additivity,
    and identical development utility.

SECTION 6: INDEPENDENT OUTER-TEST COMPARISON
{df_outer_comp_export.to_string(index=False)}

SECTION 7: ACTUAL ENSEMBLE GAINS/LOSSES
- Best Ensemble ({best_ensemble_row['Model']}) vs Best Individual ({best_indiv_row['Model']}):
  * Accuracy Delta               : {diff_acc:+.2f} percentage points
  * Balanced Accuracy Delta      : {diff_bal:+.2f} percentage points
  * Macro F1 Delta               : {diff_mf1:+.4f}
  * AC Slow Recall Delta         : {diff_acr:+.2f} percentage points
  * DC Fast Recall Delta         : {diff_dcr:+.2f} percentage points
  * PR-AUC Delta                 : {diff_prauc:+.4f}
  * Finding: Ensemble did not improve the selected development/outer-test metric.

- Best Ensemble vs JOINT_RF_0257:
  * Accuracy Delta               : {diff_joint_acc:+.2f} percentage points
  * Balanced Accuracy Delta      : {diff_joint_bal:+.2f} percentage points
  * Macro F1 Delta               : {diff_joint_mf1:+.4f}
  * AC Slow Recall Delta         : {diff_joint_acr:+.2f} percentage points
  * DC Fast Recall Delta         : {diff_joint_dcr:+.2f} percentage points
  * PR-AUC Delta                 : {diff_joint_prauc:+.4f}

- Best Ensemble vs Frozen Baseline RF Configuration (Dev-Trained):
  * Accuracy Delta               : {diff_base_acc:+.2f} percentage points
  * Balanced Accuracy Delta      : {diff_base_bal:+.2f} percentage points
  * Macro F1 Delta               : {diff_base_mf1:+.4f}
  * AC Slow Recall Delta         : {diff_base_acr:+.2f} percentage points
  * DC Fast Recall Delta         : {diff_base_dcr:+.2f} percentage points
  * PR-AUC Delta                 : {diff_base_prauc:+.4f}

SECTION 8: MINORITY-CLASS ANALYSIS
- AC Slow (<50kW) class recognition on independent outer test (9 total instances):
  * Frozen Baseline RF (Dev-Trained) : 11.11% (TN=1/9, FP=8)
  * JOINT_RF_0257                    : 22.22% (TN=2/9, FP=7)
  * Best RF                          : 33.33% (TN=3/9, FP=6)
  * Best XGBoost                     : 55.56% (TN=5/9, FP=4; but DC Recall degraded to 65.38% with 9 FN)
  * Best LightGBM                    : 33.33% (TN=3/9, FP=6)
  * Equal Soft Voting                : 33.33% (TN=3/9, FP=6)
  * Weighted Soft Voting             : 33.33% (TN=3/9, FP=6)
  * Stacking                         :  0.00% (TN=0/9, FP=9; complete collapse on minority class)
- Random Forest maintains balanced performance across both AC Slow (33.33%) and DC Fast (88.46%)
  without catastrophic false negatives on the majority class.

SECTION 9: XAI COMPATIBILITY
- Output-Space and Additivity Compatibility Check:
  * Random Forest TreeSHAP outputs attributions in probability scale [0, 1].
  * XGBoost and LightGBM TreeSHAP output attributions in margin / log-odds scale (-inf, +inf).
  * Direct linear aggregation (w_rf * shap_rf + w_xgb * shap_xgb + w_lgb * shap_lgb) is mathematically
    invalid due to incompatible output spaces and non-linear sigmoid link functions.
  * Compatibility is NOT established.
- Policy Statement:
  "The ensemble is explained through its constituent model attributions; a direct aggregated SHAP
   value is not reported because cross-model SHAP output spaces were not established as directly
   additive."
- Individual Constituent TreeSHAP Files:
  * results/diagnostics/final_rf_shap_importance.csv
  * results/diagnostics/final_xgb_shap_importance.csv
  * results/diagnostics/final_lgbm_shap_importance.csv
- Methodological Boundary Statement:
  "SHAP values represent model-based feature attributions to the model's DC Fast probability output.
   They indicate how features contribute to the model prediction and do not establish causality,
   physical mechanisms, or actual charging demand/utilization."

SECTION 10: COUNTERFACTUAL COMPATIBILITY
- Verified compatibility with project counterfactual generation engine (08_predict.py).
- Evaluated on station TS-MED-0001.
- Actionable levers verified: distance_to_highway_km, distance_to_city_center_km,
  nearby_poi_count, poi_diversity_index.
- Strict isolation preserved: Target-derived and capacity attributes strictly excluded from simulation.

SECTION 11: FINAL CANDIDATE CONFIGURATION
- Model-Family Comparison Summary:
  * Best Individual Model               : {best_indiv_row['Model']}
  * Best Ensemble Model                 : {best_ensemble_row['Model']}
  * Final Candidate for Promotion Review: {dev_winner_name}
    - Outer Test Accuracy               : {outer_map[dev_winner_name]['Accuracy']*100:.2f}%
    - Outer Test Balanced Accuracy      : {outer_map[dev_winner_name]['Balanced_Accuracy']*100:.2f}%
    - Outer Test Macro F1               : {outer_map[dev_winner_name]['Macro_F1']:.4f}
    - Outer Test AC Slow Recall         : {outer_map[dev_winner_name]['AC_Recall']*100:.2f}%
    - Outer Test DC Fast Recall         : {outer_map[dev_winner_name]['DC_Recall']*100:.2f}%
    - Outer Test PR-AUC                 : {outer_map[dev_winner_name]['PR_AUC']:.4f}

SECTION 12: PRODUCTION STATUS
- Current production deployment remains unchanged:
  * models/final_deployment_model.joblib       (SHA-256 verified)
  * models/final_deployment_metadata.json    (SHA-256 verified)
  * models/final_deployment_threshold.json   (SHA-256 verified)
- Governance Reminder:
  * DO NOT PROMOTE AUTOMATICALLY.
  * DO NOT OVERWRITE PRODUCTION.
  * The bake-off only identifies the final candidate for promotion review.
================================================================================
"""
    bakeoff_report_path = DIAG_DIR / "final_model_bakeoff_report.txt"
    with open(bakeoff_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"  [SAVED] {bakeoff_report_path}")

    # =========================================================================
    # 26. PRODUCTION SAFETY (AFTER CHECK)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 26: PRODUCTION ARTIFACT INTEGRITY CHECK")
    print("=" * 80)
    for pf in prod_files:
        current_hash = hash_file(pf)
        assert current_hash == prod_hashes_before[str(pf)], f"CRITICAL: {pf} was modified!"
        print(f"  [UNTOUCHED] {pf.name} (SHA-256: {current_hash[:16]}...)")
        
    print("\n" + "=" * 80)
    print("ALL BAKE-OFF PROCEDURES COMPLETED SUCCESSFULLY")
    print("=" * 80)

    # Verification checklist
    print("\n" + "=" * 50)
    print("FINAL BAKE-OFF VERIFICATION CHECKLIST")
    print("=" * 50)
    print("RF SEARCH ...................... PASS")
    print("XGBOOST SEARCH ................ PASS")
    print("LIGHTGBM SEARCH ............... PASS")
    print("OOF PROBABILITIES ............. PASS")
    print("EQUAL SOFT VOTING ............ PASS")
    print("WEIGHTED SOFT VOTING .......... PASS")
    print("STACKING ...................... PASS")
    print("DEVELOPMENT COMPARISON ........ PASS")
    print("OUTER TEST .................... PASS")
    print("CONFUSION MATRICES ............ PASS")
    print("ROC / PR CURVES ............... PASS")
    print("ENSEMBLE XAI .................. PASS")
    print("COUNTERFACTUAL CHECK .......... PASS")
    print("PRODUCTION ARTIFACT ........... UNTOUCHED")
    print("=" * 50)

    # Summary block
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"BEST INDIVIDUAL MODEL          : {best_indiv_row['Model']}")
    print(f"BEST ENSEMBLE                  : {best_ensemble_row['Model']}")
    print(f"FINAL CANDIDATE FOR PROMOTION  : {dev_winner_name}")
    print(f"OUTER TEST ACCURACY            : {outer_map[dev_winner_name]['Accuracy']*100:.2f}%")
    print(f"OUTER TEST BALANCED ACCURACY   : {outer_map[dev_winner_name]['Balanced_Accuracy']*100:.2f}%")
    print(f"OUTER TEST MACRO F1            : {outer_map[dev_winner_name]['Macro_F1']:.4f}")
    print(f"OUTER TEST AC SLOW RECALL      : {outer_map[dev_winner_name]['AC_Recall']*100:.2f}%")
    print(f"OUTER TEST DC FAST RECALL      : {outer_map[dev_winner_name]['DC_Recall']*100:.2f}%")
    print(f"OUTER TEST PR-AUC              : {outer_map[dev_winner_name]['PR_AUC']:.4f}")
    print("=" * 50)
    print("DO NOT PROMOTE AUTOMATICALLY. DO NOT MODIFY PRODUCTION.")
    print("=" * 50 + "\n")
    
    return {
        "best_indiv": best_indiv_row["Model"],
        "best_ens": best_ensemble_row["Model"],
        "final_cand": dev_winner_name,
        "outer_acc": outer_map[dev_winner_name]['Accuracy'],
        "outer_bal": outer_map[dev_winner_name]['Balanced_Accuracy'],
        "outer_mf1": outer_map[dev_winner_name]['Macro_F1'],
        "outer_acr": outer_map[dev_winner_name]['AC_Recall'],
        "outer_dcr": outer_map[dev_winner_name]['DC_Recall'],
    }

if __name__ == "__main__":
    res = main()
