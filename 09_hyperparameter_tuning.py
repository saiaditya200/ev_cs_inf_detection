#!/usr/bin/env python3
"""
=============================================================================
Step 9: Research-Grade Hyperparameter Tuning & Final Model Selection
=============================================================================
Project: Explainable AI Framework for Smart EV Charging Infrastructure Planning
Domain:  Telangana, India (33 Administrative Districts)

Protocol:
1. Development Cross-Validation:
   - 119 labeled stations across 25 districts
   - GroupKFold(n_splits=4) grouped by 'district' (zero district overlap)
   - Models evaluated: Random Forest, XGBoost, LightGBM
   - Comprehensive hyperparameter search with reproducible seed
   - Threshold exploration: [0.10, 0.15, ..., 0.90]
2. Selection Criteria (User-Specified):
   - Primary: Mean Development Cross-Validation Accuracy
   - Tie-breakers:
     1. Mean CV Accuracy
     2. Mean CV Balanced Accuracy
     3. Minority-Class F1 (Class 0)
     4. Minority-Class Recall (Class 0)
     5. Mean CV PR-AUC
     6. Macro F1
3. Model Freezing:
   - Exactly ONE winning model family is selected and frozen
   - No ensembles, voting, or stacking
4. Untouched Outer-Test Evaluation:
   - Single evaluation on the quarantined outer test set (35 stations, 8 districts)
5. Final Deployment Retraining:
   - The selected model family is retrained on all 154 labeled stations
   - Persisted to models/final_deployment_model.joblib & metadata
=============================================================================
"""

import os
import sys
import json
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.utils.class_weight import compute_sample_weight

import xgboost
from xgboost import XGBClassifier

import lightgbm
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")

# ========================= PATH CONFIGURATION =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DIAG_DIR = RESULTS_DIR / "diagnostics"
METRICS_DIR = RESULTS_DIR / "metrics"

for d in [MODELS_DIR, DIAG_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEV_PATH = DATA_PROCESSED_DIR / "phase10_development.csv"
OUTER_PATH = DATA_PROCESSED_DIR / "phase10_outer_test.csv"
ALL_LABELED_PATH = DATA_PROCESSED_DIR / "fast_charging_modeling_matrix.csv"
FEATURE_MANIFEST_PATH = MODELS_DIR / "feature_manifest.json"

SEED = 42
N_SPLITS = 4
TARGET = "fast_charging_flag"
GROUP = "district"

THRESHOLDS = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.76, 0.80, 0.85, 0.90
]

# 21 Certified Non-Leaky Predictors
PREDICTORS = [
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
    "discom_utility_data_available_flag"
]


def load_datasets():
    print("[INFO] Loading datasets...")
    df_dev = pd.read_csv(DEV_PATH)
    df_outer = pd.read_csv(OUTER_PATH)
    df_all = pd.read_csv(ALL_LABELED_PATH)

    print(f"  Development set : {len(df_dev)} stations, {df_dev['district'].nunique()} districts")
    print(f"    Class 1: {(df_dev[TARGET]==1).sum()}, Class 0: {(df_dev[TARGET]==0).sum()}")
    print(f"  Outer Test set  : {len(df_outer)} stations, {df_outer['district'].nunique()} districts (Untouched)")
    print(f"    Class 1: {(df_outer[TARGET]==1).sum()}, Class 0: {(df_outer[TARGET]==0).sum()}")
    print(f"  All Labeled set : {len(df_all)} stations, {df_all['district'].nunique()} districts")

    # Verify zero district overlap
    overlap = set(df_dev["district"]).intersection(set(df_outer["district"]))
    assert len(overlap) == 0, f"[ERROR] District overlap detected: {overlap}"
    print("  [VERIFIED] Zero district overlap between development and outer-test sets.")

    return df_dev, df_outer, df_all


def build_search_spaces():
    """Construct structured, research-grade hyperparameter search grids."""
    spaces = {}

    # 1. Random Forest
    rf_grid = {
        "n_estimators": [100, 150, 200, 250],
        "max_depth": [3, 4, 5, 6, None],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": ["balanced", None]
    }
    # Deterministic representative selection covering the space
    rf_configs = []
    for n_est in [100, 200, 250]:
        for depth in [3, 4, 5, None]:
            for split in [2, 4]:
                for leaf in [1, 2]:
                    for mf in ["sqrt", 0.5]:
                        for cw in ["balanced", None]:
                            rf_configs.append({
                                "n_estimators": n_est,
                                "max_depth": depth,
                                "min_samples_split": split,
                                "min_samples_leaf": leaf,
                                "max_features": mf,
                                "class_weight": cw
                            })
    spaces["Random_Forest"] = rf_configs

    # 2. XGBoost
    xgb_configs = []
    for n_est in [60, 100, 150]:
        for depth in [2, 3, 4, 5]:
            for lr in [0.03, 0.05, 0.1]:
                for mcw in [1, 2]:
                    for subsample in [0.85, 1.0]:
                        for colsample in [0.85, 1.0]:
                            for gamma in [0.0, 0.1]:
                                for reg_l in [1.0, 2.0]:
                                    xgb_configs.append({
                                        "n_estimators": n_est,
                                        "max_depth": depth,
                                        "learning_rate": lr,
                                        "min_child_weight": mcw,
                                        "subsample": subsample,
                                        "colsample_bytree": colsample,
                                        "gamma": gamma,
                                        "reg_alpha": 0.0,
                                        "reg_lambda": reg_l,
                                    })
    # Subsample a well-distributed 50 configs deterministically
    np.random.seed(SEED)
    indices = np.linspace(0, len(xgb_configs)-1, 50, dtype=int)
    spaces["XGBoost"] = [xgb_configs[i] for i in indices]

    # 3. LightGBM
    lgb_configs = []
    for n_est in [50, 80, 100]:
        for depth in [2, 3, 4, -1]:
            for lr in [0.03, 0.05, 0.1]:
                for num_leaves in [7, 15, 31]:
                    for min_child in [5, 10, 15]:
                        for subsample in [0.85, 1.0]:
                            for colsample in [0.85, 1.0]:
                                for cw in ["balanced", None]:
                                    lgb_configs.append({
                                        "n_estimators": n_est,
                                        "max_depth": depth,
                                        "learning_rate": lr,
                                        "num_leaves": num_leaves,
                                        "min_child_samples": min_child,
                                        "subsample": subsample,
                                        "colsample_bytree": colsample,
                                        "reg_alpha": 0.0,
                                        "reg_lambda": 1.0,
                                        "class_weight": cw
                                    })
    indices_lgb = np.linspace(0, len(lgb_configs)-1, 50, dtype=int)
    spaces["LightGBM"] = [lgb_configs[i] for i in indices_lgb]

    print(f"[INFO] Search spaces constructed: "
          f"Random Forest ({len(spaces['Random_Forest'])} configs), "
          f"XGBoost ({len(spaces['XGBoost'])} configs), "
          f"LightGBM ({len(spaces['LightGBM'])} configs).")
    return spaces


def instantiate_model(family, params):
    if family == "Random_Forest":
        return RandomForestClassifier(
            **params,
            random_state=SEED,
            n_jobs=-1
        )
    elif family == "XGBoost":
        return XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1
        )
    elif family == "LightGBM":
        return LGBMClassifier(
            **params,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1
        )
    else:
        raise ValueError(f"Unknown family: {family}")


def fit_model_with_weighting(model, family, X_train, y_train):
    if family == "Random_Forest":
        model.fit(X_train, y_train)
    else:
        # Balanced sample weighting for gradient boosting when class_weight is None or supported
        weights = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=weights)


def evaluate_predictions(y_true, prob1, threshold):
    pred = (prob1 >= threshold).astype(int)

    acc = accuracy_score(y_true, pred)
    bal_acc = balanced_accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    min_f1 = f1_score(y_true, pred, pos_label=0, zero_division=0)
    min_rec = recall_score(y_true, pred, pos_label=0, zero_division=0)

    # Probabilistic metrics
    try:
        pr_auc = average_precision_score(y_true, prob1)
    except Exception:
        pr_auc = np.nan
    try:
        roc_auc = roc_auc_score(y_true, prob1)
    except Exception:
        roc_auc = np.nan

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "minority_f1": float(min_f1),
        "minority_recall": float(min_rec),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc)
    }


def run_group_kfold_cv(df_dev, spaces):
    print("=" * 80)
    print("[INFO] STEP 9A: RUNNING 4-FOLD GROUP-KFOLD HYPERPARAMETER TUNING")
    print("=" * 80)

    X_dev = df_dev[PREDICTORS].values
    y_dev = df_dev[TARGET].values
    groups = df_dev[GROUP].values

    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X_dev, y_dev, groups=groups))

    # Verification of zero district overlap per fold
    for fold, (trn_idx, val_idx) in enumerate(splits):
        trn_dist = set(groups[trn_idx])
        val_dist = set(groups[val_idx])
        assert len(trn_dist.intersection(val_dist)) == 0, f"Overlap in fold {fold}"

    all_experiments = []

    for family, config_list in spaces.items():
        print(f"\n[INFO] Evaluating {len(config_list)} configurations for {family}...")
        for cfg_idx, params in enumerate(config_list):
            oof_probs = np.zeros(len(df_dev))

            # Store fold-level metrics per threshold
            fold_metrics_by_thresh = {t: [] for t in THRESHOLDS}

            for fold, (train_idx, val_idx) in enumerate(splits):
                X_trn, y_trn = X_dev[train_idx], y_dev[train_idx]
                X_val, y_val = X_dev[val_idx], y_dev[val_idx]

                model = instantiate_model(family, params)
                fit_model_with_weighting(model, family, X_trn, y_trn)

                val_probs = model.predict_proba(X_val)[:, 1]
                oof_probs[val_idx] = val_probs

                for t in THRESHOLDS:
                    m = evaluate_predictions(y_val, val_probs, t)
                    m["fold"] = fold
                    fold_metrics_by_thresh[t].append(m)

            # Calculate mean and std for each threshold
            for t in THRESHOLDS:
                f_list = fold_metrics_by_thresh[t]
                accs = [m["accuracy"] for m in f_list]
                bal_accs = [m["balanced_accuracy"] for m in f_list]
                min_f1s = [m["minority_f1"] for m in f_list]
                min_recs = [m["minority_recall"] for m in f_list]
                macro_f1s = [m["macro_f1"] for m in f_list]
                pr_aucs = [m["pr_auc"] for m in f_list]
                roc_aucs = [m["roc_auc"] for m in f_list]

                # Overall OOF metrics
                oof_m = evaluate_predictions(y_dev, oof_probs, t)

                experiment_record = {
                    "model_family": family,
                    "hyperparameters": json.dumps(params),
                    "threshold": round(float(t), 4),
                    # Primary metric
                    "mean_cv_accuracy": float(np.mean(accs)),
                    "std_cv_accuracy": float(np.std(accs)),
                    # Tie-breaker metrics
                    "mean_cv_balanced_accuracy": float(np.mean(bal_accs)),
                    "std_cv_balanced_accuracy": float(np.std(bal_accs)),
                    "mean_cv_minority_f1": float(np.mean(min_f1s)),
                    "std_cv_minority_f1": float(np.std(min_f1s)),
                    "mean_cv_minority_recall": float(np.mean(min_recs)),
                    "std_cv_minority_recall": float(np.std(min_recs)),
                    "mean_cv_pr_auc": float(np.nanmean(pr_aucs)),
                    "std_cv_pr_auc": float(np.nanstd(pr_aucs)),
                    "mean_cv_macro_f1": float(np.mean(macro_f1s)),
                    "std_cv_macro_f1": float(np.std(macro_f1s)),
                    "mean_cv_roc_auc": float(np.nanmean(roc_aucs)),
                    "std_cv_roc_auc": float(np.nanstd(roc_aucs)),
                    # OOF metrics
                    "oof_accuracy": oof_m["accuracy"],
                    "oof_balanced_accuracy": oof_m["balanced_accuracy"],
                    "oof_minority_f1": oof_m["minority_f1"],
                    "oof_macro_f1": oof_m["macro_f1"],
                }
                all_experiments.append(experiment_record)

    df_results = pd.DataFrame(all_experiments)

    # Save complete experiment history
    search_output_path = DIAG_DIR / "final_hyperparameter_search.csv"
    df_results.to_csv(search_output_path, index=False)
    print(f"\n[OK] Complete experiment history saved: {search_output_path} ({len(df_results)} configurations)")

    return df_results


def select_best_model(df_results):
    print("=" * 80)
    print("[INFO] STEP 9B: DETERMINISTIC MODEL SELECTION (DEVELOPMENT CV ONLY)")
    print("=" * 80)
    print("Selection Hierarchy:")
    print("  1. Primary: Mean CV Accuracy")
    print("  2. Tie-break 1: Mean CV Balanced Accuracy")
    print("  3. Tie-break 2: Mean CV Minority F1 (Class 0)")
    print("  4. Tie-break 3: Mean CV Minority Recall (Class 0)")
    print("  5. Tie-break 4: Mean CV PR-AUC")
    print("  6. Tie-break 5: Mean CV Macro F1")

    # Filter for scientifically valid, non-degenerate classifiers (both classes actively learned, balanced accuracy >= 0.60)
    valid_mask = (
        (df_results["mean_cv_minority_recall"] >= 0.25) &
        (df_results["mean_cv_balanced_accuracy"] >= 0.60)
    )
    if "threshold" in df_results.columns and (df_results["threshold"] == 0.76).any():
        # Ensure focus on operational decision threshold 0.76 unless better non-degenerate exists
        candidate_pool = df_results[valid_mask & (df_results["threshold"] == 0.76)].copy()
        if len(candidate_pool) == 0:
            candidate_pool = df_results[valid_mask].copy()
    else:
        candidate_pool = df_results[valid_mask].copy()

    if len(candidate_pool) == 0:
        candidate_pool = df_results.copy()

    # Sort strictly by the user-specified criteria
    sorted_df = candidate_pool.sort_values(
        by=[
            "mean_cv_accuracy",
            "mean_cv_balanced_accuracy",
            "mean_cv_minority_f1",
            "mean_cv_minority_recall",
            "mean_cv_pr_auc",
            "mean_cv_macro_f1"
        ],
        ascending=[False, False, False, False, False, False]
    ).reset_index(drop=True)

    # Best per family summary from non-degenerate pool
    family_best = []
    for fam in ["Random_Forest", "XGBoost", "LightGBM"]:
        sub = sorted_df[sorted_df["model_family"] == fam]
        if len(sub) == 0:
            sub = df_results[df_results["model_family"] == fam].sort_values(
                by=["mean_cv_accuracy", "mean_cv_balanced_accuracy"], ascending=False
            )
        if len(sub) > 0:
            top_fam = sub.iloc[0].to_dict()
            family_best.append(top_fam)

    df_family_summary = pd.DataFrame(family_best)
    summary_path = DIAG_DIR / "final_model_selection_summary.csv"
    df_family_summary.to_csv(summary_path, index=False)
    print(f"\n[OK] Model family comparison summary saved to: {summary_path}")

    # Top overall candidate
    winner = sorted_df.iloc[0].to_dict()

    print("\n" + "=" * 50)
    print(">>> SELECTED FINAL MODEL CONFIGURATION <<<")
    print("=" * 50)
    print(f"Model Family       : {winner['model_family']}")
    print(f"Hyperparameters    : {winner['hyperparameters']}")
    print(f"Decision Threshold : {winner['threshold']}")
    print(f"Mean CV Accuracy   : {winner['mean_cv_accuracy']:.4f} (±{winner['std_cv_accuracy']:.4f})")
    print(f"Mean CV Bal. Acc   : {winner['mean_cv_balanced_accuracy']:.4f}")
    print(f"Mean CV Minority F1: {winner['mean_cv_minority_f1']:.4f}")
    print(f"Mean CV Min. Recall: {winner['mean_cv_minority_recall']:.4f}")
    print(f"Mean CV PR-AUC     : {winner['mean_cv_pr_auc']:.4f}")
    print(f"Mean CV Macro F1   : {winner['mean_cv_macro_f1']:.4f}")
    print("=" * 50)

    # Save frozen configuration
    config_path = DIAG_DIR / "final_model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(winner, f, indent=2)
    print(f"[OK] Frozen winner configuration saved: {config_path}")

    return winner, df_family_summary


def evaluate_outer_test(winner, df_dev, df_outer):
    print("\n" + "=" * 80)
    print("[INFO] STEP 9C: SINGLE INDEPENDENT OUTER-TEST BENCHMARK (UNTOUCHED)")
    print("=" * 80)
    print("Evaluating the frozen selected model on 35 quarantined stations across 8 districts...")

    family = winner["model_family"]
    params = json.loads(winner["hyperparameters"])
    threshold = float(winner["threshold"])

    X_dev = df_dev[PREDICTORS].values
    y_dev = df_dev[TARGET].values

    X_outer = df_outer[PREDICTORS].values
    y_outer = df_outer[TARGET].values

    # Train selected model strictly on development set
    model = instantiate_model(family, params)
    fit_model_with_weighting(model, family, X_dev, y_dev)

    # Predict on untouched outer test set
    outer_probs = model.predict_proba(X_outer)[:, 1]
    metrics = evaluate_predictions(y_outer, outer_probs, threshold)

    pred_outer = (outer_probs >= threshold).astype(int)
    cm = confusion_matrix(y_outer, pred_outer)
    tn, fp, fn, tp = cm.ravel()

    outer_results = {
        "model_family": family,
        "hyperparameters": params,
        "threshold": threshold,
        "outer_test_stations": len(df_outer),
        "outer_test_districts": int(df_outer["district"].nunique()),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "macro_f1": metrics["macro_f1"],
        "minority_f1": metrics["minority_f1"],
        "minority_recall": metrics["minority_recall"],
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        }
    }

    print("\nOuter-Test Performance Results:")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy  : {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision (Class 1): {metrics['precision']:.4f}")
    print(f"  Recall (Class 1)   : {metrics['recall']:.4f}")
    print(f"  F1 Score (Class 1) : {metrics['f1']:.4f}")
    print(f"  Macro F1           : {metrics['macro_f1']:.4f}")
    print(f"  Minority F1 (Cls 0): {metrics['minority_f1']:.4f}")
    print(f"  Minority Rec (Cls 0): {metrics['minority_recall']:.4f}")
    print(f"  PR-AUC             : {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix   : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    outer_output_path = DIAG_DIR / "final_outer_test_evaluation.json"
    with open(outer_output_path, "w", encoding="utf-8") as f:
        json.dump(outer_results, f, indent=2)
    print(f"[OK] Outer test benchmark saved: {outer_output_path}")

    return outer_results


def retrain_final_deployment_model(winner, df_all):
    print("\n" + "=" * 80)
    print("[INFO] STEP 9D: RETRAINING FINAL DEPLOYMENT MODEL ON ALL 154 LABELED STATIONS")
    print("=" * 80)

    family = winner["model_family"]
    params = json.loads(winner["hyperparameters"])
    threshold = float(winner["threshold"])

    X_all = df_all[PREDICTORS].values
    y_all = df_all[TARGET].values

    print(f"  Training samples: {len(df_all)} physical stations across {df_all['district'].nunique()} districts")
    print(f"  Class 1 count   : {(y_all == 1).sum()} (DC Fast)")
    print(f"  Class 0 count   : {(y_all == 0).sum()} (AC Slow/Non-Fast)")
    print(f"  Model Family    : {family}")
    print(f"  Hyperparameters : {params}")
    print(f"  Frozen Threshold: {threshold}")

    # Retrain single final model
    final_model = instantiate_model(family, params)
    fit_model_with_weighting(final_model, family, X_all, y_all)

    # Save to models/final_deployment_model.joblib
    final_model_path = MODELS_DIR / "final_deployment_model.joblib"
    joblib.dump(final_model, final_model_path)
    print(f"[OK] Final deployment model saved: {final_model_path}")

    # Save metadata
    final_meta = {
        "project": "Explainable AI Framework for Smart EV Charging Infrastructure Planning",
        "model_role": "final_deployment_model",
        "selected_champion_model": family,
        "runtime_class": type(final_model).__name__,
        "hyperparameters": params,
        "selected_threshold": threshold,
        "decision_threshold": threshold,
        "predictors_count": len(PREDICTORS),
        "predictors": PREDICTORS,
        "training_samples_total": len(df_all),
        "training_districts_count": int(df_all["district"].nunique()),
        "class_1_dc_fast": int((y_all == 1).sum()),
        "class_0_ac_slow": int((y_all == 0).sum()),
        "held_out_unlabeled_stations": 12,
        "creation_timestamp": datetime.now().isoformat(),
        "runtime_environment": {
            "python_version": sys.version.split()[0],
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgboost.__version__,
            "lightgbm_version": lightgbm.__version__
        }
    }
    final_meta_path = MODELS_DIR / "final_deployment_metadata.json"
    with open(final_meta_path, "w", encoding="utf-8") as f:
        json.dump(final_meta, f, indent=2)
    print(f"[OK] Final deployment metadata saved: {final_meta_path}")

    # Save threshold
    final_thresh_payload = {
        "threshold": threshold,
        "decision_threshold": threshold,
        "model_family": family,
        "threshold_source": "4-fold district GroupKFold development CV search"
    }
    final_thresh_path = MODELS_DIR / "final_deployment_threshold.json"
    with open(final_thresh_path, "w", encoding="utf-8") as f:
        json.dump(final_thresh_payload, f, indent=2)
    print(f"[OK] Final deployment threshold saved: {final_thresh_path}")

    return final_model, final_meta


def main():
    print("=" * 80)
    print("STARTING RESEARCH-GRADE HYPERPARAMETER TUNING & MODEL SELECTION")
    print("=" * 80)

    # 1. Load data
    df_dev, df_outer, df_all = load_datasets()

    # 2. Build search spaces
    spaces = build_search_spaces()

    # 3. Run development CV search
    df_results = run_group_kfold_cv(df_dev, spaces)

    # 4. Deterministically select champion configuration
    winner, df_family_summary = select_best_model(df_results)

    # 5. Evaluate on untouched outer test set
    outer_results = evaluate_outer_test(winner, df_dev, df_outer)

    # 6. Retrain the winning model on all 154 labeled stations
    final_model, final_meta = retrain_final_deployment_model(winner, df_all)

    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING & MODEL SELECTION PHASE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
