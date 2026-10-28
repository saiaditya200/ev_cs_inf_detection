#!/usr/bin/env python3
"""
=============================================================================
Step 3: Spatial-Aware Model Training & GroupKFold Cross-Validation
=============================================================================
Project: Explainable AI Framework for Smart EV Charging Infrastructure Planning
Domain:  Telangana, India (33 Administrative Districts)

Key Functions:
- Loads the development set (119 stations across 25 districts).
- Uses unscaled natural feature representations (tree ensembles are scale-invariant).
- Implements strict District-Grouped Cross-Validation (GroupKFold, n_splits=4):
  * Groups exclusively by 'district'
  * Verifies zero district overlap across all 4 folds:
    train_districts ∩ validation_districts == empty
  * Persists detailed fold breakdown to results/metrics/cv_fold_details.csv.
- Corrects Class Imbalance Handling for minority Class 0 (14 vs 105 in dev set):
  * Random Forest: class_weight='balanced'
  * XGBoost & LightGBM: compute_sample_weight('balanced', y_train) passed to fit()
  * Zero relabeling or inverted scale_pos_weight; target semantics fully preserved.
- Trains and benchmarks:
  1. Random Forest Classifier
  2. XGBoost Classifier
  3. LightGBM Classifier
- Deterministically selects champion model based on development CV PR-AUC & Balanced Accuracy.
- Fits final models on all 119 development stations with zero outer-test leakage.
- Persists:
  * models/random_forest_model.joblib
  * models/xgboost_model.joblib
  * models/lightgbm_model.joblib
  * models/selected_model.joblib
  * models/model_metadata.json
  * results/metrics/cv_metrics.csv
  * results/metrics/cv_fold_details.csv
=============================================================================
"""

from pathlib import Path
from itertools import product
from datetime import datetime
import json
import sys
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import lightgbm

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


BASE = Path(".")
DEV_PATH = BASE / "data" / "processed" / "phase10_development.csv"
SPLIT_PATH = BASE / "data" / "processed" / "phase10_final_split_assignments.csv"

MODEL_DIR = BASE / "models"
RESULT_DIR = BASE / "results"
DIAG_DIR = RESULT_DIR / "diagnostics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 4

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

TARGET = "fast_charging_flag"
GROUP = "district"
THRESHOLDS = np.arange(0.10, 0.901, 0.01)


def make_model(family, params):
    if family == "Random_Forest":
        return RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        )

    if family == "XGBoost":
        return XGBClassifier(
            **params,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1
        )

    if family == "LightGBM":
        return LGBMClassifier(
            **params,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1
        )

    raise ValueError("Unknown model family: {}".format(family))


def fit_model(model, family, X_train, y_train):
    if family == "Random_Forest":
        model.fit(X_train, y_train)
    else:
        weights = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=weights)


def threshold_metrics(y_true, prob1, threshold):
    pred = (prob1 >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                pred,
                average="macro",
                zero_division=0
            )
        ),
        "minority_f1": float(
            f1_score(
                y_true,
                pred,
                pos_label=0,
                zero_division=0
            )
        ),
        "minority_recall": float(
            recall_score(
                y_true,
                pred,
                pos_label=0,
                zero_division=0
            )
        ),
        "minority_precision": float(
            precision_score(
                y_true,
                pred,
                pos_label=0,
                zero_division=0
            )
        ),
        "positive_predictions": int(pred.sum())
    }


def choose_threshold(y_true, prob1):
    candidates = [
        threshold_metrics(y_true, prob1, threshold)
        for threshold in THRESHOLDS
    ]

    best_balanced = max(
        candidates,
        key=lambda r: (
            r["balanced_accuracy"],
            r["minority_f1"],
            r["macro_f1"],
            r["minority_recall"],
            -r["threshold"]
        )
    )

    best_macro = max(
        candidates,
        key=lambda r: (
            r["macro_f1"],
            r["balanced_accuracy"],
            r["minority_f1"],
            r["minority_recall"],
            -r["threshold"]
        )
    )

    best_minority_f1 = max(
        candidates,
        key=lambda r: (
            r["minority_f1"],
            r["balanced_accuracy"],
            r["minority_recall"],
            r["macro_f1"],
            -r["threshold"]
        )
    )

    return best_balanced, best_macro, best_minority_f1


def build_candidates():
    candidates = []

    rf_grid = list(product(
        [100, 200],
        [3, 5, 6, 7, None],
        [2, 4, 5],
        [1, 2]
    ))

    for n_estimators, max_depth, min_samples_split, min_samples_leaf in rf_grid:
        candidates.append({
            "family": "Random_Forest",
            "params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf
            }
        })

    xgb_grid = list(product(
        [100, 200],
        [2, 4],
        [0.03, 0.05],
        [1, 3]
    ))

    for n_estimators, max_depth, learning_rate, min_child_weight in xgb_grid:
        candidates.append({
            "family": "XGBoost",
            "params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "min_child_weight": min_child_weight
            }
        })

    lgb_grid = list(product(
        [100, 200],
        [3, 4, 5],
        [0.03, 0.05],
        [7, 15, 31]
    ))

    for n_estimators, max_depth, learning_rate, num_leaves in lgb_grid:
        candidates.append({
            "family": "LightGBM",
            "params": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "num_leaves": num_leaves
            }
        })

    return candidates


def evaluate_candidate(candidate, X, y, groups, verbose=True):
    family = candidate["family"]
    params = candidate["params"]

    oof_prob = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)

    gkf = GroupKFold(n_splits=N_SPLITS)

    fold_records = []

    for fold_id, (train_idx, val_idx) in enumerate(
        gkf.split(X, y, groups=groups),
        start=1
    ):
        train_districts = set(groups[train_idx])
        val_districts = set(groups[val_idx])

        overlap = train_districts.intersection(val_districts)
        if overlap:
            raise RuntimeError(
                "District overlap detected in fold {}: {}".format(
                    fold_id,
                    sorted(overlap)
                )
            )

        model = make_model(family, params)
        fit_model(
            model,
            family,
            X[train_idx],
            y[train_idx]
        )

        oof_prob[val_idx] = model.predict_proba(
            X[val_idx]
        )[:, 1]

        fold_ids[val_idx] = fold_id

        fold_records.append({
            "fold": fold_id,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "train_class0": int((y[train_idx] == 0).sum()),
            "train_class1": int((y[train_idx] == 1).sum()),
            "validation_class0": int((y[val_idx] == 0).sum()),
            "validation_class1": int((y[val_idx] == 1).sum()),
            "train_districts": len(train_districts),
            "validation_districts": len(val_districts)
        })

    if np.isnan(oof_prob).any():
        raise RuntimeError("Missing OOF probability detected.")

    best_balanced, best_macro, best_minority_f1 = choose_threshold(
        y,
        oof_prob
    )

    minority_true = (y == 0).astype(int)
    minority_score = 1.0 - oof_prob

    probability_metrics = {
        "minority_roc_auc": float(
            roc_auc_score(
                minority_true,
                minority_score
            )
        ),
        "minority_pr_auc": float(
            average_precision_score(
                minority_true,
                minority_score
            )
        )
    }

    return {
        "family": family,
        "params": params,
        "oof_prob": oof_prob,
        "fold_ids": fold_ids,
        "fold_records": fold_records,
        "best_balanced": best_balanced,
        "best_macro": best_macro,
        "best_minority_f1": best_minority_f1,
        "probability_metrics": probability_metrics
    }


def result_row(candidate_id, evaluation):
    balanced = evaluation["best_balanced"]
    macro = evaluation["best_macro"]
    minority = evaluation["best_minority_f1"]
    probability = evaluation["probability_metrics"]

    return {
        "candidate_id": int(candidate_id),
        "model": evaluation["family"],
        "params": json.dumps(
            evaluation["params"],
            sort_keys=True
        ),
        "minority_roc_auc": probability["minority_roc_auc"],
        "minority_pr_auc": probability["minority_pr_auc"],
        "best_bal_threshold": balanced["threshold"],
        "best_bal_accuracy": balanced["accuracy"],
        "best_balanced_accuracy": balanced["balanced_accuracy"],
        "best_bal_macro_f1": balanced["macro_f1"],
        "best_bal_minority_f1": balanced["minority_f1"],
        "best_bal_minority_recall": balanced["minority_recall"],
        "best_bal_minority_precision": balanced["minority_precision"],
        "best_bal_positive_predictions": balanced["positive_predictions"],
        "best_macro_threshold": macro["threshold"],
        "best_macro_f1": macro["macro_f1"],
        "best_macro_balanced_accuracy": macro["balanced_accuracy"],
        "best_macro_minority_f1": macro["minority_f1"],
        "best_macro_minority_recall": macro["minority_recall"],
        "best_minority_f1_threshold": minority["threshold"],
        "best_minority_f1": minority["minority_f1"],
        "best_minority_f1_balanced_accuracy": minority["balanced_accuracy"],
        "best_minority_f1_recall": minority["minority_recall"],
        "best_minority_f1_precision": minority["minority_precision"]
    }


def main():
    print("=" * 80)
    print("STEP 28 - FINALIZED MODEL TRAINING WITH DEVELOPMENT-ONLY OPTIMIZATION")
    print("=" * 80)

    if not DEV_PATH.exists():
        raise FileNotFoundError(
            "Development dataset not found: {}".format(DEV_PATH)
        )

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            "Frozen split assignment file not found: {}".format(SPLIT_PATH)
        )

    df = pd.read_csv(DEV_PATH)
    assignments = pd.read_csv(SPLIT_PATH)

    required = PREDICTORS + [TARGET, GROUP]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            "Missing required development columns: {}".format(missing)
        )

    if df[PREDICTORS].isna().any().any():
        raise RuntimeError(
            "Predictor missing values detected in development data."
        )

    if df[PREDICTORS].nunique(dropna=False).eq(1).any():
        constants = df[PREDICTORS].columns[
            df[PREDICTORS].nunique(dropna=False).eq(1)
        ].tolist()
        raise RuntimeError(
            "Constant predictors detected: {}".format(constants)
        )

    X = df[PREDICTORS].values
    y = df[TARGET].astype(int).values
    groups = df[GROUP].astype(str).values

    print("Development rows:", len(df))
    print("Class 0:", int((y == 0).sum()))
    print("Class 1:", int((y == 1).sum()))
    print("Districts:", len(np.unique(groups)))
    print("Frozen split rows available:", len(assignments))

    candidates = build_candidates()

    print("Total candidate configurations:", len(candidates))
    print("Selection: threshold-optimized balanced accuracy")
    print("Tie-breaks: minority F1, minority PR-AUC, macro F1")

    records = []
    family_best = {}

    for candidate_id, candidate in enumerate(candidates, start=1):
        print(
            "[{}/{}] {} {}".format(
                candidate_id,
                len(candidates),
                candidate["family"],
                candidate["params"]
            )
        )

        evaluation = evaluate_candidate(
            candidate,
            X,
            y,
            groups,
            verbose=False
        )

        row = result_row(candidate_id, evaluation)
        records.append(row)

        current = family_best.get(candidate["family"])

        if (
            current is None
            or (
                row["best_balanced_accuracy"],
                row["best_bal_minority_f1"],
                row["minority_pr_auc"],
                row["best_bal_macro_f1"]
            )
            >
            (
                current["row"]["best_balanced_accuracy"],
                current["row"]["best_bal_minority_f1"],
                current["row"]["minority_pr_auc"],
                current["row"]["best_bal_macro_f1"]
            )
        ):
            family_best[candidate["family"]] = {
                "candidate": candidate,
                "evaluation": evaluation,
                "row": row
            }

    result = pd.DataFrame(records)

    result = result.sort_values(
        [
            "best_balanced_accuracy",
            "best_bal_minority_f1",
            "minority_pr_auc",
            "best_bal_macro_f1"
        ],
        ascending=False
    ).reset_index(drop=True)

    search_output = DIAG_DIR / (
        "development_joint_hyperparameter_threshold_search.csv"
    )
    result.to_csv(search_output, index=False)

    best_global_row = result.iloc[0]

    selected_family = str(best_global_row["model"])
    selected_entry = family_best[selected_family]

    selected_params = selected_entry["candidate"]["params"]
    selected_threshold = float(best_global_row["best_bal_threshold"])

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION BY FAMILY")
    print("=" * 80)

    family_rows = []

    for family in [
        "Random_Forest",
        "XGBoost",
        "LightGBM"
    ]:
        entry = family_best[family]
        family_rows.append(entry["row"])

        print("\n{}".format(family))
        print("Parameters:", entry["candidate"]["params"])
        print(
            "Threshold:",
            entry["row"]["best_bal_threshold"]
        )
        print(
            "Balanced Accuracy:",
            "{:.6f}".format(
                entry["row"]["best_balanced_accuracy"]
            )
        )
        print(
            "Minority F1:",
            "{:.6f}".format(
                entry["row"]["best_bal_minority_f1"]
            )
        )
        print(
            "Minority Recall:",
            "{:.6f}".format(
                entry["row"]["best_bal_minority_recall"]
            )
        )
        print(
            "Minority PR-AUC:",
            "{:.6f}".format(
                entry["row"]["minority_pr_auc"]
            )
        )

    print("\n" + "=" * 80)
    print("SELECTED FINAL MODEL")
    print("=" * 80)
    print("Model:", selected_family)
    print("Parameters:", selected_params)
    print("Decision threshold:", selected_threshold)
    print(
        "Development balanced accuracy:",
        "{:.6f}".format(
            float(best_global_row["best_balanced_accuracy"])
        )
    )
    print(
        "Development minority F1:",
        "{:.6f}".format(
            float(best_global_row["best_bal_minority_f1"])
        )
    )

    comparison_rows = []

    for family in [
        "Random_Forest",
        "XGBoost",
        "LightGBM"
    ]:
        entry = family_best[family]
        row = entry["row"]

        final_model = make_model(
            family,
            entry["candidate"]["params"]
        )

        fit_model(
            final_model,
            family,
            X,
            y
        )

        model_path = MODEL_DIR / (
            family.lower() + "_model.joblib"
        )

        joblib.dump(final_model, model_path)

        comparison_rows.append({
            "model": family,
            "selected_for_family": True,
            "parameters": json.dumps(
                entry["candidate"]["params"],
                sort_keys=True
            ),
            "decision_threshold": row["best_bal_threshold"],
            "development_balanced_accuracy": row[
                "best_balanced_accuracy"
            ],
            "development_macro_f1": row[
                "best_bal_macro_f1"
            ],
            "development_minority_f1": row[
                "best_bal_minority_f1"
            ],
            "development_minority_recall": row[
                "best_bal_minority_recall"
            ],
            "development_minority_precision": row[
                "best_bal_minority_precision"
            ],
            "development_minority_pr_auc": row[
                "minority_pr_auc"
            ],
            "development_minority_roc_auc": row[
                "minority_roc_auc"
            ]
        })

    selected_model = joblib.load(
        MODEL_DIR /
        (selected_family.lower() + "_model.joblib")
    )

    joblib.dump(
        selected_model,
        MODEL_DIR / "selected_model.joblib"
    )

    selected_threshold_payload = {
        "model": selected_family,
        "threshold": selected_threshold,
        "selection_metric": "development_groupkfold_threshold_optimized_balanced_accuracy",
        "tie_breakers": [
            "minority_f1",
            "minority_pr_auc",
            "macro_f1"
        ],
        "development_rows": int(len(df)),
        "development_districts": int(len(np.unique(groups))),
        "outer_test_accessed": False
    }

    with open(
        MODEL_DIR / "selected_threshold.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            selected_threshold_payload,
            f,
            indent=2
        )

    feature_manifest = {
        "predictors": PREDICTORS,
        "target": TARGET,
        "group_column": GROUP,
        "n_features": len(PREDICTORS),
        "scaling": False,
        "development_rows": int(len(df)),
        "development_districts": int(len(np.unique(groups))),
        "class_counts": {
            "0": int((y == 0).sum()),
            "1": int((y == 1).sum())
        },
        "outer_test_accessed_during_training": False
    }

    with open(
        MODEL_DIR / "feature_manifest.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            feature_manifest,
            f,
            indent=2
        )

    model_metadata = {
        "selected_model": selected_family,
        "selected_parameters": selected_params,
        "selected_threshold": selected_threshold,
        "selection_metric": "Development GroupKFold threshold-optimized balanced accuracy",
        "tie_breakers": [
            "minority F1",
            "minority PR-AUC",
            "macro F1"
        ],
        "grouping": "district",
        "cv_method": "GroupKFold",
        "cv_folds": N_SPLITS,
        "seed": SEED,
        "development_rows": int(len(df)),
        "development_districts": int(len(np.unique(groups))),
        "predictors": PREDICTORS,
        "target": TARGET,
        "scaling": False,
        "imbalance_strategy": {
            "Random_Forest": "class_weight=balanced",
            "XGBoost": "balanced sample weights",
            "LightGBM": "balanced sample weights"
        },
        "outer_test_accessed": False,
        "outer_test_used_for_selection": False,
        "candidate_configurations": len(candidates),
        "python_version": sys.version,
        "platform": platform.platform(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "lightgbm_version": lightgbm.__version__,
        "generated_at": datetime.now().isoformat()
    }

    with open(
        MODEL_DIR / "model_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            model_metadata,
            f,
            indent=2,
            default=str
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values(
        [
            "development_balanced_accuracy",
            "development_minority_f1",
            "development_minority_pr_auc"
        ],
        ascending=False
    )

    comparison_df.to_csv(
        RESULT_DIR / "model_comparison.csv",
        index=False
    )

    cv_metrics = comparison_df[
        [
            "model",
            "decision_threshold",
            "development_balanced_accuracy",
            "development_macro_f1",
            "development_minority_f1",
            "development_minority_recall",
            "development_minority_precision",
            "development_minority_pr_auc",
            "development_minority_roc_auc"
        ]
    ].copy()

    cv_metrics.to_csv(
        RESULT_DIR / "cv_metrics.csv",
        index=False
    )

    training_summary = RESULT_DIR / "training_summary.txt"

    with open(
        training_summary,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "FINAL MODEL TRAINING SUMMARY\n"
        )
        f.write(
            "=============================\n\n"
        )
        f.write(
            "Development rows: {}\n".format(len(df))
        )
        f.write(
            "Development districts: {}\n".format(
                len(np.unique(groups))
            )
        )
        f.write(
            "Predictors: {}\n".format(len(PREDICTORS))
        )
        f.write(
            "Cross-validation: {}-fold GroupKFold by district\n".format(
                N_SPLITS
            )
        )
        f.write(
            "Candidate configurations: {}\n".format(
                len(candidates)
            )
        )
        f.write(
            "Selection metric: threshold-optimized balanced accuracy\n"
        )
        f.write(
            "Selected model: {}\n".format(
                selected_family
            )
        )
        f.write(
            "Selected parameters: {}\n".format(
                selected_params
            )
        )
        f.write(
            "Selected decision threshold: {:.2f}\n".format(
                selected_threshold
            )
        )
        f.write(
            "Outer test accessed: NO\n"
        )
        f.write(
            "Scaling: NONE\n"
        )
        f.write(
            "=============================\n\n"
        )

        for row in comparison_rows:
            f.write(
                "{}\n".format(row["model"])
            )
            f.write(
                "Parameters: {}\n".format(
                    row["parameters"]
                )
            )
            f.write(
                "Threshold: {:.2f}\n".format(
                    row["decision_threshold"]
                )
            )
            f.write(
                "Development balanced accuracy: {:.6f}\n".format(
                    row["development_balanced_accuracy"]
                )
            )
            f.write(
                "Development macro F1: {:.6f}\n".format(
                    row["development_macro_f1"]
                )
            )
            f.write(
                "Development minority F1: {:.6f}\n".format(
                    row["development_minority_f1"]
                )
            )
            f.write(
                "Development minority recall: {:.6f}\n".format(
                    row["development_minority_recall"]
                )
            )
            f.write(
                "Development minority precision: {:.6f}\n".format(
                    row["development_minority_precision"]
                )
            )
            f.write(
                "Development minority PR-AUC: {:.6f}\n".format(
                    row["development_minority_pr_auc"]
                )
            )
            f.write(
                "Development minority ROC-AUC: {:.6f}\n\n".format(
                    row["development_minority_roc_auc"]
                )
            )

    run_metadata = {
        "python": platform.python_version(),
        "python_full": sys.version,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "dataset": str(DEV_PATH),
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "labeled_rows": int(len(df)),
        "class_0": int((y == 0).sum()),
        "class_1": int((y == 1).sum()),
        "development_districts": int(len(np.unique(groups))),
        "outer_test_accessed": False,
        "predictor_count": len(PREDICTORS),
        "selected_model": selected_family,
        "selected_threshold": selected_threshold,
        "selected_parameters": selected_params,
        "seed": SEED,
        "groupkfold_folds": N_SPLITS,
        "scaling": False,
        "hyperparameter_candidates": len(candidates),
        "generated_at": datetime.now().isoformat()
    }

    with open(
        RESULT_DIR / "run_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            run_metadata,
            f,
            indent=2,
            default=str
        )

    print("\n" + "=" * 80)
    print("STEP 28 STATUS")
    print("=" * 80)
    print("[OK] Development-only hyperparameter optimization")
    print("[OK] 4-fold GroupKFold by district")
    print("[OK] Threshold optimization integrated")
    print("[OK] RF, XGBoost and LightGBM retrained on development set")
    print("[OK] Selected model saved")
    print("[OK] Selected threshold saved")
    print("[OK] Updated model metadata")
    print("[OK] Updated comparison and CV results")
    print("[OK] Outer test NOT accessed")
    print(
        "[OK] Selected:",
        selected_family,
        "threshold={:.2f}".format(selected_threshold)
    )
    print("[OK] Search output:", search_output)
    print("=" * 80)


if __name__ == "__main__":
    main()
