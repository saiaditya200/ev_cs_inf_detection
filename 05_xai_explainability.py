#!/usr/bin/env python3
"""
=============================================================================
Step 5: Explainable AI — TreeSHAP & Constrained Counterfactual Explanations
=============================================================================
Project: Explainable AI Framework for Smart EV Charging Infrastructure Planning
Domain: Telangana, India (33 Administrative Districts)

This step is post-hoc XAI over the frozen selected classifier.

Outputs:
- results/xai/shap_feature_importance.csv
- results/xai/station_shap_values.csv
- results/counterfactuals/counterfactual_analysis.csv
- results/figures/shap_summary_bar.png
- results/figures/shap_beeswarm.png
- results/figures/shap_local_high_confidence.png
- results/figures/shap_local_low_confidence.png
- results/figures/counterfactual_comparison.png
- results/xai/xai_run_metadata.json
- results/counterfactuals/counterfactual_summary_report.txt
=============================================================================
"""

import json
import os

# Force a headless Matplotlib backend before importing pyplot.
# This avoids Tkinter/Tcl GUI objects during command-line execution on Windows.
os.environ["MPLBACKEND"] = "Agg"

import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import warnings

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
XAI_DIR = RESULTS_DIR / "xai"
COUNTERFACTUALS_DIR = RESULTS_DIR / "counterfactuals"
FIGURES_DIR = RESULTS_DIR / "figures"

for directory in [
    XAI_DIR,
    COUNTERFACTUALS_DIR,
    FIGURES_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

ACTIONABLE_BASE_LEVERS = [
    "distance_to_highway_km",
    "distance_to_city_center_km",
    "nearby_poi_count",
    "poi_diversity_index"
]

DERIVED_FEATURES = [
    "highway_proximity_score",
    "city_center_proximity_score"
]

TARGET = "fast_charging_flag"
TARGET_LABEL = {
    0: "AC Slow",
    1: "DC Fast"
}


def load_artifacts():
    print("=" * 80)
    print("[INFO] STEP 5: STARTING EXPLAINABLE AI & COUNTERFACTUAL PIPELINE")
    print("=" * 80)
    print("[INFO] Matplotlib backend:", matplotlib.get_backend())

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

    matrix_path = DATA_PROCESSED_DIR / "fast_charging_modeling_matrix.csv"

    required_paths = [
        (model_path, "Deployment Model"),
        (manifest_path, "Feature Manifest"),
        (matrix_path, "Modeling Matrix")
    ]

    for path, name in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                "[ERROR] Required artifact missing: {} at {}".format(
                    name,
                    path
                )
            )

    model = joblib.load(model_path)

    with open(
        manifest_path,
        "r",
        encoding="utf-8"
    ) as file:
        manifest = json.load(file)

    metadata = {}
    if metadata_path.exists():
        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:
            metadata = json.load(file)

    if threshold_path.exists():
        with open(
            threshold_path,
            "r",
            encoding="utf-8"
        ) as file:
            threshold_info = json.load(file)
    else:
        threshold_info = {}

    df_matrix = pd.read_csv(matrix_path)

    predictors = manifest.get(
        "predictors",
        manifest.get("certified_21_predictors")
    )

    if not predictors:
        raise RuntimeError(
            "[ERROR] No certified predictor list found in feature_manifest.json"
        )

    missing_predictors = [
        column
        for column in predictors
        if column not in df_matrix.columns
    ]

    if missing_predictors:
        raise RuntimeError(
            "[ERROR] Modeling matrix is missing predictors: {}".format(
                missing_predictors
            )
        )

    if df_matrix[predictors].isna().any().any():
        missing_counts = (
            df_matrix[predictors]
            .isna()
            .sum()
        )
        missing_columns = missing_counts[
            missing_counts > 0
        ].to_dict()

        raise RuntimeError(
            "[ERROR] Missing values found in XAI predictors: {}".format(
                missing_columns
            )
        )

    if TARGET not in df_matrix.columns:
        raise RuntimeError(
            "[ERROR] Target column missing from modeling matrix: {}".format(
                TARGET
            )
        )

    target_values = set(
        df_matrix[TARGET].astype(int).unique()
    )

    if not target_values.issubset({0, 1}):
        raise RuntimeError(
            "[ERROR] XAI matrix contains target values outside {0,1}: {}".format(
                sorted(target_values)
            )
        )

    selected_model_name = metadata.get(
        "selected_champion_model",
        metadata.get(
            "selected_model",
            threshold_info.get(
                "model_family",
                threshold_info.get(
                    "model",
                    "Random_Forest"
                )
            )
        )
    )

    selected_threshold = float(
        threshold_info.get(
            "threshold",
            threshold_info.get(
                "decision_threshold",
                metadata.get(
                    "selected_threshold",
                    metadata.get(
                        "decision_threshold",
                        0.76
                    )
                )
            )
        )
    )

    print(
        "[INFO] Loaded model: {}".format(
            type(model).__name__
        )
    )
    print(
        "[INFO] Selected model name: {}".format(
            selected_model_name
        )
    )
    print(
        "[INFO] Frozen decision threshold: {:.2f}".format(
            selected_threshold
        )
    )
    print(
        "[INFO] Loaded modeling matrix: {} stations x {} columns".format(
            len(df_matrix),
            len(df_matrix.columns)
        )
    )
    print(
        "[INFO] Predictor set: {} certified unscaled features".format(
            len(predictors)
        )
    )
    print(
        "[INFO] Target distribution: Class 0 = {}, Class 1 = {}".format(
            int((df_matrix[TARGET] == 0).sum()),
            int((df_matrix[TARGET] == 1).sum())
        )
    )

    return (
        model,
        manifest,
        metadata,
        threshold_info,
        df_matrix,
        predictors,
        selected_model_name,
        selected_threshold
    )


def extract_shap_values(model, X):
    explainer = shap.TreeExplainer(model)

    try:
        shap_raw = explainer.shap_values(
            X,
            check_additivity=False
        )
    except TypeError:
        shap_raw = explainer.shap_values(X)

    if hasattr(shap_raw, "values"):
        values = np.asarray(shap_raw.values)
    elif isinstance(shap_raw, list):
        if len(shap_raw) < 2:
            values = np.asarray(shap_raw[0])
        else:
            values = np.asarray(shap_raw[1])
    else:
        values = np.asarray(shap_raw)

    if values.ndim == 3:
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        elif values.shape[1] == 2:
            values = values[:, 1, :]
        else:
            raise RuntimeError(
                "Unexpected 3D SHAP shape: {}".format(
                    values.shape
                )
            )

    if values.ndim != 2:
        raise RuntimeError(
            "Unexpected SHAP array shape: {}".format(
                values.shape
            )
        )

    return explainer, values


def compute_shap_explanations(model, df_matrix, predictors):
    print("\n" + "=" * 80)
    print("[INFO] COMPUTING TREESHAP EXPLANATIONS ON UNSCALED FEATURES")
    print("=" * 80)

    X = df_matrix[predictors].values.astype(float)

    explainer, shap_class1 = extract_shap_values(
        model,
        X
    )

    if shap_class1.shape != (
        len(df_matrix),
        len(predictors)
    ):
        raise RuntimeError(
            "SHAP output shape {} does not match expected ({}, {})".format(
                shap_class1.shape,
                len(df_matrix),
                len(predictors)
            )
        )

    mean_abs_shap = np.mean(
        np.abs(shap_class1),
        axis=0
    )

    total_shap = float(
        mean_abs_shap.sum()
    )

    if total_shap == 0.0:
        relative_importance = np.zeros(
            len(mean_abs_shap)
        )
    else:
        relative_importance = (
            mean_abs_shap /
            total_shap *
            100.0
        )

    df_importance = pd.DataFrame({
        "feature": predictors,
        "mean_abs_shap": mean_abs_shap,
        "relative_importance_pct": np.round(
            relative_importance,
            2
        )
    }).sort_values(
        by="mean_abs_shap",
        ascending=False
    ).reset_index(drop=True)

    importance_csv = XAI_DIR / "shap_feature_importance.csv"

    df_importance.to_csv(
        importance_csv,
        index=False
    )

    print(
        "[OK] SHAP feature importance saved: {}".format(
            importance_csv
        )
    )

    print("\nTop 7 Global SHAP Predictors:")
    for rank, (_, row) in enumerate(
        df_importance.head(7).iterrows(),
        start=1
    ):
        print(
            "  {}. {:<35} Mean |SHAP| = {:.6f} ({:.2f}%)".format(
                rank,
                row["feature"],
                row["mean_abs_shap"],
                row["relative_importance_pct"]
            )
        )

    df_station_shap = pd.DataFrame(
        shap_class1,
        columns=[
            "shap_{}".format(column)
            for column in predictors
        ]
    )

    if "station_id" in df_matrix.columns:
        df_station_shap.insert(
            0,
            "station_id",
            df_matrix["station_id"].values
        )

    if "district" in df_matrix.columns:
        df_station_shap.insert(
            1 if "station_id" in df_station_shap.columns else 0,
            "district",
            df_matrix["district"].values
        )

    df_station_shap[
        "ground_truth"
    ] = df_matrix[TARGET].values

    station_shap_csv = (
        XAI_DIR /
        "station_shap_values.csv"
    )

    df_station_shap.to_csv(
        station_shap_csv,
        index=False
    )

    print(
        "[OK] Station-level SHAP attributions saved: {}".format(
            station_shap_csv
        )
    )

    top_n = min(
        12,
        len(df_importance)
    )

    plot_df = (
        df_importance
        .head(top_n)
        .sort_values(
            by="mean_abs_shap",
            ascending=True
        )
    )

    plt.figure(
        figsize=(10, 6),
        dpi=300
    )

    bars = plt.barh(
        plot_df["feature"],
        plot_df["mean_abs_shap"],
        alpha=0.85
    )

    plt.xlabel(
        "Mean |SHAP Value|",
        fontsize=11,
        fontweight="bold"
    )

    plt.title(
        "Global Feature Importance from TreeSHAP",
        fontsize=13,
        fontweight="bold",
        pad=12
    )

    plt.grid(
        axis="x",
        linestyle="--",
        alpha=0.4
    )

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            " {:.4f}".format(width),
            va="center",
            fontsize=9
        )

    plt.tight_layout()

    bar_path = (
        FIGURES_DIR /
        "shap_summary_bar.png"
    )

    plt.savefig(
        bar_path,
        dpi=300
    )

    plt.close()

    print(
        "[OK] Saved: {}".format(
            bar_path
        )
    )

    plt.figure(
        figsize=(10, 7),
        dpi=300
    )

    shap.summary_plot(
        shap_class1,
        X,
        feature_names=predictors,
        max_display=12,
        show=False
    )

    plt.title(
        "TreeSHAP Value Distribution Across Labeled Stations",
        fontsize=12,
        fontweight="bold",
        pad=15
    )

    plt.tight_layout()

    beeswarm_path = (
        FIGURES_DIR /
        "shap_beeswarm.png"
    )

    plt.savefig(
        beeswarm_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "[OK] Saved: {}".format(
            beeswarm_path
        )
    )

    probs = model.predict_proba(
        X
    )[:, 1]

    high_idx = int(
        np.argmax(probs)
    )

    low_idx = int(
        np.argmin(probs)
    )

    local_specs = [
        (
            high_idx,
            "High Probability Example",
            "shap_local_high_confidence.png"
        ),
        (
            low_idx,
            "Low Probability Example",
            "shap_local_low_confidence.png"
        )
    ]

    for (
        idx,
        label_name,
        filename
    ) in local_specs:

        station_id = (
            df_matrix.iloc[idx]["station_id"]
            if "station_id" in df_matrix.columns
            else str(idx)
        )

        district = (
            df_matrix.iloc[idx]["district"]
            if "district" in df_matrix.columns
            else "Unknown"
        )

        ground_truth = int(
            df_matrix.iloc[idx][TARGET]
        )

        station_values = shap_class1[idx]

        top_local_idx = np.argsort(
            np.abs(station_values)
        )[-8:]

        plot_features = [
            predictors[i]
            for i in top_local_idx
        ]

        plot_values = [
            station_values[i]
            for i in top_local_idx
        ]

        plt.figure(
            figsize=(10, 5),
            dpi=300
        )

        plt.barh(
            plot_features,
            plot_values,
            alpha=0.85
        )

        plt.axvline(
            0,
            linestyle="--",
            linewidth=0.8
        )

        plt.xlabel(
            "SHAP Attribution to Model Output",
            fontsize=10,
            fontweight="bold"
        )

        plt.title(
            "{}: {} ({})\nGround Truth: Class {} ({}) | P(DC Fast) = {:.1%}".format(
                label_name,
                station_id,
                district,
                ground_truth,
                TARGET_LABEL.get(
                    ground_truth,
                    "Unknown"
                ),
                probs[idx]
            ),
            fontsize=11,
            fontweight="bold"
        )

        plt.grid(
            axis="x",
            linestyle=":",
            alpha=0.5
        )

        plt.tight_layout()

        local_path = (
            FIGURES_DIR /
            filename
        )

        plt.savefig(
            local_path,
            dpi=300
        )

        plt.close()

        print(
            "[OK] Saved local explanation: {}".format(
                local_path
            )
        )

    return (
        explainer,
        shap_class1,
        df_importance,
        X,
        probs
    )


def get_feature_index(predictors, feature):
    try:
        return predictors.index(feature)
    except ValueError as exc:
        raise RuntimeError(
            "Required actionable predictor '{}' not found.".format(
                feature
            )
        ) from exc


def compute_constrained_counterfactuals(
    model,
    df_matrix,
    predictors,
    probs,
    selected_threshold
):
    print("\n" + "=" * 80)
    print(
        "[INFO] COMPUTING CONSTRAINED ACTIONABLE COUNTERFACTUAL EXPLANATIONS"
    )
    print("=" * 80)

    print(
        "[INFO] Counterfactual target threshold: {:.2f}".format(
            selected_threshold
        )
    )

    print(
        "[INFO] Immutable attributes: Coordinates, road-type indicators, district context"
    )

    print(
        "[INFO] Actionable base levers: distance to highway, distance to city center, POI count, POI diversity"
    )

    print(
        "[INFO] Derived features are recomputed deterministically"
    )

    candidates = df_matrix[
        probs < selected_threshold
    ].copy()

    if len(candidates) == 0:
        candidates = (
            df_matrix
            .assign(_pred_prob=probs)
            .sort_values(
                by="_pred_prob"
            )
            .head(15)
            .drop(
                columns=["_pred_prob"]
            )
        )

    print(
        "[INFO] Evaluating {} candidate stations below the frozen threshold.".format(
            len(candidates)
        )
    )

    bounds = {}

    for feature in ACTIONABLE_BASE_LEVERS:
        values = pd.to_numeric(
            df_matrix[feature],
            errors="coerce"
        ).values

        std = float(
            np.nanstd(values)
        )

        if std <= 0.0:
            std = 1.0

        bounds[feature] = {
            "min": float(
                np.nanpercentile(
                    values,
                    5
                )
            ),
            "max": float(
                np.nanpercentile(
                    values,
                    95
                )
            ),
            "std": std
        }

    results = []

    search_alphas = [
        0.10,
        0.20,
        0.35,
        0.50,
        0.65,
        0.80,
        1.00
    ]

    idx_hwy = get_feature_index(
        predictors,
        "distance_to_highway_km"
    )

    idx_city = get_feature_index(
        predictors,
        "distance_to_city_center_km"
    )

    idx_poi = get_feature_index(
        predictors,
        "nearby_poi_count"
    )

    idx_div = get_feature_index(
        predictors,
        "poi_diversity_index"
    )

    idx_hwy_prox = get_feature_index(
        predictors,
        "highway_proximity_score"
    )

    idx_city_prox = get_feature_index(
        predictors,
        "city_center_proximity_score"
    )

    for row_index, row in candidates.iterrows():
        station_id = row["station_id"]
        district = row["district"]

        orig_p = float(
            probs[row_index]
        )

        orig_raw = row[
            predictors
        ].values.astype(float).copy()

        orig_hwy = float(
            orig_raw[idx_hwy]
        )

        orig_city = float(
            orig_raw[idx_city]
        )

        orig_poi = float(
            orig_raw[idx_poi]
        )

        orig_div = float(
            orig_raw[idx_div]
        )

        best_cf = None
        min_cost = float("inf")

        for alpha in search_alphas:
            perturbed = orig_raw.copy()

            new_hwy = max(
                bounds[
                    "distance_to_highway_km"
                ]["min"],
                orig_hwy * (
                    1.0 - 0.40 * alpha
                )
            )

            new_city = max(
                bounds[
                    "distance_to_city_center_km"
                ]["min"],
                orig_city * (
                    1.0 - 0.30 * alpha
                )
            )

            new_poi = min(
                bounds[
                    "nearby_poi_count"
                ]["max"],
                orig_poi + np.ceil(
                    2.0 * alpha
                )
            )

            new_div = min(
                bounds[
                    "poi_diversity_index"
                ]["max"],
                orig_div + 0.15 * alpha
            )

            perturbed[idx_hwy] = new_hwy
            perturbed[idx_city] = new_city
            perturbed[idx_poi] = new_poi
            perturbed[idx_div] = new_div

            perturbed[idx_hwy_prox] = (
                1.0 /
                (1.0 + new_hwy)
            )

            perturbed[idx_city_prox] = (
                1.0 /
                (1.0 + new_city)
            )

            candidate_probability = float(
                model.predict_proba(
                    perturbed.reshape(1, -1)
                )[0, 1]
            )

            cost = (
                abs(
                    new_hwy - orig_hwy
                ) /
                bounds[
                    "distance_to_highway_km"
                ]["std"]
                +
                abs(
                    new_city - orig_city
                ) /
                bounds[
                    "distance_to_city_center_km"
                ]["std"]
                +
                abs(
                    new_poi - orig_poi
                ) /
                bounds[
                    "nearby_poi_count"
                ]["std"]
                +
                abs(
                    new_div - orig_div
                ) /
                bounds[
                    "poi_diversity_index"
                ]["std"]
            )

            if (
                candidate_probability >= selected_threshold
                and cost < min_cost
            ):
                min_cost = cost
                best_cf = {
                    "probability": candidate_probability,
                    "cost": cost,
                    "hwy_new": new_hwy,
                    "city_new": new_city,
                    "poi_new": new_poi,
                    "div_new": new_div,
                    "hwy_prox_new": perturbed[idx_hwy_prox],
                    "city_prox_new": perturbed[idx_city_prox]
                }

        if best_cf is not None:
            changes = []

            if abs(
                best_cf["hwy_new"] - orig_hwy
            ) > 0.05:
                changes.append(
                    "Highway distance: {:.2f} km -> {:.2f} km".format(
                        orig_hwy,
                        best_cf["hwy_new"]
                    )
                )

            if abs(
                best_cf["city_new"] - orig_city
            ) > 0.05:
                changes.append(
                    "City-center distance: {:.2f} km -> {:.2f} km".format(
                        orig_city,
                        best_cf["city_new"]
                    )
                )

            if (
                best_cf["poi_new"] >
                orig_poi
            ):
                changes.append(
                    "Nearby POI count: {:.0f} -> {:.0f}".format(
                        orig_poi,
                        best_cf["poi_new"]
                    )
                )

            if (
                best_cf["div_new"] >
                orig_div + 0.001
            ):
                changes.append(
                    "POI diversity: {:.3f} -> {:.3f}".format(
                        orig_div,
                        best_cf["div_new"]
                    )
                )

            results.append({
                "station_id": station_id,
                "district": district,
                "original_prob_fast": round(
                    orig_p,
                    4
                ),
                "counterfactual_prob_fast": round(
                    best_cf["probability"],
                    4
                ),
                "probability_gain": round(
                    best_cf["probability"] - orig_p,
                    4
                ),
                "decision_threshold": round(
                    selected_threshold,
                    4
                ),
                "normalized_cost": round(
                    best_cf["cost"],
                    4
                ),
                "original_distance_to_highway_km": round(
                    orig_hwy,
                    4
                ),
                "counterfactual_distance_to_highway_km": round(
                    best_cf["hwy_new"],
                    4
                ),
                "original_distance_to_city_center_km": round(
                    orig_city,
                    4
                ),
                "counterfactual_distance_to_city_center_km": round(
                    best_cf["city_new"],
                    4
                ),
                "original_nearby_poi_count": round(
                    orig_poi,
                    4
                ),
                "counterfactual_nearby_poi_count": round(
                    best_cf["poi_new"],
                    4
                ),
                "original_poi_diversity_index": round(
                    orig_div,
                    4
                ),
                "counterfactual_poi_diversity_index": round(
                    best_cf["div_new"],
                    4
                ),
                "counterfactual_highway_proximity_score": round(
                    best_cf["hwy_prox_new"],
                    6
                ),
                "counterfactual_city_center_proximity_score": round(
                    best_cf["city_prox_new"],
                    6
                ),
                "actionable_modifications": "; ".join(
                    changes
                ),
                "search_status": "Feasible"
            })

        else:
            results.append({
                "station_id": station_id,
                "district": district,
                "original_prob_fast": round(
                    orig_p,
                    4
                ),
                "counterfactual_prob_fast": round(
                    orig_p,
                    4
                ),
                "probability_gain": 0.0,
                "decision_threshold": round(
                    selected_threshold,
                    4
                ),
                "normalized_cost": np.nan,
                "original_distance_to_highway_km": round(
                    orig_hwy,
                    4
                ),
                "counterfactual_distance_to_highway_km": np.nan,
                "original_distance_to_city_center_km": round(
                    orig_city,
                    4
                ),
                "counterfactual_distance_to_city_center_km": np.nan,
                "original_nearby_poi_count": round(
                    orig_poi,
                    4
                ),
                "counterfactual_nearby_poi_count": np.nan,
                "original_poi_diversity_index": round(
                    orig_div,
                    4
                ),
                "counterfactual_poi_diversity_index": np.nan,
                "counterfactual_highway_proximity_score": np.nan,
                "counterfactual_city_center_proximity_score": np.nan,
                "actionable_modifications": (
                    "No feasible transition found within the defined local search space"
                ),
                "search_status": "High Spatial Constraint"
            })

    df_cf = pd.DataFrame(
        results
    )

    cf_csv = (
        COUNTERFACTUALS_DIR /
        "counterfactual_analysis.csv"
    )

    df_cf.to_csv(
        cf_csv,
        index=False
    )

    print(
        "[OK] Constrained counterfactual explanations saved: {}".format(
            cf_csv
        )
    )

    if len(df_cf) > 0:
        feasible_count = int(
            (
                df_cf["search_status"] ==
                "Feasible"
            ).sum()
        )
        feasible_pct = (
            feasible_count /
            len(df_cf) *
            100.0
        )
    else:
        feasible_count = 0
        feasible_pct = 0.0

    report_path = (
        COUNTERFACTUALS_DIR /
        "counterfactual_summary_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write("=" * 80 + "\n")
        file.write(
            "ACTIONABLE COUNTERFACTUAL EXPLANATION SUMMARY REPORT\n"
        )
        file.write("=" * 80 + "\n\n")
        file.write(
            "Selected model: {}\n".format(
                type(model).__name__
            )
        )
        file.write(
            "Frozen decision threshold: {:.2f}\n".format(
                selected_threshold
            )
        )
        file.write(
            "Total evaluated sites: {}\n".format(
                len(df_cf)
            )
        )
        file.write(
            "Feasible counterfactuals: {} ({:.1f}%)\n".format(
                feasible_count,
                feasible_pct
            )
        )
        file.write(
            "Actionable levers: Highway distance, City-center distance, POI count, POI diversity\n"
        )
        file.write(
            "Immutable attributes: Coordinates, Road-type indicators, District context\n"
        )
        file.write(
            "Derived features: Highway and city-center proximity recomputed deterministically\n\n"
        )

        feasible_rows = df_cf[
            df_cf["search_status"] ==
            "Feasible"
        ].head(5)

        if len(feasible_rows) > 0:
            file.write(
                "Sample Feasible Transitions:\n"
            )

            for _, result_row in feasible_rows.iterrows():
                file.write(
                    "  Station {} ({}): P(Fast) {:.2f} -> {:.2f}\n".format(
                        result_row["station_id"],
                        result_row["district"],
                        result_row["original_prob_fast"],
                        result_row["counterfactual_prob_fast"]
                    )
                )

                file.write(
                    "    Actions: {}\n".format(
                        result_row[
                            "actionable_modifications"
                        ]
                    )
                )

    print(
        "[OK] Counterfactual summary report saved: {}".format(
            report_path
        )
    )

    plot_df = df_cf[
        df_cf["search_status"] ==
        "Feasible"
    ].head(10)

    if len(plot_df) > 0:
        y_pos = np.arange(
            len(plot_df)
        )

        height = 0.35

        plt.figure(
            figsize=(10, 6),
            dpi=300
        )

        plt.barh(
            y_pos - height / 2,
            plot_df[
                "original_prob_fast"
            ],
            height,
            label="Original P(DC Fast)",
            alpha=0.85
        )

        plt.barh(
            y_pos + height / 2,
            plot_df[
                "counterfactual_prob_fast"
            ],
            height,
            label="Counterfactual P(DC Fast)",
            alpha=0.85
        )

        plt.axvline(
            selected_threshold,
            linestyle="--",
            linewidth=1,
            label="Frozen Decision Threshold ({:.2f})".format(
                selected_threshold
            )
        )

        plt.yticks(
            y_pos,
            [
                "{} ({})".format(
                    row.station_id,
                    str(row.district)[:15]
                )
                for _, row in plot_df.iterrows()
            ],
            fontsize=9
        )

        plt.xlabel(
            "Model Probability of DC Fast Class",
            fontsize=10,
            fontweight="bold"
        )

        plt.title(
            "Constrained Counterfactual Probability Shifts",
            fontsize=11,
            fontweight="bold",
            pad=12
        )

        plt.legend(
            loc="lower right"
        )

        plt.grid(
            axis="x",
            linestyle=":",
            alpha=0.5
        )

        plt.tight_layout()

        cf_fig = (
            FIGURES_DIR /
            "counterfactual_comparison.png"
        )

        plt.savefig(
            cf_fig,
            dpi=300
        )

        plt.close()

        print(
            "[OK] Saved: {}".format(
                cf_fig
            )
        )

    return df_cf


def save_xai_metadata(
    metadata,
    selected_model_name,
    selected_threshold,
    df_matrix,
    predictors,
    df_importance,
    df_cf
):
    metadata_output = {
        "step": "05_xai_explainability",
        "project_title": (
            "Explainable AI Framework for Smart EV Charging Infrastructure Planning"
        ),
        "model": selected_model_name,
        "model_class": (
            "unknown"
        ),
        "frozen_decision_threshold": float(
            selected_threshold
        ),
        "xai_method": "TreeSHAP",
        "counterfactual_method": "Constrained normalized L1 search",
        "counterfactual_target_probability": float(
            selected_threshold
        ),
        "predictor_count": int(
            len(predictors)
        ),
        "xai_rows": int(
            len(df_matrix)
        ),
        "xai_target_class_0": int(
            (df_matrix[TARGET] == 0).sum()
        ),
        "xai_target_class_1": int(
            (df_matrix[TARGET] == 1).sum()
        ),
        "top_feature": (
            str(df_importance.iloc[0]["feature"])
            if len(df_importance) > 0
            else None
        ),
        "counterfactual_rows": int(
            len(df_cf)
        ),
        "counterfactual_feasible_rows": int(
            (
                df_cf["search_status"] ==
                "Feasible"
            ).sum()
            if len(df_cf) > 0
            else 0
        ),
        "outer_test_used_for_model_selection": False,
        "generated_at": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "shap_version": shap.__version__
    }

    metadata_path = (
        XAI_DIR /
        "xai_run_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata_output,
            file,
            indent=2
        )

    print(
        "[OK] XAI metadata saved: {}".format(
            metadata_path
        )
    )


def main():
    (
        model,
        manifest,
        metadata,
        threshold_info,
        df_matrix,
        predictors,
        selected_model_name,
        selected_threshold
    ) = load_artifacts()

    (
        explainer,
        shap_values,
        df_importance,
        X,
        probs
    ) = compute_shap_explanations(
        model,
        df_matrix.copy(),
        predictors
    )

    df_cf = compute_constrained_counterfactuals(
        model,
        df_matrix,
        predictors,
        probs,
        selected_threshold
    )

    save_xai_metadata(
        metadata,
        selected_model_name,
        selected_threshold,
        df_matrix,
        predictors,
        df_importance,
        df_cf
    )

    print("\n" + "=" * 80)
    print(
        "[OK] STEP 5 XAI & COUNTERFACTUAL EXPLANATIONS COMPLETED SUCCESSFULLY"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
