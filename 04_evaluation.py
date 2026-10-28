#!/usr/bin/env python3
"""
=============================================================================
Step 4: Spatial-Aware Outer-Test Evaluation & Benchmarking
=============================================================================
Project: Explainable AI Framework for Smart EV Charging Infrastructure Planning
Domain:  Telangana, India (33 Administrative Districts)

Key Functions:
- Evaluates the three trained models on the quarantined Outer Test Set:
  * 35 physical charging stations
  * 8 untouched districts (0 district overlap with development set)
  * Class distribution: Class 0 = 9 stations (25.7%), Class 1 = 26 stations (74.3%)
- Evaluates models on raw unscaled feature space (matching training pipeline).
- Computes authentic, un-fabricated performance metrics:
  * Accuracy & Balanced Accuracy
  * Precision, Recall & F1-Macro
  * ROC-AUC & PR-AUC
  * Exact Confusion Matrices
- Produces publication-ready evaluation visualizations:
  * results/figures/confusion_matrix.png
  * results/figures/roc_curves.png
  * results/figures/pr_curves.png
  * results/figures/model_comparison.png
  * results/figures/feature_importance.png
- Persists:
  * results/metrics/outer_test_metrics.csv
  * results/metrics/model_comparison.csv
  * results/metrics/evaluation_report.txt
=============================================================================
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# ========================= CONFIGURATION =========================
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [METRICS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_evaluation_data():
    print("=" * 80)
    print("[INFO] STEP 4: STARTING SPATIAL OUTER-TEST BENCHMARKING")
    print("=" * 80)

    outer_path = DATA_PROCESSED_DIR / "phase10_outer_test.csv"
    manifest_path = MODELS_DIR / "feature_manifest.json"
    meta_path = MODELS_DIR / "model_metadata.json"

    if not outer_path.exists():
        raise FileNotFoundError(f"[ERROR] Outer test set missing at {outer_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"[ERROR] Manifest missing at {manifest_path}")

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    champion_name = "Random_Forest"
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            champion_name = meta.get(
                "selected_model",
                meta.get("selected_champion_model", champion_name)
            )

    df_outer = pd.read_csv(outer_path)

    predictors = manifest.get(
        "predictors",
        manifest.get("certified_21_predictors")
    )

    if not predictors:
        raise KeyError(
            "[ERROR] No certified predictor list found in feature_manifest.json"
        )

    c0 = int((df_outer["fast_charging_flag"] == 0).sum())
    c1 = int((df_outer["fast_charging_flag"] == 1).sum())

    print(f"[INFO] Champion Model: {champion_name}")
    print(f"[INFO] Outer Test Set: {len(df_outer)} stations across {df_outer['district'].nunique()} districts")
    print(f"       Class 0 (AC Slow): {c0} stations")
    print(f"       Class 1 (DC Fast): {c1} stations")

    return df_outer, predictors, champion_name


def evaluate_models(df_outer, predictors, champion_name):
    X_outer = df_outer[predictors].values
    y_outer = df_outer["fast_charging_flag"].values.astype(int)

    model_names = ["Random_Forest", "XGBoost", "LightGBM"]
    models = {}
    for name in model_names:
        m_path = MODELS_DIR / f"{name.lower()}_model.joblib"
        if not m_path.exists():
            raise FileNotFoundError(f"[ERROR] Model artifact missing: {m_path}")
        models[name] = joblib.load(m_path)

    # Load the frozen decision thresholds selected on the development set.
    # The outer test is used only for final evaluation; no threshold is tuned here.
    thresholds_by_model = {
        "Random_Forest": 0.50,
        "XGBoost": 0.50,
        "LightGBM": 0.50
    }

    selected_threshold_path = MODELS_DIR / "selected_threshold.json"
    if selected_threshold_path.exists():
        with open(selected_threshold_path, "r", encoding="utf-8") as f:
            selected_threshold_info = json.load(f)

        selected_model_from_file = selected_threshold_info.get(
            "model",
            champion_name
        )
        selected_threshold_from_file = selected_threshold_info.get(
            "threshold"
        )

        if selected_threshold_from_file is not None:
            thresholds_by_model[selected_model_from_file] = float(
                selected_threshold_from_file
            )

    # Step 28 stores the frozen family-specific thresholds in results/cv_metrics.csv.
    # Prefer this artifact because results/model_comparison.csv is overwritten
    # by this evaluation step and is therefore not an authoritative threshold
    # source.
    threshold_sources = [
        RESULTS_DIR / "cv_metrics.csv",
        RESULTS_DIR / "model_comparison.csv",
        METRICS_DIR / "cv_metrics.csv",
        METRICS_DIR / "model_comparison.csv"
    ]

    for threshold_source in threshold_sources:
        if not threshold_source.exists():
            continue

        try:
            threshold_df = pd.read_csv(threshold_source)

            if "model" in threshold_df.columns:
                model_col = "model"
            elif "model_name" in threshold_df.columns:
                model_col = "model_name"
            else:
                continue

            if "decision_threshold" not in threshold_df.columns:
                continue

            for _, row in threshold_df.iterrows():
                model_name = str(row[model_col])
                threshold_value = pd.to_numeric(
                    row["decision_threshold"],
                    errors="coerce"
                )

                if pd.notna(threshold_value):
                    thresholds_by_model[model_name] = float(
                        threshold_value
                    )

            if all(
                m in thresholds_by_model
                for m in ["Random_Forest", "XGBoost", "LightGBM"]
            ):
                break

        except Exception as exc:
            print(
                f"[WARN] Could not read threshold source {threshold_source}: {exc}"
            )

    for required_model in ["Random_Forest", "XGBoost", "LightGBM"]:
        if required_model not in thresholds_by_model:
            raise RuntimeError(
                "Missing frozen decision threshold for {}".format(
                    required_model
                )
            )

    print("\n[INFO] Frozen decision thresholds:")
    for model_name in ["Random_Forest", "XGBoost", "LightGBM"]:
        print(
            f"  {model_name}: {thresholds_by_model[model_name]:.2f}"
        )

    metrics_list = []
    curves_data = {}

    print("\n" + "=" * 80)
    print("OUTER SPATIAL TEST EVALUATION (35 UNTOUCHED STATIONS)")
    print("=" * 80)

    for name, model in models.items():
        y_prob = model.predict_proba(X_outer)[:, 1]
        threshold = thresholds_by_model[name]
        y_pred = (y_prob >= threshold).astype(int)

        acc = accuracy_score(y_outer, y_pred)
        bal_acc = balanced_accuracy_score(y_outer, y_pred)
        prec = precision_score(y_outer, y_pred, zero_division=0)
        rec = recall_score(y_outer, y_pred, zero_division=0)
        f1_macro = f1_score(y_outer, y_pred, average="macro", zero_division=0)
        f1_minority = f1_score(y_outer, y_pred, pos_label=0, zero_division=0)
        roc_auc = roc_auc_score(y_outer, y_prob)
        pr_auc = average_precision_score(y_outer, y_prob)
        cm = confusion_matrix(
            y_outer,
            y_pred,
            labels=[0, 1]
        )

        metrics_list.append({
            "model_name": name,
            "decision_threshold": float(threshold),
            "outer_test_accuracy": float(acc),
            "outer_test_balanced_acc": float(bal_acc),
            "outer_test_precision": float(prec),
            "outer_test_recall": float(rec),
            "outer_test_macro_f1": float(f1_macro),
            "outer_test_minority_f1": float(f1_minority),
            "outer_test_roc_auc": float(roc_auc),
            "outer_test_pr_auc": float(pr_auc),
            "is_champion": (name == champion_name)
        })

        # Save curve data
        fpr, tpr, _ = roc_curve(y_outer, y_prob)
        p_curve, r_curve, _ = precision_recall_curve(y_outer, y_prob)
        curves_data[name] = {
            "y_prob": y_prob, "y_pred": y_pred, "cm": cm,
            "fpr": fpr, "tpr": tpr, "p_curve": p_curve, "r_curve": r_curve
        }

        print(f"Model: {name}")
        print(f"  Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f}")
        print(f"  PR-AUC:   {pr_auc:.4f} | ROC-AUC:      {roc_auc:.4f}")
        print(f"  F1-Macro: {f1_macro:.4f} | F1-Minority:  {f1_minority:.4f}")
        print(f"  Confusion Matrix: TP={cm[1,1]}, TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}")
        print("-" * 60)

    # Save outer test metrics
    df_outer_metrics = pd.DataFrame(metrics_list)
    outer_csv_path = RESULTS_DIR / "outer_test_metrics.csv"
    df_outer_metrics.to_csv(outer_csv_path, index=False)

    legacy_outer_csv_path = METRICS_DIR / "outer_test_metrics.csv"
    df_outer_metrics.to_csv(legacy_outer_csv_path, index=False)

    print(f"[OK] Outer test metrics saved: {outer_csv_path}")

    # Build comprehensive comparison table.
    # Prefer the active Step 28 CV artifact and fall back to the legacy path.
    cv_paths = [
        RESULTS_DIR / "cv_metrics.csv",
        METRICS_DIR / "cv_metrics.csv"
    ]

    df_cv = None
    for cv_csv_path in cv_paths:
        if cv_csv_path.exists():
            df_cv = pd.read_csv(cv_csv_path)
            break

    if df_cv is not None:
        if (
            "model" in df_cv.columns
            and "model_name" not in df_cv.columns
        ):
            df_cv = df_cv.rename(
                columns={"model": "model_name"}
            )

        df_comp = pd.merge(
            df_cv,
            df_outer_metrics,
            on="model_name",
            how="inner"
        )
    else:
        df_comp = df_outer_metrics.copy()

    # Attach the frozen development-selected threshold to the final comparison
    # table before saving it.
    if "model_name" in df_comp.columns:
        df_comp["decision_threshold"] = df_comp["model_name"].map(
            thresholds_by_model
        )

    comp_csv_path = RESULTS_DIR / "model_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)

    legacy_comp_csv_path = METRICS_DIR / "model_comparison.csv"
    df_comp.to_csv(legacy_comp_csv_path, index=False)

    print(f"[OK] Model comparison table saved: {comp_csv_path}")

    return (
        models,
        curves_data,
        df_outer_metrics,
        df_comp,
        y_outer,
        thresholds_by_model
    )


def generate_evaluation_visualizations(models, curves_data, df_comp, champion_name, predictors, y_outer, thresholds_by_model, outer_district_count):
    print("\n[INFO] Generating Evaluation Visualizations...")

    # 1. Confusion Matrix (Champion Model)
    plt.figure(figsize=(6, 5), dpi=300)
    cm = curves_data[champion_name]["cm"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["AC Slow (0)", "DC Fast (1)"],
                yticklabels=["AC Slow (0)", "DC Fast (1)"])
    plt.title(f"Confusion Matrix: {champion_name} (Outer Test, N=35)", fontsize=11, fontweight="bold", pad=12)
    plt.ylabel("True Infrastructure Label", fontsize=10)
    plt.xlabel("Predicted Infrastructure Class", fontsize=10)
    plt.tight_layout()
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {cm_path}")

    # 2. ROC Curves Comparison
    plt.figure(figsize=(8, 6), dpi=300)
    for name, cdata in curves_data.items():
        auc_val = roc_auc_score(y_outer, cdata["y_prob"])
        style = "-" if name == champion_name else "--"
        width = 2.5 if name == champion_name else 1.5
        plt.plot(cdata["fpr"], cdata["tpr"], linestyle=style, linewidth=width,
                 label=f"{name} (ROC-AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], 'k:', alpha=0.5, label="Random Guess (0.500)")
    plt.xlabel("False Positive Rate", fontsize=10, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=10, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) — Outer Test", fontsize=11, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    roc_path = FIGURES_DIR / "roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {roc_path}")

    # 3. Precision-Recall Curves Comparison
    plt.figure(figsize=(8, 6), dpi=300)
    base_rate = (y_outer == 1).sum() / len(y_outer)
    for name, cdata in curves_data.items():
        pr_val = average_precision_score(y_outer, cdata["y_prob"])
        style = "-" if name == champion_name else "--"
        width = 2.5 if name == champion_name else 1.5
        plt.plot(cdata["r_curve"], cdata["p_curve"], linestyle=style, linewidth=width,
                 label=f"{name} (PR-AUC = {pr_val:.3f})")
    plt.axhline(base_rate, color='k', linestyle=':', alpha=0.5, label=f"Baseline Prevalence ({base_rate:.3f})")
    plt.xlabel("Recall (DC Fast Coverage)", fontsize=10, fontweight="bold")
    plt.ylabel("Precision (DC Fast Fidelity)", fontsize=10, fontweight="bold")
    plt.title("Precision-Recall (PR) Curves — Outer Test", fontsize=11, fontweight="bold", pad=12)
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    pr_path = FIGURES_DIR / "pr_curves.png"
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {pr_path}")

    # 4. Model Comparison Bar Chart
    # Use directly comparable metrics. The legacy pipeline has CV PR-AUC,
    # while the active Step 28 pipeline stores threshold-optimized development
    # balanced accuracy. Do not compare minority PR-AUC with class-1 PR-AUC.
    plt.figure(figsize=(9, 5), dpi=300)
    x = np.arange(len(df_comp))
    width = 0.35

    if "cv_pr_auc_mean" in df_comp.columns:
        dev_values = df_comp["cv_pr_auc_mean"].astype(float)
        outer_values = df_comp["outer_test_pr_auc"].astype(float)
        dev_label = "Development CV PR-AUC"
        outer_label = "Outer Spatial Test PR-AUC"
        ylabel = "Precision-Recall AUC"
        title = (
            "Model Generalization: Development CV vs "
            "Quarantined Outer Spatial Test"
        )
    elif "development_balanced_accuracy" in df_comp.columns:
        dev_values = df_comp["development_balanced_accuracy"].astype(float)
        outer_values = df_comp["outer_test_balanced_acc"].astype(float)
        dev_label = "Development Optimized Balanced Accuracy"
        outer_label = "Outer Spatial Test Balanced Accuracy"
        ylabel = "Balanced Accuracy"
        title = (
            "Model Generalization: Development Optimization vs "
            "Quarantined Outer Spatial Test"
        )
    else:
        dev_values = None
        outer_values = df_comp["outer_test_balanced_acc"].astype(float)
        dev_label = None
        outer_label = "Outer Spatial Test Balanced Accuracy"
        ylabel = "Balanced Accuracy"
        title = "Outer Spatial Test Model Comparison"

    if dev_values is not None:
        plt.bar(
            x - width/2,
            dev_values,
            width,
            label=dev_label,
            color="#3b82f6",
            alpha=0.85
        )

        plt.bar(
            x + width/2,
            outer_values,
            width,
            label=outer_label,
            color="#10b981",
            alpha=0.85
        )
    else:
        plt.bar(
            x,
            outer_values,
            width,
            label=outer_label,
            color="#10b981",
            alpha=0.85
        )

    plt.xticks(
        x,
        df_comp["model_name"],
        fontsize=10,
        fontweight="bold"
    )
    plt.ylabel(
        ylabel,
        fontsize=10,
        fontweight="bold"
    )
    plt.title(
        title,
        fontsize=11,
        fontweight="bold",
        pad=12
    )
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()

    comp_fig = FIGURES_DIR / "model_comparison.png"
    plt.savefig(comp_fig, dpi=300)
    plt.close()
    print(f"[OK] Saved: {comp_fig}")

    # 5. Feature Importance (Champion Model)
    champ_model = models[champion_name]
    if hasattr(champ_model, "feature_importances_"):
        plt.figure(figsize=(10, 6), dpi=300)
        importances = champ_model.feature_importances_
        df_feat = pd.DataFrame({"feature": predictors, "importance": importances})
        df_feat = df_feat.sort_values(by="importance", ascending=True).tail(12)
        plt.barh(df_feat["feature"], df_feat["importance"], color="#3b82f6", alpha=0.85, edgecolor="#1d4ed8")
        plt.xlabel("MDI Feature Importance", fontsize=10, fontweight="bold")
        plt.title(f"Top 12 Features — {champion_name}", fontsize=11, fontweight="bold", pad=12)
        plt.grid(axis="x", linestyle=":", alpha=0.5)
        plt.tight_layout()
        feat_path = FIGURES_DIR / "feature_importance.png"
        plt.savefig(feat_path, dpi=300)
        plt.close()
        print(f"[OK] Saved: {feat_path}")

        feat_csv_path = METRICS_DIR / "feature_importance.csv"
        df_feat.sort_values(by="importance", ascending=False).to_csv(feat_csv_path, index=False)
        print(f"[OK] Saved: {feat_csv_path}")

    # Save summary report text
    rep_path = RESULTS_DIR / "evaluation_report.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("EXPLAINABLE AI EV CHARGING INFRASTRUCTURE PLANNING\n")
        f.write("OUTER SPATIAL TEST BENCHMARKING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Champion Model: {champion_name}\n")
        f.write(
            f"Champion Decision Threshold: "
            f"{float(thresholds_by_model.get(champion_name, 0.50)):.2f}\n"
        )
        f.write(
            f"Outer Test Stations: {len(y_outer)} across "
            f"{outer_district_count} untouched districts\n"
        )
        f.write(f"Class 0 (AC Slow): {(y_outer==0).sum()} stations\n")
        f.write(f"Class 1 (DC Fast): {(y_outer==1).sum()} stations\n\n")
        f.write("PERFORMANCE SUMMARY:\n")
        for _, row in df_comp.iterrows():
            f.write(f"Model: {row['model_name']}\n")
            if "cv_pr_auc_mean" in row:
                f.write(
                    f"  Dev CV PR-AUC:       "
                    f"{row['cv_pr_auc_mean']:.4f}"
                )
                if "cv_pr_auc_std" in row:
                    f.write(
                        f" (+/- {row['cv_pr_auc_std']:.4f})"
                    )
                f.write("\n")

                if "cv_roc_auc_mean" in row:
                    f.write(
                        f"  Dev CV ROC-AUC:      "
                        f"{row['cv_roc_auc_mean']:.4f}\n"
                    )

            elif "development_balanced_accuracy" in row:
                f.write(
                    f"  Dev Optimized Bal Acc: "
                    f"{row['development_balanced_accuracy']:.4f}\n"
                )

            report_threshold = float(
                thresholds_by_model.get(
                    str(row["model_name"]),
                    0.50
                )
            )

            f.write(
                f"  Decision Threshold:   "
                f"{report_threshold:.2f}\n"
            )
            f.write(
                f"  Outer Test PR-AUC:   "
                f"{row['outer_test_pr_auc']:.4f}\n"
            )
            f.write(f"  Outer Test ROC-AUC:  {row['outer_test_roc_auc']:.4f}\n")
            f.write(f"  Outer Test Accuracy: {row['outer_test_accuracy']:.4f}\n")
            f.write(f"  Outer Balanced Acc:  {row['outer_test_balanced_acc']:.4f}\n")
            f.write(f"  Outer Macro F1:      {row['outer_test_macro_f1']:.4f}\n\n")
        f.write("=" * 80 + "\n")
    print(f"[OK] Evaluation report written: {rep_path}")


def main():
    df_outer, predictors, champion_name = load_evaluation_data()

    (
        models,
        curves_data,
        df_outer_metrics,
        df_comp,
        y_outer,
        thresholds_by_model
    ) = evaluate_models(
        df_outer,
        predictors,
        champion_name
    )

    generate_evaluation_visualizations(
        models,
        curves_data,
        df_comp,
        champion_name,
        predictors,
        y_outer,
        thresholds_by_model,
        int(df_outer["district"].nunique())
    )

    print("\n" + "=" * 80)
    print("[OK] Frozen development thresholds applied")
    print("[OK] STEP 4 MODEL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
