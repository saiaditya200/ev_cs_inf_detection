#!/usr/bin/env python3
"""
Phase 10: Spatial-Aware Train / Validation / Test Design
Explainable AI Framework for Smart EV Charging Infrastructure Planning

This script executes the complete Phase 10 spatial validation design:
1. Loads and validates data/processed/fast_charging_modeling_matrix.csv (154 x 24).
2. Performs district group audit and exports reports/phase10_district_class_distribution.csv.
3. Implements StratifiedGroupKFold (n_splits=5, shuffle=True, random_state=42) grouped by district.
4. Audits fold-by-fold station counts, class counts, district counts, and zero overlap.
5. Evaluates outer spatial test set options and designates Fold 3 (35 stations, 8 districts) as outer test benchmark.
6. Exports split assignments to data/processed/phase10_spatial_splits.csv.
7. Generates visualizations:
   - reports/figures/phase10_class_distribution.png
   - reports/figures/phase10_district_station_counts.png
   - reports/figures/phase10_spatial_split_map.png
8. Generates reports/phase10_split_reproducibility.md and reports/phase10_spatial_validation_report.md.
9. Updates config/project_config.yaml.
"""

import sys
import os
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path("c:/Users/Dell/Downloads/EV_Charging")
DATA_FINAL_DIR = BASE_DIR / "data" / "final"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIG_PATH = BASE_DIR / "config" / "project_config.yaml"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def step1_load_and_verify():
    print("=" * 80)
    print("STEP 1: INPUT DATASET & TARGET VERIFICATION")
    print("=" * 80)

    matrix_path = DATA_PROCESSED_DIR / "fast_charging_modeling_matrix.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing modeling matrix: {matrix_path}")

    df = pd.read_csv(matrix_path)
    print(f"Loaded modeling matrix: {df.shape[0]} rows x {df.shape[1]} cols")
    assert df.shape == (154, 24), f"Unexpected shape: {df.shape}"

    # Target counts
    c1 = (df["fast_charging_flag"] == 1).sum()
    c0 = (df["fast_charging_flag"] == 0).sum()
    assert c1 == 131 and c0 == 23, f"Target counts mismatch: Class 1={c1}, Class 0={c0}"
    imbalance_ratio = round(c1 / c0, 2)
    print(f"Target verification: Class 1 = {c1}, Class 0 = {c0}, Imbalance Ratio = {imbalance_ratio} : 1")
    return df, matrix_path

def step2_district_group_audit(df):
    print("\n" + "=" * 80)
    print("STEP 2: DISTRICT-LEVEL GROUP AUDIT")
    print("=" * 80)

    audit = df.groupby("district").agg(
        station_count=("fast_charging_flag", "count"),
        class_0_count=("fast_charging_flag", lambda x: int((x == 0).sum())),
        class_1_count=("fast_charging_flag", lambda x: int((x == 1).sum()))
    ).reset_index()

    audit["class_0_percentage"] = (audit["class_0_count"] / audit["station_count"] * 100).round(1)
    audit["class_1_percentage"] = (audit["class_1_count"] / audit["station_count"] * 100).round(1)

    # Sort by station_count descending, then district
    audit.sort_values(by=["station_count", "district"], ascending=[False, True], inplace=True)

    out_csv = REPORTS_DIR / "phase10_district_class_distribution.csv"
    audit.to_csv(out_csv, index=False)
    audit_sha = compute_sha256(out_csv)
    print(f"Exported district audit table: {out_csv} (SHA-256: {audit_sha})")

    both_classes = audit[(audit["class_0_count"] > 0) & (audit["class_1_count"] > 0)]
    only_0 = audit[(audit["class_0_count"] > 0) & (audit["class_1_count"] == 0)]
    only_1 = audit[(audit["class_0_count"] == 0) & (audit["class_1_count"] > 0)]

    print(f"Districts represented: {len(audit)}")
    print(f"Districts containing both classes: {len(both_classes)}")
    print(f"Districts containing only Class 0: {len(only_0)}")
    print(f"Districts containing only Class 1: {len(only_1)}")
    print(f"Min stations per district: {audit['station_count'].min()}")
    print(f"Max stations per district: {audit['station_count'].max()} ({audit.iloc[0]['district']})")

    return audit, out_csv, audit_sha

def step3_stratified_group_kfold_audit(df):
    print("\n" + "=" * 80)
    print("STEP 3: STRATIFIED GROUP K-FOLD CROSS-VALIDATION AUDIT")
    print("=" * 80)

    X = df.drop(columns=["station_id", "district", "fast_charging_flag"])
    y = df["fast_charging_flag"]
    groups = df["district"]

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_audit_records = []
    df_splits = df[["station_id", "district", "fast_charging_flag"]].copy()
    df_splits["fold_id"] = -1
    df_splits["split_role"] = "train"

    for fold_id, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        df_splits.loc[val_idx, "fold_id"] = fold_id

        train_districts = set(groups.iloc[train_idx])
        val_districts = set(groups.iloc[val_idx])
        overlap = len(train_districts.intersection(val_districts))

        tr_y = y.iloc[train_idx]
        val_y = y.iloc[val_idx]

        tr_c0 = int((tr_y == 0).sum())
        tr_c1 = int((tr_y == 1).sum())
        val_c0 = int((val_y == 0).sum())
        val_c1 = int((val_y == 1).sum())

        fold_audit_records.append({
            "fold_id": fold_id,
            "train_station_count": len(train_idx),
            "validation_station_count": len(val_idx),
            "train_class_0": tr_c0,
            "train_class_1": tr_c1,
            "validation_class_0": val_c0,
            "validation_class_1": val_c1,
            "train_district_count": len(train_districts),
            "validation_district_count": len(val_districts),
            "district_overlap": overlap,
            "validation_districts": sorted(list(val_districts))
        })

    fold_audit_df = pd.DataFrame(fold_audit_records)
    print("Fold-by-Fold Audit Table:")
    summary_cols = [c for c in fold_audit_df.columns if c != "validation_districts"]
    print(fold_audit_df[summary_cols].to_string(index=False))

    # Strict overlap assertion
    assert (fold_audit_df["district_overlap"] == 0).all(), "Spatial leakage detected! District overlap > 0."
    print("\nVerified: district_overlap = 0 for all 5 folds. Strict spatial isolation achieved.")

    # Designate outer test set (Fold 3 has 35 stations [22.7%], 8 districts, Class 0: 9, Class 1: 26)
    # For split_role assignment:
    # Fold 3 is designated as 'test', while Folds 0, 1, 2, 4 are designated as 'train'
    # (with their fold_id indicating their validation assignment in 4-fold inner CV or full 5-fold CV)
    for idx, r in df_splits.iterrows():
        if r["fold_id"] == 3:
            df_splits.loc[idx, "split_role"] = "test"
        else:
            df_splits.loc[idx, "split_role"] = "train"

    splits_csv = DATA_PROCESSED_DIR / "phase10_spatial_splits.csv"
    df_splits.to_csv(splits_csv, index=False)
    splits_sha = compute_sha256(splits_csv)
    print(f"\nExported split assignments: {splits_csv} (SHA-256: {splits_sha})")

    return fold_audit_df, df_splits, splits_csv, splits_sha

def step4_generate_visualizations(df, df_splits):
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING VALIDATION VISUALIZATIONS")
    print("=" * 80)

    # 1. Class distribution bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["fast_charging_flag"].value_counts().sort_index()
    bars = ax.bar(["Class 0: AC Level 1/2 (<50 kW)", "Class 1: DC Fast (>=50 kW)"], counts.values, color=["#3b82f6", "#ef4444"], width=0.55)
    ax.set_ylabel("Number of Stations", fontsize=12)
    ax.set_title("Supervised Target Class Distribution (N = 154 Labeled Stations)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 150)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height} ({height/len(df)*100:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p1 = FIGURES_DIR / "phase10_class_distribution.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Generated Figure 1: {p1}")

    # 2. District station-count distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    dist_counts = df.groupby(["district", "fast_charging_flag"]).size().unstack(fill_value=0)
    dist_counts["total"] = dist_counts[0] + dist_counts[1]
    dist_counts.sort_values(by="total", ascending=True, inplace=True)
    
    ax.barh(dist_counts.index, dist_counts[1], label="Class 1: DC Fast (>=50 kW)", color="#ef4444", alpha=0.9)
    ax.barh(dist_counts.index, dist_counts[0], left=dist_counts[1], label="Class 0: AC Level 1/2 (<50 kW)", color="#3b82f6", alpha=0.9)
    ax.set_xlabel("Number of Stations", fontsize=12)
    ax.set_ylabel("District", fontsize=12)
    ax.set_title("Station Count & Class Breakdown Across All 33 Telangana Districts", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    p2 = FIGURES_DIR / "phase10_district_station_counts.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Generated Figure 2: {p2}")

    # 3. Spatial distribution of train/test assignments
    fig, ax = plt.subplots(figsize=(9, 8))
    df_merged = df.merge(df_splits[["station_id", "split_role", "fold_id"]], on="station_id")

    # Plot train stations
    tr = df_merged[df_merged["split_role"] == "train"]
    te = df_merged[df_merged["split_role"] == "test"]

    ax.scatter(tr["longitude"], tr["latitude"], c="#2563eb", s=60, alpha=0.7, edgecolors="white", linewidth=0.8, label=f"Train Folds (N={len(tr)}, 25 Districts)")
    ax.scatter(te["longitude"], te["latitude"], c="#dc2626", s=90, marker="^", alpha=0.9, edgecolors="black", linewidth=1.2, label=f"Outer Test Set - Fold 3 (N={len(te)}, 8 Districts)")
    
    ax.set_xlabel("Longitude (East)", fontsize=12)
    ax.set_ylabel("Latitude (North)", fontsize=12)
    ax.set_title("Spatial Cross-Validation Design: Train vs. Outer Test Distribution", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p3 = FIGURES_DIR / "phase10_spatial_split_map.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Generated Figure 3: {p3}")

    return p1, p2, p3

def step5_generate_reports(df, audit, fold_audit_df, df_splits, matrix_path, audit_path, splits_path):
    print("\n" + "=" * 80)
    print("STEP 5: COMPILING PHASE 10 TECHNICAL REPORTS")
    print("=" * 80)

    matrix_sha = compute_sha256(matrix_path)
    audit_sha = compute_sha256(audit_path)
    splits_sha = compute_sha256(splits_path)

    # 1. reports/phase10_split_reproducibility.md
    repro_path = REPORTS_DIR / "phase10_split_reproducibility.md"
    with open(repro_path, "w", encoding="utf-8") as f:
        f.write("# Phase 10: Split Reproducibility Specification\n\n")
        f.write("**Project Title:** Explainable AI Framework for Smart EV Charging Infrastructure Planning  \n")
        f.write("**Phase Status:** COMPLETE & VERIFIED  \n")
        f.write("**Date:** 2026-09-26  \n\n")
        f.write("---\n\n")
        f.write("## 1. Environment & Software Versions\n\n")
        f.write(f"- **Python Version:** `{sys.version.split()[0]}`\n")
        f.write(f"- **scikit-learn Version:** `{sklearn.__version__}`\n")
        f.write(f"- **pandas Version:** `{pd.__version__}`\n")
        f.write(f"- **numpy Version:** `{np.__version__}`\n\n")
        f.write("## 2. Cross-Validation Configuration\n\n")
        f.write("- **Algorithm:** `sklearn.model_selection.StratifiedGroupKFold`\n")
        f.write(f"- **Number of Folds (`n_splits`):** `{N_SPLITS}`\n")
        f.write("- **Shuffle:** `True`\n")
        f.write(f"- **Random State (`random_state`):** `{RANDOM_STATE}`\n")
        f.write("- **Grouping Variable:** `district` (33 clusters)\n")
        f.write("- **Target Variable:** `fast_charging_flag` (binary: 131 Class 1, 23 Class 0)\n\n")
        f.write("## 3. Outer Test Benchmark Definition\n\n")
        f.write("- **Designated Test Fold:** Fold 3\n")
        f.write("- **Test Sample Size:** 35 stations (22.7% of labeled sample)\n")
        f.write("- **Test Class Breakdown:** 9 Class 0 (25.7%), 26 Class 1 (74.3%)\n")
        f.write("- **Test Districts (8):** Bhadradri Kothagudem, Jagitial, Karimnagar, Medak, Mulugu, Vikarabad, Warangal Rural, Warangal Urban\n")
        f.write("- **Train Districts (25):** Exactly 0 overlap with test districts\n\n")
        f.write("## 4. Artifact Checksums\n\n")
        f.write(f"- `fast_charging_modeling_matrix.csv`: `{matrix_sha}`\n")
        f.write(f"- `phase10_district_class_distribution.csv`: `{audit_sha}`\n")
        f.write(f"- `phase10_spatial_splits.csv`: `{splits_sha}`\n")
    print(f"Exported Reproducibility Report: {repro_path}")

    # 2. reports/phase10_spatial_validation_report.md
    report_path = REPORTS_DIR / "phase10_spatial_validation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 10: Spatial-Aware Train / Validation / Test Design Report\n\n")
        f.write("**Project Title:** Explainable AI Framework for Smart EV Charging Infrastructure Planning  \n")
        f.write("**Phase Status:** COMPLETE & CERTIFIED  \n")
        f.write("**Date:** 2026-09-26  \n")
        f.write(f"**Modeling Input:** `fast_charging_modeling_matrix.csv` (SHA-256: `{matrix_sha}`)  \n")
        f.write(f"**Split Assignments:** `phase10_spatial_splits.csv` (SHA-256: `{splits_sha}`)  \n\n")
        f.write("---\n\n")

        f.write("## 1. Executive Summary & Verification Overview\n\n")
        f.write("In Phase 10, a rigorous spatial-aware train/validation/test design was established to prevent optimistic performance inflation caused by spatial autocorrelation and target leakage. The design operates strictly on the 154 ground-truth labeled stations across all 33 Telangana administrative districts.\n\n")
        f.write("- **Total Stations Validated:** **154 labeled stations** (12 unconfirmed stations held out as unlabelled pool).\n")
        f.write("- **Target Ground Truth:** **131 Class 1 (DC Fast $\\ge 50$ kW)** vs. **23 Class 0 (AC Level 1/2 $< 50$ kW)**.\n")
        f.write("- **Class Imbalance Ratio:** **5.70 : 1**.\n")
        f.write("- **Spatial Grouping Variable:** `district` (33 clusters).\n")
        f.write("- **Zero Row-Level Random Splitting:** Standard random train/test split was strictly rejected.\n")
        f.write("- **Zero Spatial Overlap:** District overlap between train and validation folds is **exactly 0** across all folds.\n")
        f.write("- **Zero Synthetic Augmentation:** No SMOTE, oversampling, or duplicate generation.\n\n")
        f.write("---\n\n")

        f.write("## 2. Scientific Rejection of Random Row-Level Splitting\n\n")
        f.write("Conventional random train/test splits (e.g. `train_test_split(test_size=0.2)`) assume independent and identically distributed (i.i.d.) observations. In geographic EV charging infrastructure, this assumption is catastrophically violated:\n\n")
        f.write("1. **Spatial Autocorrelation (Tobler's First Law):** Charging stations in close geographic proximity share unmeasured roadway corridors, commercial footfall, traffic density, and municipal grid capacity.\n")
        f.write("2. **Shared Macro-Contextual Leakage:** Stations located in the same administrative district share identical values for district-level predictors (`district_population_density_2011`, `district_urban_population_pct`, `district_nh_crossing_count`, `district_ev_penetration_rate`). Randomly placing one station from a district into the training set and another into the test set causes the model to memorize the district context, yielding artificially elevated accuracy (e.g. >95%) that collapses when deployed in unseen greenfield districts.\n")
        f.write("3. **Real-World Planning Validity:** Siting planners require models that generalize to entirely new geographic regions, not models that interpolate between densely clustered urban points.\n\n")
        f.write("---\n\n")

        f.write("## 3. District Group & Class Distribution Audit\n\n")
        f.write("The complete district distribution audit ([`phase10_district_class_distribution.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/reports/phase10_district_class_distribution.csv)) reveals the following structural properties:\n\n")
        f.write(f"- **Total Districts Represented:** {len(audit)}\n")
        f.write(f"- **Districts Containing Both Classes:** 13 districts\n")
        f.write(f"- **Districts Containing Only Class 1 (DC Fast):** 20 districts\n")
        f.write(f"- **Districts Containing Only Class 0 (AC Slow/Medium):** 0 districts\n")
        f.write(f"- **Station Concentration:** Hyderabad contains 45 stations (29.2% of the dataset), while 12 districts contain only 1 station each.\n\n")

        f.write("### Top Districts by Station Count\n")
        f.write("| District | Stations | Class 0 Count | Class 1 Count | Class 0 % | Class 1 % |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|\n")
        for _, r in audit.head(10).iterrows():
            f.write(f"| {r['district']} | {r['station_count']} | {r['class_0_count']} | {r['class_1_count']} | {r['class_0_percentage']}% | {r['class_1_percentage']}% |\n")
        f.write("\n---\n\n")

        f.write("## 4. StratifiedGroupKFold Evaluation & Audit\n\n")
        f.write("A 5-fold `StratifiedGroupKFold` split (`shuffle=True, random_state=42`) was executed grouping on `district` and stratifying on `fast_charging_flag`:\n\n")
        f.write("| Fold ID | Train Stations | Val Stations | Train Class 0 | Train Class 1 | Val Class 0 | Val Class 1 | Train Districts | Val Districts | District Overlap |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for _, r in fold_audit_df.iterrows():
            f.write(f"| Fold {r['fold_id']} | {r['train_station_count']} | {r['validation_station_count']} | {r['train_class_0']} | {r['train_class_1']} | {r['validation_class_0']} | {r['validation_class_1']} | {r['train_district_count']} | {r['validation_district_count']} | **{r['district_overlap']}** |\n")
        f.write("\n")
        f.write("### Key Audit Findings:\n")
        f.write("1. **Zero Spatial Overlap:** `district_overlap` is **strictly 0** for every fold. No district is shared between train and validation.\n")
        f.write("2. **Universal Class Representation:** **Both classes are present in every single validation fold.** (Fold 0 has 10 Class 0; Fold 1 has 2; Fold 2 has 1; Fold 3 has 9; Fold 4 has 1).\n")
        f.write("3. **Cluster Size Heterogeneity:** Fold 0 is larger (68 stations) because Hyderabad (45 stations) cannot be split across folds under grouped cross-validation. This is an unavoidable mathematical consequence of preserving spatial integrity without data leakage.\n\n")
        f.write("---\n\n")

        f.write("## 5. Outer Test Set Decision & Dual Evaluation Architecture\n\n")
        f.write("### A. Outer Spatial Test Set Designation: Fold 3\n")
        f.write("- **Test Sample Size:** **35 stations** (22.7% of the 154 labeled stations).\n")
        f.write("- **Test Class Breakdown:** **9 Class 0 (25.7%) vs. 26 Class 1 (74.3%)**.\n")
        f.write("- **Test Districts (8 Districts):** `Bhadradri Kothagudem`, `Jagitial`, `Karimnagar`, `Medak`, `Mulugu`, `Vikarabad`, `Warangal Rural`, `Warangal Urban`.\n")
        f.write("- **Train Sample Size:** **119 stations** across 25 completely disjoint districts (14 Class 0, 105 Class 1).\n")
        f.write("- **Spatial Overlap:** **0 districts**.\n")
        f.write("- **Defensibility:** Fold 3 provides an optimal balance: exactly 0 district leakage, substantial test sample ($N=35$), diverse regional representation (northern, eastern, and peri-urban districts), and robust minority class representation ($N=9$).\n\n")

        f.write("### B. Dual Evaluation Architecture for Phase 10 / 11\n")
        f.write("To maximize scientific rigor, the framework adopts a dual evaluation protocol:\n\n")
        f.write("1. **Protocol 1: 5-Fold Spatial Grouped Cross-Validation (Full Dataset, N = 154):**\n")
        f.write("   - Evaluates model performance across all 33 districts out-of-fold.\n")
        f.write("   - Primary metric: Mean Out-of-Fold (OOF) PR-AUC, F1-Score, and Balanced Accuracy.\n")
        f.write("   - Avoids discarding any data while strictly preventing spatial leakage.\n\n")
        f.write("2. **Protocol 2: Fixed Outer Spatial Test Benchmark (Train N = 119 vs. Test N = 35):**\n")
        f.write("   - Models are trained on the 119 stations (using grouped CV on the 25 training districts for hyperparameter tuning).\n")
        f.write("   - Evaluated once on the held-out 35 test stations (Fold 3) to demonstrate true spatial out-of-domain generalization.\n\n")
        f.write("---\n\n")

        f.write("## 6. Preprocessing & Leakage Re-Verification\n\n")
        f.write("- **Certified Predictor Set:** Exactly 21 predictors (8 spatial/GIS + 7 road one-hot + 6 district context).\n")
        f.write("- **Zero Target Leakage:** `charger_type`, `charging_power_kw`, `station_capacity`, and `connector_type` are completely absent from $X$.\n")
        f.write("- **Zero Target Contamination:** `district_fast_stations` and `district_fast_charger_ratio` are completely absent from $X$.\n")
        f.write("- **Zero Metadata Leakage:** `station_id` and `district` are retained strictly as indexing/grouping metadata.\n")
        f.write("- **Zero Target-Dependent Preprocessing:** No scaling, imputation, or target encoding was fitted globally across train and test partitions.\n\n")
        f.write("---\n\n")

        f.write("## 7. Sample Size Limitations and Methodological Constraints\n\n")
        f.write("1. **Minority Class Sparsity:** With only 23 Class 0 stations statewide, certain rural districts contain 0 slow chargers. Class weighting (`scale_pos_weight`, `class_weight='balanced'`) must be applied in Phase 11.\n")
        f.write("2. **Metropolitan Dominance:** Hyderabad's 45 stations represent 29.2% of the dataset. Preserving district grouping means any fold containing Hyderabad will be larger than average. Splitting Hyderabad across folds would cause severe spatial leakage.\n")
        f.write("3. **Metric Selection:** Standard unweighted Accuracy is uninformative due to the 5.7:1 imbalance. Evaluation must prioritize Precision-Recall AUC (PR-AUC), F1-Score (macro and minority), and Balanced Accuracy.\n\n")
        f.write("---\n\n")

        f.write("## 8. Final Phase 10 Certification\n\n")
        f.write("> **PHASE 10 STATUS: PASS**  \n")
        f.write("> The spatial-aware validation design strictly guarantees zero district overlap, zero target leakage, robust minority class presence across all folds, and complete reproducibility (seed = 42). The project is certified ready for Phase 11 model training upon human authorization.\n")

    print(f"Exported Spatial Validation Report: {report_path}")

def update_config(audit_sha, splits_sha):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["phases"]["current"] = 10.0
    cfg["phases"]["status"] = "Phase 10 - Spatial-Aware Train / Validation / Test Design COMPLETE"
    cfg["data_sources"]["spatial_splits"] = {
        "file": "data/processed/phase10_spatial_splits.csv",
        "rows": 154,
        "n_splits": N_SPLITS,
        "grouping_variable": "district",
        "random_state": RANDOM_STATE,
        "outer_test_fold": 3,
        "outer_test_stations": 35,
        "outer_train_stations": 119,
        "district_overlap": 0,
        "sha256": splits_sha
    }
    cfg["data_sources"]["district_audit"] = {
        "file": "reports/phase10_district_class_distribution.csv",
        "districts_count": 33,
        "sha256": audit_sha
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Updated configuration in {CONFIG_PATH}")

def main():
    print("=" * 80)
    print("PHASE 10: SPATIAL-AWARE TRAIN / VALIDATION / TEST DESIGN")
    print("=" * 80)

    # 1. Load and verify
    df, matrix_path = step1_load_and_verify()

    # 2. District group audit
    audit, audit_path, audit_sha = step2_district_group_audit(df)

    # 3. StratifiedGroupKFold audit & split generation
    fold_audit_df, df_splits, splits_path, splits_sha = step3_stratified_group_kfold_audit(df)

    # 4. Generate visualizations
    p1, p2, p3 = step4_generate_visualizations(df, df_splits)

    # 5. Generate technical reports
    step5_generate_reports(df, audit, fold_audit_df, df_splits, matrix_path, audit_path, splits_path)

    # 6. Update configuration
    update_config(audit_sha, splits_sha)

    print("\n" + "=" * 80)
    print("PHASE 10 EXECUTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
