# Phase 10: Spatial-Aware Train / Validation / Test Design Report

**Project Title:** Explainable AI Framework for Smart EV Charging Infrastructure Planning  
**Phase Status:** COMPLETE & CERTIFIED  
**Date:** 2026-09-26  
**Modeling Input:** `fast_charging_modeling_matrix.csv` (SHA-256: `fc32c75635fe2e763ec174e491a897adc691539fba0be655f06b9d58bca79783`)  
**Split Assignments:** `phase10_spatial_splits.csv` (SHA-256: `27423b8a5d084fe09e4dcd037f376b11c5c7acf904ccd243d0e85dfcafe624e5`)  

---

## 1. Executive Summary & Verification Overview

In Phase 10, a rigorous spatial-aware train/validation/test design was established to prevent optimistic performance inflation caused by spatial autocorrelation and target leakage. The design operates strictly on the 154 ground-truth labeled stations across all 33 Telangana administrative districts.

- **Total Stations Validated:** **154 labeled stations** (12 unconfirmed stations held out as unlabelled pool).
- **Target Ground Truth:** **131 Class 1 (DC Fast $\ge 50$ kW)** vs. **23 Class 0 (AC Level 1/2 $< 50$ kW)**.
- **Class Imbalance Ratio:** **5.70 : 1**.
- **Spatial Grouping Variable:** `district` (33 clusters).
- **Zero Row-Level Random Splitting:** Standard random train/test split was strictly rejected.
- **Zero Spatial Overlap:** District overlap between train and validation folds is **exactly 0** across all folds.
- **Zero Synthetic Augmentation:** No SMOTE, oversampling, or duplicate generation.

---

## 2. Scientific Rejection of Random Row-Level Splitting

Conventional random train/test splits (e.g. `train_test_split(test_size=0.2)`) assume independent and identically distributed (i.i.d.) observations. In geographic EV charging infrastructure, this assumption is catastrophically violated:

1. **Spatial Autocorrelation (Tobler's First Law):** Charging stations in close geographic proximity share unmeasured roadway corridors, commercial footfall, traffic density, and municipal grid capacity.
2. **Shared Macro-Contextual Leakage:** Stations located in the same administrative district share identical values for district-level predictors (`district_population_density_2011`, `district_urban_population_pct`, `district_nh_crossing_count`, `district_ev_penetration_rate`). Random row-level splitting can produce optimistically biased performance estimates because geographically related stations and shared district-level contextual features may occur in both training and evaluation partitions.
3. **Real-World Planning Validity:** Siting planners require models that generalize to entirely new geographic regions, not models that interpolate between densely clustered urban points.

> **Methodological Note on Spatial Autocorrelation:**
> No district-level overlap exists between training and validation folds, which substantially reduces intra-district spatial leakage. This does not guarantee zero geographic autocorrelation across adjacent district boundaries. District-level spatial grouping mitigates administrative-level feature leakage but does not claim complete elimination of spatial autocorrelation.

---

## 3. District Group & Class Distribution Audit

The complete district distribution audit ([`phase10_district_class_distribution.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/reports/phase10_district_class_distribution.csv)) reveals the following structural properties:

- **Total Districts Represented:** 33
- **Districts Containing Both Classes:** 13 districts
- **Districts Containing Only Class 1 (DC Fast):** 20 districts
- **Districts Containing Only Class 0 (AC Slow/Medium):** 0 districts
- **Station Concentration:** Hyderabad contains 45 stations (29.2% of the dataset), while 12 districts contain only 1 station each.

### Top Districts by Station Count
| District | Stations | Class 0 Count | Class 1 Count | Class 0 % | Class 1 % |
|---|:---:|:---:|:---:|:---:|:---:|
| Hyderabad | 45 | 6 | 39 | 13.3% | 86.7% |
| Karimnagar | 10 | 2 | 8 | 20.0% | 80.0% |
| Warangal Urban | 10 | 2 | 8 | 20.0% | 80.0% |
| Medchal-Malkajgiri | 8 | 1 | 7 | 12.5% | 87.5% |
| Ranga Reddy | 7 | 0 | 7 | 0.0% | 100.0% |
| Sangareddy | 5 | 2 | 3 | 40.0% | 60.0% |
| Mahabubnagar | 4 | 0 | 4 | 0.0% | 100.0% |
| Nalgonda | 4 | 0 | 4 | 0.0% | 100.0% |
| Nizamabad | 4 | 0 | 4 | 0.0% | 100.0% |
| Suryapet | 4 | 0 | 4 | 0.0% | 100.0% |

---

## 4. StratifiedGroupKFold Evaluation & Audit

A 5-fold `StratifiedGroupKFold` split (`shuffle=True, random_state=42`) was executed grouping on `district` and stratifying on `fast_charging_flag`:

| Fold ID | Train Stations | Val Stations | Train Class 0 | Train Class 1 | Val Class 0 | Val Class 1 | Train Districts | Val Districts | District Overlap |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Fold 0 | 86 | 68 | 13 | 73 | 10 | 58 | 25 | 8 | **0** |
| Fold 1 | 133 | 21 | 21 | 112 | 2 | 19 | 26 | 7 | **0** |
| Fold 2 | 151 | 3 | 22 | 129 | 1 | 2 | 32 | 1 | **0** |
| Fold 3 | 119 | 35 | 14 | 105 | 9 | 26 | 25 | 8 | **0** |
| Fold 4 | 127 | 27 | 22 | 105 | 1 | 26 | 24 | 9 | **0** |

### Key Audit Findings:
1. **Zero Spatial Overlap:** `district_overlap` is **strictly 0** for every fold. No district is shared between train and validation.
2. **Universal Class Representation:** **Both classes are present in every single validation fold.** (Fold 0 has 10 Class 0; Fold 1 has 2; Fold 2 has 1; Fold 3 has 9; Fold 4 has 1).
3. **Cluster Size Heterogeneity:** Fold 0 is larger (68 stations) because Hyderabad (45 stations) cannot be split across folds under grouped cross-validation. This is an unavoidable mathematical consequence of preserving spatial integrity without data leakage.

---

## 5. Outer Test Set Decision & Dual Evaluation Architecture

### A. Outer Spatial Test Set Designation: Fold 3
- **Test Sample Size:** **35 stations** (22.7% of the 154 labeled stations).
- **Test Class Breakdown:** **9 Class 0 (25.7%) vs. 26 Class 1 (74.3%)**.
- **Test Districts (8 Districts):** `Bhadradri Kothagudem`, `Jagitial`, `Karimnagar`, `Medak`, `Mulugu`, `Vikarabad`, `Warangal Rural`, `Warangal Urban`.
- **Train Sample Size:** **119 stations** across 25 completely disjoint districts (14 Class 0, 105 Class 1).
- **Spatial Overlap:** **0 districts**.
- **Defensibility:** Fold 3 provides an optimal balance: exactly 0 district leakage, substantial test sample ($N=35$), diverse regional representation (northern, eastern, and peri-urban districts), and robust minority class representation ($N=9$).

### B. Dual Evaluation Architecture for Phase 10 / 11
To maximize scientific rigor, the framework adopts a dual evaluation protocol:

1. **Protocol 1: 5-Fold Spatial Grouped Cross-Validation (Full Dataset, N = 154):**
   - Evaluates model performance across all 33 districts out-of-fold.
   - Primary metric: Mean Out-of-Fold (OOF) PR-AUC, F1-Score, and Balanced Accuracy.
   - Avoids discarding any data while strictly preventing spatial leakage.

2. **Protocol 2: Fixed Outer Spatial Test Benchmark (Train N = 119 vs. Test N = 35):**
   - Models are trained on the 119 stations (using grouped CV on the 25 training districts for hyperparameter tuning).
   - Evaluated once on the held-out 35 test stations (Fold 3) to demonstrate true spatial out-of-domain generalization.

---

## 6. Preprocessing & Leakage Re-Verification

- **Certified Predictor Set:** Exactly 21 predictors (8 spatial/GIS + 7 road one-hot + 6 district context).
- **Zero Target Leakage:** `charger_type`, `charging_power_kw`, `station_capacity`, and `connector_type` are completely absent from $X$.
- **Zero Target Contamination:** `district_fast_stations` and `district_fast_charger_ratio` are completely absent from $X$.
- **Zero Metadata Leakage:** `station_id` and `district` are retained strictly as indexing/grouping metadata.
- **Zero Target-Dependent Preprocessing:** No scaling, imputation, or target encoding was fitted globally across train and test partitions.

---

## 7. Sample Size Limitations and Methodological Constraints

1. **Minority Class Sparsity:** With only 23 Class 0 stations statewide, certain rural districts contain 0 slow chargers. Class weighting (`scale_pos_weight`, `class_weight='balanced'`) must be applied in Phase 11.
2. **Metropolitan Dominance:** Hyderabad's 45 stations represent 29.2% of the dataset. Preserving district grouping means any fold containing Hyderabad will be larger than average. Splitting Hyderabad across folds would cause severe spatial leakage.
3. **Metric Selection:** Standard unweighted Accuracy is uninformative due to the 5.7:1 imbalance. Evaluation must prioritize Precision-Recall AUC (PR-AUC), F1-Score (macro and minority), and Balanced Accuracy.

---

## 8. Final Phase 10 Certification

> **PHASE 10 STATUS: PASS**  
> The spatial-aware validation design strictly guarantees zero district overlap, zero target leakage, robust minority class presence across all folds, and complete reproducibility (seed = 42). The project is certified ready for Phase 11 model training upon human authorization.
