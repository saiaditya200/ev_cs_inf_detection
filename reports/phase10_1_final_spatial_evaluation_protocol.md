# Phase 10.1 & Phase 10.2: Final Spatial Evaluation Protocol Report

**Project Title:** Explainable AI Framework for Smart EV Charging Infrastructure Planning  
**Phase Status:** COMPLETE & CERTIFIED  
**Date:** 2026-09-26  
**Frozen Master Dataset:** `data/final/FINAL_PROJECT_DATASET.csv` (SHA-256: `a1999524323860397e7255b5ff7884d7f77394b09abd69a168a306bf7eae9788`)  
**Outer Test Dataset:** `data/processed/phase10_outer_test.csv` (SHA-256: `a2a24c8c3e6477f631be850dd182472e481ac0edb0e4a57c2e1d2caaa98038ea`)  
**Development Dataset:** `data/processed/phase10_development.csv` (SHA-256: `50afe820f9e753a9eb410c39009e32527ddedc9606ccad0c7e4f7675296ab75e`)  
**Final Split Assignments:** `data/processed/phase10_final_split_assignments.csv` (SHA-256: `2fda0c0a16feea5bf998c8f11ff91f3f23cb011d41e2b1b4e53add453127e581`)  

---

## 1. Original Phase 10 Diagnostic

The original Phase 10 evaluation conducted an initial feasibility audit using 5-fold `StratifiedGroupKFold` on all 154 labeled stations across 33 districts. That audit confirmed:
- Spatial grouping at the district level is structurally feasible.
- District-level overlap between train and validation folds was 0 across all five folds.
- Fold 3 contained 35 stations across 8 districts with both classes represented (9 Class 0, 26 Class 1).
- The diagnostic artifacts (`reports/phase10_district_class_distribution.csv`, `reports/phase10_spatial_validation_report.md`, and `data/processed/phase10_spatial_splits.csv`) remain preserved as diagnostic references.

## 2. Why Random Row Splitting Is Rejected

Standard random train/test splits (e.g. `train_test_split(test_size=0.2)`) assume independent and identically distributed (i.i.d.) observations. For geographic infrastructure planning, this assumption is fundamentally invalid:
1. **Spatial Autocorrelation:** Siting locations in geographical proximity share unobserved physical corridors, commuter flows, commercial density, and power grid infrastructure.
2. **Shared Macro-Contextual Features:** Stations within the same administrative district share identical Census demographics, urbanization percentages, national highway counts, and EV penetration rates. Splitting stations from the same district across train and test partitions allows algorithms to memorize district-level contextual predictors.
3. **Real-World Policy Generalization:** Infrastructure planners require models that evaluate novel locations in unseen or underserved districts, rather than interpolating between existing high-density stations.

## 3. Correction to Unsupported ">95%" Performance Claim

> **METHODOLOGICAL CORRECTION ON PERFORMANCE CLAIMS:**
> Random row-level splitting can produce optimistically biased performance estimates because geographically related stations and shared district-level contextual features may occur in both training and evaluation partitions.
> No specific numerical claims regarding future model performance (such as ">95% accuracy") are asserted prior to actual model training.

## 4. Outer Test Selection Methodology

The outer test partition was selected via a pre-specified, deterministic selection rule established prior to observing any model performance:
- **Selection Rule:** Fold 3 of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouping on `district`.
- **Selection Criteria Satisfied (A through G):**
  - A. **0 District Overlap:** Exactly 0 districts shared with the development partition.
  - B. **Both Classes Present:** Both Class 0 (AC Slow/Medium) and Class 1 (DC Fast) are represented.
  - C. **Adequate Minority Count:** Contains 9 Class 0 stations (39.1% of all Class 0 stations statewide).
  - D. **Deterministic Reproducibility:** Fixed random state = 42.
  - E. **Independence from Model Performance:** Specified in Phase 10 before any machine learning algorithms are fitted.
  - F. **No Preprocessing Fitted on Test:** No scalers, imputers, or transformers fitted using test observations.
  - G. **No Feature Engineering Using Test Labels:** Certified 21 predictors are strictly exogenous.

## 5. Outer Test Class Distribution

- **Total Test Stations ($N$):** **35 physical stations** (22.73% of labeled sample)
- **Class 0 Count:** **9 stations** (25.71% of outer test)
- **Class 1 Count:** **26 stations** (74.29% of outer test)
- **Number of Test Districts:** **8 districts**
- **Test District Roster:** `Bhadradri Kothagudem, Jagitial, Karimnagar, Medak, Mulugu, Vikarabad, Warangal Rural, Warangal Urban`
- **Geographic Breadth:** Spans northern agricultural and coal belt districts (Jagitial, Karimnagar, Bhadradri Kothagudem), peri-urban districts (Warangal Urban, Warangal Rural, Medak, Vikarabad), and rural forest districts (Mulugu).

## 6. Development Dataset Dimensions

- **Total Development Stations ($N$):** **119 stations** (77.27% of labeled sample)
- **Total Development Features:** **24 columns** (`station_id`, `district`, `fast_charging_flag`, plus the 21 certified predictors)
- **Class 0 Count:** **14 stations** (11.76% of development sample)
- **Class 1 Count:** **105 stations** (88.24% of development sample)
- **Station Conservation:** 119 (Development) + 35 (Outer Test) = 154 (Total Labeled). The 12 unlabelled stations remain held out.

## 7. Development District Distribution

- **Total Development Districts:** **25 districts**
- **Development District Roster:** `Adilabad, Hyderabad, Jangaon, Jayashankar Bhupalapally, Jogulamba Gadwal, Kamareddy, Khammam, Kumurambheem Asifabad, Mahabubabad, Mahabubnagar, Mancherial, Medchal-Malkajgiri, Nagarkurnool, Nalgonda, Narayanpet, Nirmal, Nizamabad, Peddapalli, Rajanna Sircilla, Ranga Reddy, Sangareddy, Siddipet, Suryapet, Wanaparthy, Yadadri Bhuvanagiri`
- **Metropolitan Center:** Hyderabad (45 stations) resides entirely within the development partition.
- **Minority Class Sparsity:** Among the 25 development districts, only 5 districts contain Class 0 stations (`Hyderabad`: 6, `Sangareddy`: 2, `Kamareddy`: 2, `Mahabubabad`: 2, `Medchal-Malkajgiri`: 1, `Siddipet`: 1).

## 8. Development-Only Grouped Cross-Validation Comparison

Cross-validation strategies were audited **strictly on the 119 development stations across 25 districts**, with outer-test stations completely excluded:

### Strategy 1: StratifiedGroupKFold (n_splits = 5, shuffle = True, random_state = 42)
| fold_id | train_N | val_N | train_c0 | train_c1 | val_c0 | val_c1 | train_dists | val_dists | district_overlap |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 117 | 2 | 14 | 103 | 0 | 2 | 24 | 1 | 0 |
| Fold 1 | 106 | 13 | 13 | 93 | 1 | 12 | 19 | 6 | 0 |
| Fold 2 | 104 | 15 | 11 | 93 | 3 | 12 | 20 | 5 | 0 |
| Fold 3 | 54 | 65 | 8 | 46 | 6 | 59 | 18 | 7 | 0 |
| Fold 4 | 95 | 24 | 10 | 85 | 4 | 20 | 19 | 6 | 0 |

*Defect:* Fold 0 has only 2 stations and contains **0 Class 0 stations** (`val_c0 = 0`), creating a degenerate validation fold.

### Strategy 2: StratifiedGroupKFold (n_splits = 4, shuffle = True, random_state = 42)
| fold_id | train_N | val_N | train_c0 | train_c1 | val_c0 | val_c1 | train_dists | val_dists | district_overlap |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 117 | 2 | 14 | 103 | 0 | 2 | 24 | 1 | 0 |
| Fold 1 | 97 | 22 | 14 | 83 | 0 | 22 | 17 | 8 | 0 |
| Fold 2 | 94 | 25 | 12 | 82 | 2 | 23 | 17 | 8 | 0 |
| Fold 3 | 49 | 70 | 2 | 47 | 12 | 58 | 17 | 8 | 0 |

*Defect:* Fold 0 and Fold 1 both contain **0 Class 0 stations** (`val_c0 = 0`), making minority-class validation impossible across half the folds.

### Strategy 3: GroupKFold (n_splits = 4) [SELECTED PRIMARY PROTOCOL]
| fold_id | train_N | val_N | train_c0 | train_c1 | val_c0 | val_c1 | train_dists | val_dists | district_overlap |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 74 | 45 | 8 | 66 | 6 | 39 | 24 | 1 | 0 |
| Fold 1 | 94 | 25 | 11 | 83 | 3 | 22 | 17 | 8 | 0 |
| Fold 2 | 94 | 25 | 11 | 83 | 3 | 22 | 17 | 8 | 0 |
| Fold 3 | 95 | 24 | 12 | 83 | 2 | 22 | 17 | 8 | 0 |

*Performance Audit:* Every fold has at least 2 Class 0 stations (`val_c0`: 6, 3, 3, 2). Validation sample sizes are balanced (45, 25, 25, 24 stations). Zero district overlap across all folds.

### Strategy 4: GroupKFold (n_splits = 5) [SECONDARY ALTERNATIVE]
| fold_id | train_N | val_N | train_c0 | train_c1 | val_c0 | val_c1 | train_dists | val_dists | district_overlap |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 74 | 45 | 8 | 66 | 6 | 39 | 24 | 1 | 0 |
| Fold 1 | 100 | 19 | 10 | 90 | 4 | 15 | 19 | 6 | 0 |
| Fold 2 | 100 | 19 | 13 | 87 | 1 | 18 | 19 | 6 | 0 |
| Fold 3 | 101 | 18 | 12 | 89 | 2 | 16 | 19 | 6 | 0 |
| Fold 4 | 101 | 18 | 13 | 88 | 1 | 17 | 19 | 6 | 0 |

*Performance Audit:* All folds have at least 1 Class 0 station (`val_c0`: 6, 4, 1, 2, 1), but 4-fold CV provides stronger minority support per fold.

## 9. Final Development CV Choice

**Selected Method:** `GroupKFold(n_splits=4)` on the 119 development stations grouped by `district`.

### Technical Justification:
1. **Guaranteed Minority Representation:** In small, highly imbalanced grouped datasets ($N=119$, 14 minority instances across 25 groups), `StratifiedGroupKFold` fails because individual group sizes (e.g. Hyderabad with 45 stations) overwhelm the stratification optimizer. `GroupKFold(n_splits=4)` naturally isolates Hyderabad into Fold 0 (6 Class 0, 39 Class 1) and distributes the remaining 24 districts across Folds 1, 2, and 3, ensuring each fold contains 2 to 3 Class 0 stations.
2. **Balanced Sample Sizes:** Folds 1, 2, and 3 have 25, 25, and 24 validation stations respectively, providing uniform fold weights.
3. **Non-Degenerate Loss Computation:** Every single validation split contains both positive and negative cases, permitting valid computation of ROC-AUC, PR-AUC, and F1-score across all 4 folds.

## 10. District Overlap Audit

An automated intersection check was performed across all partitions:
$$\text{Outer Test Districts} \cap \text{Development Districts} = \emptyset$$
- Outer Test District Count: **8**
- Development District Count: **25**
- Total Districts: **33**
- District Overlap: **0 districts**

Within the development-CV folds (`GroupKFold(n_splits=4)`):
- District overlap between training and validation folds is **0 in every single fold**.

## 11. Test-Set Isolation Statement

The outer test dataset (`data/processed/phase10_outer_test.csv`) is **strictly quarantined and completely untouched** during model development:
- **Zero Feature Selection Influence:** Outer test observations are never used to rank, select, or drop features.
- **Zero Hyperparameter Tuning Influence:** Grid search, Bayesian search, and early stopping evaluate solely on development-CV folds.
- **Zero Algorithm Selection Influence:** The selection among Random Forest, XGBoost, or LightGBM is decided on development cross-validation metrics.
- **Zero Threshold Optimization Influence:** Decision thresholds are tuned inside development folds.
- **Single Final Evaluation:** Outer test data will be scored **exactly once** in Phase 11 as the final out-of-domain benchmark.

## 12. Leakage Prevention Protocol

To ensure zero mathematical or target leakage in Phase 11:
1. **Fit Transformations Inside Folds:** Any learned preprocessor (e.g. `StandardScaler`, `QuantileTransformer`) must be fitted solely on the training fold indices of each CV fold and applied to validation/test folds.
2. **Exclusion of Contaminated Attributes:** Direct equipment specifications (`charging_power_kw`, `charger_type`, `station_capacity`, `connector_type`, `number_of_chargers`) and target-derived aggregations (`district_fast_stations`, `district_fast_charger_ratio`) are strictly excluded from $X$.
3. **Complete Certified Features:** All 21 certified predictors contain 0 missing values; no imputation step is required.

## 13. Reproducibility

Complete pipeline reproducibility is guaranteed:
- **Random Seed:** `random_state = 42`.
- **Software Versions:** Python `3.8.10`, scikit-learn `1.3.0`, pandas `2.0.3`, numpy `1.24.3`.
- **Master Dataset SHA-256:** `a1999524323860397e7255b5ff7884d7f77394b09abd69a168a306bf7eae9788`
- **Outer Test SHA-256:** `a2a24c8c3e6477f631be850dd182472e481ac0edb0e4a57c2e1d2caaa98038ea`
- **Development SHA-256:** `50afe820f9e753a9eb410c39009e32527ddedc9606ccad0c7e4f7675296ab75e`
- **Final Split Assignments SHA-256:** `2fda0c0a16feea5bf998c8f11ff91f3f23cb011d41e2b1b4e53add453127e581`

## 14. Limitations Due to Sample Size (N = 154) and Class Imbalance

1. **Minority Class Sparsity:** With only 23 Class 0 stations statewide (14 in development, 9 in outer test), validation folds contain 2 to 6 minority cases. Minor classification errors have discrete effects on point metrics.
2. **Evaluation Metric Selection:** Unweighted accuracy is invalid under a 5.7:1 class imbalance. Phase 11 must prioritize Precision-Recall AUC (PR-AUC), F1-Score (macro & minority), and Balanced Accuracy.
3. **Algorithmic Weighting:** Class-weight adjustments (`scale_pos_weight`, `class_weight='balanced'`) will be essential during model training.

## 15. Limitations of District-Level Grouping

To maintain scientific precision and avoid overstated methodological claims:
1. **Intra-District vs. Inter-District Autocorrelation:** No district-level overlap exists between training and validation folds, which substantially reduces intra-district spatial leakage. However, this does not guarantee zero geographic autocorrelation across adjacent district boundaries. Siting locations positioned near district borders (e.g. between Hyderabad and Ranga Reddy or Medchal-Malkajgiri) may still experience spatial correlation across administrative lines.
2. **Administrative Boundary Arbitrariness:** District boundaries are administrative rather than physical barriers; traffic flows and commuter corridors cross borders continuously.
3. **Clustering Concentration:** Hyderabad's 45 stations represent 37.8% of the development partition. Grouping by district requires keeping all 45 stations within a single fold (Fold 0) to avoid intra-district contamination, resulting in uneven fold station counts (45 vs. 25, 25, 24).
4. **Acknowledgment:** These limitations of district-level spatial grouping and district-held-out evaluation are acknowledged rather than overstated.

## 16. Final Phase 10.1 Certification

> **PHASE 10.1 STATUS: PASS**  
> The spatial evaluation protocol strictly isolates the 35 outer-test stations from model development, establishes a non-degenerate 4-fold grouped cross-validation strategy across the 119 development stations with zero district overlap, corrects all performance claims and spatial isolation language, and guarantees complete mathematical reproducibility (seed = 42). The project is certified ready for Phase 11 model training upon human authorization.
