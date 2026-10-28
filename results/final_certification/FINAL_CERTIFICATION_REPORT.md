# Final Certification Report

**Project**: Explainable AI Framework for Smart EV Charging Infrastructure Planning  
**Execution Date**: 2026-09-28 02:06:59  
**Methodology Status**: FROZEN  
**Production Status**: UNCHANGED & HASH-VERIFIED  
**Research Search Status**: CLOSED  

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Research Problem](#2-research-problem)
3. [Dataset](#3-dataset)
4. [Target](#4-target)
5. [Feature Contract](#5-feature-contract)
6. [Leakage Control](#6-leakage-control)
7. [Spatial Validation](#7-spatial-validation)
8. [Model Development](#8-model-development)
9. [Final Deployment Model](#9-final-deployment-model)
10. [Outer-Test Evidence](#10-outer-test-evidence)
11. [AC Signal Investigation](#11-ac-signal-investigation)
12. [Spatial Covariate Shift](#12-spatial-covariate-shift)
13. [XAI](#13-xai)
14. [Counterfactuals](#14-counterfactuals)
15. [EIPS](#15-eips)
16. [GIS](#16-gis)
17. [Application](#17-application)
18. [Limitations](#18-limitations)
19. [Research Contribution](#19-research-contribution)
20. [Production Governance](#20-production-governance)

---

## 1. Project Identity

- **Full Title**: Explainable AI Framework for Smart EV Charging Infrastructure Planning
- **Geographic Domain**: State of Telangana, India (all 33 Administrative Districts)
- **Primary Stakeholder Focus**: Regional energy authorities, urban planners, municipal corporations, and EV charging station operators.

## 2. Research Problem

Electric vehicle charging infrastructure in developing economies is frequently deployed in an ad-hoc, intuitive manner, leading to spatial coverage gaps, grid stress, and low utilization. Existing machine learning models for charging infrastructure suffer from three critical flaws:

1. Target and hardware specification leakage (using charger specs to predict charger specs).
2. Spatial autocorrelation overfitting (random cross-validation mixing nearby stations).
3. Black-box opacity that fails to provide actionable explanations or multi-criteria district planning.

This project resolves these flaws through an explainable, leakage-controlled, spatially validated dual-level framework.

## 3. Dataset

- **Master Dataset Path**: `data/final/FINAL_PROJECT_DATASET.csv`
- **Total Station Population**: 166 stations across 78 attributes.
- **Supervised Training Population**: 154 labeled stations across all 33 Telangana districts.
- **Class Distribution**:
  - Class 1 (DC Fast Charging $\ge 50$ kW): 131 stations (85.06%)
  - Class 0 (AC Slow / Non-Fast $< 50$ kW): 23 stations (14.94%)
  - Unlabeled Stations: 12 stations reserved for descriptive visualization.
- **Geographic Representation**: All 33 Telangana districts are represented with zero missing district labels.

## 4. Target

- **Target Variable**: `fast_charging_flag`
- **Class Definitions**:
  - `1` = DC Fast Charging Hub ($\ge 50$ kW charging capacity)
  - `0` = AC Slow / Medium Destination Charging (< 50 kW charging capacity)
- **Target Boundary**: Binary classification of observed infrastructure type. It is NOT continuous future kWh demand forecasting and NOT station utilization.

## 5. Feature Contract

Exactly 21 certified exogenous predictors across three distinct feature groups:

- **Spatial/GIS (8)**: `latitude`, `longitude`, `distance_to_highway_km`, `distance_to_city_center_km`, `nearby_poi_count`, `highway_proximity_score`, `city_center_proximity_score`, `poi_diversity_index`.
- **Road Topology (7)**: `road_type_motorway`, `road_type_primary`, `road_type_residential`, `road_type_secondary`, `road_type_service`, `road_type_trunk`, `road_type_unclassified`.
- **District Context (6)**: `district_population_density_2011`, `district_urban_population_pct`, `district_projected_density_fy24`, `district_nh_crossing_count`, `district_ev_penetration_rate`, `discom_utility_data_available_flag`.

## 6. Leakage Control

The following variables are strictly excluded from the predictor space:
`charger_type`, `charging_power_kw`, `station_capacity`, `connector_type`, `number_of_chargers`, `operator`, `district_fast_stations`, `district_fast_charger_ratio`, and `traffic_density`.

Zero target-derived, equipment, or operator variables enter model training or prospective inference.

## 7. Spatial Validation

- **Development Cross-Validation**: 4-fold `GroupKFold` strictly grouped by `district` across 119 development stations in 25 districts. Ensures stations from the same district never appear in both training and validation folds.
- **Quarantined Outer Test**: 35 stations across 8 held-out districts completely quarantined from all tuning, hyperparameter search, and threshold selection. Zero district overlap.

## 8. Model Development

Research progression across 7 stages:

1. Baseline Comparison: Random Forest established superior balanced performance over XGBoost, LightGBM, and Logistic Regression.
2. Joint Optimization: 30,628 evaluations mapped depth, leaf, split, and threshold parameter spaces.
3. Ensemble Bake-off: Evaluated soft-voting and stacking ensembles; single Random Forest proved superior and preserved TreeSHAP fidelity.
4. AC-Focused Search: Evaluated cost-weighting; unweighted RF with calibrated threshold showed superior cross-fold stability.
5. Pareto Optimization: 97,188 evaluations mapped complete Pareto trade-off curve, identifying non-dominated leaders `RF_0002` and `RF_0039`.
6. AC Signal Enrichment: Evaluated site-context additions; discovered severe spatial covariate shift.
7. Provenance Reconciliation: Audited and certified 100% computational concordance.

## 9. Final Deployment Model

- **Runtime Class**: `RandomForestClassifier`
- **Parameters**: `n_estimators=250, max_depth=3, min_samples_split=2, min_samples_leaf=2, max_features=0.5, random_state=42, bootstrap=True, criterion='gini'`
- **Decision Threshold**: `0.760` (from `models/final_deployment_threshold.json`)
- **Status**: FROZEN and UNCHANGED in `models/final_deployment_model.joblib`.

## 10. Outer-Test Evidence

Authoritative evaluation on the 35 quarantined outer-test stations across 8 held-out districts:

- **Current Bakeoff Best RF (Benchmark)**: Accuracy = 74.29%, Balanced Accuracy = 60.90%, AC Recall = 33.33% (3/9), DC Recall = 88.46% (23/26), Macro F1 = 0.6182, PR-AUC = 0.8013 ($T = 0.755$).
- **RF_0039 (Secondary Benchmark)**: Accuracy = 80.00%, Balanced Accuracy = 64.74%, AC Recall = 33.33% (3/9), DC Recall = 96.15% (25/26), Macro F1 = 0.6694, PR-AUC = 0.8759 ($T = 0.720$).
- **RF_0002 (Secondary Benchmark)**: Accuracy = 71.43%, Balanced Accuracy = 51.71%, AC Recall = 11.11% (1/9), DC Recall = 92.31% (24/26), Macro F1 = 0.4971, PR-AUC = 0.7723 ($T = 0.740$).
- **Active Production Model (Frozen)**: Accuracy = 71.43%, Balanced Accuracy = 51.71%, AC Recall = 11.11% (1/9), DC Recall = 92.31% (24/26), Macro F1 = 0.4971 ($T = 0.760$).

## 11. AC Signal Investigation

- **Strongest Individual Development Feature**: `road_type_residential` exhibits Cohen's $d = -1.6074$ ($p = 5.20 \times 10^{-7}$).
- **Development Distribution**: AC stations on residential roads = 35.71%, DC stations on residential roads = 1.90%.
- **Certified Scientific Interpretation**: "Empirical feature distributions show that road_type_residential has the largest observed effect size separating AC Slow from DC Fast in the development data." (Not described as causal, universally predictive, or a deployment rule).

## 12. Spatial Covariate Shift

- **Empirical Shift**: In development districts, 35.71% of AC stations were on residential roads. In quarantined outer-test districts, **100.0% of AC stations (9/9) were located on trunk highways** and **88.9% (8/9) were located at government office campuses** with high commercial POI densities (mean 32.6), physically mimicking urban DC Fast hubs.
- **Certified Scientific Finding**: "Spatial covariate shift was observed between the development and held-out districts. The tested exogenous spatial feature space was insufficient to reliably distinguish AC Slow from DC Fast under the observed held-out distribution."
- **Reconciliation Conclusion**: "Site-context enrichment did not demonstrate improved cross-district AC Slow transfer under the independent outer-test protocol."

## 13. XAI

- **Explainability Engine**: `shap.TreeExplainer` applied directly to Random Forest probability outputs.
- **Certified Interpretation**:
  > "SHAP values represent model-based feature attributions to the Random Forest's DC Fast probability output. They indicate how features contribute to the model prediction and do not establish causality, physical mechanisms, or actual charging demand/utilization."
- **Verification**: Explanations use the identical 21-feature contract without leakage variables.

## 14. Counterfactuals

- **Approved Levers**: `distance_to_highway_km`, `distance_to_city_center_km`, `nearby_poi_count`, `poi_diversity_index`.
- **Certified Terminology**: **"model-based planning scenario"** (Never described as guaranteed action, causal intervention, or guaranteed demand).

## 15. EIPS

- **Methodology**: Analytical Hierarchy Process (AHP) $\to$ TOPSIS $\to$ Electric Infrastructure Planning Score.
- **Criteria**: C1 EV Adoption (0.2565), C2 Infrastructure Gap (0.3449), C3 Corridor Importance (0.1329), C4 Grid Context (0.1329), C5 Spatial Suitability (0.1329).
- **Consistency Ratio**: $\text{CR} = 0.000891$ ($\le 0.10$).
- **Certified Statement**: "EIPS identifies districts with higher relative planning priority for investigation and potential infrastructure expansion. It does not prescribe an exact station location or exact station count."

## 16. GIS

- **Layers**: 166 station GeoJSON features (`results/gis/station_gis_layer.geojson`), 33 district GIS polygons (`results/gis/district_eips_layer.geojson`), and interactive web map (`results/gis/interactive_planning_map.html`).
- **Integrity**: Station coordinates preserved, district boundaries validated without fabricated centroids, EIPS rankings displayed separately from station inference.

## 17. Application

- **Audited Modules**: `08_predict.py` and `app.py`.
- **Inference Checkpoints**: Both load `final_deployment_model.joblib` and `final_deployment_threshold.json` (0.76). Exactly 21 predictors enforced. Zero scaler loaded in RF path. Target and equipment fields stripped before prediction.
- **Dual Inference Semantic Modes**: Audited and confirmed on `TS-MED-0001`, `TS-HYD-0003`, and `TS_SAMPLE_PROSPECTIVE_01`.

## 18. Limitations

Documented all 10 explicit research boundaries: ground-truth size (154 labeled), AC class imbalance, outer-test AC transfer limitation, spatial covariate shift, traffic density unavailability, absence of site-level electrical data, EIPS macro-planning scope, classification vs continuous demand distinction, non-causality of SHAP, and absence of mathematical Bayes error bounds.

## 19. Research Contribution

**Final Contribution Statement**:

> "The project develops an explainable, spatially aware EV charging infrastructure planning framework that combines leakage-controlled station-level machine learning, TreeSHAP explanations, constrained counterfactual reasoning, EV adoption context, infrastructure-gap analysis, grid context, AHP-TOPSIS multi-criteria synthesis, EIPS district prioritization, GIS visualization, and deployable web-based decision support."

## 20. Production Governance

- Active production artifacts remain 100% frozen, untouched, and unpromoted:
  - `models/final_deployment_model.joblib`
  - `models/final_deployment_metadata.json`
  - `models/final_deployment_threshold.json`
- SHA-256 pre-execution and post-execution hashes verified identical.
- Methodology status: **FROZEN**. Research search status: **CLOSED**.
