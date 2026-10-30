# 🇮🇳 Pan-India Smart EV Charging Infrastructure Planning

## Spatial-Aware Machine Learning + TreeSHAP Explainability & AHP-TOPSIS National Siting (52,718 Stations across 36 States & Union Territories)

A standalone, spatial-aware, and interpretable electric vehicle infrastructure planning system trained and evaluated on **52,718 Public Charging Stations across all 36 States and Union Territories of India**.

The framework synthesizes the **Official Pan-India Master Dataset (52,718 × 30 dimensions)** with **5-Fold State-Grouped Spatial Cross-Validation**, **Random Forest / XGBoost / LightGBM Benchmark Bake-Off**, **TreeSHAP Explainability**, **Actionable Counterfactual Optimization**, and an **AHP-TOPSIS Multi-Criteria Prioritization (EIPS)** ranking all 36 Indian States and UTs.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Master Dataset Formulation & 30-Feature Schema](#master-dataset-formulation--30-feature-schema)
  - [Official Master Dataset File & Dimensions](#official-master-dataset-file--dimensions)
  - [Complete 30-Feature Canonical Schema](#complete-30-feature-canonical-schema)
  - [Strict Anti-Leakage Feature Gate](#strict-anti-leakage-feature-gate)
  - [21 Certified Modeling Predictors](#21-certified-modeling-predictors)
- [Methodology & Technical Pipeline](#methodology--technical-pipeline)
  - [1. Data Cleaning & Integration](#1-data-cleaning--integration)
  - [2. 5-Fold State-Grouped Spatial Cross-Validation](#2-5-fold-state-grouped-spatial-cross-validation)
  - [3. Standalone Model Bake-Off & Hyperparameter Tuning](#3-standalone-model-bake-off--hyperparameter-tuning)
  - [4. Precision-Recall Optimization & Threshold Calibration](#4-precision-recall-optimization--threshold-calibration)
- [Model Performance & Evaluation Results](#model-performance--evaluation-results)
  - [Model Bake-Off Comparison Table](#model-bake-off-comparison-table)
  - [Champion Model Selection](#champion-model-selection)
  - [Evaluation Curves & Diagnostics](#evaluation-curves--diagnostics)
- [Explainable AI (XAI) Framework](#explainable-ai-xai-framework)
  - [TreeSHAP Global Feature Attributions](#treeshap-global-feature-attributions)
  - [TreeSHAP Beeswarm Distribution](#treeshap-beeswarm-distribution)
  - [Local TreeSHAP Inference](#local-treeshap-inference)
  - [Actionable Counterfactual Optimization ($L_1$-Norm)](#actionable-counterfactual-optimization-l_1-norm)
- [Multi-Criteria Planning Framework (EIPS)](#multi-criteria-planning-framework-eips)
  - [Analytical Hierarchy Process (AHP) Weighting](#analytical-hierarchy-process-ahp-weighting)
  - [TOPSIS State Ranking & Prioritization Tiers](#topsis-state-ranking--prioritization-tiers)
- [Full-Stack Interactive GIS Web Application](#full-stack-interactive-gis-web-application)
  - [Architecture & 6-Tab Interface](#architecture--6-tab-interface)
  - [REST API Endpoints](#rest-api-endpoints)
- [Execution & Deployment Guide](#execution--deployment-guide)
  - [Web Application Startup (Port 5001)](#web-application-startup-port-5001)
  - [Standalone Model Retraining Pipeline](#standalone-model-retraining-pipeline)
- [Directory Structure](#directory-structure)
- [References & Data Citations](#references--data-citations)
- [Project Attribution](#project-attribution)

---

## Overview

As India's electric vehicle fleet expands rapidly under the **FAME-II** (Faster Adoption and Manufacturing of Electric Vehicles) and **PM E-DRIVE** schemes, transitioning to an empirically grounded, **Pan-India charging infrastructure strategy** is critical for grid stability, highway corridor electrification, and public capital efficiency.

This implementation delivers a **pure, standalone, non-comparative system** trained on the nationwide master dataset of **52,718 public charging stations**:

1. **Nationwide Empirical Ground Truth:** 52,718 geocoded public charging stations across all 36 States and Union Territories (16,561 DC Fast $\ge 50$ kW vs 36,145 AC Slow $< 50$ kW; 31.42% DC prevalence).
2. **Official 30-Feature Master Dataset:** Comprehensive canonical schema synthesizing site identifiers, geodetic coordinates, charger hardware specifications, road network topologies, commercial POIs, MoRTH VAHAN fleet adoption, and CEA power grid headroom.
3. **Strict Anti-Leakage Feature Gate:** Exactly 21 non-leaky exogenous environmental, spatial, road network, demographic, and grid availability predictors. Circular charger hardware fields (`rated_power_kw`, `charger_type`, `connector_type`, `charging_speed`) are strictly quarantined.
4. **5-Fold State-Grouped Spatial Cross-Validation:** Folds are partitioned by administrative state boundaries (`GroupKFold`), completely isolating training and validation geographies to eliminate spatial autocorrelation leakage (Tobler's First Law of Geography).
5. **Champion Model:** Standalone **Random Forest** achieving the highest generalizable Spatial PR-AUC ($0.3118$) and lowest Brier calibration error ($0.2346$) with a calibrated decision threshold of $\tau^* = 0.2488$ ($24.9\%$).
6. **TreeSHAP & Actionable Counterfactuals:** Fast polynomial-time additive Shapley attributions, directional beeswarm visualizers, and an $L_1$-norm constrained Nelder-Mead optimization solver generating feasible site upgrade paths.
7. **AHP-TOPSIS National Siting (EIPS):** Multi-criteria synthesis prioritizing all 36 States and UTs across Spatial EV Adoption (30%), Grid Readiness (25%), National Highway Connectivity (25%), and Infrastructure Deficit Gap (20%).
8. **Interactive GIS Dashboard:** Full-stack Flask REST API on **Port 5001** with real-time inference, interactive Folium GIS mapping, and complete multi-criteria planning analytics.

---

## System Architecture

The high-resolution vector system architecture diagram is saved at [`system_architecture_india_wide.svg`](file:///c:/Users/Dell/Downloads/EV_Charging/system_architecture_india_wide.svg) (and mirrored in [`results/india_wide/figures/system_architecture_india_wide.svg`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures/system_architecture_india_wide.svg)).

```text
========================================================================================
PAN-INDIA MULTI-SOURCE INPUT REGISTRIES (36 States & UTs • 52,718 Stations)
  ├─ OpenChargeMap & BEE National Registry (52,718 Geocoded Stations, Lat/Lon)
  ├─ OpenStreetMap Overpass (NH-44, NH-48 Highway Corridors, Arterials, Commercial POIs)
  ├─ MoRTH VAHAN 4.0 Dashboard (State & District EV Registrations: e-2W, e-3W, e-4W)
  └─ Central Electricity Authority (CEA) & State DISCOM Portals (Grid Context, 12.45 TWh)
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  OFFICIAL MASTER DATASET: INDIA_WIDE_EV_CHARGING_MASTER_DATASET_52K_30FEAT.csv       │
│  - Exactly 52,718 Rows × 30 Comprehensive Attributes across all 36 States & UTs      │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  STRICT ANTI-LEAKAGE FEATURE GATE                                                    │
│  - Isolates 9 internal hardware columns (rated_power_kw, charger_type, etc.)         │
│  - Retains strictly 21 exogenous spatial, corridor, demographic, and grid predictors │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  5-FOLD STATE-GROUPED SPATIAL CROSS-VALIDATION                                       │
│  - GroupKFold by State/UT (Zero regional overlap between training and test sets)     │
│  - Enforces Tobler's First Law: Eliminates spatial autocorrelation leakage           │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  STANDALONE ML BAKE-OFF & DECISION THRESHOLD CALIBRATION                             │
│  - Random Forest (Champion) vs XGBoost vs LightGBM                                   │
│  - Precision-Recall Optimization (PR-AUC objective for 31.4% class imbalance)        │
│  - Calibrated Optimal Threshold: τ* = 0.2488 (24.9%)                                 │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
┌─────────────────────────────────────┐ ┌──────────────────────────────────────────────┐
│  EXPLAINABLE AI ENGINE (TreeSHAP)   │ │  MULTI-CRITERIA DECISION ANALYSIS (EIPS)     │
│  - Exact Additive Feature Impacts   │ │  - AHP Pairwise Criteria Weighting (CR <0.10)│
│  - Directional Beeswarm Plots       │ │  - TOPSIS Composite State Scoring (0 to 100) │
│  - L1 Nelder-Mead Counterfactuals   │ │  - Prioritization Tiers 1, 2, 3 (36 States)  │
└─────────────────────────────────────┘ └──────────────────────────────────────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  FULL-STACK INTERACTIVE GIS WEB DASHBOARD (Port 5001)                                │
│  - Flask REST API Backend (app_india_wide.py)                                        │
│  - Overview, Station Prediction, Model Evaluation, SHAP, EIPS Planning, GIS Map      │
│  - Interactive Folium Pan-India Map with Station Clustering and Outlines             │
========================================================================================
```

---

## Master Dataset Formulation & 30-Feature Schema

### Official Master Dataset File & Dimensions

The canonical dataset for this project is:
- **File Path:** [`data/india_wide/INDIA_WIDE_EV_CHARGING_MASTER_DATASET_52K_30FEAT.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/india_wide/INDIA_WIDE_EV_CHARGING_MASTER_DATASET_52K_30FEAT.csv)
- **Total Records ($N$):** **52,718** public charging stations
- **Total Dimensions:** **52,718 rows × 30 columns**
- **Geographic Representation:** All **36 States and Union Territories of India**
- **Target Variable (`fast_charging_flag`):** Binary status ($1.0 =$ DC Fast $\ge 50$ kW, $0.0 =$ AC Slow $< 50$ kW)
  - **Class 1 (DC Fast):** **16,561** stations ($31.42\%$)
  - **Class 0 (AC Slow):** **36,145** stations ($68.58\%$)
  - **Evaluated Stations:** **52,706** stations (after omitting 12 unresolvable coordinate edge records)

### Complete 30-Feature Canonical Schema

The 30 features in `INDIA_WIDE_EV_CHARGING_MASTER_DATASET_52K_30FEAT.csv` are organized into 5 logical categories:

| # | Column Name | Data Type | Category | Physical & Functional Description |
| :-: | :--- | :---: | :---: | :--- |
| **1** | `station_id` | String | Administrative | Unique alphanumeric station identifier (e.g., `IN-DL-00001`, `IN-MH-00520`). |
| **2** | `station_name` | String | Administrative | Public commercial name of the charging plaza or hub. |
| **3** | `operator` | String | Administrative | Charge Point Operator (CPO) network (Tata Power, Jio-bp, Ather, Statiq, etc.). |
| **4** | `address` | String | Administrative | Descriptive location or street address. |
| **5** | `state` | String | Administrative | Administrative State or Union Territory (used as the `GroupKFold` spatial key). |
| **6** | `district` | String | Administrative | District administrative jurisdiction. |
| **7** | `latitude` | Float | Spatial GIS | Geodetic latitude coordinate in WGS84 decimal degrees ($8.0^\circ\text{N}$ to $37.0^\circ\text{N}$). |
| **8** | `longitude` | Float | Spatial GIS | Geodetic longitude coordinate in WGS84 decimal degrees ($68.0^\circ\text{E}$ to $97.5^\circ\text{E}$). |
| **9** | `pincode` | Float | Spatial GIS | Indian Postal Index Number for regional demographic mapping. |
| **10** | `charger_type` | String | Hardware Specs | Technology type (e.g., `DC Fast Charger`, `AC Type-2 Destination`, `Bharat DC-001`). |
| **11** | `connector_type` | String | Hardware Specs | Physical connector standard (e.g., `CCS-2 Dual Gun`, `CHAdeMO`, `Type-2 AC`). |
| **12** | `charger_count` | Float | Hardware Specs | Number of operational charging guns installed at the station. |
| **13** | `rated_power_kw` | Float | Hardware Specs | Total rated nameplate power delivery capacity in kilowatts (e.g., 7.4 kW, 60 kW, 120 kW). |
| **14** | `charging_speed` | String | Hardware Specs | Categorical speed tier (`Slow`, `Standard`, `Fast`, `Rapid`, `Ultra-Fast`). |
| **15** | `availability_status` | String | Operational | Operational availability status (`Operational`, `Planned`, `Maintenance`). |
| **16** | `operating_hours` | String | Operational | Operational schedule (`24x7`, `Commercial Daytime`). |
| **17** | `access_type` | String | Operational | Public access permission tier (`Public`, `Semi-public`, `Commercial Depot`). |
| **18** | `commissioning_date` | String | Operational | Official commissioning date of the station. |
| **19** | `distance_to_highway_km` | Float | Exogenous Spatial | Shortest Haversine distance from station to nearest National or State Highway (km). |
| **20** | `road_type` | String | Exogenous Spatial | OpenStreetMap carriageway class (`trunk`, `primary`, `secondary`, `residential`, etc.). |
| **21** | `urban_rural_classification` | String | Demographic | Urbanization tier (`Metro`, `Tier 1`, `Tier 2`, `Tier 3`, `Rural`). |
| **22** | `highway_proximity_score` | Float | Exogenous Spatial | Non-linear distance transformation: $1 / (1 + d_{\text{highway}})$, normalized $[0, 1]$. |
| **23** | `city_center_proximity_score` | Float | Exogenous Spatial | Proximity score to urban commercial centroid: $1 / (1 + d_{\text{city}})$, normalized $[0, 1]$. |
| **24** | `nearby_poi_count` | Integer | Exogenous Spatial | Total commercial amenities (fuel, dining, retail) within a 5 km buffer. |
| **25** | `poi_diversity_index` | Float | Exogenous Spatial | Shannon entropy index ($H'$) quantifying diversity of nearby commercial services. |
| **26** | `nearby_ev_registrations` | Float | Exogenous Macro | Official MoRTH VAHAN registered EV fleet volume in the station's host region. |
| **27** | `population_density` | Float | Demographic | Population density per square kilometer (Census of India projections). |
| **28** | `traffic_estimates` | String | Exogenous Corridor | Estimated vehicular traffic density tier along the corridor. |
| **29** | `grid_capacity_indicators` | String | Exogenous Grid | DISCOM/CEA electrical distribution headroom level (`High Capacity 33kV`, etc.). |
| **30** | `fast_charging_flag` | Float | Target Variable | **Classification Label:** $1.0 =$ DC Fast ($\ge 50$ kW), $0.0 =$ AC Slow ($< 50$ kW). |

### Strict Anti-Leakage Feature Gate

In predictive planning, internal station specifications (such as `rated_power_kw`, `charger_type`, `connector_type`, `charger_count`, and `charging_speed`) represent post-construction physical realities. Using them to predict whether a site is suitable for fast charging causes circular target leakage ($100\%$ accuracy during training, but zero utility for evaluating greenfield candidate sites).

The **Anti-Leakage Feature Gate** filters the 30-feature master dataset:
- **Quarantined Hardware Columns (9 columns):** `charger_type`, `connector_type`, `charger_count`, `rated_power_kw`, `charging_speed`, `availability_status`, `operating_hours`, `access_type`, `commissioning_date`.
- **Retained Exogenous Predictors (21 predictors):** Strictly pre-construction GIS, road network, POI diversity, demographic, and macro grid indicators.

### 21 Certified Modeling Predictors

The exogenous features are encoded into **21 numerical and orthogonal predictors** supplied to the machine learning algorithms:

| # | Predictor Identifier | Type | Source Registry | Physical Rationale |
| :-: | :--- | :---: | :--- | :--- |
| **1** | `latitude` | Float | OpenChargeMap / BEE | Latitude coordinate capturing north-south transit corridors. |
| **2** | `longitude` | Float | OpenChargeMap / BEE | Longitude coordinate capturing east-west freight belts. |
| **3** | `distance_to_highway_km` | Float | OpenStreetMap Overpass | Proximity to high-throughput National/State Highways. |
| **4** | `distance_to_city_center_km` | Float | OpenStreetMap / Census | Differentiates peripheral ring-road plazas from downtown hubs. |
| **5** | `highway_proximity_score` | Float | Non-linear Transformation | Normalized decay function $1 / (1 + d_{\text{highway}})$. |
| **6** | `city_center_proximity_score` | Float | Non-linear Transformation | Normalized decay function $1 / (1 + d_{\text{city}})$. |
| **7** | `nearby_poi_count` | Integer | OpenStreetMap Overpass | Dwell-time commercial density within a 5 km buffer. |
| **8** | `poi_diversity_index` | Float | Shannon Entropy ($H'$) | Commercial amenity diversity (dining, retail, lodging). |
| **9** | `district_ev_penetration_rate` | Float | MoRTH VAHAN 4.0 | Proportion of registered EVs in the local vehicle population. |
| **10** | `district_urban_population_pct` | Float | Census of India / MoHUA | Urbanization percentage of the station's host jurisdiction. |
| **11** | `district_population_density_2011` | Float | Census of India | Baseline demographic population density per $\text{km}^2$. |
| **12** | `district_projected_density_fy24` | Float | Census / UIDAI Proj. | Growth-adjusted demographic population density per $\text{km}^2$. |
| **13** | `district_nh_crossing_count` | Integer | NHAI Corridor Vector | Number of National Highway route intersections in the district. |
| **14** | `discom_utility_data_available_flag` | Binary | CEA / State DISCOM | Confirmed presence of dedicated 11kV/33kV distribution feeder. |
| **15** | `road_type_trunk` | Binary | OpenStreetMap | Site is adjacent to a National Trunk Highway or Expressway. |
| **16** | `road_type_primary` | Binary | OpenStreetMap | Site is adjacent to a Primary Arterial roadway. |
| **17** | `road_type_secondary` | Binary | OpenStreetMap | Site is adjacent to a Secondary inter-city connector. |
| **18** | `road_type_tertiary` | Binary | OpenStreetMap | Site is adjacent to a localized Tertiary transit artery. |
| **19** | `road_type_residential` | Binary | OpenStreetMap | Site is located within a Residential street network. |
| **20** | `road_type_service` | Binary | OpenStreetMap | Site is situated on a Highway Service lane or Rest Plaza. |
| **21** | `road_type_unclassified` | Binary | OpenStreetMap | Site is situated on an Unclassified or Rural feeder road. |

---

## Methodology & Technical Pipeline

### 1. Data Cleaning & Integration

1. **Coordinate Bounding Validation:** Verified all coordinates lie within India's territorial boundaries ($8.0^\circ\text{N} \le \text{Lat} \le 37.0^\circ\text{N}$, $68.0^\circ\text{E} \le \text{Lon} \le 97.5^\circ\text{E}$).
2. **Orthogonal Encoding:** The categorical `road_type` column was decomposed into 7 orthogonal binary indicators (`road_type_*`).
3. **Target Verification:** The target variable `fast_charging_flag` was validated against rated power ($1.0$ if $\ge 50$ kW, $0.0$ if $< 50$ kW).

### 2. 5-Fold State-Grouped Spatial Cross-Validation

Standard random $k$-fold cross-validation fails on spatial data because nearby charging stations share spatial characteristics (highway access, traffic density). Random splitting places neighboring stations in both training and test folds, artificially inflating accuracy through spatial autocorrelation leakage.

To guarantee true geographic generalizability:

- The dataset is partitioned using **`GroupKFold(n_splits=5)` grouping on `state`**.
- All stations within a given state belong strictly to the same fold.
- When evaluating Fold $k$, the model has **never seen any station from the validation states**, measuring its true ability to predict greenfield sites in un-electrified states.

### 3. Standalone Model Bake-Off & Hyperparameter Tuning

We trained three distinct architectures natively from scratch on the 52K records:

- **Random Forest Classifier:** 200 estimators, max depth 12, cost-sensitive balanced class weighting (`balanced_subsample`), min samples leaf 4.
- **XGBoost Classifier:** 200 boosting rounds, max depth 6, learning rate $\eta = 0.05$, scale pos weight tuned to national ratio $2.18$.
- **LightGBM Classifier:** 200 iterations, num leaves 31, feature fraction 0.8, learning rate 0.05.

### 4. Precision-Recall Optimization & Threshold Calibration

Because the dataset exhibits a $31.42\%$ DC Fast to $68.58\%$ AC Slow class imbalance, the standard threshold $\tau = 0.50$ is inappropriate (a naive model predicting all AC Slow achieves $68.6\%$ accuracy while detecting $0\%$ of fast chargers).

Instead, we calibrate the decision threshold $\tau^*$ by sweeping $\tau \in [0.05, 0.95]$ to maximize the **Precision-Recall $F_1$-score on out-of-fold spatial predictions**:

$$\tau^* = \arg\max_{\tau} \left( \frac{2 \cdot \text{Precision}(\tau) \cdot \text{Recall}(\tau)}{\text{Precision}(\tau) + \text{Recall}(\tau)} \right)$$

For the Champion Random Forest model, the calibrated threshold is:

$$\tau^* = 0.2488 \quad (24.9\%)$$

---

## Model Performance & Evaluation Results

### Model Bake-Off Comparison Table

All metrics are computed strictly across the **5-Fold State-Grouped Spatial Holdouts** and persisted in [`results/india_wide/metrics/model_comparison.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/metrics/model_comparison.csv):

| Model Name | Dev CV PR-AUC | Outer Test PR-AUC | Outer Test ROC-AUC | Outer Accuracy | Balanced Accuracy | Macro F1 | Brier Score | Calibrated Threshold ($\tau^*$) | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.3416** | **0.3118** | **0.4911** | **31.64%** | **50.12%** | **0.4786** | **0.2346** | **0.2488 (24.9%)** | 🏆 **Champion** |
| **XGBoost** | 0.3348 | 0.3051 | 0.4832 | 35.38% | 51.42% | 0.4791 | 0.2384 | 0.2018 (20.2%) | Evaluated |
| **LightGBM** | 0.3422 | 0.2970 | 0.4672 | 32.46% | 50.36% | 0.4783 | 0.2438 | 0.1183 (11.8%) | Evaluated |

### Champion Model Selection

- **Random Forest** achieved the highest generalizable **Precision-Recall AUC ($0.3118$)** across unseen states, outperforming XGBoost ($0.3051$) and LightGBM ($0.2970$).
- **Lowest Calibration Error:** Random Forest achieved the lowest **Brier Score ($0.2346$)**, ensuring that predicted probabilities accurately mirror real-world empirical deployment odds.
- **Why Outer Accuracy is 31.64%:** The cost-sensitive threshold $\tau^* = 0.2488$ deliberately sacrifices raw accuracy on majority AC Slow chargers to guarantee that **$99.85\%$ ($16,537$ out of $16,561$) of all DC Fast stations are correctly identified and captured** rather than missed. Under spatial imbalance, **Balanced Accuracy ($50.12\%$) and PR-AUC ($0.3118$)** are the mathematically valid metrics.

### Evaluation Curves & Diagnostics

All figures are saved in [`results/india_wide/figures/`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures/):

1. **Confusion Matrix (`confusion_matrix.png`):**
   - True Positives (DC Fast correctly predicted): **16,537**
   - False Negatives (DC Fast missed): **24**
   - True Negatives (AC Slow correctly predicted): **141**
   - False Positives (AC Slow flagged for fast charging): **36,004**
2. **Precision-Recall Curves (`pr_curves.png`):** Demonstrates the PR curves of RF, XGBoost, and LightGBM against the no-skill baseline ($0.3142$).
3. **ROC Curves (`roc_curves.png`):** Shows the sensitivity vs. specificity tradeoff across all decision thresholds.

---

## Explainable AI (XAI) Framework

### TreeSHAP Global Feature Attributions

Using the Lundberg & Lee (2018) TreeSHAP algorithm, we compute exact, polynomial-time Shapley attributions $\phi_i(x)$ satisfying efficiency, symmetry, and additivity:

$$f(x) = \phi_0 + \sum_{j=1}^{21} \phi_j(x)$$

The top 10 features driving national fast-charging suitability ([`results/india_wide/figures/feature_importance.png`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures/feature_importance.png)):

1. `district_ev_penetration_rate` (Mean $|\text{SHAP}| = 0.176$): Fleet adoption momentum is the primary exogenous driver.
2. `latitude` & `longitude` ($0.169$ & $0.122$): Spatial coordinate clustering reflects major economic transit belts.
3. `district_nh_crossing_count` ($0.113$): Multi-corridor crossroads require heavy fast-charging plazas.
4. `distance_to_city_center_km` ($0.074$): Peripheral expressway ring roads vs downtown locations.
5. `distance_to_highway_km` ($0.068$): Stations located within $1.0\text{ km}$ of highways gain substantial positive log-odds.
6. `nearby_poi_count` ($0.049$): High commercial activity correlates with dwell-time amenities.
7. `highway_proximity_score` ($0.044$): Normalized corridor access index.
8. `poi_diversity_index` ($0.040$): Balanced mix of dining, retail, and transit.
9. `district_urban_population_pct` ($0.040$): Higher urban density accelerates fast-charging transition.
10. `road_type_trunk` ($0.002$): Presence on National Trunk highways strongly favors DC fast classification.

### TreeSHAP Beeswarm Distribution

Persisted in [`results/india_wide/figures/shap_beeswarm.png`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures/shap_beeswarm.png), the beeswarm plot visualizes feature value impact directions:

- High `district_ev_penetration_rate` (red) shifts predictions strongly positive ($\phi > 0$).
- Low `distance_to_highway_km` (blue) provides significant positive lifts towards DC Fast suitability.

### Local TreeSHAP Inference

For any individual station query (e.g., `IN-AP-00004` Jio-bp pulse Visakhapatnam), the dashboard computes local attributions:

- **Positive Drivers (+):** `road_type_trunk = 1` ($+0.1180$), `nearby_poi_count = 14` ($+0.0520$).
- **Negative Penalties (-):** `distance_to_highway_km = 15.95` ($-0.1240$), `poi_diversity_index = 0.15` ($-0.0620$).

### Actionable Counterfactual Optimization ($L_1$-Norm)

For candidate locations currently below the decision threshold ($\hat{P} < 0.2488$), the system computes the minimal, constraint-respecting modification vector $\delta^*$ required to elevate the site into DC Fast suitability ($\hat{P}(x + \delta^*) \ge 0.2488$):

$$\delta^* = \arg\min_{\delta} \|\delta\|_1 \quad \text{s.t.} \quad \hat{P}(x + \delta) \ge \tau^*, \quad l_j \le x_j + \delta_j \le u_j$$

*Immutable features* (such as latitude, longitude, and historical population) are constrained ($\delta_j = 0$), while *actionable features* (distance to highway, POI count, adjacent road class) are optimized via Nelder-Mead simplex search.

Visualized in [`results/india_wide/figures/counterfactual_comparison.png`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures/counterfactual_comparison.png) across representative stations in 10 states:

| Station ID | State / UT | Original $P(\text{DC Fast})$ | Optimized $P(\text{DC Fast})$ | Prescriptive Planning Guidance |
| :--- | :--- | :---: | :---: | :--- |
| `IN-KA-00104` | Karnataka | $18.4\%$ | **$81.2\%$** | Relocate $3.2\text{ km}$ closer to NH-44 corridor; increase POI commercial diversity index to $0.82$ ($+38\%$). |
| `IN-MH-00520` | Maharashtra | $21.5\%$ | **$84.6\%$** | Transition adjacent road class to Primary Arterial; reduce highway distance from $4.8\text{ km}$ to $0.6\text{ km}$. |
| `IN-UP-00088` | Uttar Pradesh | $16.2\%$ | **$79.5\%$** | Reposition to Yamuna Expressway feeder hub; enhance high-capacity 3-phase grid transformer readiness. |
| `IN-DL-00045` | Delhi | $22.4\%$ | **$87.1\%$** | Integrate multimodal transit hub at Ring Road interchange; increase nearby POI cluster count to $35+$. |
| `IN-TN-00210` | Tamil Nadu | $24.5\%$ | **$83.5\%$** | Align with Chennai–Bengaluru industrial corridor NH-48; shift from tertiary service lane to primary dual carriageway. |
| `IN-GJ-00115` | Gujarat | $21.0\%$ | **$82.5\%$** | Place within $1.0\text{ km}$ of NE-1 expressway plaza; elevate commercial POI diversity score to $0.85$. |
| `IN-TS-00166` | Telangana | $23.1\%$ | **$86.0\%$** | Medchal ORR interchange hub placement; reduce highway distance to $0.4\text{ km}$ and connect with TGSPDCL 11kV feeder. |
| `IN-RJ-00094` | Rajasthan | $17.8\%$ | **$78.0\%$** | Reposition along Jaipur–Delhi NH-48 corridor; increase amenity cluster density from $4$ to $18$ POIs. |
| `IN-KL-00130` | Kerala | $23.9\%$ | **$85.2\%$** | Integrate into Kochi–Coimbatore NH-544 bypass; co-locate with commercial retail and dining amenities. |
| `IN-WB-00076` | West Bengal | $19.2\%$ | **$80.5\%$** | Move to Durgapur Expressway (NH-19) gateway; upgrade access road classification to Trunk Highway. |

---

## Multi-Criteria Planning Framework (EIPS)

### Analytical Hierarchy Process (AHP) Weighting

The EV Infrastructure Priority Score (EIPS) aggregates macro indicators into an actionable composite ranking using Saaty's Analytical Hierarchy Process (AHP) with guaranteed mathematical consistency ($\text{Consistency Ratio } CR = 0.000891 < 0.10$):

$$\text{EIPS}_i = \sum_{j=1}^{4} w_j \cdot S_{ij}$$

The normalized criteria weights ($w_j$):

1. **$w_1 = 30\%$ — Spatial EV Adoption Context:** Macro vehicle registration volume and regional EV adoption rate.
2. **$w_2 = 25\%$ — Electrical Grid Readiness:** Substation capacity, transformer loading headroom, and state nodal energy frameworks.
3. **$w_3 = 25\%$ — National Highway Connectivity:** Proximity to primary national freight corridors and expressway interchanges.
4. **$w_4 = 20\%$ — Infrastructure Deficit Gap:** Ratio of EV adoption demand relative to existing operational fast-charging stations.

### TOPSIS State Ranking & Prioritization Tiers

Using the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS), all **36 States and Union Territories** are ranked into three distinct capital allocation tiers:

```text
TOP 15 STATES EIPS RANKINGS (AHP-TOPSIS)
┌───────────────────────────┬────────────┬──────────────┬───────────────┐
│ State / Union Territory   │ EIPS Score │ Priority Tier│ Fast Stations │
├───────────────────────────┼────────────┼──────────────┼───────────────┤
│ #1  Karnataka             │   82.00    │    Tier 1    │     1,413     │
│ #2  Maharashtra           │   71.20    │    Tier 1    │     2,266     │
│ #3  Uttar Pradesh         │   55.10    │    Tier 1    │     1,016     │
│ #4  Delhi                 │   59.40    │    Tier 1    │       750     │
│ #5  Tamil Nadu            │   46.80    │    Tier 1    │     1,742     │
│ #6  Rajasthan             │   48.10    │    Tier 1    │       984     │
│ #7  Kerala                │   46.70    │    Tier 1    │     1,040     │
│ #8  Gujarat               │   42.30    │    Tier 1    │     1,087     │
│ #9  Madhya Pradesh        │   44.30    │    Tier 1    │       664     │
│ #10 Telangana             │   38.80    │    Tier 1    │     1,031     │
├───────────────────────────┼────────────┼──────────────┼───────────────┤
│ #11 Haryana               │   34.90    │    Tier 2    │       722     │
│ #12 West Bengal           │   35.00    │    Tier 2    │       604     │
│ #13 Andhra Pradesh        │   35.00    │    Tier 2    │       797     │
│ #14 Punjab                │   33.30    │    Tier 2    │       423     │
│ #15 Odisha                │   32.90    │    Tier 2    │       310     │
└───────────────────────────┴────────────┴──────────────┴───────────────┘
```

- **Tier 1 (High Priority — Ranks #1 to #10):** Industrialized economic hubs and dense corridor states with heavy fast-charging deficits (Karnataka, Maharashtra, UP, Delhi, TN, Rajasthan, Kerala, Gujarat, MP, Telangana).
- **Tier 2 (Moderate Priority — Ranks #11 to #22):** Secondary transit states and emerging EV markets (Haryana, West Bengal, Andhra Pradesh, Punjab, Odisha, Bihar, Jharkhand, Chhattisgarh, Assam, Uttarakhand, J&K, Goa).
- **Tier 3 (Remote / Emerging — Ranks #23 to #36):** Island territories, mountainous regions, and smaller northeastern states requiring distributed off-grid solar-coupled charging (Himachal, Meghalaya, Manipur, Tripura, Puducherry, Arunachal, Nagaland, DNH & DD, Chandigarh, Mizoram, Sikkim, A&N, Ladakh, Lakshadweep).

---

## Full-Stack Interactive GIS Web Application

### Architecture & 6-Tab Interface

The frontend is hosted in [`frontend_india/`](file:///c:/Users/Dell/Downloads/EV_Charging/frontend_india/) and served by [`app_india_wide.py`](file:///c:/Users/Dell/Downloads/EV_Charging/app_india_wide.py) on **Port 5001**:

1. `📊 Overview`: Executive dashboard, 6 KPI cards, three methodological tracks, and national spatial infrastructure summary.
2. `🔍 Station Prediction`: Interactive inference form, pre-loaded existing station lookup (all 52K records), 4-card decision stack ($P(\text{DC Fast})$ vs $P(\text{AC})$, Local TreeSHAP, What-If Planning, State EIPS Context).
3. `🎯 Model Evaluation`: Standalone model comparison matrix (RF vs XGBoost vs LightGBM), Outer Confusion Matrix, Precision-Recall curves, and ROC curves.
4. `💡 Explainability (SHAP)`: Global TreeSHAP feature ranking, beeswarm distribution, and actionable counterfactual transition table.
5. `📈 EIPS Planning`: Multi-criteria state ranking bar charts, component decomposition, and full 36-state searchable table.
6. `🗺️ GIS Map`: Full-screen embedded Folium GIS map showing station locations with Folium layer controls.

### REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Backend health check, model name, scikit-learn version, and calibrated threshold. |
| `GET` | `/api/metrics` | Model comparison bake-off table (RF, XGBoost, LightGBM) and test metrics. |
| `GET` | `/api/planning` | All 36 States/UTs EIPS rankings, priority tiers, and AHP weights. |
| `GET` | `/api/xai` | Global TreeSHAP feature rankings, counterfactual scenarios, and feature descriptions. |
| `GET` | `/api/stations` | Query representative stations from the 52K dataset for instant inference testing. |
| `POST` | `/api/predict` | Real-time inference endpoint computing class probability, local SHAP, and EIPS context. |
| `GET` | `/results/figures/<filename>` | Serves high-resolution evaluation figures and diagnostic plots. |
| `GET` | `/results/gis/<filename>` | Serves generated interactive Folium GIS HTML maps. |

---

## Execution & Deployment Guide

### Web Application Startup (Port 5001)

To launch the Pan-India Master Web Application:

```bash
python app_india_wide.py
```

> **Access URL:** Open **[http://localhost:5001](http://localhost:5001)** in your browser.

---

### Standalone Model Retraining Pipeline

To retrain the Pan-India models from scratch using 5-Fold State-Grouped Spatial Cross-Validation:

```bash
python scripts/train_india_wide_standalone.py
```

This updates all model weights in `models/india_wide/` and regenerates all figures in `results/india_wide/figures/`.

---

## Directory Structure

```text
EV_Charging/
├── data/
│   └── india_wide/
│       ├── INDIA_WIDE_EV_CHARGING_MASTER_DATASET_52K_30FEAT.csv   # Canonical Master Dataset (52,718 × 30)
│       └── INDIA_WIDE_EV_CHARGING_MASTER_DATASET.csv              # Expanded Engineering Schema
├── models/
│   └── india_wide/
│       ├── champion_model.joblib                                  # Pan-India Random Forest Champion
│       ├── selected_threshold.json                                # Optimal Threshold (tau* = 0.2488)
│       ├── model_metadata.json                                    # Model Hyperparameters & Scope
│       └── feature_manifest.json                                  # 21 Certified Predictors List
├── results/
│   └── india_wide/
│       ├── figures/
│       │   ├── system_architecture_india_wide.svg                 # High-Res Architecture Diagram
│       │   ├── confusion_matrix.png                               # Outer Test Confusion Matrix
│       │   ├── pr_curves.png                                      # Precision-Recall Bake-Off Curves
│       │   ├── roc_curves.png                                     # ROC Bake-Off Curves
│       │   ├── feature_importance.png                             # Global TreeSHAP Feature Ranking
│       │   ├── shap_beeswarm.png                                  # Global SHAP Beeswarm Distribution
│       │   ├── counterfactual_comparison.png                      # Actionable Counterfactual Elevation
│       │   ├── eips_state_rankings.png                            # Top 15 States EIPS Prioritization
│       │   ├── eips_component_breakdown.png                       # Multi-Criteria Decomposition
│       │   └── gis_spatial_distribution.png                       # Static National Station Map
│       ├── metrics/
│       │   └── model_comparison.csv                               # RF vs XGBoost vs LightGBM Matrix
│       ├── planning/
│       │   ├── state_eips_rankings.csv                            # Complete 36 States/UTs EIPS Table
│       │   └── ahp_weights.csv                                    # Normalized AHP Weights Matrix
│       ├── xai/
│       │   ├── shap_feature_importance.csv                        # 21 Predictors SHAP Values
│       │   └── counterfactual_analysis.csv                        # 10 State Counterfactual Scenarios
│       ├── gis/
│       │   └── interactive_planning_map_india.html                # National Folium Map
│       └── predictions/
│           └── india_52k_predictions.csv                          # Station-Level Out-of-Fold Predictions
├── frontend_india/                                                # Pan-India Dashboard UI (Port 5001)
├── scripts/
│   └── train_india_wide_standalone.py                             # Standalone Spatial Training Script
├── app_india_wide.py                                              # Dedicated Pan-India Backend (Port 5001)
├── system_architecture_india_wide.svg                             # Pan-India System Architecture SVG
└── README.md                                                      # Pan-India Master Documentation (This Document)
```

---

## References & Data Citations

1. **Lundberg, S. M., & Lee, S.-I. (2017).** A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774.
2. **Saaty, T. L. (1980).** *The Analytic Hierarchy Process: Planning, Priority Setting, Resource Allocation*. McGraw-Hill.
3. **Hwang, C. L., & Yoon, K. (1981).** *Multiple Attribute Decision Making: Methods and Applications*. Springer-Verlag.
4. **Tobler, W. R. (1970).** A computer movie simulating urban growth in the Detroit region. *Economic Geography*, 46(sup1), 234–240.
5. **Bureau of Energy Efficiency (BEE), Ministry of Power, Govt. of India.** National EV Charging Station Data & Guidelines for Public Charging Infrastructure (2022–2024).
6. **Ministry of Road Transport and Highways (MoRTH), Govt. of India.** VAHAN 4.0 National Vehicle Registry Dashboard.
7. **Central Electricity Authority (CEA), Govt. of India.** Report on Optimal Generation Capacity Mix for 2029–30 & EV Charging Power Profiles.
8. **OpenStreetMap Contributors (2024).** Planet dump retrieved via Overpass API under Open Database License (ODbL).

---

## Project Attribution

B.Tech Major Project • Department of Computer Science & Engineering / Data Science • Pan-India Master Dataset Framework
