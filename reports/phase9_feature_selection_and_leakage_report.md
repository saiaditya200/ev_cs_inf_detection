# Phase 9 / 9.1: Feature Selection and Data Leakage Audit Report

**Project Title:** Explainable AI Framework for Smart EV Charging Infrastructure Planning  
**Phase Status:** COMPLETE & CERTIFIED  
**Date:** 2026-09-26  
**Supervised Modeling Matrix:** `fast_charging_modeling_matrix.csv` (SHA-256: `fc32c75635fe2e763ec174e491a897adc691539fba0be655f06b9d58bca79783`)  
**Feature Manifest:** `phase9_feature_manifest.csv` (SHA-256: `030054b58d24d63011407eeb90ee44a151e816295f0a7795a7df07af25d874d6`)  

---

## 1. Executive Summary & X Matrix Dimensionality

Following Phase 9.1 feature representation correction, the supervised predictor matrix $X$ has an **exact dimensionality of 21 certified predictors** across $N = 154$ ground-truth labeled stations. Redundant categorical representations (`road_type` raw string and `road_type_code`) were removed from $X$, leaving the seven one-hot dummy variables as the sole model representation of roadway accessibility.

## 2. Definitive 21-Predictor Specification

| # | Predictor Name | Feature Group | Dtype | Range | Siting Rationale |
|---|---|---|---|---|---|
| 1 | `latitude` | Spatial / GIS | `float64` | [15.89, 19.65] | Certified exogenous predictor |
| 2 | `longitude` | Spatial / GIS | `float64` | [77.50, 80.89] | Certified exogenous predictor |
| 3 | `distance_to_highway_km` | Spatial / GIS | `float64` | [0.00, 23.15] | Certified exogenous predictor |
| 4 | `distance_to_city_center_km` | Spatial / GIS | `float64` | [0.03, 72.31] | Certified exogenous predictor |
| 5 | `nearby_poi_count` | Spatial / GIS | `int64` | [0, 38] | Certified exogenous predictor |
| 6 | `highway_proximity_score` | Spatial / GIS | `float64` | [0.04, 1.00] | Certified exogenous predictor |
| 7 | `city_center_proximity_score` | Spatial / GIS | `float64` | [0.01, 0.97] | Certified exogenous predictor |
| 8 | `poi_diversity_index` | Spatial / GIS | `float64` | [0.00, 0.82] | Certified exogenous predictor |
| 9 | `road_type_motorway` | Road Accessibility (One-Hot) | `int32` | [0, 1] | Certified exogenous predictor |
| 10 | `road_type_primary` | Road Accessibility (One-Hot) | `int32` | [0, 1] | Certified exogenous predictor |
| 11 | `road_type_residential` | Road Accessibility (One-Hot) | `int32` | [0, 1] | Certified exogenous predictor |
| 12 | `road_type_secondary` | Road Accessibility (One-Hot) | `int32` | [0, 1] | Certified exogenous predictor |
| 13 | `road_type_service` | Road Accessibility (One-Hot) | `int32` | [0, 1] | Certified exogenous predictor |
| 14 | `road_type_trunk` | Road Accessibility (One-Hot) | `int32` | [0, 1] | Certified exogenous predictor |
| 15 | `road_type_unclassified` | Road Accessibility (One-Hot) | `int32` | [0, 1] | Certified exogenous predictor |
| 16 | `district_population_density_2011` | District Demographic / Context | `int64` | [81, 18172] | Certified exogenous predictor |
| 17 | `district_urban_population_pct` | District Demographic / Context | `float64` | [6.80, 100.00] | Certified exogenous predictor |
| 18 | `district_projected_density_fy24` | District Demographic / Context | `float64` | [88.13, 19767.20] | Certified exogenous predictor |
| 19 | `district_nh_crossing_count` | District Demographic / Context | `int64` | [0, 5] | Certified exogenous predictor |
| 20 | `district_ev_penetration_rate` | District Demographic / Context | `float64` | [0.98, 5.05] | Certified exogenous predictor |
| 21 | `discom_utility_data_available_flag` | District Demographic / Context | `int64` | [0, 1] | Certified exogenous predictor |

---

## 3. Provenance Clarification for `district_ev_penetration_rate`

> **CRITICAL METHODOLOGICAL DISTINCTION:**
> `district_ev_penetration_rate` is a **macro district-level contextual EV adoption indicator** derived as:  
> $$\text{district\_ev\_penetration\_rate} = \left( \frac{\text{district\_cumulative\_evs\_fy24}}{\text{district\_projected\_population\_fy24}} \right) \times 1000$$
>
> - It is **not** a station-level observed measurement.
> - It is **not** station charging demand or energy throughput.
> - It must **not** be interpreted as direct charger occupancy or utilization telemetry.
> - Its provenance derives from official Telangana state VAHAN registration totals scaled by district economic/vehicle weights and normalized by projected Census population. It enters $X$ strictly as macro adoption context.

---

## 4. Exhaustive Leakage Exclusions

| Excluded Field | Leakage Category | Detailed Rationale for Exclusion |
|---|---|---|
| `charging_power_kw` | **Direct Identity Leakage** | Threshold $\ge 50$ kW is the mathematical definition of fast_charging_flag. Would yield trivial 100% accuracy. |
| `charger_type` | **Direct Identity Leakage** | `DC_FAST` is the ground-truth target definition. Direct identity duplicate. |
| `station_capacity` | **Direct Identity Leakage** | Product of power and chargers; directly embeds power rating. |
| `connector_type` | **Direct Identity Leakage** | Physical coupling like `CCS (Type 2)` uniquely implies DC Fast charging. |
| `number_of_chargers` | **Post-Outcome Engineering** | Selected during site construction, not prior to location siting. |
| `operator` | **Post-Outcome Engineering** | CPO selected after site approval. |
| `district_fast_stations` | **Target-Contaminated Aggregate** | District sum of `fast_charging_flag`; embeds the current station's outcome. |
| `district_fast_charger_ratio` | **Target-Contaminated Aggregate** | Ratio of fast chargers in district; directly embeds target outcome. |
| `district_total_stations` | **Current-Station Aggregate** | Aggregates all operating stations in district, embedding station's presence. |
| `district_total_capacity_kw` | **Current-Station Aggregate** | Sum of station capacities in district; embeds current station capacity. |
| `district_station_density_*` | **Current-Station Aggregate** | Linear transform of total stations. |
| `district_ev_to_charger_ratio` | **Current-Station Aggregate** | Contains district total stations in denominator. |
| `station_id`, `district` | **Metadata / CV Index** | Retained in CSV for fold tracking, but strictly excluded from predictor matrix $X$. |
| `station_name`, `city` | **Non-Generalizable Text** | Site titles and municipality names; non-generalizable to coordinates. |
| `state`, `state_key` | **Zero-Variance Constants** | Constant 'Telangana' across all records. |
| `availability_status` | **Zero-Variance Constant** | 100% 'OPERATIONAL' across all records. |
| `traffic_density` | **100% Missingness** | No physical sensor telemetry available; un-synthesized. |
| `pincode` | **High Missingness (92.2%)** | Spatial coordinates fully capture physical geography. |
| `distance_to_major_road_km`, `poi_*_count` | **High Missingness (84.9%)** | Phase 4 baseline only (25 stations); universal spatial context captured by master POI count. |
| `data_timestamp`, `source_registry*`, `osm_id*` | **Engineering Provenance** | Ingestion metadata; non-predictive. |
| `road_type` (raw string), `road_type_code` | **Redundant Representation** | Replaced by the seven one-hot dummy variables to avoid colinearity. |

---

## 5. Spatial GroupKFold Cross-Validation Protocol

- **Grouping Key:** `district` (33 administrative districts).
- **Rationale:** Stations within the same district share identical macro-contextual values. Random K-fold causes spatial autocorrelation leakage. Spatial GroupKFold ensures complete spatial out-of-fold validation.
- **Class Imbalance:** 131 Class 1 vs 23 Class 0 (ratio 5.70:1). Handled in Phase 10 via cost-sensitive class weights (`scale_pos_weight` in XGBoost/LightGBM, `class_weight='balanced'` in Random Forest).
- **Evaluation Metrics:** Evaluated primarily on PR-AUC, F1-Score (macro & minority), ROC-AUC, and Balanced Accuracy.

## 6. Certification

> **The 21-feature predictor matrix $X$ and modeling matrix `fast_charging_modeling_matrix.csv` contain ZERO target leakage fields, ZERO target-contaminated aggregates, ZERO synthetic observations, and EXACTLY 21 certified predictors. Phase 9.1 is certified PASS.**
