# Phase 8.8: Final Analytical Dataset Integrity, Leakage Correction, and Freeze Report

**Project Title:** Explainable AI Framework for Smart EV Charging Infrastructure Planning  
**Phase Status:** COMPLETE & FROZEN  
**Date:** 2026-09-26  
**Target Dataset:** `FINAL_PROJECT_DATASET.csv`  
**SHA-256 Checksum:** `ba5cc7335d4405646db13e3386a62228ceec0c6952e6c38309a991bb8c5a7476`  

---

## 1. Final Dataset Dimensions and Overview

- **Final Validated Row Count:** **166** rows
- **Final Validated Column Count:** **78** columns
- **Row Grain:** **Strictly 1 row = 1 physical EV charging station** facility ($N = 166$).
- **District Coverage:** Complete spatial coverage across all **33 administrative districts** of Telangana.
- **Zero Synthetic Demand:** Strictly 0 rows or columns of synthetic charging demand, fake station kWh, or artificial data augmentation.
- **No Duplicates:** Exactly 166 unique `station_id` values and 0 duplicate physical coordinate pairs.

## 2. Complete Column Enumeration (78 Columns)

| Col # | Column Name | Data Type | Null Count | Missing % | Analytical Role | Practical ML Usability | Leakage Classification | Model Status |
|---|---|---|---|---|---|---|---|---|
| 1 | `station_id` | `object` | 0 | 0.0% | IDENTIFIER | METADATA | IDENTIFIER | `EXCLUDED_METADATA_ID` |
| 2 | `station_name` | `object` | 0 | 0.0% | METADATA | METADATA | METADATA | `EXCLUDED_METADATA_ID` |
| 3 | `state` | `object` | 0 | 0.0% | METADATA | METADATA | METADATA | `EXCLUDED_METADATA_ID` |
| 4 | `district` | `object` | 0 | 0.0% | DISTRICT_CONTEXT | CORE_USABLE | SAFE | `FEATURE_X` |
| 5 | `city` | `object` | 4 | 2.4% | METADATA | METADATA | METADATA | `EXCLUDED_METADATA_ID` |
| 6 | `latitude` | `float64` | 0 | 0.0% | CORE_STATION_FEATURE | CORE_USABLE | SAFE | `FEATURE_X` |
| 7 | `longitude` | `float64` | 0 | 0.0% | CORE_STATION_FEATURE | CORE_USABLE | SAFE | `FEATURE_X` |
| 8 | `pincode` | `float64` | 153 | 92.2% | METADATA | HIGH_MISSINGNESS | HIGH_MISSINGNESS | `EXCLUDED` |
| 9 | `location_type` | `object` | 25 | 15.1% | CORE_STATION_FEATURE | OPTIONAL | SAFE | `FEATURE_X` |
| 10 | `charger_type` | `object` | 12 | 7.2% | EXCLUDED_FROM_ML | EXCLUDED | TARGET_LEAKAGE | `EXCLUDED_FROM_FAST_CHARGING_MODEL` |
| 11 | `connector_type` | `object` | 14 | 8.4% | EXCLUDED_FROM_ML | EXCLUDED | TARGET_LEAKAGE | `EXCLUDED_FROM_FAST_CHARGING_MODEL` |
| 12 | `number_of_chargers` | `float64` | 5 | 3.0% | EXCLUDED_FROM_ML | OPTIONAL | POST_OUTCOME | `EXCLUDED_FROM_FAST_CHARGING_MODEL` |
| 13 | `charging_power_kw` | `float64` | 12 | 7.2% | EXCLUDED_FROM_ML | EXCLUDED | TARGET_LEAKAGE | `EXCLUDED_FROM_FAST_CHARGING_MODEL` |
| 14 | `station_capacity` | `float64` | 12 | 7.2% | EXCLUDED_FROM_ML | EXCLUDED | TARGET_LEAKAGE | `EXCLUDED_FROM_FAST_CHARGING_MODEL` |
| 15 | `availability_status` | `object` | 0 | 0.0% | METADATA | METADATA | METADATA | `EXCLUDED_METADATA_ID` |
| 16 | `operator` | `object` | 12 | 7.2% | CORE_STATION_FEATURE | OPTIONAL | POST_OUTCOME | `EXCLUDED_FROM_FAST_CHARGING_MODEL` |
| 17 | `24x7_availability` | `object` | 25 | 15.1% | CORE_STATION_FEATURE | OPTIONAL | SAFE | `FEATURE_X` |
| 18 | `road_type` | `object` | 0 | 0.0% | GIS_FEATURE | CORE_USABLE | SAFE | `FEATURE_X` |
| 19 | `highway_name` | `object` | 0 | 0.0% | GIS_FEATURE | METADATA | METADATA | `EXCLUDED_METADATA_ID` |
| 20 | `distance_to_highway_km` | `float64` | 0 | 0.0% | GIS_FEATURE | CORE_USABLE | SAFE | `FEATURE_X` |
| 21 | `distance_to_city_center_km` | `float64` | 0 | 0.0% | GIS_FEATURE | CORE_USABLE | SAFE | `FEATURE_X` |
| 22 | `traffic_density` | `float64` | 166 | 100.0% | CORE_STATION_FEATURE | HIGH_MISSINGNESS | HIGH_MISSINGNESS | `EXCLUDED` |
| 23 | `nearby_poi_count` | `int64` | 0 | 0.0% | GIS_FEATURE | CORE_USABLE | SAFE | `FEATURE_X` |
| 24 | `parking_availability` | `object` | 25 | 15.1% | CORE_STATION_FEATURE | OPTIONAL | SAFE | `FEATURE_X` |
| 25 | `data_timestamp` | `object` | 0 | 0.0% | PROVENANCE | METADATA | IDENTIFIER | `EXCLUDED_METADATA_ID` |
| 26 | `highway_proximity_score` | `float64` | 0 | 0.0% | GIS_FEATURE | CORE_USABLE | SAFE | `FEATURE_X` |
| 27 | `city_center_proximity_score` | `float64` | 0 | 0.0% | GIS_FEATURE | CORE_USABLE | SAFE | `FEATURE_X` |
| 28 | `poi_diversity_index` | `float64` | 0 | 0.0% | GIS_FEATURE | CORE_USABLE | SAFE | `FEATURE_X` |
| 29 | `distance_to_major_road_km` | `float64` | 141 | 84.9% | GIS_FEATURE | HIGH_MISSINGNESS | HIGH_MISSINGNESS | `EXCLUDED` |
| 30 | `nearest_major_road_type` | `object` | 141 | 84.9% | GIS_FEATURE | HIGH_MISSINGNESS | HIGH_MISSINGNESS | `EXCLUDED` |
| 31 | `nearest_road_distance_m` | `float64` | 141 | 84.9% | GIS_FEATURE | HIGH_MISSINGNESS | HIGH_MISSINGNESS | `EXCLUDED` |
| 32 | `poi_fuel_count` | `float64` | 141 | 84.9% | GIS_FEATURE | HIGH_MISSINGNESS | HIGH_MISSINGNESS | `EXCLUDED` |
| 33 | `poi_parking_count` | `float64` | 141 | 84.9% | GIS_FEATURE | HIGH_MISSINGNESS | HIGH_MISSINGNESS | `EXCLUDED` |
| 34 | `poi_food_count` | `float64` | 141 | 84.9% | GIS_FEATURE | HIGH_MISSINGNESS | HIGH_MISSINGNESS | `EXCLUDED` |
| 35 | `poi_shopping_count` | `float64` | 141 | 84.9% | GIS_FEATURE | HIGH_MISSINGNESS | HIGH_MISSINGNESS | `EXCLUDED` |
| 36 | `poi_healthcare_count` | `float64` | 141 | 84.9% | GIS_FEATURE | HIGH_MISSINGNESS | HIGH_MISSINGNESS | `EXCLUDED` |
| 37 | `poi_lodging_count` | `float64` | 141 | 84.9% | GIS_FEATURE | HIGH_MISSINGNESS | HIGH_MISSINGNESS | `EXCLUDED` |
| 38 | `poi_transport_count` | `float64` | 141 | 84.9% | GIS_FEATURE | HIGH_MISSINGNESS | HIGH_MISSINGNESS | `EXCLUDED` |
| 39 | `poi_financial_count` | `float64` | 141 | 84.9% | GIS_FEATURE | HIGH_MISSINGNESS | HIGH_MISSINGNESS | `EXCLUDED` |
| 40 | `poi_commercial_count` | `float64` | 141 | 84.9% | GIS_FEATURE | HIGH_MISSINGNESS | HIGH_MISSINGNESS | `EXCLUDED` |
| 41 | `gis_search_radius_m` | `float64` | 141 | 84.9% | GIS_FEATURE | METADATA | METADATA | `EXCLUDED_METADATA_ID` |
| 42 | `gis_projection_crs` | `object` | 141 | 84.9% | GIS_FEATURE | METADATA | METADATA | `EXCLUDED_METADATA_ID` |
| 43 | `nearest_highway_name_osm` | `object` | 155 | 93.4% | GIS_FEATURE | METADATA | METADATA | `EXCLUDED_METADATA_ID` |
| 44 | `nearest_road_name_osm` | `object` | 158 | 95.2% | GIS_FEATURE | METADATA | METADATA | `EXCLUDED_METADATA_ID` |
| 45 | `nearest_road_osm_id` | `float64` | 141 | 84.9% | GIS_FEATURE | METADATA | IDENTIFIER | `EXCLUDED_METADATA_ID` |
| 46 | `nearest_nh_osm_id` | `float64` | 141 | 84.9% | GIS_FEATURE | METADATA | IDENTIFIER | `EXCLUDED_METADATA_ID` |
| 47 | `state_key` | `object` | 0 | 0.0% | PROVENANCE | METADATA | METADATA | `EXCLUDED_METADATA_ID` |
| 48 | `district_key` | `object` | 0 | 0.0% | DISTRICT_CONTEXT | CORE_USABLE | IDENTIFIER | `FEATURE_X` |
| 49 | `fast_charging_flag` | `int64` | 0 | 0.0% | TARGET | CORE_USABLE | TARGET_LEAKAGE | `TARGET_Y` |
| 50 | `operational_completeness_pct` | `float64` | 0 | 0.0% | METADATA | METADATA | POST_OUTCOME | `EXCLUDED` |
| 51 | `source_registry` | `object` | 0 | 0.0% | PROVENANCE | METADATA | IDENTIFIER | `EXCLUDED_METADATA_ID` |
| 52 | `source_registry_id` | `object` | 0 | 0.0% | PROVENANCE | METADATA | IDENTIFIER | `EXCLUDED_METADATA_ID` |
| 53 | `expansion_batch` | `object` | 0 | 0.0% | PROVENANCE | METADATA | IDENTIFIER | `EXCLUDED_METADATA_ID` |
| 54 | `district_population_census_2011` | `int64` | 0 | 0.0% | DISTRICT_CONTEXT | CORE_USABLE | SAFE | `FEATURE_X` |
| 55 | `district_area_sq_km` | `int64` | 0 | 0.0% | DISTRICT_CONTEXT | CORE_USABLE | SAFE | `FEATURE_X` |
| 56 | `district_population_density_2011` | `int64` | 0 | 0.0% | DISTRICT_CONTEXT | CORE_USABLE | SAFE | `FEATURE_X` |
| 57 | `district_urban_population_pct` | `float64` | 0 | 0.0% | DISTRICT_CONTEXT | CORE_USABLE | SAFE | `FEATURE_X` |
| 58 | `district_urban_tier` | `object` | 0 | 0.0% | DISTRICT_CONTEXT | CORE_USABLE | SAFE | `FEATURE_X` |
| 59 | `district_projected_population_fy24` | `int64` | 0 | 0.0% | DISTRICT_CONTEXT | CORE_USABLE | SAFE | `FEATURE_X` |
| 60 | `district_projected_density_fy24` | `float64` | 0 | 0.0% | DISTRICT_CONTEXT | CORE_USABLE | SAFE | `FEATURE_X` |
| 61 | `district_nh_crossing_count` | `int64` | 0 | 0.0% | DISTRICT_CONTEXT | CORE_USABLE | SAFE | `FEATURE_X` |
| 62 | `district_nh_routes` | `object` | 4 | 2.4% | DISTRICT_CONTEXT | METADATA | METADATA | `EXCLUDED_METADATA_ID` |
| 63 | `district_ev_registrations_fy24` | `int64` | 0 | 0.0% | DISTRICT_CONTEXT | CORE_USABLE | SAFE | `FEATURE_X` |
| 64 | `district_cumulative_evs_fy24` | `int64` | 0 | 0.0% | DISTRICT_CONTEXT | CORE_USABLE | SAFE | `FEATURE_X` |
| 65 | `district_ev_penetration_rate` | `float64` | 0 | 0.0% | DISTRICT_CONTEXT | CORE_USABLE | SAFE | `FEATURE_X` |
| 66 | `district_total_stations` | `int64` | 0 | 0.0% | DISTRICT_CONTEXT | OPTIONAL | CURRENT_STATION_AGGREGATE | `EXCLUDED_FROM_FAST_CHARGING_MODEL` |
| 67 | `district_fast_stations` | `int64` | 0 | 0.0% | DISTRICT_CONTEXT | EXCLUDED | TARGET_LEAKAGE | `EXCLUDED_FROM_FAST_CHARGING_MODEL` |
| 68 | `district_fast_charger_ratio` | `float64` | 0 | 0.0% | DISTRICT_CONTEXT | EXCLUDED | TARGET_LEAKAGE | `EXCLUDED_FROM_FAST_CHARGING_MODEL` |
| 69 | `district_total_capacity_kw` | `float64` | 0 | 0.0% | DISTRICT_CONTEXT | OPTIONAL | CURRENT_STATION_AGGREGATE | `EXCLUDED_FROM_FAST_CHARGING_MODEL` |
| 70 | `district_station_density_per_1000_sqkm` | `float64` | 0 | 0.0% | DISTRICT_CONTEXT | OPTIONAL | CURRENT_STATION_AGGREGATE | `EXCLUDED_FROM_FAST_CHARGING_MODEL` |
| 71 | `district_station_density_per_lakh_pop` | `float64` | 0 | 0.0% | DISTRICT_CONTEXT | OPTIONAL | CURRENT_STATION_AGGREGATE | `EXCLUDED_FROM_FAST_CHARGING_MODEL` |
| 72 | `district_measured_discom_ev_kwh` | `float64` | 69 | 41.6% | DISTRICT_CONTEXT | OPTIONAL | SAFE | `FEATURE_X` |
| 73 | `district_measured_discom_connections` | `float64` | 69 | 41.6% | DISTRICT_CONTEXT | OPTIONAL | SAFE | `FEATURE_X` |
| 74 | `district_ev_to_charger_ratio` | `float64` | 0 | 0.0% | DISTRICT_CONTEXT | OPTIONAL | CURRENT_STATION_AGGREGATE | `EXCLUDED_FROM_FAST_CHARGING_MODEL` |
| 75 | `discom_utility_data_available_flag` | `int32` | 0 | 0.0% | DISTRICT_CONTEXT | CORE_USABLE | SAFE | `FEATURE_X` |
| 76 | `district_discom_billed_revenue_inr_fy24` | `float64` | 69 | 41.6% | DISTRICT_CONTEXT | OPTIONAL | SAFE | `FEATURE_X` |
| 77 | `district_discom_contracted_load_mean_kw` | `float64` | 69 | 41.6% | DISTRICT_CONTEXT | OPTIONAL | SAFE | `FEATURE_X` |
| 78 | `district_discom_billed_peak_load_mean_kw` | `float64` | 69 | 41.6% | DISTRICT_CONTEXT | OPTIONAL | SAFE | `FEATURE_X` |

---

## 3. Original 25 Core Attributes Verification

The first 25 columns of `FINAL_PROJECT_DATASET.csv` match the frozen baseline (`telangana_ev_charging_stations_25_attributes.csv`) **verbatim and in exact positional order**:

 1. `station_id` -> `station_id` [PASS]
 2. `station_name` -> `station_name` [PASS]
 3. `state` -> `state` [PASS]
 4. `district` -> `district` [PASS]
 5. `city` -> `city` [PASS]
 6. `latitude` -> `latitude` [PASS]
 7. `longitude` -> `longitude` [PASS]
 8. `pincode` -> `pincode` [PASS]
 9. `location_type` -> `location_type` [PASS]
10. `charger_type` -> `charger_type` [PASS]
11. `connector_type` -> `connector_type` [PASS]
12. `number_of_chargers` -> `number_of_chargers` [PASS]
13. `charging_power_kw` -> `charging_power_kw` [PASS]
14. `station_capacity` -> `station_capacity` [PASS]
15. `availability_status` -> `availability_status` [PASS]
16. `operator` -> `operator` [PASS]
17. `24x7_availability` -> `24x7_availability` [PASS]
18. `road_type` -> `road_type` [PASS]
19. `highway_name` -> `highway_name` [PASS]
20. `distance_to_highway_km` -> `distance_to_highway_km` [PASS]
21. `distance_to_city_center_km` -> `distance_to_city_center_km` [PASS]
22. `traffic_density` -> `traffic_density` [PASS]
23. `nearby_poi_count` -> `nearby_poi_count` [PASS]
24. `parking_availability` -> `parking_availability` [PASS]
25. `data_timestamp` -> `data_timestamp` [PASS]

---

## 4. Multi-Layer Data Source Mapping and Provenance

| Layer | Scope | Granularity | Data Sources | Key Variables |
|---|---|---|---|---|
| **Core Stations** | Baseline & Expansion | Station ($N=166$) | Open Charge Map (OCM), OpenStreetMap (OSM), TGREDCO FAME Directory | Coordinates, operator, charger count, power kW, availability |
| **GIS & Spatial** | Accessibility & Road Network | Station ($N=166$) | OpenStreetMap Overpass API, MoRTH / NHAI National Highway Network | Highway distance/score, city center distance/score, road type, POI counts, POI diversity index |
| **Demographics & Geography** | District Panel | District ($N=33$) | Census of India 2011, Telangana Statistical Abstract, NDAP | Population, area, population density, urban percentage, urban tier, projected FY24 pop/density |
| **Corridors & Highways** | Transport Infrastructure | District ($N=33$) | MoRTH / NHAI Official Highway Gazette | NH route names, NH crossing count |
| **EV Fleet Adoption** | Vehicle Registrations | District ($N=33$) | MoRTH VAHAN Registry (State) & Economic Weight Scaling | FY24 EV registrations, cumulative EV fleet, EV penetration rate |
| **Grid & Utility Telemetry** | Power Utility Demand | District ($N=33$) | TGSPDCL Official Commercial LT-IX Billing Ledger (FY2023-24) | Metered EV kWh, active connections, billed revenue, contracted load, peak demand load |

---

## 5. Missing Value Analysis and Zero-Fabrication Integrity

Missing values reflect authentic real-world data collection boundaries and are preserved transparently without artificial imputation:

- `traffic_density`: **100% missing (166/166)**. Preserved as NA. No synthetic traffic values have been or will be fabricated.
- `pincode`: **92.2% missing (153/166)**. Omitted by raw registries where GPS coordinates were provided. Classified `HIGH_MISSINGNESS` and excluded from ML.
- Spatial Sub-category POIs (`distance_to_major_road_km`, `poi_*_count`): **84.9% missing (141/166)**. Collected for the Phase 4 baseline (25 stations). Classified `HIGH_MISSINGNESS` and excluded from ML. The master spatial feature `nearby_poi_count` (100% complete across all 166 stations) and `poi_diversity_index` serve as the universal predictors.
- TGSPDCL Utility Consumption (`district_measured_discom_ev_kwh`, etc.): **41.6% missing (69/166)**. TGSPDCL serves southern/central Telangana (14 districts). The remaining 19 northern districts fall under TSNPDCL jurisdiction where section ledgers were unavailable. Explicitly tracked by `discom_utility_data_available_flag` (1 for TGSPDCL, 0 for TSNPDCL).
- Station Technical Attributes (`charger_type`, `power_kw`, `capacity`): **7.2% missing (12/166)**. Missing in basic crowd-sourced OSM nodes where only charging location was confirmed.

---

## 6. Comprehensive Data Leakage Audit and Mitigation Strategy

To ensure a scientifically defensible Phase 9 supervised modeling workflow, a column-by-column leakage audit was conducted:

### Summary of Leakage Audit Categories
- **SAFE:** 30 columns
- **HIGH_MISSINGNESS:** 14 columns
- **METADATA:** 11 columns
- **IDENTIFIER:** 8 columns
- **TARGET_LEAKAGE:** 7 columns
- **CURRENT_STATION_AGGREGATE:** 5 columns
- **POST_OUTCOME:** 3 columns

### Critical Leakage Findings & Modeling Rules

1. **Direct Equipment Identity Leakage:**
   - `charging_power_kw`, `charger_type`, `station_capacity`, and `connector_type` mathematically define or directly imply `fast_charging_flag` (DC Fast >= 50 kW).
   - **Rule:** These 4 columns are classified `TARGET_LEAKAGE` and must be **strictly excluded from feature matrix $X$** during Phase 9.

2. **Target-Contaminated District Aggregates:**
   - `district_fast_stations` and `district_fast_charger_ratio` aggregate the `fast_charging_flag` of stations in the district, including the current station record.
   - **Rule:** Classified `TARGET_LEAKAGE` and marked `EXCLUDED_FROM_FAST_CHARGING_MODEL`.

3. **Current-Station District Infrastructure Aggregates:**
   - `district_total_stations`, `district_total_capacity_kw`, `district_station_density_per_1000_sqkm`, `district_station_density_per_lakh_pop`, and `district_ev_to_charger_ratio` derive their values from the sum of stations in the district, thereby containing self-referential information.
   - **Rule:** Classified `CURRENT_STATION_AGGREGATE` and marked `EXCLUDED_FROM_FAST_CHARGING_MODEL`.

4. **Valid Siting Predictor Matrix ($X$):**
   - Exactly **31 legitimate features** are certified as `SAFE` and designated as `FEATURE_X` for predicting `fast_charging_flag`.
   - These include spatial/GIS accessibility scores (`highway_proximity_score`, `city_center_proximity_score`, `nearby_poi_count`, `poi_diversity_index`, `road_type`) and exogenous district demographic/economic context (`district_population_density_2011`, `district_urban_population_pct`, `district_urban_tier`, `district_nh_crossing_count`, `district_ev_penetration_rate`, `discom_utility_data_available_flag`).

---

## 7. Machine Learning Target Architecture

### Primary Supervised Siting Target: `fast_charging_flag`
- **Definition:** $1 = \text{DC Fast Charger } (\ge 50\text{ kW}),\; 0 = \text{AC Level 1/Level 2 Charger} (< 50\text{ kW})$.
- **Class Balance:** **131 Class 1 (78.9%) vs 35 Class 0 (21.1%)**.
- **Target Nature:** Fully observed ground-truth engineering classification.
- **Planning Rationale:** Predicts whether an optimal station location demands high-throughput fast DC infrastructure or slow destination charging based purely on local spatial accessibility and district demographic context.

### Contextual Planning Metrics (NOT Station Supervised Targets)
- `district_ev_penetration_rate`: District adoption pressure metric.
- `district_measured_discom_ev_kwh`: Official utility energy throughput at district scale.
- These metrics serve as macro-level inputs into the multi-criteria EIPS framework in Phase 12, never as fake station-level regression targets.

---

## 8. Multi-Level Granularity and Spatial Validation

- **Station-Level Features ($N=166$):** Physical coordinates, road type, distance to highway, distance to city center, nearby POI counts, POI diversity index, and ground-truth charging speed.
- **District-Level Context ($m:1$ mapping across 33 districts):** Official Census population, land area, urbanization rate, NH corridor crossings, estimated EV stock, and TGSPDCL grid load.
- **Validation Protocol:** In Phase 9, cross-validation must use **Spatial GroupKFold grouped by `district`** to prevent spatial autocorrelation leakage between stations in the same administrative area.

---

## 9. Estimated vs. Observed Variable Classification

| Variable | Observation Nature | Methodology & Provenance |
|---|---|---|
| `fast_charging_flag` | **OBSERVED** | Extracted directly from validated CPO equipment specs in OCM/OSM/TGREDCO |
| `latitude`, `longitude` | **OBSERVED** | GPS coordinates verified within Telangana administrative boundary |
| `nearby_poi_count` | **DERIVED_SPATIAL** | OpenStreetMap 1000m buffer spatial query in EPSG:32644 |
| `highway_proximity_score` | **DERIVED_SPATIAL** | Projected distance to MoRTH NH network: $\exp(-0.2 \times d_{\text{NH}})$ |
| `district_population_census_2011` | **OBSERVED** | Official Registrar General Census of India 2011 |
| `district_ev_registrations_fy24` | **ESTIMATED_DISTRICT_CONTEXT** | Scaled from official Telangana state VAHAN totals using district economic/vehicle weights |
| `district_cumulative_evs_fy24` | **ESTIMATED_DISTRICT_CONTEXT** | Scaled cumulative historical VAHAN registrations |
| `district_ev_penetration_rate` | **ESTIMATED_DISTRICT_CONTEXT** | $(\text{cumulative\_evs} / \text{projected\_pop}) \times 1000$ |
| `district_measured_discom_ev_kwh` | **OBSERVED** | Metered LT-IX billing records from TGSPDCL utility database |

---

## 10. Features Excluded from ML Training and Rationale

| Feature | Rationale for Exclusion |
|---|---|
| `station_id`, `station_name`, `city` | Non-generalizable unique identifiers and text labels |
| `state`, `state_key` | Constant zero-variance administrative strings ('Telangana') |
| `availability_status` | Constant zero-variance attribute (100% 'OPERATIONAL') |
| `traffic_density` | 100% missing; no synthetic fabrication permitted |
| `pincode` | 92.2% missing; spatial location fully captured by lat/lon |
| `charging_power_kw`, `charger_type`, `station_capacity`, `connector_type` | Strict identity leakage for `fast_charging_flag` |
| `district_fast_stations`, `district_fast_charger_ratio` | Target-contaminated district aggregates |
| `district_total_stations`, `district_total_capacity_kw`, `district_station_density_*`, `district_ev_to_charger_ratio` | Current-station aggregates |
| `distance_to_major_road_km`, `poi_*_count` (sub-types) | 84.9% missing; baseline only; master `nearby_poi_count` retained |
| `source_registry`, `source_registry_id`, `expansion_batch`, `data_timestamp`, `gis_*`, `osm_id` | Engineering provenance and metadata |

---

## 11. Known Limitations and Scientific Transparency

1. **Sample Size:** $N = 166$ stations is representative of Telangana's early-stage public EV infrastructure in 2024. While sufficient for tree-based ensemble models with spatial cross-validation, models must be regularized to prevent overfitting.
2. **Utility Metering Coverage:** TGSPDCL metered electricity data covers 14 southern/central districts (55.73 GWh total). The 19 northern districts under TSNPDCL lack section-level utility metering and are handled via the `discom_utility_data_available_flag`.
3. **Absence of Real Traffic Counts:** In the absence of state-wide continuous traffic sensor feeds, roadway functional class (`road_type`), National Highway proximity (`highway_proximity_score`), and POI diversity serve as valid spatial accessibility proxies.

---

## 12. Final Certification Statement

> **FINAL_PROJECT_DATASET.csv is the single reviewed master dataset for the project and is ready for Phase 9 feature/target modeling after human review.**

---
*Report generated automatically by Phase 8.8 dataset integrity and freeze engine at 2026-09-26T18:32:50.480081.*
