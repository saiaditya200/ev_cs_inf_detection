# Phase 8.7 Report: Single Final Master Project Dataset Review & Audit

**Project Title:** Explainable AI Framework for Smart EV Charging Infrastructure Planning  
**Phase:** Phase 8.7 — Reframe and Finalize the Single Master Project Dataset  
**Status:** Certified, Harmonized, Audited & Formally Frozen  
**Date:** September 2026  
**Final Master File:** `data/final/FINAL_PROJECT_DATASET.csv`  
**Master Data Dictionary:** `data/final/FINAL_PROJECT_DATA_DICTIONARY.csv`  

---

## 1. Critical Final Declaration & Milestone Governance

In accordance with the foundational project mandate:

> [!IMPORTANT]
> **CRITICAL FINAL STATEMENT:**  
> **`FINAL_PROJECT_DATASET.csv` is the single final master dataset for the project and is the dataset that will be reviewed before Phase 9.**  
> 
> The project does not maintain multiple fragmented final CSV datasets. All subsequent research phases—Phase 9 (Supervised Model Training), Phase 10 (Model Cross-Validation & Metric Evaluation), Phase 11 (Explainability via TreeSHAP and Counterfactuals), and Phase 12 (Electric Vehicle Infrastructure Planning Score formulation)—will ingest exclusively from this single, validated master table.
> 
> *Scope Statement:* This dataset represents **166 collected and verified EV charging stations covering all 33 Telangana districts**. It does not guarantee good model performance in advance of Phase 9 experiments, nor does it claim to capture every private residential wallbox or captive commercial fleet charger in the state.

---

## 2. Final Dataset Dimensions & Row Grain

- **Final Row Count:** **166 rows**
- **Final Column Count:** **78 columns**
- **Exact Row Grain:** **ONE ROW = ONE PHYSICAL EV CHARGING STATION**
- **Scientific Defense of Grain:** Physical charging stations are static spatial infrastructure assets. Unlike vehicle fleet telemetry, physical charging facilities do not multiply across fiscal years. Duplicating station observations across 7 historical fiscal years to force-fit macroeconomic panel datasets would introduce artificial sample inflation, duplicate spatial coordinates, and violate standard cross-validation independence assumptions. Macroeconomic, demographic, and utility grid variables are therefore joined cleanly at the district contextual level ($m:1$), strictly preserving the natural physical station grain ($N = 166$).

---

## 3. The 25 Original Core Attributes (Columns 1–25)

The first 25 columns are preserved in their exact original snake_case names and exact order from the frozen Phase 8.5 baseline:

| Index | Attribute Name | Data Type | Non-Null | Status | Source / Description |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | `station_id` | `object` | 166 (100%) | **AVAILABLE** | Unique project station identifier (`TS-{PREFIX}-{INDEX}`) |
| **2** | `station_name` | `object` | 166 (100%) | **AVAILABLE** | Verified commercial or public transit host facility title |
| **3** | `state` | `object` | 166 (100%) | **AVAILABLE** | Fixed administrative boundary constant (`Telangana`) |
| **4** | `district` | `object` | 166 (100%) | **AVAILABLE** | Harmonized 33 Census districts (post-2016 structure) |
| **5** | `city` | `object` | 162 (97.6%) | **PARTIAL** | Locality or town; NA for 4 remote highway nodes |
| **6** | `latitude` | `float64` | 166 (100%) | **AVAILABLE** | WGS-84 latitude decimal degrees $[15.8, 19.9] \text{ N}$ |
| **7** | `longitude` | `float64` | 166 (100%) | **AVAILABLE** | WGS-84 longitude decimal degrees $[77.2, 81.8] \text{ E}$ |
| **8** | `pincode` | `float64` | 13 (7.8%) | **PARTIAL** | Postal PIN code where available in address records |
| **9** | `location_type` | `object` | 141 (84.9%) | **PARTIAL** | Host venue (`public_transit_hub`, `government_office`) |
| **10** | `charger_type` | `object` | 154 (92.8%) | **PARTIAL** | Equipment category (`DC_FAST`, `AC_LEVEL2`, `AC_LEVEL1`) |
| **11** | `connector_type` | `object` | 152 (91.6%) | **PARTIAL** | Physical plug standard (`CCS (Type 2)`, `Type 2`, `Type 1`) |
| **12** | `number_of_chargers`| `float64` | 161 (97.0%) | **PARTIAL** | Count of physical charging dispensing guns/bays |
| **13** | `charging_power_kw` | `float64` | 154 (92.8%) | **PARTIAL** | Rated power output per gun ($3.0$ to $240.0$ kW) |
| **14** | `station_capacity` | `float64` | 154 (92.8%) | **PARTIAL** | Sanctioned nameplate power capacity in kW |
| **15** | `availability_status`| `object` | 166 (100%) | **AVAILABLE** | Verified operating state (`OPERATIONAL`) |
| **16** | `operator` | `object` | 154 (92.8%) | **PARTIAL** | Operating CPO (TGREDCO, REIL, EESL, NTPC, private CPOs) |
| **17** | `24x7_availability` | `object` | 141 (84.9%) | **PARTIAL** | FAME-II public mandate (`YES` for public sites; NA for baseline) |
| **18** | `road_type` | `object` | 166 (100%) | **DERIVED** | OSM functional road hierarchy (`trunk`, `primary`, etc.) |
| **19** | `highway_name` | `object` | 166 (100%) | **DERIVED** | Nearest National Highway corridor reference (NH 65, 44, etc.) |
| **20** | `distance_to_highway_km` | `float64` | 166 (100%) | **DERIVED** | Great-circle distance to nearest NH centerline (km) |
| **21** | `distance_to_city_center_km` | `float64` | 166 (100%) | **DERIVED** | Great-circle distance to District Headquarters centroid (km) |
| **22** | `traffic_density` | `float64` | 0 (0.0%) | **UNAVAILABLE**| Preserved strictly as NA per anti-fabrication mandate |
| **23** | `nearby_poi_count` | `int64` | 166 (100%) | **DERIVED** | Count of commercial, transit, and food POIs within 1km |
| **24** | `parking_availability` | `object` | 141 (84.9%) | **PARTIAL** | Dedicated EV parking bays (`YES` for transit/depot sites) |
| **25** | `data_timestamp` | `object` | 166 (100%) | **AVAILABLE** | ISO 8601 collection / verification snapshot timestamp |

---

## 4. Complete List of Additional Features (Columns 26–78)

The 53 additional features provide deep spatial, demographic, infrastructure, utility grid, and target context:

### A. Station-Level Spatial & Accessibility Features (Columns 26–48)
26. `highway_proximity_score`: Exponential decay score: $\exp(-0.5 \times d_{\text{highway}}) \in (0, 1]$.
27. `city_center_proximity_score`: Exponential decay score: $\exp(-0.1 \times d_{\text{city}}) \in (0, 1]$.
28. `poi_diversity_index`: Normalized Shannon entropy across 9 POI categories within 1,000m.
29. `distance_to_major_road_km`: Distance to nearest primary/trunk/motorway road (km).
30. `nearest_major_road_type`: Road classification of nearest major corridor.
31. `nearest_road_distance_m`: Distance in meters to nearest accessible OSM way.
32. `poi_fuel_count`: Fuel stations (petrol/diesel/CNG) within 1km buffer.
33. `poi_parking_count`: Public parking lots within 1km buffer.
34. `poi_food_count`: Restaurants, cafes, and food courts within 1km buffer.
35. `poi_shopping_count`: Retail shopping malls and supermarkets within 1km buffer.
36. `poi_healthcare_count`: Hospitals, clinics, and pharmacies within 1km buffer.
37. `poi_lodging_count`: Hotels, guest houses, and highway motels within 1km buffer.
38. `poi_transport_count`: Bus stops, railway stations, and metro stations within 1km buffer.
39. `poi_financial_count`: Commercial banks and ATMs within 1km buffer.
40. `poi_commercial_count`: Corporate office parks and commercial plazas within 1km buffer.
41. `gis_search_radius_m`: Spatial POI query buffer radius (1,000m).
42. `gis_projection_crs`: Coordinate Reference System (`EPSG:32644` - UTM Zone 44N).
43. `nearest_highway_name_osm`: Detailed OSM way name corresponding to nearest NH corridor.
44. `nearest_road_name_osm`: Detailed OSM street name for nearest accessible road.
45. `nearest_road_osm_id`: OpenStreetMap way ID for nearest road segment.
46. `nearest_nh_osm_id`: OpenStreetMap way ID for nearest National Highway segment.
47. `state_key`: Canonical lowercase state slug (`telangana`).
48. `district_key`: Canonical lowercase district slug (`slugify(district)`).

### B. Target Variable, Data Quality & Provenance (Columns 49–53)
49. **`fast_charging_flag`:** **Primary Station-Level Supervised ML Target** ($1 = \text{DC Fast } \ge 50\text{ kW}, 0 = \text{AC Level 1/2}$).
50. `operational_completeness_pct`: Quality audit metric (% of non-null core fields).
51. `source_registry`: Originating source authority (`OCM`, `OSM`, `TGREDCO_FAME`).
52. `source_registry_id`: Native registry identifier.
53. `expansion_batch`: Pipeline collection batch (`Phase4_Baseline`, `TGREDCO_FAME_Directory`).

### C. District Contextual Demographics & Macroeconomic Structure (Columns 54–62)
*(Joined from Census of India 2011 and official geometric projections for FY2023-24)*
54. `district_population_census_2011`: Total district population (Census 2011).
55. `district_area_sq_km`: Geographic area of district in square kilometers.
56. `district_population_density_2011`: Census population density (persons/sq km).
57. `district_urban_population_pct`: Proportion of population residing in statutory urban areas.
58. `district_urban_tier`: Urbanization classification (`Metropolitan`, `Urban`, `Semi-Urban`, `Rural`).
59. `district_projected_population_fy24`: Projected district population for FY2023-24.
60. `district_projected_density_fy24`: Projected population density for FY2023-24 (persons/sq km).
61. `district_nh_crossing_count`: Number of major National Highway corridors traversing the district.
62. `district_nh_routes`: List of specific intersecting National Highway routes.

### D. District Contextual EV Adoption & Infrastructure Density (Columns 63–71)
63. `district_ev_registrations_fy24`: Annual new EV registrations in the district for FY2023-24 (VAHAN).
64. `district_cumulative_evs_fy24`: Cumulative EV fleet stock in the district up to FY2023-24.
65. **`district_ev_penetration_rate`:** **District-Level Contextual Target** (EVs per 1,000 population).
66. `district_total_stations`: Total operational public charging stations in the district ($N=166$).
67. `district_fast_stations`: Total DC Fast charging stations in the district.
68. `district_fast_charger_ratio`: Proportion of stations offering fast charging ($[0.0, 1.0]$).
69. `district_total_capacity_kw`: Cumulative commissioned charging capacity in the district (kW).
70. `district_station_density_per_1000_sqkm`: Spatial charging station density per 1,000 sq km.
71. `district_station_density_per_lakh_pop`: Per-capita charging station density per 100,000 residents.
72. `district_ev_to_charger_ratio`: Ratio of cumulative EVs to public charging stations.

### E. District Contextual Utility Electricity & Grid Load Indicators (Columns 73–78)
*(From TGSPDCL LT-IX metered utility ledger for FY2023-24)*
73. **`district_measured_discom_ev_kwh`:** **DISCOM-Level Contextual Target** (Annual metered electricity consumed by LT-IX EV charging connections in the district in kWh).
74. `district_measured_discom_connections`: Count of active billed utility service connections in the district.
75. `discom_utility_data_available_flag`: Binary indicator (1 if district has metered TGSPDCL utility data in FY23-24, 0 otherwise).
76. `district_discom_billed_revenue_inr_fy24`: Total annual utility billed revenue from EV charging tariff LT-IX in the district (INR).
77. `district_discom_contracted_load_mean_kw`: Average contracted transformer capacity across sections in the district (kW).
78. `district_discom_billed_peak_load_mean_kw`: Average registered monthly peak load demand in the district (kW).

---

## 5. Target Variable Justification & Analytical Framing

The master dataset contains three distinct, observed targets preserved at their native grains:

```
+---------------------------------------------------------------------------------+
|                    TARGET VARIABLE SYSTEM IN MASTER DATASET                     |
+---------------------------------------------------------------------------------+
   1. Station-Level Supervised Target: fast_charging_flag (Col 49)
      - Unit: Binary classification (1 = DC Fast >= 50kW, 0 = AC Level 1/2)
      - Class Distribution: 131 DC Fast (78.9%) vs 35 AC Standard (21.1%)
      - Use Case: Predicts optimal charging technology siting (Random Forest, XGBoost, LightGBM)
   
   2. District Contextual Target: district_ev_penetration_rate (Col 65)
      - Unit: Continuous (EVs per 1,000 population, Range: [0.98, 5.05])
      - Use Case: Evaluates regional vehicle electrification intensity and subsidy allocation
   
   3. DISCOM Contextual Target: district_measured_discom_ev_kwh (Col 73)
      - Unit: Continuous (Metered annual kWh, Mean: 11.43 GWh across covered districts)
      - Use Case: Informs Phase 12 Grid Readiness scoring without synthetic station disaggregation
+---------------------------------------------------------------------------------+
```

### Why `fast_charging_flag` is the Primary Siting Target for Phase 9
1. **Grain Alignment:** It is observed directly at the **individual physical station site** ($N = 166$).
2. **Physical Capital Siting Decision:** CPOs and government agencies face a discrete spatial investment choice: deploy capital-intensive DC Fast charging infrastructure (requiring dedicated distribution transformers and high highway proximity) versus low-cost AC destination charging.
3. **Zero Fabrication:** It is an authoritative hardware attribute directly commissioned in the physical field.
4. **Non-Circularity:** As long as direct hardware capacity attributes (`charging_power_kw`, `charger_type`) are excluded from predictor matrix $X$, the spatial prediction of `fast_charging_flag` from local GIS and demographic context is completely non-circular and leakage-free.

---

## 6. Comprehensive Data Leakage Audit

A strict leakage audit was conducted across every candidate predictor against the primary target `fast_charging_flag`:

| Feature Name | Column | Leakage Status | Decision for Phase 9 ML Modeling |
| :--- | :---: | :--- | :--- |
| `charging_power_kw` | 13 | **Direct Identity Leakage** | **MUST BE EXCLUDED FROM $X$** (Defines `fast_charging_flag` mathematically) |
| `charger_type` | 10 | **Direct Identity Leakage** | **MUST BE EXCLUDED FROM $X$** (Directly encodes `DC_FAST` label) |
| `station_capacity` | 14 | **Direct Identity Leakage** | **MUST BE EXCLUDED FROM $X$** (Collinear with total charger power) |
| `connector_type` | 11 | **High Post-Outcome Leakage** | **MUST BE EXCLUDED FROM $X$** (`CCS (Type 2)` implies DC Fast) |
| `number_of_chargers` | 12 | **Equipment Scale Leakage** | **MUST BE EXCLUDED FROM $X$** (Facility scale determined after siting) |
| `district_fast_charger_ratio` | 68 | **Contextual Target Feedback** | **RECOMMENDED EXCLUSION** (Encodes aggregate district fast charging rate) |
| `district_fast_stations` | 67 | **Contextual Target Feedback** | **RECOMMENDED EXCLUSION** (Direct count of fast stations in district) |
| `distance_to_highway_km` | 20 | **CLEAN PREDICTOR** | **VALID IN $X$** (Spatial exogenous predictor) |
| `highway_proximity_score` | 26 | **CLEAN PREDICTOR** | **VALID IN $X$** (Spatial accessibility index) |
| `distance_to_city_center_km` | 21 | **CLEAN PREDICTOR** | **VALID IN $X$** (Urban center accessibility) |
| `poi_diversity_index` | 28 | **CLEAN PREDICTOR** | **VALID IN $X$** (Land-use mix index) |
| `nearby_poi_count` | 23 | **CLEAN PREDICTOR** | **VALID IN $X$** (Local commercial activity density) |
| `road_type` | 18 | **CLEAN PREDICTOR** | **VALID IN $X$** (OSM road hierarchy) |
| `district_urban_population_pct` | 57 | **CLEAN PREDICTOR** | **VALID IN $X$** (Exogenous demographic context) |
| `district_population_density_2011` | 56 | **CLEAN PREDICTOR** | **VALID IN $X$** (Demographic demand density) |
| `district_ev_penetration_rate` | 65 | **CLEAN PREDICTOR** | **VALID IN $X$** (Exogenous vehicle adoption pressure) |
| `district_nh_crossing_count` | 61 | **CLEAN PREDICTOR** | **VALID IN $X$** (Regional transit corridor density) |

---

## 7. Data Quality & Geographic Validation

- **Primary Key Uniqueness:** **PASS** (166 unique `station_id` values, zero duplicates).
- **Coordinate Bounding Box:** **PASS** (Latitudes strictly in $[15.8^\circ, 19.9^\circ \text{ N}]$, Longitudes in $[77.2^\circ, 81.8^\circ \text{ E}]$).
- **District Completeness:** **PASS** (All 33 official Telangana districts represented).
- **Non-Negative Metrics:** **PASS** (Distances, POI counts, population densities, and power metrics are all $\ge 0$).
- **Zero Fan-Out / Cardinality:** **PASS** (1:1 station joins and m:1 district joins; exactly 166 rows before and after merge).

---

## 8. Known Limitations

1. **Traffic Density:** Preserved as 100% `NA` because no open telemetry feed (such as Google Traffic API or automated vehicle counters) exists under open government licensing. It was strictly NOT synthesized.
2. **Northern Discom Utility Gap:** Official metered electricity ledger covers the southern and central discom (TGSPDCL). Northern rural districts under TSNPDCL lack open utility billing ledgers and are preserved as `NaN` (flagged via `discom_utility_data_available_flag = 0`).
3. **Private Wallbox Exclusion:** Captures verified public stations; excludes private home wallboxes and private commercial fleet depots.

---

## 9. File Deliverables & Cryptographic Signatures

| File Role | Path | Dimensions | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| **Final Master Dataset** | `data/final/FINAL_PROJECT_DATASET.csv` | 166 rows $\times$ 78 cols | `ba5cc7335d4405646db13e3386a62228ceec0c6952e6c38309a991bb8c5a7476` |
| **Master Data Dictionary** | `data/final/FINAL_PROJECT_DATA_DICTIONARY.csv` | 78 rows $\times$ 13 cols | `ec2309c7511ebfdd995d17d5184db289ec88fb49de434ba9ee2fce11469174e6` |
| **Phase 8.7 Review Report** | `reports/phase8_7_final_dataset_review.md` | Markdown Report | *This Document* |

---

## 10. Milestone Sign-Off & Execution Halt

> [!IMPORTANT]
> **Boundary Enforcement Directive:**  
> In strict accordance with the Phase 8.7 instructions:
> - **NO Machine Learning models (Random Forest, XGBoost, LightGBM, Deep Learning) were trained.**
> - **NO train/test splitting was performed.**
> - **NO cross-validation, SHAP, counterfactuals, or EIPS calculations were conducted.**
> - **Phase 9 has NOT been started.**
> 
> **EXECUTION STOPPED AFTER COMPLETING PHASE 8.7.**

---
*Certified and signed by Antigravity Phase 8.7 Lead Data Architect.*
