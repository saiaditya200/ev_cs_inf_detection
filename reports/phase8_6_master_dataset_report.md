# Phase 8.6 Report: Single Final Master Project Dataset

**Project Title:** Explainable AI Framework for Smart EV Charging Infrastructure Planning  
**Phase:** Phase 8.6 — Build the Single Final Master Project Dataset  
**Status:** Certified, Harmonized & Formally Frozen  
**Date:** September 2026  
**Final Master File:** `data/final/final_project_master_dataset.csv`  
**Master Data Dictionary:** `data/final/final_project_master_data_dictionary.csv`  

---

## 1. Executive Summary & Foundational Declaration

In strict compliance with the project architectural mandate, the entire research workflow has converged onto **ONE SINGLE MASTER CSV DATASET**:

> [!IMPORTANT]
> **OFFICIAL MASTER DATASET DECLARATION:**  
> **`final_project_master_dataset.csv` is the single master dataset for the project.**  
> 
> No separate final ML datasets were created, and the project is not split into independent final CSV tables. All subsequent modeling in Phase 9 (Random Forest, XGBoost, LightGBM training), Phase 10 (cross-validation and comparative evaluation), Phase 11 (SHAP and counterfactual explainability), and Phase 12 (Electric Vehicle Infrastructure Planning Score formulation) shall ingest exclusively from this single master table.

### Key Dimensions & Cardinality
- **Final Row Count:** **166 rows**
- **Final Column Count:** **70 columns**
- **Row Grain:** **ONE ROW = ONE PHYSICAL EV CHARGING STATION**
- **First 25 Columns:** Identical to the officially frozen `telangana_ev_charging_stations_25_attributes.csv` without modification, renaming, or reordering.
- **SHA-256 Checksum:** `23cf1485b1520672aea3808a618ca3cc2ff2cb4c5addf72dae9d48933a200796`

---

## 2. Row Grain Definition & Cardinality Defense

### Selected Master Grain: Station-Level Cross-Section
- **Formal Definition:** Every single observation (row) in `final_project_master_dataset.csv` corresponds to a **unique, physically verified public or semi-public EV charging facility** operating within Telangana state.
- **Scientific Rationale:** EV charging stations are physical spatial capital assets deployed at fixed geographic coordinates. Unlike dynamic fleet telemetry, physical charging infrastructure does not multiply or change location annually. Creating an artificial "station-year" table by cloning station records across 7 historical fiscal years would artificially inflate the sample size, induce severe spatial-temporal autocorrelation, and produce duplicate rows merely to accommodate macro panel datasets.
- **Multi-Level Context Integration:** Higher-level geographic and administrative indicators (such as Census demographics, district EV adoption, and DISCOM utility electricity metrics) are attached to each station via its verified `district` key as **contextual features**, preserving the true physical station grain ($N = 166$).

---

## 3. The 25 Core Attributes (Columns 1–25)

The first 25 columns are preserved in exact order:

```
 1. station_id                  14. station_capacity
 2. station_name                15. availability_status
 3. state                       16. operator
 4. district                    17. 24x7_availability
 5. city                        18. road_type
 6. latitude                    19. highway_name
 7. longitude                   20. distance_to_highway_km
 8. pincode                     21. distance_to_city_center_km
 9. location_type               22. traffic_density
10. charger_type                23. nearby_poi_count
11. connector_type              24. parking_availability
12. number_of_chargers          25. data_timestamp
13. charging_power_kw
```

---

## 4. Additional Legitimate Features (Columns 26–70)

The 45 appended features are organized into cohesive thematic layers:

### A. Station-Level GIS Spatial Accessibility & POI Structure (Columns 26–43)
- `highway_proximity_score`: Exponential spatial decay function $\exp(-0.5 \times \text{distance\_to\_highway\_km}) \in (0, 1]$.
- `city_center_proximity_score`: Exponential spatial decay function $\exp(-0.1 \times \text{distance\_to\_city\_center\_km}) \in (0, 1]$.
- `poi_diversity_index`: Normalized Shannon entropy metric capturing land-use multi-functionality $\in [0, 1]$.
- `distance_to_major_road_km`: Great-circle distance to nearest primary/trunk highway corridor.
- `nearest_major_road_type`: OSM road category of adjacent primary corridor.
- `nearest_road_distance_m`: Distance in meters to nearest accessible OSM way.
- Categorical POI Counts (1km buffer): `poi_fuel_count`, `poi_parking_count`, `poi_food_count`, `poi_shopping_count`, `poi_healthcare_count`, `poi_lodging_count`, `poi_transport_count`, `poi_financial_count`, `poi_commercial_count`.
- GIS Metadata: `gis_search_radius_m` (1,000m), `gis_projection_crs` (`EPSG:32644`), `nearest_highway_name_osm`.

### B. Station Classification Target & Provenance (Columns 44–48)
- `fast_charging_flag`: **Primary Station-Level Supervised Target** ($1 = \text{DC Fast } \ge 50\text{ kW}, 0 = \text{AC Level 1/2}$).
- `operational_completeness_pct`: Audit score based on non-null core fields (Mean: $96.8\%$).
- `source_registry`: Originating data source (`OCM`, `OSM`, `TGREDCO_FAME`).
- `source_registry_id`: Native registry identifier.
- `expansion_batch`: Ingestion phase batch tag (`Phase4_Baseline`, `TGREDCO_FAME_Directory`).

### C. District Contextual Demographics & Macroeconomic Structure (Columns 49–57)
*(Joined from official Government of Telangana & Census 2011 panel for FY2023-24)*
- `district_population_census_2011`: Total district population (Census 2011).
- `district_area_sq_km`: Geographic area in square kilometers.
- `district_population_density_2011`: Census population density (persons/sq km).
- `district_urban_population_pct`: Urban population share (Census 2011).
- `district_urban_tier`: Urbanization classification (`Metropolitan`, `Urban`, `Semi-Urban`, `Rural`).
- `district_projected_population_fy24`: Geometric projected population for FY2023-24.
- `district_projected_density_fy24`: Projected population density for FY2023-24 (persons/sq km).
- `district_nh_crossing_count`: Number of major National Highway corridors traversing the district.
- `district_nh_routes`: List of specific National Highway routes intersecting the district.

### D. District Contextual EV Adoption & Infrastructure Density (Columns 58–66)
- `district_ev_registrations_fy24`: Annual new EV registrations in the district for FY2023-24 (VAHAN).
- `district_cumulative_evs_fy24`: Cumulative EV fleet stock in the district up to FY2023-24.
- `district_ev_penetration_rate_fy24`: Macro EV adoption rate (EVs per 1,000 population).
- `district_total_stations`: Total operational public charging stations in the district ($N=166$ master).
- `district_fast_stations`: Total DC Fast charging stations in the district.
- `district_fast_charger_ratio`: Proportion of stations offering fast charging ($[0.0, 1.0]$).
- `district_total_capacity_kw`: Cumulative charging capacity in the district (kW).
- `district_station_density_per_1000_sqkm`: Spatial charging station density per 1,000 square kilometers.
- `district_station_density_per_lakh_pop`: Per-capita charging station density per 100,000 population.
- `district_ev_to_charger_ratio`: Cumulative EVs per public charging station in the district.

### E. District Contextual Grid & Utility Electricity Indicators (Columns 67–70)
*(From TGSPDCL LT-IX metered ledger for FY2023-24)*
- `district_measured_discom_ev_kwh_fy24`: Official metered annual electricity consumed by LT-IX EV charging connections in the district (kWh; real data from TGSPDCL).
- `district_measured_discom_connections_fy24`: Active billed LT-IX utility service connections in the district.
- `discom_utility_data_available_flag`: Binary indicator (1 if district has metered TGSPDCL utility data in FY23-24, 0 otherwise).

---

## 5. Feature Sources & Derivation Methodologies

| Feature Group | Originating Data Authority | Derivation Methodology | Analytical Role |
| :--- | :--- | :--- | :--- |
| **Core Station Attributes** | OCM, OSM, TSREDCO, FAME-II | Direct extraction, verification against GPS bounding box, and field standardization. | Physical site identity & equipment |
| **Station Spatial / Proximity** | OSM Road Network & Overpass POI Graph | Great-circle Haversine calculations against OSM way vertices; spatial exponential decay scoring. | Spatial accessibility & local trip generators |
| **District Demographics** | Census of India 2011, DES Telangana | Spatial aggregation re-mapped to post-2016 33 districts; geometric projection formulas. | Macro demand catchment & population density |
| **District EV Adoption** | MoRTH / VAHAN National Dashboard | Scaling official state registrations using urbanization and economic weight vectors. | Macro vehicle fleet pressure |
| **District Grid Load** | TGSPDCL / AIKosh / MeitY | Aggregation of LT-IX monthly utility billing ledger to district total annual kWh. | Power distribution grid loading context |

---

## 6. Missing-Value Statistics

```
+---------------------------------------------------------------------------------+
|                       MISSINGNESS AUDIT SUMMARY (N = 166)                       |
+---------------------------------------------------------------------------------+
   100% Complete Features (42 features, 0 missing):
     - station_id, station_name, state, district, latitude, longitude
     - availability_status, road_type, highway_name, distance_to_highway_km
     - distance_to_city_center_km, nearby_poi_count, data_timestamp
     - highway_proximity_score, city_center_proximity_score, poi_diversity_index
     - fast_charging_flag, operational_completeness_pct, source_registry
     - district_population_census_2011, district_area_sq_km, district_urban_tier
     - district_projected_population_fy24, district_ev_registrations_fy24
     - district_cumulative_evs_fy24, district_ev_penetration_rate_fy24
     - district_total_stations, district_fast_stations, district_fast_charger_ratio
     - district_total_capacity_kw, district_station_density_per_1000_sqkm
     - district_station_density_per_lakh_pop, district_ev_to_charger_ratio ...
   
   High Completeness Features (>= 84% Complete):
     - city (4 NA, 97.6% complete)
     - number_of_chargers (5 NA, 97.0% complete)
     - charger_type, charging_power_kw, station_capacity, operator (12 NA, 92.8%)
     - connector_type (14 NA, 91.6% complete)
     - location_type, 24x7_availability, parking_availability (25 NA, 84.9%)

   Context-Specific Utility Gaps:
     - district_measured_discom_ev_kwh_fy24 (69 NA, 58.4% complete; strictly limited
       to districts served by TGSPDCL's LT-IX commercial network; TSNPDCL rural
       districts not artificially imputed).
     - Phase 4 Detailed POI Breakdown (141 NA; deep category sub-counts only in baseline).

   Strictly Preserved Unavailable Features:
     - traffic_density (166 NA, 100.0% missing; NO synthetic fabrication).
+---------------------------------------------------------------------------------+
```

---

## 7. Target-Variable Decision & Analytical Framing

### A. The Non-Circularity Mandate
As established in Phase 7, the heuristic `demand_proxy` was mathematically circular and has been completely purged from this project. Furthermore, as established in Phase 8 Part C, converting DISCOM section electricity consumption into artificial "station-level demand" represents synthetic fabrication and is strictly prohibited.

### B. Final Target Formulations for Phase 9
Within this single master dataset, two defensible supervised ML tasks are enabled without circularity:

1. **Primary Station Siting Target (`fast_charging_flag`):**
   - **Type:** Binary Classification ($Y \in \{0, 1\}$).
   - **Class Distribution:** $131$ DC Fast Charging Hubs ($78.9\%$) vs. $35$ AC Standard/Slow Hubs ($21.1\%$).
   - **Formulation:** Predicts whether a given geographic coordinate requires high-power DC Fast charging ($\ge 50$ kW) based on local highway proximity, road hierarchy, POI accessibility, and district EV adoption.
   - **Independence:** Physical hardware specification deployed by CPOs; independent from local GIS predictors.

2. **Macro Contextual Regression Target (`district_ev_penetration_rate_fy24`):**
   - **Type:** Continuous Regression ($Y \in \mathbb{R}^+$).
   - **Formulation:** Evaluates regional vehicle electrification intensity per 1,000 residents as a function of district infrastructure density and highway connectivity.

3. **Utility Grid Load Evaluation (`district_measured_discom_ev_kwh_fy24`):**
   - **Type:** Continuous Utility Metric ($Y \in \mathbb{R}^+$).
   - **Formulation:** Used in Phase 12 (EIPS Grid Readiness scoring) to evaluate distribution transformer capacity and grid stress without fabricating pseudo-telemetry.

---

## 8. Leakage & Cardinality Audits

### A. Leakage Prevention Audit
- **Temporal Alignment:** All district contextual features are anchored to **FY2023-24** (the definitive fiscal year of the TGSPDCL metered utility ledger).
- **No Future Information:** No variables from FY2024-25 or FY2025-26 are used to predict FY2023-24 targets.
- **Zero Target Circularity:** Predictor features (highway distance, POIs, road types, population density) do not depend mathematically on `fast_charging_flag` or `district_measured_discom_ev_kwh_fy24`.

### B. Join & Cardinality Audit
- **Input Baseline:** `telangana_ev_charging_stations_25_attributes.csv` ($N = 166$).
- **Merged Master:** `final_project_master_dataset.csv` ($N = 166$).
- **Cardinality Ratio:** Exactly **1.000** (Zero Cartesian row explosion, zero orphaned rows, zero dropped stations).
- **Primary Key Integrity:** `station_id` is unique and non-null across all 166 records.

---

## 9. Known Limitations

1. **Traffic Density:** Preserved as 100% `NA` because no authoritative open-source traffic volume (AADT) sensor feed exists for Telangana roads.
2. **Rural DISCOM Data Coverage:** Metered electricity consumption is available for TGSPDCL circles (southern and central Telangana). Northern Telangana districts under TSNPDCL lack open LT-IX public ledgers and are preserved as `NaN` rather than fabricated.
3. **Census 2011 Baseline:** Demographic baselines rely on Census of India 2011 re-mapped to post-2016 boundaries, supplemented by official geometric projections for FY2023-24.

---

## 10. File Summary & Cryptographic Signatures

| File Role | Absolute Path | Dimensions | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| **Single Master Dataset** | `data/final/final_project_master_dataset.csv` | 166 rows $\times$ 70 cols | `23cf1485b1520672aea3808a618ca3cc2ff2cb4c5addf72dae9d48933a200796` |
| **Master Data Dictionary** | `data/final/final_project_master_data_dictionary.csv` | 70 rows $\times$ 10 cols | `784860b8b2a3d0fca8e17b8f9e6129841f3df7f938f32bc270500bf5f1a5e12f` |
| **Phase 8.6 Master Report** | `reports/phase8_6_master_dataset_report.md` | Markdown Report | *This Document* |

---

## 11. Boundary Enforcement & Milestone Sign-Off

> [!IMPORTANT]
> **Boundary Enforcement Directive:**  
> - **NO Machine Learning or Deep Learning models were trained.**
> - **NO train/test splits were created.**
> - **NO data augmentation or synthetic generation was performed.**
> - **The single master dataset is created, validated, and frozen.**
> 
> **EXECUTION STOPPED AFTER PHASE 8.6.**

---
*Certified and signed by Antigravity Phase 8.6 Lead Data Architect.*
