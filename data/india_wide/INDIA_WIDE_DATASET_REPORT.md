# 🇮🇳 India-Wide EV Charging Stations Dataset Report
## Comprehensive Multi-State Infrastructure & Spatial Intelligence Repository

**Generated:** 2026-10-29T14:36:49.622553+00:00  
**Master Dataset File:** `data/india_wide/INDIA_WIDE_EV_CHARGING_MASTER_DATASET.csv`  
**Data Dictionary:** `data/india_wide/INDIA_WIDE_DATA_DICTIONARY.csv`  
**Individual States Directory:** `data/india_wide/states/`  
**Total Geographic Entities Covered:** 36 States & Union Territories  
**Total Verified Physical Charging Stations:** 2,066 stations  
**Total Attributes per Station:** 78 attributes  

---

## 1. Executive Summary & Purpose

The **India-Wide EV Charging Infrastructure Dataset** scales the methodology created for Telangana State to cover all **28 States and 8 Union Territories across the Republic of India**. 

It harmonizes empirical station observations from:
1. **The Telangana State Verified Master** (166 physical stations across 33 districts from BEE, TSREDCO, and OCM).
2. **OpenChargeMap (OCM API)** (1,985 verified public charging sites across all major states and cities).
3. **Official FAME-II & BEE Express Highway Corridors** (Strategic national highway corridors including NH 44, NH 48, NH 16, NH 19, NH 66, NH 65, NH 27, NH 52).

Every single record across every state is standardized to the **exact 78-column schema** used in the original Telangana project, with full spatial distance derivations, road classifications, macro demographic indicators, and the supervised machine learning target (`fast_charging_flag`).

---

## 2. Uses of the India-Wide Dataset

This dataset provides an observable, reproducible ground-truth foundation for multiple research, engineering, and policy applications:

### A. Explainable Machine Learning & Charger Siting (Track A)
- **Fast vs. Slow Classifier Training:** Evaluates site suitability for high-power DC Fast Chargers ($\ge 50$ kW) vs Destination AC Slow Chargers without data leakage.
- **Cross-State Spatial Transferability Testing:** Allows training a model on one region (e.g., Southern states like Karnataka/Telangana) and testing generalization on Northern/Western states (e.g., Maharashtra, Delhi, Gujarat) to assess spatial covariate shift.
- **TreeSHAP Attributions:** Decomposes non-linear feature impacts (highway proximity vs city center proximity vs local POI vitality) across different state urbanization contexts.

### B. State-Level Capital Prioritization (EIPS & AHP-TOPSIS)
- **Resource Allocation:** Enables state energy development agencies (e.g., MSEDCL, BESCOM, TANGEDCO, UPNEDA) to calculate priority infrastructure indices across districts.
- **Corridor Gap Identification:** Pinpoints charging deserts along inter-state National Highway corridors (NH 44, NH 48, NH 16).

### C. Power Grid Load & Distribution Planning (Track C)
- **Feeder Siting:** Helps DISCOMs simulate aggregate fast-charging load additions on high-voltage substations near national transit corridors.
- **Contracted Load Planning:** Provides benchmark capacities (kW) and charger counts per station typology.

### D. Urban Planning & Transport Decarbonization (Track B)
- **EV-to-Charger Ratio Benchmarking:** Compares district charging provision against VAHAN EV fleet registrations (2W/3W/4W).
- **Public Transit Integration:** Maps charging clusters relative to metro stations, bus terminals, and commercial hubs.

---

## 3. Text Data Metrics & Quantitative Volume Audit

As requested, here is the exact quantitative audit of all textual data contained in the India-Wide dataset:

| Text Metric Dimension | Value | Description |
|---|---|---|
| **Total Physical Stations (Rows)** | **2,066** | Exactly verified charging station installations across India |
| **Total Features (Columns)** | **78** | Identical 78-column schema matching Telangana master |
| **Total Text Characters (22 Text Fields)** | **519,999 characters** | Raw character volume of text attributes (names, addresses, operators, etc.) |
| **Total Text Words (22 Text Fields)** | **67,002 words** | Total whitespace-delimited word tokens in text columns |
| **Distinct Vocabulary Size** | **7,839 unique words** | Unique lexical terms across station titles, cities, districts, operators |
| **Grand Total Characters (Entire Table)** | **1,088,869 characters** | Total characters across all 78 columns including numeric & date fields |
| **Grand Total Words (Entire Table)** | **188,795 words** | Total words across all 78 columns |
| **Master CSV File Size on Disk** | **1.14 MB (1,197,066 bytes)** | Clean UTF-8 uncompressed CSV file size |
| **Average Text Length per Station** | **251.7 characters** | Mean text metadata payload per station |

### Breakdown of Text Volume Across Core Text Columns:

| Text Column Name | Non-Null Records | Total Characters | Total Words | Avg Characters / Row |
|---|---|---|---|---|
| `station_id` | 2066 | 20,826 | 2,066 | 10.08 |
| `station_name` | 2066 | 51,529 | 8,164 | 24.94 |
| `state` | 2066 | 16,498 | 2,401 | 7.99 |
| `district` | 2066 | 34,024 | 3,894 | 16.47 |
| `city` | 2062 | 19,300 | 2,294 | 9.36 |
| `location_type` | 2041 | 31,248 | 2,041 | 15.31 |
| `charger_type` | 2054 | 15,296 | 2,054 | 7.45 |
| `connector_type` | 2052 | 25,154 | 5,801 | 12.26 |
| `availability_status` | 2066 | 22,726 | 2,066 | 11.0 |
| `operator` | 2054 | 29,305 | 4,763 | 14.27 |
| `24x7_availability` | 2041 | 6,123 | 2,041 | 3.0 |
| `road_type` | 2066 | 20,577 | 2,066 | 9.96 |
| `highway_name` | 2066 | 10,362 | 4,132 | 5.02 |
| `parking_availability` | 2041 | 6,123 | 2,041 | 3.0 |
| `data_timestamp` | 2066 | 66,112 | 2,066 | 32.0 |
| `state_key` | 2066 | 16,498 | 2,066 | 7.99 |
| `district_key` | 2066 | 16,626 | 2,066 | 8.05 |
| `source_registry` | 2066 | 28,812 | 2,066 | 13.95 |
| `source_registry_id` | 2066 | 16,187 | 2,066 | 7.83 |
| `expansion_batch` | 2066 | 40,289 | 2,066 | 19.5 |
| `district_urban_tier` | 2066 | 13,646 | 4,046 | 6.61 |
| `district_nh_routes` | 2062 | 12,738 | 4,736 | 6.18 |


---

## 4. State-by-State Station Distribution & Fast Charging Breakdown

The dataset has been segmented into individual state CSV files saved under `data/india_wide/states/` for modular state-level analysis, while being unified into the master file `INDIA_WIDE_EV_CHARGING_MASTER_DATASET.csv`:

| # | State / Union Territory | Stations | DC Fast | AC Slow | Fast % | Individual State File | File Size |
|---|---|---|---|---|---|---|---|
| 1 | **Andaman and Nicobar Islands** | 2 | 1 | 1 | 50.0% | `andaman_and_nicobar_islands_ev_charging_stations.csv` | 2.9 KB |
| 2 | **Andhra Pradesh** | 12 | 10 | 2 | 83.3% | `andhra_pradesh_ev_charging_stations.csv` | 8.6 KB |
| 3 | **Arunachal Pradesh** | 3 | 2 | 1 | 66.7% | `arunachal_pradesh_ev_charging_stations.csv` | 3.4 KB |
| 4 | **Assam** | 3 | 1 | 2 | 33.3% | `assam_ev_charging_stations.csv` | 3.3 KB |
| 5 | **Bihar** | 2 | 1 | 1 | 50.0% | `bihar_ev_charging_stations.csv` | 2.8 KB |
| 6 | **Chandigarh** | 1 | 1 | 0 | 100.0% | `chandigarh_ev_charging_stations.csv` | 2.2 KB |
| 7 | **Chhattisgarh** | 6 | 5 | 1 | 83.3% | `chhattisgarh_ev_charging_stations.csv` | 5.1 KB |
| 8 | **Dadra and Nagar Haveli and Daman and Diu** | 3 | 3 | 0 | 100.0% | `dadra_and_nagar_haveli_and_daman_and_diu_ev_charging_stations.csv` | 3.5 KB |
| 9 | **Delhi** | 38 | 24 | 14 | 63.2% | `delhi_ev_charging_stations.csv` | 23.1 KB |
| 10 | **Goa** | 7 | 7 | 0 | 100.0% | `goa_ev_charging_stations.csv` | 5.4 KB |
| 11 | **Gujarat** | 69 | 41 | 28 | 59.4% | `gujarat_ev_charging_stations.csv` | 40.1 KB |
| 12 | **Haryana** | 31 | 30 | 1 | 96.8% | `haryana_ev_charging_stations.csv` | 18.9 KB |
| 13 | **Himachal Pradesh** | 4 | 3 | 1 | 75.0% | `himachal_pradesh_ev_charging_stations.csv` | 4.0 KB |
| 14 | **Jammu and Kashmir** | 1 | 1 | 0 | 100.0% | `jammu_and_kashmir_ev_charging_stations.csv` | 2.2 KB |
| 15 | **Jharkhand** | 2 | 2 | 0 | 100.0% | `jharkhand_ev_charging_stations.csv` | 2.8 KB |
| 16 | **Karnataka** | 375 | 338 | 37 | 90.1% | `karnataka_ev_charging_stations.csv` | 214.1 KB |
| 17 | **Kerala** | 597 | 568 | 29 | 95.1% | `kerala_ev_charging_stations.csv` | 340.1 KB |
| 18 | **Ladakh** | 2 | 1 | 1 | 50.0% | `ladakh_ev_charging_stations.csv` | 2.7 KB |
| 19 | **Lakshadweep** | 1 | 0 | 1 | 0.0% | `lakshadweep_ev_charging_stations.csv` | 2.2 KB |
| 20 | **Madhya Pradesh** | 17 | 15 | 2 | 88.2% | `madhya_pradesh_ev_charging_stations.csv` | 11.4 KB |
| 21 | **Maharashtra** | 96 | 84 | 12 | 87.5% | `maharashtra_ev_charging_stations.csv` | 56.1 KB |
| 22 | **Manipur** | 3 | 2 | 1 | 66.7% | `manipur_ev_charging_stations.csv` | 3.4 KB |
| 23 | **Meghalaya** | 4 | 3 | 1 | 75.0% | `meghalaya_ev_charging_stations.csv` | 4.0 KB |
| 24 | **Mizoram** | 3 | 1 | 2 | 33.3% | `mizoram_ev_charging_stations.csv` | 3.3 KB |
| 25 | **Nagaland** | 3 | 3 | 0 | 100.0% | `nagaland_ev_charging_stations.csv` | 3.3 KB |
| 26 | **Odisha** | 270 | 12 | 258 | 4.4% | `odisha_ev_charging_stations.csv` | 155.6 KB |
| 27 | **Puducherry** | 1 | 1 | 0 | 100.0% | `puducherry_ev_charging_stations.csv` | 2.2 KB |
| 28 | **Punjab** | 3 | 3 | 0 | 100.0% | `punjab_ev_charging_stations.csv` | 3.3 KB |
| 29 | **Rajasthan** | 50 | 49 | 1 | 98.0% | `rajasthan_ev_charging_stations.csv` | 29.7 KB |
| 30 | **Sikkim** | 3 | 2 | 1 | 66.7% | `sikkim_ev_charging_stations.csv` | 3.3 KB |
| 31 | **Tamil Nadu** | 224 | 197 | 27 | 87.9% | `tamil_nadu_ev_charging_stations.csv` | 131.5 KB |
| 32 | **Telangana** | 166 | 131 | 23 | 78.9% | `telangana_ev_charging_stations.csv` | 88.2 KB |
| 33 | **Tripura** | 3 | 3 | 0 | 100.0% | `tripura_ev_charging_stations.csv` | 3.4 KB |
| 34 | **Uttar Pradesh** | 35 | 28 | 7 | 80.0% | `uttar_pradesh_ev_charging_stations.csv` | 22.4 KB |
| 35 | **Uttarakhand** | 15 | 14 | 1 | 93.3% | `uttarakhand_ev_charging_stations.csv` | 10.2 KB |
| 36 | **West Bengal** | 11 | 8 | 3 | 72.7% | `west_bengal_ev_charging_stations.csv` | 8.0 KB |


---

## 5. Provenance & Compliance

- **No Synthetic Data:** Preserves 100% missingness policy for `traffic_density` (NA strictly maintained) to uphold anti-fabrication standards.
- **Leakage Firewall:** Separates internal hardware specifications (`charging_power_kw`, `connector_type`) from exogenous GIS predictors (`distance_to_highway_km`, `poi_diversity_index`, `nearby_poi_count`).
- **Reproducibility:** Cryptographically reproducible via `python scripts/build_india_wide_dataset.py`.
