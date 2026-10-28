# Phase 6 — Integrated Analytical Dataset Report

## Executive Summary

Phase 6 marks the critical transition from isolated data collections into a unified, mathematically defensible, multi-layered analytical database for the Telangana EV Charging Infrastructure Planning project.

By systematically harmonizing **Phase 4 GIS-enriched charging stations** with **Phase 5 official government datasets**, this phase creates the project's first integrated analytical dataset while **strictly preserving the original geographic and temporal granularity** of every input source.

> [!IMPORTANT]
> **Data Scope Limitation**: The current station dataset contains **25 verified charging stations** collected and validated through Phases 2–4 (Open Charge Map primary + OpenStreetMap supplementary). This represents the sample collected to date, **not a complete census** of all EV charging stations in Telangana. All derived density, accessibility, and demand proxy figures reflect this sample boundary.

---

## 1. Input vs. Output Summary

| Dataset Name | File Path | Granularity Level | Temporal Coverage | Input Rows | Output Rows | Primary Key |
|---|---|---|---|---|---|---|
| **Station Master** | [`data/final/station_master.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/final/station_master.csv) | Station Point (lat/lon) | Sep 2026 Snapshot | 25 | 25 | `station_id` |
| **District × Year Features** | [`data/final/district_year_features.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/final/district_year_features.csv) | District | FY2019-20 to FY2025-26 | 33 districts × 7 years | 231 | `district_key`, `financial_year` |
| **State × Year Features** | [`data/final/state_year_features.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/final/state_year_features.csv) | State | FY2019-20 to FY2025-26 | 7 years | 7 | `state_key`, `financial_year` |
| **Integrated Analytical Dataset** | [`data/final/integrated_analytical_dataset.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/final/integrated_analytical_dataset.csv) | Station Point (Enriched) | 2026 Contemporaneous | 25 | 25 | `station_id` |

---

## 2. Logical Data Architecture & Granularity Layers

Rather than forcing disparate datasets into a single flat file that conflates point measurements with macro-regional indicators, Phase 6 partitions the project data into **5 distinct logical layers**:

```
+-----------------------------------------------------------------------------------+
|                           LOGICAL DATA ARCHITECTURE                               |
+-----------------------------------------------------------------------------------+
|  Layer A: Station-Level Master Layer                                              |
|    - 25 rows (1 row per physical EV charging station)                             |
|    - Attributes: station_id, lat/lon, power, capacity, charger_type, operator     |
+-----------------------------------------------------------------------------------+
|  Layer B: District x Year Contextual Layer                                        |
|    - 231 rows (33 Telangana districts x 7 Financial Years)                        |
|    - Attributes: Census population, projected pop, urban tier, sample station cnt |
+-----------------------------------------------------------------------------------+
|  Layer C: State x Year Contextual Layer                                           |
|    - 7 rows (Telangana State x 7 Financial Years)                                 |
|    - Attributes: VAHAN EV registrations, CEA electricity, FAME chargers, policy   |
+-----------------------------------------------------------------------------------+
|  Layer D: Highway / Corridor Layer                                                |
|    - 6 corridors (NH 44, NH 65, NH 163, NH 765, NH 161, NH 167)                  |
|    - Attributes: length in TS, lane config, entry/exit points                     |
+-----------------------------------------------------------------------------------+
|  Layer E: Static Spatial / GIS Layer                                              |
|    - High-precision OSM road proximity, distance to NH, 1000m POI amenity counts |
+-----------------------------------------------------------------------------------+
```

### Layer Details:

1. **Layer A (Station Master)**:
   - Contains immutable hardware specifications, geographic coordinates, operator identifiers, and verification metadata.
   - Preserves raw OCM/OSM IDs, connector configurations, and data completeness scores.

2. **Layer B (District × Year)**:
   - Solves the temporal gap between the decennial Census (2011) and the contemporary observation window by applying official National Commission on Population (NCP) growth rates to project annual district populations.
   - Defines district urbanization tiers and tracks sample station density.

3. **Layer C (State × Year)**:
   - Tracks macroeconomic macro-adoption: official VAHAN EV registrations (which grew from 1,842 in FY20 to 158,000+ cumulative by FY26).
   - Tracks grid capacity and per-capita power consumption from the Central Electricity Authority (CEA).
   - Tracks public charging rollout under FAME-II and PM E-DRIVE.

4. **Layer D (Highway Corridor)**:
   - Captures national highway freight and transit spines traversing Telangana.

5. **Layer E (Spatial GIS)**:
   - Detailed point-to-network topology: distance to nearest road, nearest road class, distance to designated National Highway, and 9-category POI access breakdown within a 1,000m buffer.

---

## 3. Standardized Harmonization Keys

To eliminate join discrepancies caused by varying administrative spellings and formatting across ministries, four standardized keys were established:

| Standard Key | Data Type | Permitted Format | Example Values | Purpose |
|---|---|---|---|---|
| `state_key` | string | Lowercase slug | `telangana` | Unifies state-level joins across VAHAN, CEA, Census, and MHI |
| `district_key` | string | Lowercase snake_case | `hyderabad`, `medchal_malkajgiri`, `ranga_reddy`, `sangareddy` | Canonical identifier for all 33 Telangana districts |
| `year` | integer | 4-digit YYYY | `2020`, `2021`, `2022`, `2023`, `2024`, `2025`, `2026` | Standard calendar ending year for annual joins |
| `financial_year` | string | `FYyyyy-yy` | `FY2019-20`, `FY2020-21`, ..., `FY2025-26` | Official Indian fiscal reporting period (April–March) |

All 25 stations were matched to their respective district keys with **0 unmatched keys (100% match rate)**.

---

## 4. Contextual Joins & The Replication Effect

### Why Direct Many-to-One Merges Create Replication

When macro-level contextual features (district population, state EV registrations) are joined onto micro-level station points, multiple stations located in the same geographic unit receive **identical copies** of the regional value.

For example, all 9 stations located in Hyderabad share:
- The exact same district population ($4,356,351$ projected for 2026)
- The exact same district urbanization percentage ($100.0\%$)
- The exact same state EV registration growth rate ($+92.0\%$)

### Statistical & Econometric Consequences:
1. **Artificially Deflated Variance**: Treating these 9 records as independent observations in standard Ordinary Least Squares (OLS) or un-regularized machine learning models violates the *Independent and Identically Distributed (i.i.d.)* assumption.
2. **Spurious Precision**: Standard errors will be severely underestimated, leading to false statistical significance on regional variables.
3. **Multicollinearity**: State-level variables have zero within-year station-level variance ($\text{Var} = 0$), causing rank deficiency if included alongside a constant intercept without panel variation.

### Mitigation in the Analytical Dataset:
- Every row in [`integrated_analytical_dataset.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/final/integrated_analytical_dataset.csv) includes explicit tracking columns:
  - `_contextual_district_replication_factor`: Tracks how many sample stations share the same district context (e.g., 9 for Hyderabad, 4 for Khammam, 2 for Sangareddy).
  - `_contextual_state_replication_factor`: 25 (all stations share state context).
  - `_contextual_join_warning`: Explicit programmatic flag advising downstream modelers to apply clustered standard errors (clustered by `district_key`) or hierarchical/multilevel models ($Y_{sd} = \alpha + \beta X_s + \gamma Z_d + \epsilon_{sd}$).

---

## 5. Mathematically Defensible Derived Features

Phase 6 implements 10 derived variables designed according to physical transport and socioeconomic principles:

### A. Station-Level Spatial Features
1. **Highway Proximity Score**:
   $$\text{highway\_proximity\_score} = \frac{1}{1 + \text{distance\_to\_highway\_km}}$$
   *Properties*: Bounded in $(0, 1]$; equals $1.0$ at highway roadside ($0\text{ km}$), $0.5$ at $1\text{ km}$, $0.2$ at $4\text{ km}$. Avoids division-by-zero singularities.
2. **City Center Proximity Score**:
   $$\text{city\_center\_proximity\_score} = \frac{1}{1 + \text{distance\_to\_city\_center\_km}}$$
3. **POI Diversity Index (Simpson's Diversity Index)**:
   $$D = 1 - \sum_{i=1}^9 \left(\frac{n_i}{N}\right)^2 \quad (\text{for } N > 1 \text{ else } 0)$$
   *Properties*: Measures the multi-functional richness of the immediate amenity environment across 9 categories (food, retail, parking, lodging, etc.). Bounded in $[0, 1)$.
4. **Fast Charging Indicator**:
   $$\text{fast\_charging\_flag} = \mathbb{I}(\text{charger\_type} = \text{'DC\_FAST'} \lor \text{charging\_power\_kw} \ge 50.0)$$

### B. District-Level Contextual Features
5. **Projected Population Density**:
   $$\text{projected\_population\_density} = \frac{\text{projected\_population}_{d,t}}{\text{area\_sq\_km}_d}$$
6. **Sample Station Density per $1,000\text{ km}^2$**:
   $$\text{sample\_station\_density\_per\_1000\_sq\_km} = \frac{\text{sample\_station\_count}_d}{\text{area\_sq\_km}_d / 1000}$$
7. **Sample Station Density per $100,000$ Population**:
   $$\text{sample\_station\_density\_per\_lakh\_pop} = \frac{\text{sample\_station\_count}_d}{\text{projected\_population}_{d,t} / 100000}$$
8. **Estimated District EV Adoption Share Proxy**:
   $$\text{ev\_dist\_weight}_d = 0.70 \times \left(\frac{\text{urban\_pop}_d}{\sum_k \text{urban\_pop}_k}\right) + 0.30 \times \left(\frac{\text{total\_pop}_d}{\sum_k \text{total\_pop}_k}\right)$$

### C. State-Level Macro Features
9. **Annual EV-to-FAME Charger Ratio**:
   $$\text{ev\_to\_fame\_charger\_ratio} = \frac{\text{cumulative\_ev\_stock}_t}{\text{fame\_installed\_stations}_t}$$
   *Finding*: Surged from $421.4$ EVs/charger in FY21 to **$1,591.1$ EVs/charger in FY26**, indicating infrastructure lagging behind EV adoption.
10. **State Highway Density**:
    $$\text{nh\_density} = \frac{4983\text{ km}}{112.077\text{ thousand sq km}} = 44.46\text{ km / } 1,000\text{ km}^2$$

---

## 6. Target Candidate: `demand_proxy`

### Rationale & Formula
In compliance with project specifications:
- **No variable is named "charging_demand"**.
- True charging session throughput (kWh delivered, charging events/day) is proprietary commercial telemetry held by CPOs and is unavailable in open public directories.
- To provide a scientifically valid target candidate for exploratory modeling, a composite **Potential Utilization Index** (`demand_proxy`) was formulated:

$$\text{demand\_proxy} = \left(\frac{\text{Capacity}_{\text{eff}}}{50}\right) \times \left(\frac{1}{1 + \text{Dist}_{\text{NH}}}\right) \times \ln(e + \text{POI}_{\text{count}}) \times \left(0.5 + 0.5 \times \frac{\text{UrbanPct}}{100}\right)$$

Where:
- $\text{Capacity}_{\text{eff}}$: Station capacity in kW (or median benchmark imputation if missing: $60\text{ kW}$ DC, $7.4\text{ kW}$ AC-2).
- $\text{Dist}_{\text{NH}}$: Distance to nearest National Highway in km.
- $\text{POI}_{\text{count}}$: Count of commercial/amenity POIs within $1,000\text{ m}$.
- $\text{UrbanPct}$: District urban population percentage.

### Summary Statistics of `demand_proxy`:

| Metric | Value |
|---|---|
| **Count** | 25 |
| **Mean** | 2.4515 |
| **Std Dev** | 5.2612 |
| **Min** | 0.0115 |
| **25th Percentile** | 0.3430 |
| **Median (50th)** | 0.7200 |
| **75th Percentile** | 1.5520 |
| **Max** | 20.8561 |
| **Missing Values** | 0 (0.0%) |

### Top Stations by `demand_proxy`:
1. **TS-RAN-0001** (SASS EV charging station, Ranga Reddy): `20.856` (High capacity 240 kW, near NH 65, urban hub)
2. **TS-MED-0001** (Bharat Charge, Medchal-Malkajgiri): `18.462` (Mega hub 1680 kW, high urban corridor)
3. **TS-JOG-0001** (NH44 Hub, Jogulamba Gadwal): `2.967` (1440 kW high-speed highway plaza)
4. **TS-HYD-0009** (TGRedCo, Hyderabad): `2.489` (On NH 65, dense commercial POI cluster)

> [!WARNING]
> **Exploratory Status**: This variable is an un-calibrated synthetic proxy. It must **NOT** be treated as verified ground truth for production charging demand forecasting until real-world CPO session telemetry can be integrated in later phases.

---

## 7. Limitations & Scientific Constraints

1. **Sample Representativeness**: The 25 stations represent the validated harvest from OCM and OSM across 10 districts. They do not constitute an exhaustive census of all public or private chargers in Telangana.
2. **Temporal Snapshot vs. Panel**: Stations were harvested as a 2026 cross-sectional snapshot. Historical station commissioning dates are incomplete in open directories; therefore, longitudinal station-level panels cannot be fabricated without introducing artificial data.
3. **Spatial Granularity of EV Registrations**: Official VAHAN EV registration data is published at the state level (State × FY). District-level allocations are based on an urban population weighting model and serve as contextual indicators, not metered RTO census totals.
4. **Demand Proxy Approximations**: The demand proxy reflects supply-side capacity and environmental exposure. It does not measure grid draw, electricity tariffs, queue lengths, or charger downtime.

---

## 8. Summary of Produced Datasets

All files are stored in [`data/final/`](file:///c:/Users/Dell/Downloads/EV_Charging/data/final):

1. [`data/final/station_master.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/final/station_master.csv): 25 rows × 64 columns. Clean station master with standardized keys, coordinates, equipment, and GIS metrics.
2. [`data/final/district_year_features.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/final/district_year_features.csv): 231 rows × 25 columns. Complete 33-district panel over 7 financial years.
3. [`data/final/state_year_features.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/final/state_year_features.csv): 7 rows × 27 columns. Telangana state macro-adoption, power grid, and policy time series.
4. [`data/final/integrated_analytical_dataset.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/final/integrated_analytical_dataset.csv): 25 rows × 97 columns. The unified analytical dataset ready for exploratory spatial analysis and feature selection.
