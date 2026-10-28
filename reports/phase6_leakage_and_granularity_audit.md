# Phase 6 — Leakage and Granularity Audit Report

## Overview

This report provides a formal scientific audit of data leakage, geographic and temporal granularity mismatches, join integrity, and contextual replication effects for the Telangana EV Charging Infrastructure dataset.

Every dataset utilized in Phase 6 was inspected against rigorous data-engineering and econometrics standards to ensure that downstream modeling will not suffer from data snooping, lookahead bias, or invalid cross-level inferences.

---

## 1. Comprehensive Dataset Inventory & Schema Audit

| # | Dataset Identifier | Geographic Granularity | Temporal Granularity | Primary Key | Standard Units | Duplicate Keys | Missing Values Summary | Source Authority & Provenance |
|---|---|---|---|---|---|---|---|---|
| **1** | `telangana_station_gis_enriched.csv` | Point (Lat/Lon) | Sep 2026 Snapshot | `station_id` | kW, km, count, meters | **0** | 12 charger types missing in raw directory; 0 missing in GIS spatial columns | Open Charge Map (OCM) + OpenStreetMap (OSM) |
| **2** | `population_telangana_districts.csv` | District (33 districts) | 2011 Baseline | `district` / `district_key` | Persons, sq km, persons/sq km, % | **0** | **0** (100% complete across all 33 districts) | Census of India 2011 Primary Census Abstract (PCA) |
| **3** | `population_telangana_projections.csv` | State (Telangana) | Multi-year (2011, 2021, 2026) | `state`, `year` | Persons count | **0** | **0** | National Commission on Population (NCP) |
| **4** | `ev_registrations_telangana.csv` | State (Telangana) | Annual (FY2019-20 to FY2025-26) | `state`, `financial_year` | Vehicles count | **0** | **0** | VAHAN Dashboard / Ministry of Road Transport & Highways (MoRTH) |
| **5** | `ev_registrations_india.csv` | National (India) | Annual (FY2019-20 to FY2024-25) | `state`, `financial_year` | Vehicles count | **0** | **0** | VAHAN Dashboard / MoRTH |
| **6** | `electricity_telangana.csv` | State & National | Annual (FY2019-20 to FY2023-24) | `state`, `financial_year` | kWh, MU, MW | **0** | **0** | Central Electricity Authority (CEA) General Review |
| **7** | `ev_charging_infrastructure_fame.csv` | State & National | 2024 / 2025-26 Milestones | `state`, `scheme`, `year` | Station count | **0** | **0** | Ministry of Heavy Industries (MHI) / FAME Dashboard |
| **8** | `highway_infrastructure_telangana.csv` | State & National | FY2024-25 | `state`, `metric`, `year` | km, count | **0** | **0** | MoRTH / NHAI Basic Road Statistics |
| **9** | `nh_corridors_telangana.csv` | Highway Corridor | Static | `nh_number` | km, lane count | **0** | **0** | National Highways Authority of India (NHAI) |
| **10** | `ev_policy_milestones.csv` | State & National | Event Date | `scope`, `policy_name` | Date (YYYY-MM-DD) | **0** | **0** | Government of Telangana (GoTS) / GoI Gazette |

---

## 2. Granularity Mismatch Audit

The project synthesizes data spanning **three spatial resolutions** and **three temporal frequencies**:

```
+--------------------------------------------------------------------------------+
|                             GRANULARITY MATRIX                                 |
+----------------------+--------------------+------------------------------------+
| Spatial Resolution   | Temporal Frequency | Datasets Involved                  |
+----------------------+--------------------+------------------------------------+
| Station Point (m)    | 2026 Snapshot      | Station Master, Road Topology, POI |
| District Polygon     | Decennial / Annual | Census Population, NCP Projections |
| State Polygon        | Annual Fiscal Year | VAHAN EV Regs, CEA Grid, FAME      |
+----------------------+--------------------+------------------------------------+
```

### Challenges & Methodological Safeguards:
1. **Ecological Fallacy Risk**: Inferring station-level utilization patterns solely from district-level population density risks committing the ecological fallacy.
   - *Safeguard*: Layer B and Layer C contextual features are segregated into standalone panel tables (`district_year_features.csv`, `state_year_features.csv`). In the integrated table, they are explicitly prefixed with `district_` and `state_` to avoid semantic confusion with station-level attributes.
2. **Temporal Alignment**: Station observations reflect contemporary physical reality (2026). Joining earlier baseline data (e.g. 2011 Census) directly without adjustment introduces a 15-year temporal lag.
   - *Safeguard*: NCP official population projections were interpolated annually ($r = 0.615\%$ to $0.767\%$ p.a.) to align demographic denominators to the exact observation year.

---

## 3. Replication Effect Audit & Clustered Standard Errors

### Quantification of Contextual Replication

When regional context is merged into the station dataset (25 rows), identical district values are duplicated across stations located in the same district:

| District Key | Station Count in Sample | Contextual District Replication Factor | Stations Represented |
|---|---|---|---|
| `hyderabad` | **9** | **9×** | TS-HYD-0001 to TS-HYD-0009 |
| `khammam` | **4** | **4×** | TS-KHA-0001 to TS-KHA-0004 |
| `ranga_reddy` | **2** | **2×** | TS-RAN-0001, TS-RAN-0002 |
| `sangareddy` | **2** | **2×** | TS-SAN-0001, TS-SAN-0002 |
| `nizamabad` | **2** | **2×** | TS-NIZ-0001, TS-NIZ-0002 |
| `suryapet` | **2** | **2×** | TS-SUR-0001, TS-SUR-0002 |
| `medchal_malkajgiri` | **1** | **1×** | TS-MED-0001 |
| `nalgonda` | **1** | **1×** | TS-NAL-0001 |
| `mahabubnagar` | **1** | **1×** | TS-MAH-0001 |
| `jogulamba_gadwal` | **1** | **1×** | TS-JOG-0001 |
| **Total** | **25** | **State Context: 25×** | **All 25 Stations** |

### Statistical Implication:
If an OLS regression or un-grouped tree model is trained using `district_projected_population_2026` to predict a station metric, the effective sample size for the district feature is **$N_{\text{eff}} = 10$ districts**, NOT $N = 25$ stations.

### Formal Guidance for Downstream Modeling:
1. **Cluster-Robust Variance Estimation (CRVE)**: Any linear regression must cluster errors at the `district_key` level:
   $$\hat{\text{Var}}(\hat{\beta}) = (X'X)^{-1} \left( \sum_{g=1}^G X_g' \hat{e}_g \hat{e}_g' X_g \right) (X'X)^{-1}$$
2. **Multilevel / Hierarchical Models**: Use random-intercept specifications:
   $$Y_{ij} = \beta_0 + \beta_1 X_{ij} + u_j + \epsilon_{ij}, \quad u_j \sim \mathcal{N}(0, \sigma_u^2)$$
   where $i$ indexes stations and $j$ indexes districts.
3. **Explicit Data Flags**: Each row in `integrated_analytical_dataset.csv` contains `_contextual_district_replication_factor` and `_contextual_join_warning` to enforce compliance.

---

## 4. Temporal Leakage Audit

A strict zero-tolerance policy for lookahead bias and temporal leakage was executed:

### Test 4.1: Monotonic Cumulative Stock Test
- **Assertion**: Cumulative EV registration estimates over financial years must be strictly non-decreasing ($S_t \ge S_{t-1}$).
- **Evaluation**:
  - FY20: 1,842
  - FY21: 5,057 (+3,215)
  - FY22: 16,507 (+11,450)
  - FY23: 40,887 (+24,380)
  - FY24: 88,617 (+47,730)
  - FY25: 180,254 (+91,637)
  - FY26: 246,617 (+66,363 in Q1)
- **Result**: **PASSED** ($\Delta S_t > 0$ for all $t$).

### Test 4.2: Contemporaneous Slice Alignment
- **Assertion**: Stations observed in 2026 must only receive contextual indicators from calendar year 2026 / FY2025-26. No forward-looking post-2026 data may be merged into station rows.
- **Evaluation**: Verified that `integrated_analytical_dataset.csv` extracts exactly `year == 2026` from `district_year_features.csv` and `state_year_features.csv`.
- **Result**: **PASSED**.

### Test 4.3: Historical Projection Non-Contamination
- **Assertion**: In the historical district panel (e.g. FY2019-20), population figures must reflect 2020 estimates ($36,990,000$), strictly lower than 2026 estimates ($38,670,000$). No 2026 values may be back-cast into historical records.
- **Evaluation**: $\text{Pop}_{d, 2020} < \text{Pop}_{d, 2026}$ holds across all 33 districts.
- **Result**: **PASSED**.

---

## 5. Join Integrity & Cartesian Validation

Exhaustive checks were executed during table synthesis:

| Validation Test | Expected Result | Observed Result | Status |
|---|---|---|---|
| **Unmatched Station States** | 0 unmatched | 0 unmatched (`telangana`) | **PASSED** |
| **Unmatched Station Districts** | 0 unmatched | 0 unmatched (all 10 in 33 Census list) | **PASSED** |
| **Duplicate Station IDs** | 0 duplicate IDs | 0 duplicate IDs (`station_id` 100% unique) | **PASSED** |
| **District Panel Key Uniqueness** | `(district_key, financial_year)` unique | 231 / 231 unique combinations | **PASSED** |
| **State Panel Key Uniqueness** | `(state_key, financial_year)` unique | 7 / 7 unique combinations | **PASSED** |
| **Row Multiplication (Cartesian Explosion)** | Station Master (25) == Integrated (25) | Exactly 25 rows (1:1 mapping) | **PASSED** |
| **Spatial Proximity Bounds** | $0 \le \text{dist} \le 100\text{ km}$ | Min: 0.00 km, Max: 6.53 km | **PASSED** |
| **POI Counts Non-Negativity** | $\text{count} \ge 0$ | Min: 0, Max: 354 | **PASSED** |

---

## 6. Target Candidate Audit: `demand_proxy`

### Rule Compliance:
- **No variable is named "charging_demand"**: Confirmed. Neither `station_master.csv`, `district_year_features.csv`, `state_year_features.csv`, nor `integrated_analytical_dataset.csv` contains any column named `charging_demand`.
- **Target Candidate Status**: The created variable is explicitly titled `demand_proxy`, annotated with metadata column `_target_status = "CANDIDATE_PROXY_EXPLORATORY_ONLY (Not final ground-truth metered demand)"`.

### Grounded Mathematical Formulation:
$$\text{demand\_proxy} = \left(\frac{\text{Capacity}_{\text{eff}}}{50}\right) \times \left(\frac{1}{1 + \text{Dist}_{\text{NH}}}\right) \times \ln(e + \text{POI}_{\text{count}}) \times \left(0.5 + 0.5 \times \frac{\text{UrbanPct}}{100}\right)$$

### Validation & Sensitivity:
1. **Capacity Term ($\text{Cap}/50$)**: Linearly scales with available physical power. A 240 kW fast charger provides $4.8\times$ higher throughput potential than a 50 kW benchmark.
2. **Highway Decay Term ($1 / (1 + d)$)**: Rapidly down-weights remote stations. A station on the highway ($d = 0$) receives factor $1.0$; at $1\text{ km}$, factor $0.5$; at $4\text{ km}$, factor $0.2$.
3. **Amenity Attraction Term ($\ln(e + \text{POI})$)**: Logarithmic dampening prevents stations in ultra-dense commercial centers (e.g. 354 POIs) from artificially dwarfing highway charging plazas.
4. **Urban Concentration Term ($0.5 + 0.5 \times \text{UrbanPct}/100$)**: Incorporates the empirical reality that EV adoption in Telangana is 70%+ concentrated in urban agglomerations.

### Why ML Modeling Must Not Finalize This Target Yet:
While `demand_proxy` provides a clean, well-behaved numerical target for exploratory feature selection, it remains a **synthetic heuristic**. Finalizing an ML pipeline on this target risks fitting a model to an assumed formula rather than real-world driver charging behavior. Empirical CPO telemetry or tariff/session data in subsequent phases will be necessary to benchmark or replace this proxy before final model deployment.

---

## 7. Audit Certification

The Phase 6 analytical dataset has been fully validated:
- **Zero data leakage detected.**
- **Zero join discrepancies or duplicate records.**
- **Replication factors fully documented.**
- **Granularities preserved across all 5 layers.**
- **Ready for exploratory spatial analysis and Phase 7.**
