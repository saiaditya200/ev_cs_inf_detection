# Phase 7 — ML Target Validation & Target Audit Report

## Executive Summary

Phase 7 serves as the formal scientific gatekeeper before any machine learning modeling is attempted in the Telangana EV Charging Infrastructure Planning project. 

The objective of this audit is to rigorously determine **whether the current dataset can support a scientifically defensible supervised machine learning experiment**, evaluate candidate target variables, audit mathematical circularity and target leakage, and establish a clear target decision matrix.

> [!IMPORTANT]
> **Key Finding & Gatekeeper Verdict**: 
> 1. The composite heuristic variable `demand_proxy` created in Phase 6 **cannot be used as a final machine learning target** because it is a **deterministic algebraic function** of candidate input features. Training a supervised model on this target constitutes **100% mathematical circularity (tautological leakage)**.
> 2. True charging-demand telemetry (sessions, kWh delivered, occupancy) is strictly proprietary to commercial Charge Point Operators (CPOs) and is **not available in open public directories**.
> 3. Supervised ML regression on charging demand with the current 25-station sample is **statistically invalid and must not proceed** until the dataset is expanded and the target is appropriately reformulated.

---

## 1. Candidate Target Audit

Seven potential target candidates were identified and evaluated across the project's data layers:

| # | Target Candidate | Target Nature | Primary Source | Geographic Granularity | Temporal Granularity | Measurement Type | Leakage Risk Level |
|---|---|---|---|---|---|---|---|
| **1** | `demand_proxy` | Continuous Index | Phase 6 Composite Formula | Station Point (lat/lon) | 2026 Snapshot | Algebraic Heuristic | **Severe (100% Circular)** |
| **2** | `real_charging_sessions_telemetry` | Count / Rate | Commercial CPO Telemetry | Station Socket | Session / Hourly / Daily | Real Empirical Metering | **None (True Ground Truth)** |
| **3** | `discom_billed_ev_consumption_kwh` | Continuous Billed kWh | TGSPDCL / AIKosh Open Data | Discom Subdivision / Section | Monthly Panel | Real Utility Billing Meter | **Low (Real Utility Billing)** |
| **4** | `station_capacity_kw` | Continuous Capacity | OCM / Station Master | Station Point | Current Snapshot | Hardware Specification | **None (Observable Hardware)** |
| **5** | `fast_charging_flag` | Binary Class (DC Fast vs AC) | OCM / Station Master | Station Point | Current Snapshot | Hardware Classification | **None (Observable Hardware)** |
| **6** | `district_ev_penetration_rate` | Rate per 1,000 Pop | VAHAN + Census / NCP Panel | District (33 Districts) | Annual Panel (7 FYs) | Government Derived Metric | **Low (Contemporaneous Panel)** |
| **7** | `eips_suitability_score` | Composite Index (0–100) | Multi-Criteria Decision Model | Spatial Grid / Site | Planning Horizon | Multi-Criteria Planning Index | **None (Unsupervised / MCDA)** |

---

## 2. Mathematical Examination of `demand_proxy`

### A. Algebraic Specification
In Phase 6, `demand_proxy` was formulated to represent a normalized Potential Utilization Index:

$$\text{demand\_proxy} = \left(\frac{\text{Capacity}_{\text{eff}}}{50}\right) \times \left(\frac{1}{1 + \text{Dist}_{\text{NH}}}\right) \times \ln(e + \text{POI}_{\text{count}}) \times \left(0.5 + 0.5 \times \frac{\text{UrbanPct}}{100}\right)$$

Where:
- $\text{Capacity}_{\text{eff}}$ is station capacity in kW (imputed by charger type if missing)
- $\text{Dist}_{\text{NH}}$ is distance to the nearest National Highway in km
- $\text{POI}_{\text{count}}$ is total commercial and amenity POIs within 1,000m
- $\text{UrbanPct}$ is district urban population percentage

### B. Proof of Deterministic Circularity (Tautology Test)
Taking the natural logarithm of both sides converts the multiplicative formula into an exact linear identity:

$$\ln(\text{demand\_proxy}) = \ln\left(\frac{\text{Capacity}_{\text{eff}}}{50}\right) - \ln(1 + \text{Dist}_{\text{NH}}) + \ln(\ln(e + \text{POI}_{\text{count}})) + \ln\left(0.5 + 0.5 \times \frac{\text{UrbanPct}}{100}\right)$$

An empirical test on the 25 station records in [`integrated_analytical_dataset.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/final/integrated_analytical_dataset.csv) confirmed:
- **Maximum absolute difference between $\ln(\text{demand\_proxy})$ and component sum**: **$0.0026$** (attributable strictly to rounding of published CSV floats).
- **Correlation with $\text{station\_capacity}$**: **$r = 0.9107$**.
- **Correlation with $\text{charging\_power\_kw}$**: **$r = 0.8842$**.

### C. Diagnosis of Target Leakage & Circularity
If a machine learning model (e.g. Random Forest or XGBoost) is trained with `demand_proxy` as the target and given access to station capacity, distance to highway, POI count, or district urbanization:

```
[Candidate Features]                             [Synthetic Target]
  - station_capacity        ----\
  - distance_to_highway_km   ----\   Formula     demand_proxy
  - nearby_poi_count         ---->  --------->   (Algebraic Combination)
  - district_urban_pct      ----/
                                        |
  Model trained:   Features  ---------> | (Learns the algebraic formula)
                   R² = 0.90 - 1.00     | (100% Circular Tautology)
```

1. **The model learns the algebraic formula, not empirical reality**: The high $R^2$ ($\approx 0.90$ on un-transformed trees, $1.00$ in log space) is a statistical illusion. The model is merely reconstructing the arithmetic equation we wrote.
2. **Feature importance is pre-determined by construction**: Capacity was assigned a linear weight, so it accounts for $\sim 70\%$ of tree feature importance by definition, not because real EV drivers prioritize capacity in that exact ratio.
3. **No inductive generalization**: The model cannot predict true charging demand; it only predicts what our formula would output for a new station.

> [!CAUTION]
> **Formal Rule**: `demand_proxy` must **NEVER** be used as the dependent variable ($Y$) in a supervised ML pipeline where its constituent variables are present as independent predictors ($X$). It is **REJECTED** as a final ML target.

---

## 3. Investigation of Real Charging-Demand Data in India & Telangana

A comprehensive investigation of external data repositories, government disclosures, academic literature, and commercial CPOs was conducted:

### A. Commercial Charge Point Operators (CPOs)
- **Entities Investigated**: Tata Power EZ Charge, Statiq, Jio-bp Pulse, Chargezone, Kazam, Bolt.Earth, TGREDCO.
- **Data Status**: **Strictly Confidential / Proprietary**.
- **Findings**:
  - Individual charging sessions, plug-in timestamps, kWh delivered per transaction, and socket occupancy are held in secure Open Charge Point Protocol (OCPP) management systems.
  - No Indian CPO publishes open station-level session telemetry due to commercial competition and customer privacy.
  - Aggregated numbers appear only in selective press releases (e.g. "Tata Power crossed 100M green kilometers").

### B. Official Utility Metered Data (AIKosh / MeitY / TGSPDCL)
- **Entity**: Telangana State Southern Power Distribution Company Limited (TGSPDCL) via **AIKosh (IndiaAI / MeitY)**.
- **Data Title**: *Telangana EV Charging Stations Electricity Consumption Dataset*.
- **License**: Open Government License (OGL), India.
- **Granularity**: Discom administrative units (Circle, Division, Subdivision, Section, Area).
- **Attributes**: Total and billed EV connections, monthly billed electricity load (kW), and total monthly electricity units consumed (kWh / MWh).
- **Assessment**:
  - This is **real, metered empirical data** from the power grid in Telangana.
  - However, it is aggregated at the **utility section / feeder level**, not mapped to individual latitude/longitude station points.
  - **Verdict**: Highly valuable for district/subdivision grid impact modeling in Phase 8, but cannot serve as a station-level socket utilization target for the current 25-station dataset.

### C. Kaggle & Academic Open Data Portals
- **Portals Investigated**: Kaggle ("Electric Vehicle Charging Dataset India", "Plug & Play EV Mobility Usage"), Harvard Dataverse, Zenodo.
- **Findings**:
  - Available Indian session-level datasets on Kaggle are **synthetic simulations** generated using Monte Carlo draws across assumed probability distributions.
  - In accordance with project instructions: *"Do NOT fabricate a target. Do not create synthetic historical demand."* Synthetic datasets are disqualified from serving as ground-truth targets.

---

## 4. Evaluation of Legitimate Alternative Targets

To establish a scientifically defensible machine learning foundation, five alternative target formulations were evaluated:

### Alternative A: Real Measured Charging Demand Telemetry
- *Status*: **UNAVAILABLE** in open public data.
- *Feasibility*: Requires future formal data-sharing agreements with TGREDCO or private CPOs.

### Alternative B: Discom Metered Billed Consumption (TGSPDCL)
- *Status*: **AVAILABLE** via AIKosh (OGL India).
- *Feasibility*: Requires spatial alignment between Discom electrical sections and administrative districts.
- *Best Use*: Feeder-level grid load forecasting (Phase 8).

### Alternative C: Hardware Infrastructure Classification (`fast_charging_flag`)
- *Definition*: Binary classification predicting whether a location warrants a **DC Fast Charger ($\ge 50\text{ kW}$)** vs an **AC Slow/Destination Charger ($< 50\text{ kW}$)**.
- *Inputs*: Road class, distance to National Highway, POI density, distance to city center, district urban percentage.
- *Scientific Validity*: **100% Defensible**. This is a standard spatial classification problem in transport planning: identifying which spatial environments attract high-power highway transit chargers vs commercial destination chargers.
- *Leakage Risk*: **Zero**. Target is an observable physical hardware characteristic, not an algebraic combination of spatial features.

### Alternative D: District-Level EV Adoption Rate (`district_ev_penetration_rate`)
- *Definition*: Panel regression on the 33 Telangana districts over 7 financial years ($N = 231$).
- *Target*: Cumulative EV registrations per 1,000 population.
- *Inputs*: Urbanization tier, population density, highway connectivity, baseline demographics, state policy milestone flags.
- *Scientific Validity*: **High**. Macro-level spatial panel econometric model with zero circularity.

### Alternative E: Multi-Criteria Planning Index (EIPS)
- *Definition*: EV Infrastructure Priority Score calculated via **Multi-Criteria Decision Analysis (AHP / TOPSIS)**.
- *Scientific Validity*: **High**. MCDA is an *unsupervised decision-science framework*, not a supervised ML model. It does not require a synthetic target. It ranks spatial candidate sites based on weighted spatial multi-criteria.

---

## 5. Formal Target Decision Matrix

| Evaluation Dimension | Candidate 1: `demand_proxy` | Candidate 2: `fast_charging_flag` | Candidate 3: `discom_billed_kwh` | Candidate 4: `district_ev_rate` | Candidate 5: `eips_suitability` |
|---|---|---|---|---|---|
| **Problem Type** | Supervised Regression | Supervised Classification | Supervised Regression | Supervised Panel Regression | Multi-Criteria Ranking (MCDA) |
| **Target Nature** | Synthetic Heuristic | Observable Hardware Class | Real Metered Billing | Government Statistics | Weighted Planning Index |
| **Leakage / Circularity** | **Severe (Fatal)** | **None** | **None** | **None** | **None** |
| **Current Sample Size** | $N = 25$ | $N = 25$ (12 labeled) | $N = 0$ (Requires Ingest) | $N = 231$ (33 dist × 7 yrs) | Spatial Grid ($N > 1,000$) |
| **Statistical Viability with Current Data** | **Invalid (Overfitting / Circular)** | **Insufficient ($N=25$)** | **Needs Ingestion** | **Viable ($N=231$)** | **Viable (Unsupervised)** |
| **Minimum Required $N$** | N/A (Methodologically Flawed) | $N \ge 200$ (TS) / $1,500$ (India) | $N \ge 100$ Discom Sections | $N \ge 231$ (Already Met) | $N \ge 500$ Spatial Grids |
| **Supervised ML Suitability** | **REJECTED** | **RECOMMENDED (Post-Expansion)** | **FEASIBLE (Phase 8)** | **FEASIBLE (Macro ML)** | **CORE PROJECT GOAL** |
| **Actionable Decision** | Retain strictly as descriptive baseline | Expand stations $\to$ Train Classifier | Ingest AIKosh in Phase 8 | Develop district panel model | Compute via AHP/TOPSIS |

---

## 6. Phase 7 Gatekeeper Verdict

1. **`demand_proxy` is formally disqualified as a supervised ML target**.
2. **Supervised ML training (Random Forest, XGBoost, LightGBM) is POSTPONED** until the dataset is expanded to a sample size that supports non-overfit cross-validation.
3. **The target for future supervised station-level ML must be reformulated** as a spatial infrastructure classification problem (`fast_charging_flag`) or infrastructure capacity sizing model, rather than predicting an artificial demand proxy.
4. **The project EIPS index must be constructed using Multi-Criteria Decision Analysis (MCDA)**, preserving scientific credibility without circular machine learning.
