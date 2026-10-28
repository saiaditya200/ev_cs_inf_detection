# Phase 5 — Data Harmonization Report

## Objective

Establish standard geographic identifiers, temporal formats, and unit conventions so that all Phase 5 government datasets can be seamlessly joined with the Phase 3/4 station dataset for downstream modelling.

---

## 1. Geographic Harmonization

### 1.1 State Name
| Dataset | Raw Value | Harmonized Value |
|---|---|---|
| Station dataset (Phase 3/4) | `Telangana` | `Telangana` |
| EV registrations | `Telangana` | `Telangana` |
| Population | `Telangana` | `Telangana` |
| Electricity | `Telangana` | `Telangana` |

**Status:** ✅ Already consistent — no transformation needed.

### 1.2 District Names
The Phase 3/4 station dataset uses the following district names:

| Station District | Matches Census District? | Notes |
|---|---|---|
| `Hyderabad` | ✅ Yes | 8 stations |
| `Medchal-Malkajgiri` | ✅ Yes | 1 station |
| `Ranga Reddy` | ⚠️ Minor variant | Census uses `Ranga Reddy`; station data uses same |
| `Nalgonda` | ✅ Yes | 1 station |
| `Sangareddy` | ✅ Yes | 2 stations |
| `Nizamabad` | ✅ Yes | 2 stations |
| `Suryapet` | ✅ Yes | 2 stations |
| `Khammam` | ✅ Yes | 4 stations |
| `Mahabubnagar` | ✅ Yes | 1 station |
| `Jogulamba Gadwal` | ✅ Yes | 1 station |

**Status:** ✅ All 10 unique station districts match Census district names exactly.

### 1.3 Geographic Join Key
```
JOIN KEY: state + district
Example:  "Telangana" + "Hyderabad" → unique district identifier
```

This join enables linking station-level data to district-level population, density, and urbanization metrics.

---

## 2. Temporal Harmonization

### 2.1 Financial Year Convention
Indian government datasets use Financial Years (Apr-Mar). Our standard format:

```
Format:   FYyyyy-yy
Examples: FY2023-24, FY2024-25
```

All EV registration and electricity datasets use this convention consistently.

### 2.2 Calendar Year vs Financial Year Mapping

| FY Period | Calendar Overlap | Data Available |
|---|---|---|
| FY2023-24 | Apr 2023 – Mar 2024 | EV reg, Electricity |
| FY2024-25 | Apr 2024 – Mar 2025 | EV reg, Electricity |
| FY2025-26 | Apr 2025 – Mar 2026 | EV reg (partial) |

### 2.3 Census Year
Population data is fixed at **Census 2011**. For analyses requiring contemporary population estimates, use the NCP projections in `population_telangana_projections.csv` for state-level scaling.

---

## 3. Unit Standardization

| Metric | Raw Unit | Standard Unit | Conversion |
|---|---|---|---|
| Population | persons | persons | None |
| Area | sq km | sq km | None |
| Density | persons/sq km | persons/sq km | None |
| EV registrations | count | count | None |
| Highway length | km | km | None |
| Electricity consumption | MU (GWh) | MU | None (1 MU = 1 GWh) |
| Per capita electricity | kWh | kWh | None |
| Installed capacity | MW | MW | None |
| Charging stations | count | count | None |
| Charging power | kW | kW | None |

**Status:** ✅ All units are already in standard SI-compatible format.

---

## 4. Cross-Dataset Join Specification

### 4.1 Station → Population (District-level)
```
station_gis_enriched.district = population_telangana_districts.district
```
Enables: Population density, urban % as station-level contextual features.

### 4.2 Station → EV Registrations (State-level)
```
station_gis_enriched.state = ev_registrations_telangana.state
```
Enables: State-level EV adoption trends as temporal features. Cannot disaggregate to district (VAHAN limitation).

### 4.3 Station → Highway Corridors
```
station_gis_enriched.highway_name → nh_corridors_telangana.nh_number
```
Enables: Linking stations to specific NH corridor metadata (length, lane config).

### 4.4 Station → Electricity (State-level)
```
station_gis_enriched.state = electricity_telangana.state
```
Enables: Grid readiness proxy features (per capita consumption, installed capacity).

---

## 5. Data Completeness Matrix

| Dataset | Geographic Completeness | Temporal Completeness | Key Gaps |
|---|---|---|---|
| EV Registrations (TS) | State only | FY2019-20 to FY2025-26 | No district-level breakdown |
| Population | All 33 districts | 2011 only | No 2021+ district counts |
| Electricity | State + India | FY2019-20 to FY2023-24 | No district-level data |
| FAME Infra | State + India | 2024 snapshot | No historical progression |
| Highway | State + India | 2024-25 | No historical NH length |
| NH Corridors | 6 major NHs | Current | Minor NHs not captured |
| EV Policy | State + National | 2019-2026 | Minor amendments omitted |

---

## 6. Recommendations for Downstream Use

1. **Demand Proxy:** Use `ev_registrations_telangana` / `population` ratio as the primary EV adoption density proxy per district (assuming proportional registration distribution weighted by urban population).

2. **Grid Readiness Proxy:** Use `per_capita_consumption_kwh` + `installed_capacity_mw` as grid readiness indicators.

3. **Infrastructure Gap:** Compare `ev_charging_infrastructure_fame.stations_installed` against `ev_registrations` to compute charging-station-per-EV ratios.

4. **Highway Corridor Analysis:** Use `nh_corridors_telangana` to identify underserved highway segments by cross-referencing with station `highway_name` + `distance_to_highway_km`.

> [!IMPORTANT]
> Do NOT create artificial station-level historical records from state-level aggregates. The temporal mismatch between state-level FY registration data and station-level point-in-time snapshots must be handled explicitly in the modelling phase.
