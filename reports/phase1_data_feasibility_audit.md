# Phase 1: Data Feasibility Audit Report
## Explainable AI Framework for Smart EV Charging Infrastructure Planning

**Date:** 2026-09-25  
**Status:** COMPLETE  
**Next Phase:** Phase 2 (Environment/Project Setup)

---

## 1. Schema-to-Source Mapping

The 25-field target station schema is mapped to data sources across **three tiers**: direct API, government/open data, and GIS-derived.

### Tier 1: Google Places API (New) — Direct Fields

| # | Field | Google Places Field | Availability | Notes |
|---|-------|-------------------|--------------|-------|
| 1 | `station_name` | `displayName` | HIGH | Standard field |
| 2 | `latitude` | `location.latitude` | HIGH | Standard field |
| 3 | `longitude` | `location.longitude` | HIGH | Standard field |
| 4 | `charger_type` | Inferred from `evChargeOptions` | MEDIUM | Fast/Slow classification |
| 5 | `connector_type` | `evChargeOptions.connectorAggregation[].type` | MEDIUM | CCS2, Type2, CHAdeMO etc. |
| 6 | `number_of_chargers` | `evChargeOptions.connectorCount` | MEDIUM | Total connectors |
| 7 | `charging_power_kw` | `evChargeOptions.connectorAggregation[].maxChargeRateKw` | MEDIUM | Max rate per connector type |
| 8 | `operator` | `displayName` + editorial analysis | MEDIUM | May need parsing |
| 9 | `nearby_poi_count` | Nearby Search (multi-type) | HIGH | Separate API calls per POI type |
| 10 | `parking_availability` | Place types / attributes | LOW | Not always tagged |

**Note:** `evChargeOptions` is an **Enterprise-tier** field — higher billing per request. Budget accordingly.

### Tier 2: Government / Open Data — Supplementary Fields

| # | Field | Source | Format | Availability |
|---|-------|--------|--------|-------------|
| 11 | `state` | Reverse geocoding / manual | - | HIGH |
| 12 | `district` | Census / Admin boundary GIS | GeoJSON/SHP | HIGH |
| 13 | `city` | Google Places / Admin data | - | HIGH |
| 14 | `pincode` | India Post / reverse geocoding | CSV | HIGH |
| 15 | `availability_status` | BEE / e-AMRIT reports | PDF/Web | LOW |
| 16 | `24x7_availability` | BEE / e-AMRIT reports | PDF/Web | LOW |

### Tier 3: GIS-Derived / Computed Fields

| # | Field | Derivation Method | Input Data |
|---|-------|--------------------|-----------|
| 17 | `station_id` | Auto-generated (state prefix + serial) | — |
| 18 | `location_type` | Spatial join with land-use / POI analysis | OSM + Google Places |
| 19 | `road_type` | Nearest road classification from OSM | Geofabrik OSM roads |
| 20 | `highway_name` | Nearest NH/SH name from road network | OSM / NHAI data |
| 21 | `distance_to_highway_km` | Geodesic distance to nearest NH | GeoPandas + Shapely |
| 22 | `distance_to_city_center_km` | Geodesic distance to district HQ | Census towns + Shapely |
| 23 | `traffic_density` | Road density proxy (OSM road segments per km²) | OSM data |
| 24 | `station_capacity` | `number_of_chargers x charging_power_kw` | Computed |
| 25 | `data_timestamp` | Collection timestamp | System-generated |

---

## 2. Government Data Sources Assessment

### 2.1 EV Registration Data (ML Demand Proxy)

| Source | URL | Data | Format | Granularity | Status |
|--------|-----|------|--------|-------------|--------|
| **VAHAN Dashboard** | analytics.parivahan.gov.in | Vehicle registrations by fuel type | Web dashboard | State/RTO | AVAILABLE |
| **Clean Mobility Shift** | cleanmobilityshift.com | EV registrations (2W/3W/4W) | Web dashboard | State/monthly | AVAILABLE |
| **India Data Portal** | indiadataportal.com | VAHAN data exports | CSV/XLSX | State/RTO | AVAILABLE |
| **IEMI (NITI Aayog)** | iemi.niti.gov.in | E-mobility indices | Web | State | AVAILABLE |

### 2.2 Charging Infrastructure Data

| Source | URL | Data | Status |
|--------|-----|------|--------|
| **e-AMRIT** | e-amrit.niti.gov.in | Station locator (map-based) | NO API — map-only, no bulk download |
| **BEE Reports** | beeindia.gov.in | Charger deployment reports | PDF reports — manual extraction |
| **PIB Press Releases** | pib.gov.in | State-wise charger counts | AVAILABLE — periodic releases |
| **CEA Reports** | cea.nic.in | EV charging power consumption | PDF reports — aggregated data |

### 2.3 Population & Census Data

| Source | Data | Format | Notes |
|--------|------|--------|-------|
| **Census 2011** (censusindia.gov.in) | District-wise population | XLS/PDF | Most recent completed census |
| **Census 2021** | — | — | NOT AVAILABLE — postponed due to COVID |

**WARNING:** Census 2021 was never conducted. We will use Census 2011 data with appropriate disclaimers.

### 2.4 Highway & Road Network Data

| Source | Data | Format | Status |
|--------|------|--------|--------|
| **Geofabrik (OSM)** | Full India road network | SHP/PBF | AVAILABLE |
| **Overpass Turbo** | Selective NH queries | GeoJSON | AVAILABLE |
| **PMGSY GeoSadak** | Rural roads | SHP | AVAILABLE |

### 2.5 Electricity Grid Data

| Source | Data | Granularity | Status |
|--------|------|-------------|--------|
| **National Power Portal** | Generation, capacity, demand | State | AVAILABLE |
| **CEA** | Peak demand vs. peak met | State | AVAILABLE |
| **NITI Aayog ICED** | Energy data with downloads | State | AVAILABLE |
| District-level grid data | Feeder monitoring, substation data | District | VERY LIMITED |

### 2.6 GIS Boundary Data

| Source | Data | Format | Notes |
|--------|------|--------|-------|
| **Telangana boundaries (GitHub)** | District/mandal boundaries | GeoJSON | 33 districts, 632 mandals |
| **AIKosh** | Telangana district + mandal shapefiles | SHP | Open Government Data License |
| **DataMeet** | All-India admin boundaries | GeoJSON | Community-maintained |

---

## 3. ML Target Variable — Critical Assessment

**CRITICAL: No open, station-level, real historical charging demand/session/energy/utilization dataset exists for India.**

### What IS Available:
1. **Kaggle "simulated" datasets** — explicitly synthetic, not measured
2. **CEA aggregate consumption** — national/state level, not station-level
3. **Telangana DISCOM billing data** — monthly aggregated kWh, not per-session
4. **Industry reports** — estimated utilization percentages (~10-15%)

### Recommended Approach:
Since no defensible real historical demand target exists, we will:

1. **Primary proxy: EV Registration Density** — State/district-level EV registrations from VAHAN, normalized by area and population
2. **Secondary proxy: Station Capacity Utilization Estimate** — Derived from CEA aggregate power consumption / installed capacity
3. **Clear labeling**: Target will be called `demand_proxy` (never `charging_demand`), with full construction documentation

### What We Will NOT Do:
- Fabricate session-level demand data
- Present proxy as measured demand
- Claim "real-time" utilization without actual real-time data

---

## 4. Data Gaps & Limitations

| Gap | Impact | Mitigation |
|-----|--------|-----------|
| No station-level demand data | Cannot train on actual utilization | Use documented proxy |
| No Census 2021 | Population figures are 15 years old | Use Census 2011 + state estimates |
| No e-AMRIT API | Cannot bulk-download station data | Use Google Places API |
| Limited district-level grid data | Grid readiness hard to measure locally | Use state-level CEA proxy |
| `availability_status`/`24x7_availability` | Not consistently reported | Treat as missing unless from BEE |
| Traffic density | No real-time traffic feed | Use road density proxy from OSM |

---

## 5. Telangana-Specific Notes

- **Formation:** Telangana carved from Andhra Pradesh on 2 June 2014
- **Current districts:** 33 districts (post-2016 reorganization from 10)
- **Handling:** All data will use post-2014 boundaries; no pre-formation values invented
- **Focus justification:** High EV adoption, good data availability, project team's home state

---

## 6. Data Provenance Classification

Every variable in the final dataset will be tagged with one of:

| Tag | Meaning | Example |
|-----|---------|---------|
| `REAL_API` | Direct from Google Places API | `station_name`, `latitude` |
| `REAL_GOV` | Direct from government source | `state`, Census population |
| `GIS_DERIVED` | Calculated via geospatial operations | `distance_to_highway_km` |
| `PROXY` | Defensible substitute for unavailable data | `demand_proxy` |
| `COMPUTED` | Arithmetically derived from other fields | `station_capacity` |
| `CURRENT_SNAPSHOT` | Point-in-time observation, not historical | Google Places data |
| `MANUAL` | Hand-entered from reports | `24x7_availability` |

---

## 7. Phase 1 Deliverables

| Deliverable | File | Status |
|-------------|------|--------|
| Project directory structure | All 15 directories | CREATED |
| Project configuration | config/project_config.yaml | CREATED |
| Python dependencies | requirements.txt | CREATED |
| Environment check script | scripts/phase1_env_check.py | CREATED |
| Data feasibility audit | This report | CREATED |

---

## 8. Actions Required Before Phase 2

1. **Install missing Python packages**: `pip install -r requirements.txt`
2. **Obtain Google Maps API key**: Enable Places API (New) in Google Cloud Console
3. **Create `.env` file** in project root with `GOOGLE_MAPS_API_KEY=your_key`
4. **Confirm focus area**: Telangana (primary) with option to expand later

---

## 9. Next Phase: Phase 2 — Environment/Project Setup

Phase 2 will:
- Install all dependencies
- Validate API key with a live test call
- Create `.env` and `.gitignore`
- Set up logging infrastructure
- Create base utility modules (`config_loader`, `data_utils`, `geo_utils`)
