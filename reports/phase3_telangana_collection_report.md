# Phase 3: Telangana EV Charging Station Collection Report

**Collection Start:** 2026-09-26T07:36:59.801124+00:00
**Collection End:**   2026-09-26T07:36:59.838307+00:00
**Status:** COMPLETE

---

## 1. Collection Summary

| Metric | Count |
|--------|-------|
| OCM records collected (raw) | 19 |
| OSM records collected (raw) | 9 |
| Total raw records | 28 |
| Duplicates detected & removed | 3 |
| **Final unique stations** | **25** |

### Source Breakdown

| Source | Stations | Percentage |
|--------|----------|------------|
| OCM | 17 | 68.0% |
| OSM | 8 | 32.0% |

---

## 2. Field Coverage (25-Field Schema)

| # | Field | Populated | Total | Coverage | Status |
|---|-------|-----------|-------|----------|--------|
| 1 | `station_id` | 25 | 25 | 100.0% | ✅ |
| 2 | `station_name` | 25 | 25 | 100.0% | ✅ |
| 3 | `state` | 25 | 25 | 100.0% | ✅ |
| 4 | `district` | 25 | 25 | 100.0% | ✅ |
| 5 | `city` | 21 | 25 | 84.0% | ✅ |
| 6 | `latitude` | 25 | 25 | 100.0% | ✅ |
| 7 | `longitude` | 25 | 25 | 100.0% | ✅ |
| 8 | `pincode` | 13 | 25 | 52.0% | ⚠️ |
| 9 | `location_type` | 0 | 25 | 0.0% | ❌ |
| 10 | `charger_type` | 13 | 25 | 52.0% | ⚠️ |
| 11 | `connector_type` | 11 | 25 | 44.0% | ⚠️ |
| 12 | `number_of_chargers` | 20 | 25 | 80.0% | ✅ |
| 13 | `charging_power_kw` | 13 | 25 | 52.0% | ⚠️ |
| 14 | `station_capacity` | 13 | 25 | 52.0% | ⚠️ |
| 15 | `availability_status` | 25 | 25 | 100.0% | ✅ |
| 16 | `operator` | 13 | 25 | 52.0% | ⚠️ |
| 17 | `24x7_availability` | 0 | 25 | 0.0% | ❌ |
| 18 | `road_type` | 0 | 25 | 0.0% | ❌ |
| 19 | `highway_name` | 0 | 25 | 0.0% | ❌ |
| 20 | `distance_to_highway_km` | 0 | 25 | 0.0% | ❌ |
| 21 | `distance_to_city_center_km` | 25 | 25 | 100.0% | ✅ |
| 22 | `traffic_density` | 0 | 25 | 0.0% | ❌ |
| 23 | `nearby_poi_count` | 0 | 25 | 0.0% | ❌ |
| 24 | `parking_availability` | 0 | 25 | 0.0% | ❌ |
| 25 | `data_timestamp` | 25 | 25 | 100.0% | ✅ |

**Fields with data: 17/25 (68.0%)**

> Missing fields are either GIS-derived (Phase 4) or not available from OCM/OSM sources.

---

## 3. Geographic Coverage

### District Distribution

| District | Stations |
|----------|----------|
| Hyderabad | 9 |
| Khammam | 4 |
| Ranga Reddy | 2 |
| Sangareddy | 2 |
| Nizamabad | 2 |
| Suryapet | 2 |
| Medchal-Malkajgiri | 1 |
| Nalgonda | 1 |
| Mahabubnagar | 1 |
| Jogulamba Gadwal | 1 |

**Districts with stations: 10**
**Districts in Telangana: 33**

---

## 4. Operator Distribution

| Operator | Stations |
|----------|----------|
| Unknown | 12 |
| Salzer NexCharge (IN) | 2 |
| Chargezone (India) | 2 |
| JIO BP Pulse (India) | 2 |
| Bolt.Earth (IN) | 1 |
| BMW | 1 |
| TGRedCo | 1 |
| Électricité de France | 1 |
| Nuego | 1 |
| Ather grid | 1 |
| FreshBus Charging Point | 1 |

---

## 5. Charger Type Distribution

| Type | Count |
|------|-------|
| Unknown | 12 |
| DC_FAST | 8 |
| AC_LEVEL1 | 1 |
| AC_LEVEL2 | 4 |

---

## 6. Validation Checks

| Check | Result |
|-------|--------|
| unique_station_ids | ✅ PASS |
| all_state_telangana | ✅ PASS |
| coordinates_in_bounds | ✅ PASS |
| all_have_coordinates | ✅ PASS |

---

## 7. Sample Records (Manual Inspection)

### Record 1 (Source: OCM)

| Field | Value |
|-------|-------|
| `station_id` | TS-HYD-0001 |
| `station_name` | Fortum Mahindra |
| `state` | Telangana |
| `district` | Hyderabad |
| `city` | Hyderabad |
| `latitude` | 17.44366 |
| `longitude` | 78.46395 |
| `pincode` | 500016 |
| `number_of_chargers` | 1 |
| `availability_status` | OPERATIONAL |
| `distance_to_city_center_km` | 7.6 |
| `data_timestamp` | 2026-09-26T07:36:59.834106+00:00 |
| `_source_id` | 149534 |

### Record 2 (Source: OCM)

| Field | Value |
|-------|-------|
| `station_id` | TS-RAN-0001 |
| `station_name` | SASS EV charging station⛽ |
| `state` | Telangana |
| `district` | Ranga Reddy |
| `city` | Nallagandla |
| `latitude` | 17.46701200951925 |
| `longitude` | 78.30910573683411 |
| `pincode` | 500019 |
| `charger_type` | DC_FAST |
| `connector_type` | CCS (Type 2) |
| `number_of_chargers` | 4 |
| `charging_power_kw` | 60.0 |
| `station_capacity` | 240.0 |
| `availability_status` | OPERATIONAL |
| `distance_to_city_center_km` | 16.39 |
| `data_timestamp` | 2026-09-26T07:36:59.835187+00:00 |
| `_source_id` | 494161 |

### Record 3 (Source: OCM)

| Field | Value |
|-------|-------|
| `station_id` | TS-NIZ-0001 |
| `station_name` | Sri Sai Kanchetty Family Restaurant |
| `state` | Telangana |
| `district` | Nizamabad |
| `city` | Dichpally |
| `latitude` | 18.632170838340482 |
| `longitude` | 78.20817022692324 |
| `charger_type` | DC_FAST |
| `connector_type` | CCS (Type 2) |
| `number_of_chargers` | 2 |
| `charging_power_kw` | 60.0 |
| `station_capacity` | 120.0 |
| `availability_status` | OPERATIONAL |
| `operator` | Chargezone (India) |
| `distance_to_city_center_km` | 13.14 |
| `data_timestamp` | 2026-09-26T07:36:59.835352+00:00 |
| `_source_id` | 312412 |

### Record 4 (Source: OCM)

| Field | Value |
|-------|-------|
| `station_id` | TS-JOG-0001 |
| `station_name` | NH44 |
| `state` | Telangana |
| `district` | Jogulamba Gadwal |
| `latitude` | 15.886334147441593 |
| `longitude` | 78.01636638843978 |
| `pincode` | 509128 |
| `charger_type` | DC_FAST |
| `connector_type` | CCS (Type 2) |
| `number_of_chargers` | 6 |
| `charging_power_kw` | 240.0 |
| `station_capacity` | 1440.0 |
| `availability_status` | OPERATIONAL |
| `operator` | Chargezone (India) |
| `distance_to_city_center_km` | 44.66 |
| `data_timestamp` | 2026-09-26T07:36:59.835664+00:00 |
| `_source_id` | 475512 |

### Record 5 (Source: OSM)

| Field | Value |
|-------|-------|
| `station_id` | TS-RAN-0002 |
| `station_name` | FreshBus Charging Point |
| `state` | Telangana |
| `district` | Ranga Reddy |
| `city` | Ranga Reddy |
| `latitude` | 17.40474957142857 |
| `longitude` | 78.33329995714287 |
| `availability_status` | OPERATIONAL |
| `operator` | FreshBus Charging Point |
| `distance_to_city_center_km` | 9.53 |
| `data_timestamp` | 2026-09-26T07:36:59.836236+00:00 |
| `_source_id` | 1299079174 |

---

## 8. Known Limitations

1. **OCM coverage depends on community contributions** — not all stations in Telangana may be listed.
2. **OSM data is sparse for India** — very few EV stations are tagged in OpenStreetMap for Telangana.
3. **Operator data** — many OCM entries list "(Unknown Operator)".
4. **Connector details** — some entries lack specific connector type / power rating.
5. **GIS-derived fields** (7 fields) are intentionally null — these require OSM road network data, spatial analysis, and POI queries (Phase 4).
6. **24x7 availability** — rarely captured in OCM/OSM data.
7. **This is NOT a claim of complete Telangana coverage** — it represents all stations available in OCM + OSM as of 2026-09-26T07:36:59.838307+00:00.

---

## 9. Output Files

| File | Description |
|------|-------------|
| `data/raw/telangana_ocm_raw.json` | Raw OCM API responses (19 stations) |
| `data/raw/telangana_osm_raw.json` | Raw OSM Overpass response (9 elements) |
| `data/cleaned/telangana_station_dataset.csv` | Unified dataset (25 stations, 25-field schema) |
| `reports/phase3_telangana_collection_report.md` | This report |
| `data/raw/phase3_collection.log` | Detailed collection log |

---

## 10. Data Provenance

Every record includes provenance columns:
- `_data_source`: OCM or OSM
- `_source_id`: Original ID from source system
- `_query_center`: Which search center found this record
- `_duplicate_of`: If duplicate, ID of the primary record
- `_raw_connector_detail`: Full connector info from API
- `_data_quality_level`: OCM quality rating (1-5)
- `_date_last_verified`: Last verification date from source
- `_missing_fields_reason`: Why each null field is missing
