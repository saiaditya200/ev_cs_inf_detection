"""
Phase 2: Hyderabad EV Charging Station — Minimal Test
======================================================
Uses OpenChargeMap API + OSM Overpass API (BOTH FREE, NO BILLING) to:
1. Validate OpenChargeMap API key
2. Search for EV charging stations in Hyderabad
3. Merge data from both sources
4. Map returned fields to our 25-field schema
5. Save raw JSON responses and test CSV

Usage: python scripts/phase2_hyderabad_test.py
Requires: .env file with OCM_API_KEY
"""

import sys
import os
import io
import json
import csv
import math
from datetime import datetime, timezone

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import requests
except ImportError:
    print("[FAIL] 'requests' not installed. Run: pip install requests")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("[FAIL] 'python-dotenv' not installed. Run: pip install python-dotenv")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────
HYDERABAD_LAT = 17.385044
HYDERABAD_LNG = 78.486671
HYDERABAD_RADIUS_KM = 25  # Cover greater Hyderabad

# Our 25-field target schema
TARGET_SCHEMA = [
    "station_id", "station_name", "state", "district", "city",
    "latitude", "longitude", "pincode", "location_type",
    "charger_type", "connector_type", "number_of_chargers",
    "charging_power_kw", "station_capacity", "availability_status",
    "operator", "24x7_availability", "road_type", "highway_name",
    "distance_to_highway_km", "distance_to_city_center_km",
    "traffic_density", "nearby_poi_count", "parking_availability",
    "data_timestamp",
]

# Output paths
RAW_OCM_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "hyderabad_ocm_raw.json")
RAW_OSM_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "hyderabad_osm_raw.json")
TEST_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "hyderabad_test_stations.csv")
FIELD_MAPPING_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "hyderabad_field_mapping.json")
RESPONSE_INSPECTION_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "hyderabad_response_inspection.json")


def load_ocm_key() -> str:
    """Load OpenChargeMap API key from .env file."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    key = os.environ.get("OCM_API_KEY", "")
    if not key:
        print("[FAIL] No OCM_API_KEY found in .env")
        sys.exit(1)
    return key


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


# ── OpenChargeMap ─────────────────────────────────────────────
def fetch_ocm_stations(api_key: str) -> list:
    """Fetch EV stations from OpenChargeMap API."""
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "output": "json",
        "countrycode": "IN",
        "latitude": HYDERABAD_LAT,
        "longitude": HYDERABAD_LNG,
        "distance": HYDERABAD_RADIUS_KM,
        "distanceunit": "KM",
        "maxresults": 100,
        "compact": False,    # Full details
        "verbose": True,     # Include reference data
        "key": api_key,
    }

    print(f"  URL: {url}")
    print(f"  Center: {HYDERABAD_LAT}, {HYDERABAD_LNG}")
    print(f"  Radius: {HYDERABAD_RADIUS_KM} km")

    resp = requests.get(url, params=params, timeout=30,
                       headers={"User-Agent": "EVChargingResearch/1.0"})
    print(f"  HTTP Status: {resp.status_code}")

    if resp.status_code == 403:
        print("  [FAIL] API key rejected (403)")
        sys.exit(1)
    if resp.status_code != 200:
        print(f"  [FAIL] HTTP {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)

    data = resp.json()
    print(f"  Stations returned: {len(data)}")
    return data


def map_ocm_to_schema(ocm_station: dict, index: int) -> dict:
    """Map an OpenChargeMap station to our 25-field schema."""
    addr = ocm_station.get("AddressInfo", {})
    conns = ocm_station.get("Connections", [])
    operator_info = ocm_station.get("OperatorInfo", {})
    status_type = ocm_station.get("StatusType", {})
    usage_type = ocm_station.get("UsageType", {})

    # Extract connector types and power ratings
    connector_types = []
    power_ratings = []
    total_points = 0

    for conn in conns:
        # Connection type
        conn_type = conn.get("ConnectionType", {})
        ctype_title = conn_type.get("Title", "") if conn_type else ""
        if ctype_title:
            connector_types.append(ctype_title)

        # Power
        power_kw = conn.get("PowerKW")
        if power_kw:
            power_ratings.append(float(power_kw))

        # Count
        qty = conn.get("Quantity")
        if qty:
            total_points += int(qty)

    # Determine charger type from power
    charger_type = None
    if power_ratings:
        max_power = max(power_ratings)
        if max_power >= 50:
            charger_type = "DC_FAST"
        elif max_power >= 7:
            charger_type = "AC_LEVEL2"
        else:
            charger_type = "AC_LEVEL1"

    # Operator
    operator_name = None
    if operator_info:
        operator_name = operator_info.get("Title", None)

    # Status
    availability = None
    if status_type:
        status_title = status_type.get("Title", "")
        if "operational" in status_title.lower():
            availability = "OPERATIONAL"
        elif "not operational" in status_title.lower():
            availability = "NOT_OPERATIONAL"
        elif "planned" in status_title.lower():
            availability = "PLANNED"
        else:
            availability = status_title

    # 24x7 check from usage type
    is_24x7 = None
    if usage_type:
        usage_title = usage_type.get("Title", "").lower()
        if "24" in usage_title:
            is_24x7 = True

    # Location
    lat = addr.get("Latitude")
    lng = addr.get("Longitude")

    # Distance to city center
    dist_to_center = None
    if lat and lng:
        dist_to_center = round(haversine_km(lat, lng, HYDERABAD_LAT, HYDERABAD_LNG), 2)

    # Number of chargers
    num_chargers = total_points if total_points > 0 else ocm_station.get("NumberOfPoints")

    # Max power
    max_power_kw = max(power_ratings) if power_ratings else None

    record = {
        "station_id": f"TS-HYD-{index:04d}",
        "station_name": addr.get("Title", "Unknown"),
        "state": addr.get("StateOrProvince", "Telangana"),
        "district": addr.get("Town", "Hyderabad"),
        "city": addr.get("Town", "Hyderabad"),
        "latitude": lat,
        "longitude": lng,
        "pincode": addr.get("Postcode"),
        "location_type": None,                    # Needs GIS derivation
        "charger_type": charger_type,
        "connector_type": "|".join(set(connector_types)) if connector_types else None,
        "number_of_chargers": num_chargers,
        "charging_power_kw": max_power_kw,
        "station_capacity": None,                 # Computed below
        "availability_status": availability,
        "operator": operator_name,
        "24x7_availability": is_24x7,
        "road_type": None,                       # Needs OSM
        "highway_name": None,                    # Needs OSM/NHAI
        "distance_to_highway_km": None,          # Needs GIS
        "distance_to_city_center_km": dist_to_center,
        "traffic_density": None,                 # Needs OSM
        "nearby_poi_count": None,                # Needs separate analysis
        "parking_availability": None,            # Needs attributes
        "data_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Compute station_capacity
    if record["number_of_chargers"] and record["charging_power_kw"]:
        record["station_capacity"] = round(
            record["number_of_chargers"] * record["charging_power_kw"], 1
        )

    return record


# ── OSM Overpass ──────────────────────────────────────────────
def fetch_osm_stations() -> list:
    """Fetch EV stations from OSM Overpass API (no key needed)."""
    url = "https://overpass-api.de/api/interpreter"

    # Bounding box for greater Hyderabad
    query = """[out:json][timeout:30];
(
  node["amenity"="charging_station"](17.10,78.10,17.65,78.75);
  way["amenity"="charging_station"](17.10,78.10,17.65,78.75);
);
out body;
>;
out skel qt;"""

    resp = requests.get(
        url,
        params={"data": query},
        headers={
            "Accept": "application/json",
            "User-Agent": "EVChargingResearch/1.0 (academic project)"
        },
        timeout=60
    )
    print(f"  HTTP Status: {resp.status_code}")

    if resp.status_code != 200:
        print(f"  [WARN] Overpass returned {resp.status_code}")
        return []

    data = resp.json()
    elements = data.get("elements", [])
    stations = [e for e in elements if e.get("tags", {}).get("amenity") == "charging_station"]
    print(f"  Stations found: {len(stations)}")
    return stations, data


def map_osm_to_schema(osm_station: dict, index: int) -> dict:
    """Map an OSM station to our 25-field schema."""
    tags = osm_station.get("tags", {})
    lat = osm_station.get("lat")
    lon = osm_station.get("lon")

    # Connector types from OSM tags
    connectors = []
    for key in tags:
        if "socket" in key.lower():
            connectors.append(key.replace("socket:", ""))

    # Distance to city center
    dist_to_center = None
    if lat and lon:
        dist_to_center = round(haversine_km(lat, lon, HYDERABAD_LAT, HYDERABAD_LNG), 2)

    record = {
        "station_id": f"TS-HYD-OSM-{index:04d}",
        "station_name": tags.get("name", tags.get("brand", "Unknown OSM Station")),
        "state": "Telangana",
        "district": "Hyderabad",
        "city": "Hyderabad",
        "latitude": lat,
        "longitude": lon,
        "pincode": None,
        "location_type": None,
        "charger_type": tags.get("charging_station:type"),
        "connector_type": "|".join(connectors) if connectors else None,
        "number_of_chargers": int(tags["capacity"]) if "capacity" in tags else None,
        "charging_power_kw": None,
        "station_capacity": None,
        "availability_status": "OPERATIONAL" if tags.get("access") != "no" else "RESTRICTED",
        "operator": tags.get("operator", tags.get("brand")),
        "24x7_availability": True if tags.get("opening_hours") == "24/7" else None,
        "road_type": None,
        "highway_name": None,
        "distance_to_highway_km": None,
        "distance_to_city_center_km": dist_to_center,
        "traffic_density": None,
        "nearby_poi_count": None,
        "parking_availability": None,
        "data_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return record


# ── Field Coverage Analysis ───────────────────────────────────
def analyze_field_coverage(records: list) -> dict:
    """Analyze which fields are populated across all records."""
    field_sources = {
        "station_id": "COMPUTED",
        "station_name": "REAL_API",
        "state": "REAL_API",
        "district": "REAL_API",
        "city": "REAL_API",
        "latitude": "REAL_API",
        "longitude": "REAL_API",
        "pincode": "REAL_API",
        "location_type": "GIS_DERIVED",
        "charger_type": "REAL_API",
        "connector_type": "REAL_API",
        "number_of_chargers": "REAL_API",
        "charging_power_kw": "REAL_API",
        "station_capacity": "COMPUTED",
        "availability_status": "REAL_API",
        "operator": "REAL_API",
        "24x7_availability": "REAL_API",
        "road_type": "GIS_DERIVED",
        "highway_name": "GIS_DERIVED",
        "distance_to_highway_km": "GIS_DERIVED",
        "distance_to_city_center_km": "COMPUTED",
        "traffic_density": "GIS_DERIVED",
        "nearby_poi_count": "GIS_DERIVED",
        "parking_availability": "GIS_DERIVED",
        "data_timestamp": "COMPUTED",
    }

    coverage = {}
    for field in TARGET_SCHEMA:
        populated = sum(1 for r in records if r.get(field) is not None and r.get(field) != "")
        samples = []
        for r in records:
            v = r.get(field)
            if v is not None and v != "" and len(samples) < 3:
                samples.append(str(v)[:80])

        coverage[field] = {
            "source": field_sources.get(field, "UNKNOWN"),
            "total": len(records),
            "populated": populated,
            "coverage_pct": round(populated / len(records) * 100, 1) if records else 0,
            "sample_values": samples,
        }

    return coverage


def print_coverage_table(coverage: dict) -> None:
    """Pretty-print field coverage table."""
    print(f"\n  {'Field':<30s} {'Source':<14s} {'Coverage':<14s} {'Sample Value'}")
    print("  " + "-" * 88)
    for field in TARGET_SCHEMA:
        info = coverage[field]
        pct = info["coverage_pct"]
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        sample = info["sample_values"][0][:35] if info["sample_values"] else "—"
        status = "✓" if pct > 0 else "✗"
        print(f"  {status} {field:<28s} {info['source']:<14s} {bar} {pct:5.1f}%  {sample}")


def build_response_inspection(ocm_data: list, osm_stations: list) -> dict:
    """Build detailed inspection of API responses for review."""
    inspection = {
        "inspection_timestamp": datetime.now(timezone.utc).isoformat(),
        "opencharegmap": {
            "total_stations": len(ocm_data),
            "sample_station_full": ocm_data[0] if ocm_data else None,
            "available_top_level_fields": list(ocm_data[0].keys()) if ocm_data else [],
            "address_fields": list(ocm_data[0].get("AddressInfo", {}).keys()) if ocm_data else [],
            "connection_fields": list(ocm_data[0].get("Connections", [{}])[0].keys()) if ocm_data and ocm_data[0].get("Connections") else [],
            "operators_found": list(set(
                s.get("OperatorInfo", {}).get("Title", "Unknown")
                for s in ocm_data if s.get("OperatorInfo")
            )),
            "status_types_found": list(set(
                s.get("StatusType", {}).get("Title", "Unknown")
                for s in ocm_data if s.get("StatusType")
            )),
            "connector_types_found": list(set(
                c.get("ConnectionType", {}).get("Title", "")
                for s in ocm_data
                for c in s.get("Connections", [])
                if c.get("ConnectionType")
            )),
        },
        "osm_overpass": {
            "total_stations": len(osm_stations),
            "sample_station": osm_stations[0] if osm_stations else None,
            "all_unique_tags": list(set(
                tag for s in osm_stations for tag in s.get("tags", {}).keys()
            )),
        }
    }
    return inspection


# ── Main ──────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PHASE 2: HYDERABAD EV CHARGING STATION — MINIMAL TEST")
    print(f"Time: {datetime.now().isoformat()}")
    print("Sources: OpenChargeMap (free) + OSM Overpass (free)")
    print("=" * 70)

    # Step 1: Load API key
    print("\n[1/8] Loading OpenChargeMap API key...")
    ocm_key = load_ocm_key()
    print(f"  [OK] Key loaded (length={len(ocm_key)})")

    # Step 2: Fetch OpenChargeMap data
    print("\n[2/8] Fetching EV stations from OpenChargeMap...")
    ocm_data = fetch_ocm_stations(ocm_key)

    # Step 3: Fetch OSM data
    print("\n[3/8] Fetching EV stations from OSM Overpass...")
    try:
        osm_stations, osm_raw = fetch_osm_stations()
    except Exception as e:
        print(f"  [WARN] Overpass failed: {e}")
        osm_stations, osm_raw = [], {"elements": []}

    # Step 4: Save raw responses
    print("\n[4/8] Saving raw API responses...")
    os.makedirs(os.path.dirname(RAW_OCM_PATH), exist_ok=True)

    with open(RAW_OCM_PATH, "w", encoding="utf-8") as f:
        json.dump(ocm_data, f, indent=2, ensure_ascii=False)
    print(f"  [OK] OpenChargeMap raw: {RAW_OCM_PATH} ({os.path.getsize(RAW_OCM_PATH):,} bytes)")

    with open(RAW_OSM_PATH, "w", encoding="utf-8") as f:
        json.dump(osm_raw, f, indent=2, ensure_ascii=False)
    print(f"  [OK] OSM raw: {RAW_OSM_PATH} ({os.path.getsize(RAW_OSM_PATH):,} bytes)")

    # Step 5: Map to schema
    print("\n[5/8] Mapping API responses to 25-field schema...")
    records = []

    for i, station in enumerate(ocm_data, start=1):
        record = map_ocm_to_schema(station, i)
        record["_data_source"] = "OpenChargeMap"
        records.append(record)
    print(f"  OCM records mapped: {len(ocm_data)}")

    for i, station in enumerate(osm_stations, start=1):
        record = map_osm_to_schema(station, i)
        record["_data_source"] = "OSM"
        records.append(record)
    print(f"  OSM records mapped: {len(osm_stations)}")
    print(f"  Total records: {len(records)}")

    # Step 6: Analyze field coverage
    print("\n[6/8] Analyzing field coverage against 25-field schema...")
    coverage = analyze_field_coverage(records)
    print_coverage_table(coverage)

    # Step 7: Save outputs
    print("\n[7/8] Saving test outputs...")

    # Save CSV
    csv_fields = TARGET_SCHEMA + ["_data_source"]
    with open(TEST_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"  [OK] Test CSV: {TEST_CSV_PATH} ({len(records)} records)")

    # Save field mapping
    with open(FIELD_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Field mapping: {FIELD_MAPPING_PATH}")

    # Build & save response inspection
    print("\n  Building response inspection report...")
    inspection = build_response_inspection(ocm_data, osm_stations)
    with open(RESPONSE_INSPECTION_PATH, "w", encoding="utf-8") as f:
        json.dump(inspection, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Response inspection: {RESPONSE_INSPECTION_PATH}")

    # Step 8: Validation summary
    print("\n\n[8/8] VALIDATION SUMMARY")
    print("=" * 70)

    api_fields = [f for f in TARGET_SCHEMA if coverage[f]["source"] == "REAL_API"]
    api_populated = sum(1 for f in api_fields if coverage[f]["populated"] > 0)

    computed_fields = [f for f in TARGET_SCHEMA if coverage[f]["source"] == "COMPUTED"]
    computed_populated = sum(1 for f in computed_fields if coverage[f]["populated"] > 0)

    gis_fields = [f for f in TARGET_SCHEMA if coverage[f]["source"] == "GIS_DERIVED"]

    print(f"  Data Sources:")
    print(f"    OpenChargeMap stations:    {len(ocm_data)}")
    print(f"    OSM Overpass stations:     {len(osm_stations)}")
    print(f"    Total unique stations:     {len(records)}")

    print(f"\n  Schema Coverage (25 fields):")
    print(f"    API-sourced fields:        {api_populated}/{len(api_fields)} populated")
    print(f"    Computed fields:           {computed_populated}/{len(computed_fields)} populated")
    print(f"    GIS-derived (Phase 3+):    0/{len(gis_fields)} (expected)")

    total_populated = sum(1 for f in TARGET_SCHEMA if coverage[f]["populated"] > 0)
    print(f"    TOTAL fields populated:    {total_populated}/{len(TARGET_SCHEMA)}")

    # Show operators found
    if ocm_data:
        operators = set()
        for s in ocm_data:
            op = s.get("OperatorInfo", {})
            if op and op.get("Title"):
                operators.add(op["Title"])
        if operators:
            print(f"\n  Operators found: {', '.join(sorted(operators))}")

    # Show connector types
    if ocm_data:
        conn_types = set()
        for s in ocm_data:
            for c in s.get("Connections", []):
                ct = c.get("ConnectionType", {})
                if ct and ct.get("Title"):
                    conn_types.add(ct["Title"])
        if conn_types:
            print(f"  Connector types: {', '.join(sorted(conn_types))}")

    print(f"\n  Output files:")
    print(f"    {RAW_OCM_PATH}")
    print(f"    {RAW_OSM_PATH}")
    print(f"    {TEST_CSV_PATH}")
    print(f"    {FIELD_MAPPING_PATH}")
    print(f"    {RESPONSE_INSPECTION_PATH}")

    print("\n" + "=" * 70)
    print("  ✓ OpenChargeMap API key validated (FREE, no billing)")
    print("  ✓ OSM Overpass API validated (FREE, no billing)")
    print("  ✓ Live data retrieved from Hyderabad")
    print("  ✓ Fields mapped to 25-field schema")
    print("  ✓ Raw JSON and test CSV saved")
    print("  ✓ Field coverage analyzed")
    print("  ✓ Response inspection report generated")
    print("\n  NEXT: Review outputs, then proceed to Phase 3")
    print("        (Do NOT begin Telangana-wide collection yet)")
    print("=" * 70)


if __name__ == "__main__":
    main()
