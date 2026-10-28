"""
Phase 3: Telangana-Wide EV Charging Station Collection
=======================================================
Systematic collection from OpenChargeMap (primary) + OSM Overpass (supplementary).

Strategy:
  1. OCM statewide search from Telangana center with large radius
  2. OCM per-city searches to catch edge cases
  3. OSM Overpass bounding-box query for entire Telangana
  4. Deduplicate across OCM results (by OCM ID)
  5. Deduplicate across OCM + OSM (by coordinate proximity + name similarity)
  6. Map to 25-field schema with provenance
  7. Validate and save

Usage:
  python scripts/phase3_telangana_collection.py
  python scripts/phase3_telangana_collection.py --resume   (resume from checkpoint)
"""

import sys
import os
import io
import json
import csv
import math
import time
import logging
import hashlib
import argparse
from datetime import datetime, timezone
from difflib import SequenceMatcher
from collections import Counter

# Force UTF-8 on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("[FAIL] 'requests' not installed"); sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("[FAIL] 'python-dotenv' not installed"); sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("[FAIL] 'pandas' not installed"); sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────
CONFIG_PATH     = os.path.join(PROJECT_ROOT, "config", "telangana_districts.json")
RAW_OCM_PATH    = os.path.join(PROJECT_ROOT, "data", "raw", "telangana_ocm_raw.json")
RAW_OSM_PATH    = os.path.join(PROJECT_ROOT, "data", "raw", "telangana_osm_raw.json")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "telangana_checkpoint.json")
CLEANED_CSV     = os.path.join(PROJECT_ROOT, "data", "cleaned", "telangana_station_dataset.csv")
REPORT_PATH     = os.path.join(PROJECT_ROOT, "reports", "phase3_telangana_collection_report.md")

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

PROVENANCE_COLS = [
    "_data_source",          # OCM or OSM
    "_source_id",            # OCM ID or OSM node ID
    "_query_center",         # Search center used
    "_duplicate_of",         # ID of primary record if duplicate
    "_raw_connector_detail", # Full connector info from API
    "_data_quality_level",   # OCM data quality level (1-5)
    "_date_last_verified",   # When station was last verified
    "_missing_fields_reason",# Why fields are null
]

# ── Setup logging ─────────────────────────────────────────────
os.makedirs(os.path.join(PROJECT_ROOT, "data", "raw"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "data", "cleaned"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)

log_path = os.path.join(PROJECT_ROOT, "data", "raw", "phase3_collection.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("phase3")


# ── Utility functions ─────────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two points."""
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def haversine_km(lat1, lon1, lat2, lon2):
    return haversine_m(lat1, lon1, lat2, lon2) / 1000.0


def name_similarity(a: str, b: str) -> float:
    """Normalized name similarity (0-1)."""
    if not a or not b:
        return 0.0
    a_clean = a.lower().strip()
    b_clean = b.lower().strip()
    if a_clean == b_clean:
        return 1.0
    return SequenceMatcher(None, a_clean, b_clean).ratio()


def build_session(retries=3, backoff=1.0):
    """Build a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "EVChargingResearch/1.0 (BTech academic project)",
        "Accept": "application/json",
    })
    return session


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_searches": [], "ocm_stations_raw": [], "osm_done": False}


def save_checkpoint(cp):
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(cp, f, indent=2, ensure_ascii=False)


def point_in_telangana(lat, lng, config):
    """Check if a point falls within Telangana's bounding box (loose check)."""
    bb = config["bounding_box"]
    margin = 0.15  # ~16km margin
    return (bb["south"] - margin <= lat <= bb["north"] + margin
            and bb["west"] - margin <= lng <= bb["east"] + margin)


def find_nearest_district(lat, lng, config):
    """Find the nearest district HQ to a given point."""
    best = None
    best_dist = float("inf")
    for d in config["districts"]:
        dist = haversine_km(lat, lng, d["lat"], d["lng"])
        if dist < best_dist:
            best_dist = dist
            best = d["name"]
    return best, best_dist


# ── OCM Collection ────────────────────────────────────────────
def fetch_ocm_search(session, api_key, lat, lng, radius_km, max_results=500):
    """Single OCM API search. Returns list of station dicts."""
    params = {
        "output": "json",
        "countrycode": "IN",
        "latitude": lat,
        "longitude": lng,
        "distance": radius_km,
        "distanceunit": "KM",
        "maxresults": max_results,
        "compact": False,
        "verbose": True,
        "key": api_key,
    }
    resp = session.get(
        "https://api.openchargemap.io/v3/poi/",
        params=params,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def collect_ocm(session, api_key, config, checkpoint, use_raw=False):
    """
    Collect all OCM stations across Telangana.
    Strategy:
      1. Single statewide search from center with 300km radius
      2. Per-major-city searches to catch edge cases
      3. Deduplicate by OCM ID
    """
    if use_raw and os.path.exists(RAW_OCM_PATH):
        logger.info("  [SKIP] Loading OCM from raw file")
        with open(RAW_OCM_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    all_stations = {}  # keyed by OCM ID to auto-deduplicate
    rate_limit = config["ocm_api"]["rate_limit_seconds"]

    # Load previously collected stations from checkpoint
    for s in checkpoint.get("ocm_stations_raw", []):
        ocm_id = s.get("ID")
        if ocm_id:
            all_stations[ocm_id] = s

    search_tasks = []

    # Task 1: Statewide search from center
    center = config["center"]
    search_tasks.append({
        "name": "Statewide-Center",
        "lat": center["latitude"],
        "lng": center["longitude"],
        "radius_km": 300,  # Covers entire Telangana from center
    })

    # Task 2: Per-city searches
    for city in config["major_cities"]:
        search_tasks.append({
            "name": city["name"],
            "lat": city["lat"],
            "lng": city["lng"],
            "radius_km": city["search_radius_km"],
        })

    completed = set(checkpoint.get("completed_searches", []))

    for task in search_tasks:
        task_key = f"OCM:{task['name']}"
        if task_key in completed:
            logger.info(f"  [SKIP] {task['name']} (already completed)")
            continue

        logger.info(f"  Searching: {task['name']} "
                     f"({task['lat']:.3f}, {task['lng']:.3f}, r={task['radius_km']}km)")

        try:
            results = fetch_ocm_search(
                session, api_key,
                task["lat"], task["lng"], task["radius_km"],
                max_results=config["ocm_api"]["max_results_per_call"],
            )

            new_count = 0
            for station in results:
                ocm_id = station.get("ID")
                if ocm_id and ocm_id not in all_stations:
                    # Check if it's in/near Telangana
                    addr = station.get("AddressInfo", {})
                    lat = addr.get("Latitude")
                    lng = addr.get("Longitude")
                    state = (addr.get("StateOrProvince") or "").strip()

                    if state.lower() in ("telangana", "ts", ""):
                        # If state is empty, check coordinates
                        if lat and lng:
                            if point_in_telangana(lat, lng, config) or state.lower() == "telangana":
                                all_stations[ocm_id] = station
                                station["_query_center"] = task["name"]
                                new_count += 1
                        elif state.lower() == "telangana":
                            all_stations[ocm_id] = station
                            station["_query_center"] = task["name"]
                            new_count += 1
                    elif state.lower() == "telangana":
                        all_stations[ocm_id] = station
                        station["_query_center"] = task["name"]
                        new_count += 1

            logger.info(f"    → {len(results)} results, {new_count} new Telangana stations")

            # Update checkpoint
            completed.add(task_key)
            checkpoint["completed_searches"] = list(completed)
            checkpoint["ocm_stations_raw"] = list(all_stations.values())
            save_checkpoint(checkpoint)

        except requests.exceptions.RequestException as e:
            logger.error(f"    [ERROR] {task['name']}: {e}")
            # Continue with next search

        time.sleep(rate_limit)

    logger.info(f"  OCM collection complete: {len(all_stations)} unique Telangana stations")
    return list(all_stations.values())


def extract_osm_stations(elements: list) -> list:
    """Extract charging stations and compute centroids for ways from member nodes."""
    node_coords = {
        e["id"]: (e["lat"], e["lon"])
        for e in elements
        if e.get("type") == "node" and "lat" in e and "lon" in e
    }
    stations = []
    for e in elements:
        if e.get("tags", {}).get("amenity") == "charging_station":
            # If way and lacks lat/lon, compute centroid from member nodes
            if e.get("type") == "way" and "nodes" in e and ("lat" not in e or "lon" not in e):
                pts = [node_coords[nid] for nid in e["nodes"] if nid in node_coords]
                if pts:
                    e["lat"] = sum(p[0] for p in pts) / len(pts)
                    e["lon"] = sum(p[1] for p in pts) / len(pts)
            stations.append(e)
    return stations


# ── OSM Collection ────────────────────────────────────────────
def collect_osm(session, config, checkpoint, use_raw=False):
    """
    Collect all OSM EV charging stations in Telangana via Overpass.
    Single bounding-box query for the entire state.
    """
    if use_raw or checkpoint.get("osm_done"):
        logger.info("  [SKIP] Loading OSM from raw file")
        if os.path.exists(RAW_OSM_PATH):
            with open(RAW_OSM_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            elements = data.get("elements", [])
            return extract_osm_stations(elements)
        return []

    bb = config["bounding_box"]
    query = f"""[out:json][timeout:90];
(
  node["amenity"="charging_station"]({bb['south']},{bb['west']},{bb['north']},{bb['east']});
  way["amenity"="charging_station"]({bb['south']},{bb['west']},{bb['north']},{bb['east']});
);
out body;
>;
out skel qt;"""

    logger.info(f"  Overpass query: bbox=({bb['south']},{bb['west']},"
                f"{bb['north']},{bb['east']})")

    resp = session.get(
        config["osm_overpass"]["base_url"],
        params={"data": query},
        timeout=config["osm_overpass"]["timeout_seconds"],
    )
    resp.raise_for_status()
    data = resp.json()

    # Save raw
    with open(RAW_OSM_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"  OSM raw saved: {RAW_OSM_PATH} ({os.path.getsize(RAW_OSM_PATH):,} bytes)")

    elements = data.get("elements", [])
    stations = extract_osm_stations(elements)

    logger.info(f"  OSM stations found: {len(stations)}")

    checkpoint["osm_done"] = True
    save_checkpoint(checkpoint)

    return stations


# ── Schema Mapping ────────────────────────────────────────────
def map_ocm_record(station: dict, config: dict) -> dict:
    """Map an OCM station to the 25-field schema + provenance columns."""
    addr = station.get("AddressInfo", {})
    conns = station.get("Connections", [])
    op_info = station.get("OperatorInfo", {})
    status = station.get("StatusType", {})
    usage = station.get("UsageType", {})

    lat = addr.get("Latitude")
    lng = addr.get("Longitude")

    # Connector analysis
    connector_types = []
    power_ratings = []
    total_points = 0
    connector_detail = []

    for c in conns:
        ct = c.get("ConnectionType", {})
        ct_title = ct.get("Title", "") if ct else ""
        if ct_title and ct_title != "Unknown":
            connector_types.append(ct_title)

        pwr = c.get("PowerKW")
        if pwr:
            power_ratings.append(float(pwr))

        qty = c.get("Quantity")
        if qty:
            total_points += int(qty)

        connector_detail.append({
            "type": ct_title,
            "powerKW": pwr,
            "qty": qty,
            "amps": c.get("Amps"),
            "voltage": c.get("Voltage"),
        })

    # Charger type classification
    charger_type = None
    if power_ratings:
        mx = max(power_ratings)
        if mx >= 50:
            charger_type = "DC_FAST"
        elif mx >= 7:
            charger_type = "AC_LEVEL2"
        else:
            charger_type = "AC_LEVEL1"

    # Operator
    operator_name = None
    if op_info:
        t = op_info.get("Title", "")
        if t and t != "(Unknown Operator)":
            operator_name = t

    # Availability
    avail = None
    if status:
        st = status.get("Title", "")
        if "operational" in st.lower():
            avail = "OPERATIONAL"
        elif "not" in st.lower():
            avail = "NOT_OPERATIONAL"
        elif "planned" in st.lower():
            avail = "PLANNED"
        elif st:
            avail = st.upper()

    # 24x7
    is_24x7 = None
    if usage:
        ut = usage.get("Title", "").lower()
        if "24" in ut:
            is_24x7 = True

    # District
    town = addr.get("Town", "")
    district = town
    dist_to_center = None
    if lat and lng:
        nearest_dist, nearest_km = find_nearest_district(lat, lng, config)
        if nearest_km < 60:
            district = nearest_dist
        dist_to_center = round(nearest_km, 2)

    # Number of chargers
    num_chargers = total_points if total_points > 0 else station.get("NumberOfPoints")

    # Max power
    max_pwr = max(power_ratings) if power_ratings else None

    # Missing fields reason
    missing_reasons = []
    gis_fields = ["location_type", "road_type", "highway_name",
                  "distance_to_highway_km", "traffic_density",
                  "nearby_poi_count", "parking_availability"]
    for f in gis_fields:
        missing_reasons.append(f"{f}=GIS_DERIVED_PHASE4")
    if not is_24x7:
        missing_reasons.append("24x7_availability=NOT_IN_OCM_DATA")

    record = {
        # 25-field schema
        "station_id": None,  # Assigned later
        "station_name": addr.get("Title", "Unknown").strip(),
        "state": "Telangana",
        "district": district,
        "city": town.strip() if town else None,
        "latitude": lat,
        "longitude": lng,
        "pincode": addr.get("Postcode"),
        "location_type": None,
        "charger_type": charger_type,
        "connector_type": "|".join(sorted(set(connector_types))) if connector_types else None,
        "number_of_chargers": num_chargers,
        "charging_power_kw": max_pwr,
        "station_capacity": None,
        "availability_status": avail,
        "operator": operator_name,
        "24x7_availability": is_24x7,
        "road_type": None,
        "highway_name": None,
        "distance_to_highway_km": None,
        "distance_to_city_center_km": dist_to_center,
        "traffic_density": None,
        "nearby_poi_count": None,
        "parking_availability": None,
        "data_timestamp": datetime.now(timezone.utc).isoformat(),
        # Provenance
        "_data_source": "OCM",
        "_source_id": str(station.get("ID", "")),
        "_query_center": station.get("_query_center", ""),
        "_duplicate_of": None,
        "_raw_connector_detail": json.dumps(connector_detail, ensure_ascii=False),
        "_data_quality_level": station.get("DataQualityLevel"),
        "_date_last_verified": station.get("DateLastVerified"),
        "_missing_fields_reason": "; ".join(missing_reasons),
    }

    # Computed: station_capacity
    if record["number_of_chargers"] and record["charging_power_kw"]:
        record["station_capacity"] = round(
            record["number_of_chargers"] * record["charging_power_kw"], 1
        )

    return record


def map_osm_record(element: dict, config: dict) -> dict:
    """Map an OSM element to the 25-field schema + provenance."""
    tags = element.get("tags", {})
    lat = element.get("lat")
    lon = element.get("lon")

    # Connector types from socket tags
    connectors = []
    for k in tags:
        if "socket" in k.lower():
            connectors.append(k.replace("socket:", ""))

    # District
    district = "Unknown"
    dist_to_center = None
    if lat and lon:
        district, dist_km = find_nearest_district(lat, lon, config)
        dist_to_center = round(dist_km, 2)

    # Charger type
    charger_type = tags.get("charging_station:type")

    # Power from tags
    power_kw = None
    for k in tags:
        if "output" in k.lower() and "kw" in tags[k].lower():
            try:
                power_kw = float(tags[k].replace("kW", "").replace("kw", "").strip())
            except ValueError:
                pass

    missing_reasons = [
        "pincode=NOT_IN_OSM",
        "location_type=GIS_DERIVED_PHASE4",
        "road_type=GIS_DERIVED_PHASE4",
        "highway_name=GIS_DERIVED_PHASE4",
        "distance_to_highway_km=GIS_DERIVED_PHASE4",
        "traffic_density=GIS_DERIVED_PHASE4",
        "nearby_poi_count=GIS_DERIVED_PHASE4",
        "parking_availability=GIS_DERIVED_PHASE4",
    ]

    num_chargers = int(tags["capacity"]) if "capacity" in tags else None

    record = {
        "station_id": None,
        "station_name": tags.get("name", tags.get("brand", "Unknown OSM Station")).strip(),
        "state": "Telangana",
        "district": district,
        "city": tags.get("addr:city", district),
        "latitude": lat,
        "longitude": lon,
        "pincode": tags.get("addr:postcode"),
        "location_type": None,
        "charger_type": charger_type,
        "connector_type": "|".join(connectors) if connectors else None,
        "number_of_chargers": num_chargers,
        "charging_power_kw": power_kw,
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
        # Provenance
        "_data_source": "OSM",
        "_source_id": str(element.get("id", "")),
        "_query_center": "Telangana-BBox",
        "_duplicate_of": None,
        "_raw_connector_detail": json.dumps(
            {k: v for k, v in tags.items() if "socket" in k or "output" in k},
            ensure_ascii=False
        ),
        "_data_quality_level": None,
        "_date_last_verified": None,
        "_missing_fields_reason": "; ".join(missing_reasons),
    }

    if record["number_of_chargers"] and record["charging_power_kw"]:
        record["station_capacity"] = round(
            record["number_of_chargers"] * record["charging_power_kw"], 1
        )

    return record


# ── Deduplication ─────────────────────────────────────────────
def deduplicate_records(records: list, config: dict) -> list:
    """
    Deduplicate stations across OCM and OSM.
    Two records are duplicates if:
      - Within coordinate_threshold_meters AND
      - Name similarity >= name_similarity_threshold
    OCM records are preferred (richer data).
    """
    threshold_m = config["deduplication"]["coordinate_threshold_meters"]
    sim_threshold = config["deduplication"]["name_similarity_threshold"]

    # Sort: OCM first (preferred), then OSM
    records.sort(key=lambda r: (0 if r["_data_source"] == "OCM" else 1))

    kept = []
    duplicate_count = 0

    for rec in records:
        is_dup = False
        lat = rec.get("latitude")
        lng = rec.get("longitude")

        if lat and lng:
            for existing in kept:
                ex_lat = existing.get("latitude")
                ex_lng = existing.get("longitude")
                if ex_lat and ex_lng:
                    dist = haversine_m(lat, lng, ex_lat, ex_lng)
                    if dist < threshold_m:
                        sim = name_similarity(
                            rec.get("station_name", ""),
                            existing.get("station_name", "")
                        )
                        if sim >= sim_threshold or dist < 50:
                            # Mark as duplicate
                            rec["_duplicate_of"] = existing["_source_id"]
                            is_dup = True
                            duplicate_count += 1
                            logger.debug(
                                f"  Duplicate: '{rec['station_name']}' "
                                f"≈ '{existing['station_name']}' "
                                f"(dist={dist:.0f}m, sim={sim:.2f})"
                            )
                            break

        if not is_dup:
            kept.append(rec)

    logger.info(f"  Deduplication: {len(records)} → {len(kept)} "
                f"({duplicate_count} duplicates removed)")
    return kept, duplicate_count


# ── Station ID Generation ─────────────────────────────────────
def assign_station_ids(records: list) -> list:
    """
    Assign stable, unique station_id values.
    Format: TS-{DISTRICT_CODE}-{SERIAL}
    District code: first 3 chars of district, uppercased.
    """
    # Group by district
    district_counters = Counter()

    for rec in records:
        dist = (rec.get("district") or "UNK").upper()
        code = dist[:3].replace(" ", "")
        if len(code) < 3:
            code = code.ljust(3, "X")
        district_counters[code] += 1
        serial = district_counters[code]
        rec["station_id"] = f"TS-{code}-{serial:04d}"

    return records


# ── Validation ────────────────────────────────────────────────
def validate_dataset(records: list, config: dict) -> dict:
    """Run validation checks and return results dict."""
    results = {
        "total_records": len(records),
        "checks": {},
        "issues": [],
    }

    # 1. Unique station_ids
    ids = [r["station_id"] for r in records]
    unique_ids = len(set(ids))
    dup_ids = len(ids) - unique_ids
    results["checks"]["unique_station_ids"] = dup_ids == 0
    if dup_ids > 0:
        results["issues"].append(f"{dup_ids} duplicate station_id values")

    # 2. All state = Telangana
    non_ts = [r for r in records if r.get("state") != "Telangana"]
    results["checks"]["all_state_telangana"] = len(non_ts) == 0
    if non_ts:
        results["issues"].append(f"{len(non_ts)} records with state != Telangana")

    # 3. Coordinates within Telangana bounding box
    bb = config["bounding_box"]
    out_of_bounds = []
    no_coords = []
    for r in records:
        lat = r.get("latitude")
        lng = r.get("longitude")
        if lat is None or lng is None:
            no_coords.append(r["station_id"])
        elif not point_in_telangana(lat, lng, config):
            out_of_bounds.append(r["station_id"])

    results["checks"]["coordinates_in_bounds"] = len(out_of_bounds) == 0
    results["checks"]["all_have_coordinates"] = len(no_coords) == 0
    if out_of_bounds:
        results["issues"].append(f"{len(out_of_bounds)} stations outside Telangana bbox: {out_of_bounds}")
    if no_coords:
        results["issues"].append(f"{len(no_coords)} stations missing coordinates: {no_coords}")

    # 4. Field coverage
    field_coverage = {}
    for field in TARGET_SCHEMA:
        populated = sum(1 for r in records if r.get(field) is not None and r.get(field) != "")
        field_coverage[field] = {
            "populated": populated,
            "total": len(records),
            "pct": round(populated / len(records) * 100, 1) if records else 0,
        }
    results["field_coverage"] = field_coverage

    # 5. Source breakdown
    source_counts = Counter(r["_data_source"] for r in records)
    results["source_breakdown"] = dict(source_counts)

    # 6. District breakdown
    district_counts = Counter(r.get("district", "Unknown") for r in records)
    results["district_breakdown"] = dict(
        sorted(district_counts.items(), key=lambda x: -x[1])
    )

    # 7. Operator breakdown
    op_counts = Counter(r.get("operator") or "Unknown" for r in records)
    results["operator_breakdown"] = dict(
        sorted(op_counts.items(), key=lambda x: -x[1])
    )

    # 8. Charger type breakdown
    ct_counts = Counter(r.get("charger_type") or "Unknown" for r in records)
    results["charger_type_breakdown"] = dict(ct_counts)

    # 9. Manual inspection of sample records
    results["sample_records"] = []
    for i in [0, len(records)//3, len(records)//2, 2*len(records)//3, len(records)-1]:
        if 0 <= i < len(records):
            r = records[i]
            sample = {k: r[k] for k in TARGET_SCHEMA if r.get(k) is not None}
            sample["_data_source"] = r["_data_source"]
            sample["_source_id"] = r["_source_id"]
            results["sample_records"].append(sample)

    return results


# ── Report Generation ─────────────────────────────────────────
def generate_report(validation: dict, ocm_raw_count: int, osm_raw_count: int,
                    duplicate_count: int, collection_start: str, collection_end: str):
    """Generate the Phase 3 collection report."""
    fc = validation["field_coverage"]

    report = f"""# Phase 3: Telangana EV Charging Station Collection Report

**Collection Start:** {collection_start}
**Collection End:**   {collection_end}
**Status:** COMPLETE

---

## 1. Collection Summary

| Metric | Count |
|--------|-------|
| OCM records collected (raw) | {ocm_raw_count} |
| OSM records collected (raw) | {osm_raw_count} |
| Total raw records | {ocm_raw_count + osm_raw_count} |
| Duplicates detected & removed | {duplicate_count} |
| **Final unique stations** | **{validation['total_records']}** |

### Source Breakdown

| Source | Stations | Percentage |
|--------|----------|------------|
"""
    for src, cnt in validation["source_breakdown"].items():
        pct = round(cnt / validation["total_records"] * 100, 1)
        report += f"| {src} | {cnt} | {pct}% |\n"

    report += f"""
---

## 2. Field Coverage (25-Field Schema)

| # | Field | Populated | Total | Coverage | Status |
|---|-------|-----------|-------|----------|--------|
"""
    for i, field in enumerate(TARGET_SCHEMA, 1):
        info = fc[field]
        status = "✅" if info["pct"] >= 80 else ("⚠️" if info["pct"] > 0 else "❌")
        report += (f"| {i} | `{field}` | {info['populated']} | {info['total']} "
                   f"| {info['pct']}% | {status} |\n")

    # Compute totals
    populated_fields = sum(1 for f in fc if fc[f]["pct"] > 0)
    report += f"""
**Fields with data: {populated_fields}/25 ({round(populated_fields/25*100,1)}%)**

> Missing fields are either GIS-derived (Phase 4) or not available from OCM/OSM sources.

---

## 3. Geographic Coverage

### District Distribution

| District | Stations |
|----------|----------|
"""
    for dist, cnt in validation["district_breakdown"].items():
        report += f"| {dist} | {cnt} |\n"

    report += f"""
**Districts with stations: {len(validation['district_breakdown'])}**
**Districts in Telangana: 33**

---

## 4. Operator Distribution

| Operator | Stations |
|----------|----------|
"""
    for op, cnt in validation["operator_breakdown"].items():
        report += f"| {op} | {cnt} |\n"

    report += f"""
---

## 5. Charger Type Distribution

| Type | Count |
|------|-------|
"""
    for ct, cnt in validation["charger_type_breakdown"].items():
        report += f"| {ct} | {cnt} |\n"

    report += f"""
---

## 6. Validation Checks

| Check | Result |
|-------|--------|
"""
    for check, passed in validation["checks"].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        report += f"| {check} | {status} |\n"

    if validation["issues"]:
        report += "\n### Issues Found\n\n"
        for issue in validation["issues"]:
            report += f"- ⚠️ {issue}\n"

    report += f"""
---

## 7. Sample Records (Manual Inspection)

"""
    for i, sample in enumerate(validation["sample_records"], 1):
        report += f"### Record {i} (Source: {sample.get('_data_source', 'N/A')})\n\n"
        report += "| Field | Value |\n|-------|-------|\n"
        for k, v in sample.items():
            if k not in ("_data_source", "_source_id"):
                report += f"| `{k}` | {v} |\n"
        report += f"| `_source_id` | {sample.get('_source_id', 'N/A')} |\n\n"

    report += f"""---

## 8. Known Limitations

1. **OCM coverage depends on community contributions** — not all stations in Telangana may be listed.
2. **OSM data is sparse for India** — very few EV stations are tagged in OpenStreetMap for Telangana.
3. **Operator data** — many OCM entries list "(Unknown Operator)".
4. **Connector details** — some entries lack specific connector type / power rating.
5. **GIS-derived fields** (7 fields) are intentionally null — these require OSM road network data, spatial analysis, and POI queries (Phase 4).
6. **24x7 availability** — rarely captured in OCM/OSM data.
7. **This is NOT a claim of complete Telangana coverage** — it represents all stations available in OCM + OSM as of {collection_end}.

---

## 9. Output Files

| File | Description |
|------|-------------|
| `data/raw/telangana_ocm_raw.json` | Raw OCM API responses ({ocm_raw_count} stations) |
| `data/raw/telangana_osm_raw.json` | Raw OSM Overpass response ({osm_raw_count} elements) |
| `data/cleaned/telangana_station_dataset.csv` | Unified dataset ({validation['total_records']} stations, 25-field schema) |
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
"""

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Report saved: {REPORT_PATH}")


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 3: Telangana EV Station Collection")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--use-raw", action="store_true",
                        help="Process from existing raw JSON files without re-querying APIs")
    args = parser.parse_args()

    collection_start = datetime.now(timezone.utc).isoformat()

    print("=" * 70)
    print("PHASE 3: TELANGANA-WIDE EV CHARGING STATION COLLECTION")
    print(f"Time: {datetime.now().isoformat()}")
    print("Primary: OpenChargeMap | Supplementary: OSM Overpass")
    print("=" * 70)

    # Load config and API key
    config = load_config()
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    api_key = os.environ.get("OCM_API_KEY", "")
    if not api_key:
        logger.error("No OCM_API_KEY in .env")
        sys.exit(1)

    # Checkpoint
    checkpoint = load_checkpoint() if args.resume else {
        "completed_searches": [], "ocm_stations_raw": [], "osm_done": False
    }

    session = build_session(
        retries=config["ocm_api"]["retry_max"],
        backoff=config["ocm_api"]["retry_backoff_seconds"],
    )

    # ── Step 1: Collect from OCM ──────────────────────────────
    logger.info("\n[1/6] Collecting from OpenChargeMap...")
    ocm_stations = collect_ocm(session, api_key, config, checkpoint, use_raw=args.use_raw)

    # Save OCM raw
    with open(RAW_OCM_PATH, "w", encoding="utf-8") as f:
        json.dump(ocm_stations, f, indent=2, ensure_ascii=False)
    logger.info(f"  OCM raw saved: {RAW_OCM_PATH} ({os.path.getsize(RAW_OCM_PATH):,} bytes)")

    # ── Step 2: Collect from OSM ──────────────────────────────
    logger.info("\n[2/6] Collecting from OSM Overpass...")
    osm_stations = collect_osm(session, config, checkpoint, use_raw=args.use_raw)

    # ── Step 3: Map to schema ─────────────────────────────────
    logger.info("\n[3/6] Mapping to 25-field schema...")
    records = []

    for station in ocm_stations:
        records.append(map_ocm_record(station, config))
    logger.info(f"  OCM records mapped: {len(ocm_stations)}")

    for element in osm_stations:
        records.append(map_osm_record(element, config))
    logger.info(f"  OSM records mapped: {len(osm_stations)}")

    # ── Step 4: Deduplicate ───────────────────────────────────
    logger.info("\n[4/6] Deduplicating across sources...")
    records, dup_count = deduplicate_records(records, config)

    # Assign stable IDs
    records = assign_station_ids(records)

    # ── Step 5: Save cleaned dataset ──────────────────────────
    logger.info("\n[5/6] Saving cleaned dataset...")
    all_cols = TARGET_SCHEMA + PROVENANCE_COLS
    with open(CLEANED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"  Dataset saved: {CLEANED_CSV} ({len(records)} records)")

    # ── Step 6: Validate & Report ─────────────────────────────
    logger.info("\n[6/6] Running validation checks...")
    validation = validate_dataset(records, config)

    collection_end = datetime.now(timezone.utc).isoformat()

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"  Total stations:     {validation['total_records']}")
    print(f"  Sources:            {validation['source_breakdown']}")
    print(f"  Duplicates removed: {dup_count}")
    print(f"\n  Validation checks:")
    for check, passed in validation["checks"].items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"    {check:35s} {status}")

    if validation["issues"]:
        print(f"\n  Issues:")
        for issue in validation["issues"]:
            print(f"    ⚠️ {issue}")

    print(f"\n  Field coverage:")
    for field in TARGET_SCHEMA:
        info = validation["field_coverage"][field]
        bar = "█" * int(info["pct"] / 10) + "░" * (10 - int(info["pct"] / 10))
        status = "✓" if info["pct"] > 0 else "✗"
        print(f"    {status} {field:<30s} {bar} {info['pct']:5.1f}%")

    print(f"\n  Districts covered: {len(validation['district_breakdown'])}")
    for d, c in validation["district_breakdown"].items():
        print(f"    {d:<30s} {c} stations")

    # Generate report
    generate_report(
        validation, len(ocm_stations), len(osm_stations),
        dup_count, collection_start, collection_end,
    )

    # Clean up checkpoint on success
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        logger.info("Checkpoint removed (collection complete)")

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print(f"  ✓ OCM: {len(ocm_stations)} Telangana stations collected")
    print(f"  ✓ OSM: {len(osm_stations)} Telangana stations collected")
    print(f"  ✓ Duplicates: {dup_count} removed")
    print(f"  ✓ Final dataset: {validation['total_records']} unique stations")
    print(f"  ✓ Output: {CLEANED_CSV}")
    print(f"  ✓ Report: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
