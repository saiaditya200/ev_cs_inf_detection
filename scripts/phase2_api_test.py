"""
Phase 2: Fixed API Connectivity Test
=====================================
Tests both Overpass (OSM) and OpenChargeMap with corrected parameters.
Overpass requires no API key. OCM requires a free key.
"""

import sys
import io
import json
import requests
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print("=" * 70)
print("PHASE 2: API CONNECTIVITY TEST (FIXED)")
print(f"Time: {datetime.now().isoformat()}")
print("=" * 70)

# ── Test 1: OSM Overpass (NO KEY NEEDED) ──────────────────────
print("\n[1/2] Testing OSM Overpass API (no API key required)...")
print("-" * 50)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Bounding box for Hyderabad metro area
# South-West: 17.20, 78.20  North-East: 17.55, 78.65
overpass_query = """[out:json][timeout:30];
(
  node["amenity"="charging_station"](17.20,78.20,17.55,78.65);
  way["amenity"="charging_station"](17.20,78.20,17.55,78.65);
);
out body;
>;
out skel qt;"""

try:
    resp = requests.get(
        OVERPASS_URL,
        params={"data": overpass_query},
        headers={
            "Accept": "application/json",
            "User-Agent": "EVChargingResearch/1.0 (academic project)"
        },
        timeout=60
    )
    print(f"  HTTP Status: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        elements = data.get("elements", [])
        
        # Filter for actual stations (nodes/ways with tags, not member nodes)
        stations = [e for e in elements if e.get("tags", {}).get("amenity") == "charging_station"]
        print(f"  Total elements: {len(elements)}")
        print(f"  EV stations:    {len(stations)}")
        
        if stations:
            # Collect all unique tag keys across stations
            all_tags = set()
            for s in stations:
                all_tags.update(s.get("tags", {}).keys())
            
            print(f"\n  All tag keys found across {len(stations)} stations:")
            for tag in sorted(all_tags):
                count = sum(1 for s in stations if tag in s.get("tags", {}))
                print(f"    {tag:40s}  present in {count}/{len(stations)} stations")
            
            print(f"\n  Sample stations (first 5):")
            for i, station in enumerate(stations[:5]):
                tags = station.get("tags", {})
                print(f"\n    --- Station {i+1} (OSM ID: {station.get('id')}) ---")
                if "lat" in station:
                    print(f"    Location: {station['lat']}, {station['lon']}")
                print(f"    Name:     {tags.get('name', 'N/A')}")
                print(f"    Operator: {tags.get('operator', 'N/A')}")
                print(f"    Network:  {tags.get('network', 'N/A')}")
                print(f"    Socket types: {', '.join(k for k in tags if 'socket' in k.lower())}")
                print(f"    Capacity: {tags.get('capacity', 'N/A')}")
                print(f"    All tags:")
                for k, v in sorted(tags.items()):
                    print(f"      {k:35s} = {v}")
            
            # Save raw data
            raw_path = "data/raw/hyderabad_osm_raw.json"
            import os
            os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n  [OK] Raw OSM data saved to: {raw_path}")
            print(f"       Size: {os.path.getsize(raw_path):,} bytes")
        
        print(f"\n  [OK] OSM Overpass API is WORKING — {len(stations)} stations found!")
    else:
        print(f"  [FAIL] HTTP {resp.status_code}")
        print(f"  Response: {resp.text[:500]}")
except Exception as e:
    print(f"  [FAIL] Error: {type(e).__name__}: {e}")

# ── Test 2: OpenChargeMap ─────────────────────────────────────
print(f"\n\n[2/2] Testing OpenChargeMap API...")
print("-" * 50)

OCM_URL = "https://api.openchargemap.io/v3/poi/"
ocm_params = {
    "output": "json",
    "countrycode": "IN",
    "latitude": 17.385044,
    "longitude": 78.486671,
    "distance": 20,
    "distanceunit": "KM",
    "maxresults": 10,
    "compact": True,
    "verbose": False,
}

# Try without API key first
try:
    resp = requests.get(
        OCM_URL,
        params=ocm_params,
        headers={
            "User-Agent": "EVChargingResearch/1.0",
            "Accept": "application/json",
        },
        timeout=30
    )
    print(f"  HTTP Status (no key): {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        print(f"  [OK] OpenChargeMap works WITHOUT key! Results: {len(data)}")
        
        if data:
            first = data[0]
            print(f"\n  Top-level fields:")
            for key in sorted(first.keys()):
                val = first[key]
                if isinstance(val, dict):
                    print(f"    {key:30s} = dict({len(val)} keys)")
                elif isinstance(val, list):
                    print(f"    {key:30s} = list({len(val)} items)")
                else:
                    print(f"    {key:30s} = {str(val)[:60]}")
            
            # AddressInfo
            addr = first.get("AddressInfo", {})
            if addr:
                print(f"\n  AddressInfo fields:")
                for k, v in sorted(addr.items()):
                    if isinstance(v, dict):
                        print(f"    {k:30s} = dict: {v}")
                    else:
                        print(f"    {k:30s} = {v}")
            
            # Connections
            conns = first.get("Connections", [])
            if conns:
                print(f"\n  Connections ({len(conns)}):")
                for i, c in enumerate(conns[:3]):
                    print(f"    Connection {i+1}:")
                    for k, v in sorted(c.items()):
                        if isinstance(v, dict):
                            title = v.get("Title", v.get("FormalName", str(v)))
                            print(f"      {k:28s} = {title}")
                        else:
                            print(f"      {k:28s} = {v}")
            
            # Save raw
            raw_path = "data/raw/hyderabad_ocm_raw.json"
            import os
            os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n  [OK] Raw OCM data saved to: {raw_path}")
    elif resp.status_code == 403:
        print("  [INFO] 403 — API key required (free registration)")
        print("  → Register at https://openchargemap.org (free, no billing)")
        print("  → Go to My Profile → My Apps → Generate API Key")
        print("  → Then set OCM_API_KEY in .env file")
    else:
        print(f"  Response: {resp.text[:300]}")
except Exception as e:
    print(f"  [FAIL] Error: {type(e).__name__}: {e}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
