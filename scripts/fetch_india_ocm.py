import os
import sys
import json
import urllib.request
from pathlib import Path

BASE_DIR = Path("C:/Users/Dell/Downloads/EV_Charging")
RAW_DIR = BASE_DIR / "data" / "india_wide" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = RAW_DIR / "india_ocm_raw.json"

print(f"[INFO] Fetching full OpenChargeMap India repository...")
url = "https://api.openchargemap.io/v3/poi/?output=json&countrycode=IN&maxresults=5000&key=86b9b4a9-8f28-4a1d-9aa3-60ff0e2b4031"
req = urllib.request.Request(url, headers={"User-Agent": "EV-Research-Agent/1.0"})

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print(f"[SUCCESS] Fetched {len(data)} stations from OpenChargeMap API.")
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[SAVED] Saved raw OCM India data to: {OUT_FILE} ({OUT_FILE.stat().st_size / 1024:.1f} KB)")
except Exception as e:
    print(f"[ERROR] Failed to fetch OCM data: {e}")
    sys.exit(1)

