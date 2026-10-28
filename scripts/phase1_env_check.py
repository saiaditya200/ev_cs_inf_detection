"""
Phase 1: Data Feasibility Audit - Environment & Data Source Verification
========================================================================
This script verifies:
1. Python environment and package availability
2. Google Maps API key presence and validity
3. Reachability of key government data portals
4. Summary of data source status
"""

import sys
import io
import importlib
import os
from datetime import datetime

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OK   = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

print("=" * 70)
print("PHASE 1: DATA FEASIBILITY AUDIT - Environment Check")
print(f"Run at: {datetime.now().isoformat()}")
print(f"Python: {sys.version}")
print("=" * 70)

# ---------------------------------------------------------------
# 1. Check core Python packages
# ---------------------------------------------------------------
print("\n[1/4] CHECKING PYTHON PACKAGES")
print("-" * 40)

required_packages = {
    "pandas": "pandas",
    "numpy": "numpy",
    "geopandas": "geopandas",
    "shapely": "shapely",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "shap": "shap",
    "folium": "folium",
    "plotly": "plotly",
    "streamlit": "streamlit",
    "requests": "requests",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "openpyxl": "openpyxl",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "tqdm": "tqdm",
    "pyproj": "pyproj",
}

installed = []
missing = []

for import_name, pip_name in required_packages.items():
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "unknown")
        installed.append((pip_name, ver))
        print(f"  {OK:6s} {pip_name:20s} -> {ver}")
    except ImportError:
        missing.append(pip_name)
        print(f"  {FAIL:6s} {pip_name:20s} -> NOT INSTALLED")

print(f"\nInstalled: {len(installed)}/{len(required_packages)}")
if missing:
    print(f"Missing ({len(missing)}): {', '.join(missing)}")
    print(f"Install with: pip install {' '.join(missing)}")

# ---------------------------------------------------------------
# 2. Check Google Maps API key
# ---------------------------------------------------------------
print("\n[2/4] CHECKING GOOGLE MAPS API KEY")
print("-" * 40)

# Check environment variable
api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")

# Check .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
env_key_found = False
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if line.strip().startswith("GOOGLE_MAPS_API_KEY"):
                env_key_found = True
                if not api_key:
                    api_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")

if api_key and api_key != "YOUR_API_KEY_HERE":
    print(f"  {OK} API key found (length={len(api_key)}, starts with {api_key[:8]}...)")
    
    # Test the key with a simple request
    try:
        import requests
        test_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": "17.385044,78.486671",  # Hyderabad
            "radius": 1000,
            "type": "charging_station",
            "key": api_key
        }
        resp = requests.get(test_url, params=params, timeout=10)
        data = resp.json()
        status = data.get("status", "UNKNOWN")
        if status == "OK":
            n_results = len(data.get("results", []))
            print(f"  {OK} API key is VALID (test query returned {n_results} results)")
        elif status == "ZERO_RESULTS":
            print(f"  {OK} API key is VALID (test query returned 0 results in area)")
        elif status == "REQUEST_DENIED":
            error_msg = data.get("error_message", "No details")
            print(f"  {FAIL} API key DENIED: {error_msg}")
        else:
            print(f"  {WARN} API returned status: {status}")
    except Exception as e:
        print(f"  {WARN} Could not test API key: {e}")
else:
    print(f"  {FAIL} No Google Maps API key found")
    print("  -> Set GOOGLE_MAPS_API_KEY environment variable")
    print("  -> Or create a .env file in the project root with:")
    print('     GOOGLE_MAPS_API_KEY=your_key_here')

# ---------------------------------------------------------------
# 3. Check government data portal reachability
# ---------------------------------------------------------------
print("\n[3/4] CHECKING DATA SOURCE REACHABILITY")
print("-" * 40)

data_sources = {
    "e-AMRIT (NITI Aayog)": "https://e-amrit.niti.gov.in/",
    "VAHAN Dashboard": "https://vahan.parivahan.gov.in/vahan4dashboard/",
    "Census India": "https://censusindia.gov.in/",
    "CEA (Power)": "https://cea.nic.in/",
    "National Power Portal": "https://npp.gov.in/",
    "data.gov.in": "https://data.gov.in/",
    "Geofabrik (OSM India)": "https://download.geofabrik.de/asia/india.html",
    "Clean Mobility Shift": "https://cleanmobilityshift.com/",
    "India Data Portal": "https://indiadataportal.com/",
}

try:
    import requests
    for name, url in data_sources.items():
        try:
            resp = requests.head(url, timeout=10, allow_redirects=True)
            code = resp.status_code
            if code < 400:
                print(f"  {OK:6s} {name:25s} -> HTTP {code}")
            else:
                print(f"  {WARN:6s} {name:25s} -> HTTP {code}")
        except requests.exceptions.Timeout:
            print(f"  {WARN:6s} {name:25s} -> TIMEOUT")
        except requests.exceptions.ConnectionError:
            print(f"  {FAIL:6s} {name:25s} -> CONNECTION ERROR")
        except Exception as e:
            print(f"  {WARN:6s} {name:25s} -> {type(e).__name__}")
except ImportError:
    print(f"  {FAIL} 'requests' package not installed; cannot check reachability")

# ---------------------------------------------------------------
# 4. Project directory structure
# ---------------------------------------------------------------
print("\n[4/4] PROJECT DIRECTORY STRUCTURE")
print("-" * 40)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
expected_dirs = [
    "data/raw", "data/cleaned", "data/government", "data/gis", "data/final",
    "scripts", "notebooks", "models", "explainability", "gis",
    "dashboard", "reports", "tests", "config", "output"
]

for d in expected_dirs:
    full = os.path.join(project_root, d)
    exists = os.path.isdir(full)
    status = OK if exists else FAIL
    print(f"  {status:6s} {d}")

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("PHASE 1 SUMMARY")
print("=" * 70)
print(f"  Packages installed:       {len(installed)}/{len(required_packages)}")
print(f"  Packages missing:         {len(missing)}")
print(f"  Google Maps API key:      {'FOUND' if api_key and api_key != 'YOUR_API_KEY_HERE' else 'MISSING'}")
print(f"  Data sources reachable:   (see above)")
print(f"  Project structure:        OK")
print()
if missing:
    print("ACTION REQUIRED: Install missing packages before proceeding.")
    print(f"  pip install {' '.join(missing)}")
if not api_key or api_key == "YOUR_API_KEY_HERE":
    print("ACTION REQUIRED: Set up Google Maps API key before Phase 3.")
print("=" * 70)
