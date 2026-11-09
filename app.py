#!/usr/bin/env python3
"""
=============================================================================
app.py — Master Pan-India EV Charging Dashboard Web Server
=============================================================================
Project: Explainable AI Framework for Smart EV Charging Infrastructure Planning
Domain: All 36 States & Union Territories of India (52,718 Stations)
Usage: python app.py [--port 5000]
=============================================================================
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app_india_wide import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print("=" * 80)
    print("PAN-INDIA EXPLAINABLE AI EV CHARGING INFRASTRUCTURE PLANNING DASHBOARD")
    print("Interactive Leaflet GIS • TreeSHAP Diagnostics • Saaty AHP EIPS")
    print(f"Local Server URL: http://127.0.0.1:{port}")
    print("Press Ctrl+C to terminate the dashboard server.")
    print("=" * 80)
    try:
        app.run(host="0.0.0.0", port=port, debug=False)
    except OSError:
        fallback_port = 5001
        print(f"[WARN] Port {port} in use. Launching on fallback port {fallback_port}...")
        print(f"Local Server URL: http://127.0.0.1:{fallback_port}")
        app.run(host="0.0.0.0", port=fallback_port, debug=False)
