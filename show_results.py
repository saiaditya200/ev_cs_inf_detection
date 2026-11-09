#!/usr/bin/env python3
"""
show_results.py — Master Root Entrypoint for Verification & Terminal Output
=============================================================================
Usage: python show_results.py (or python scripts/show_results.py)
=============================================================================
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ROOT_DIR))

from scripts.show_results import display_results

if __name__ == "__main__":
    display_results()
