#!/usr/bin/env python3
"""Root forwarder -> scripts/02_feature_extraction.py"""
import subprocess
import sys
from pathlib import Path

target = Path(__file__).resolve().parent / "scripts" / "02_feature_extraction.py"
sys.exit(subprocess.call([sys.executable, str(target)] + sys.argv[1:]))
