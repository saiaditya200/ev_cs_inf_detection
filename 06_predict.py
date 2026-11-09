#!/usr/bin/env python3
"""Root forwarder -> scripts/06_predict.py"""
import subprocess
import sys
from pathlib import Path

target = Path(__file__).resolve().parent / "scripts" / "06_predict.py"
sys.exit(subprocess.call([sys.executable, str(target)] + sys.argv[1:]))
