#!/usr/bin/env python3
"""Root forwarder -> scripts/05_xai_explainability.py"""
import subprocess
import sys
from pathlib import Path

target = Path(__file__).resolve().parent / "scripts" / "05_xai_explainability.py"
sys.exit(subprocess.call([sys.executable, str(target)] + sys.argv[1:]))
