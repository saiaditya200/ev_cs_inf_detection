#!/usr/bin/env python3
"""Root forwarder -> scripts/03_model_training.py"""
import subprocess
import sys
from pathlib import Path

target = Path(__file__).resolve().parent / "scripts" / "03_model_training.py"
sys.exit(subprocess.call([sys.executable, str(target)] + sys.argv[1:]))
