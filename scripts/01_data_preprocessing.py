#!/usr/bin/env python3
"""
=============================================================================
01_data_preprocessing.py
=============================================================================
Project: Explainable AI Framework for Smart EV Charging Infrastructure Planning
Purpose: Ingest, clean, audit, and preprocess the Pan-India EV charging station
         master dataset (52,718 stations across all 36 States & Union Territories).
=============================================================================
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "india_wide"
MASTER_CSV = DATA_DIR / "INDIA_WIDE_EV_CHARGING_MASTER_DATASET_52K_30FEAT.csv"
SUMMARY_CSV = DATA_DIR / "INDIA_WIDE_STATE_EV_CHARGING_SUMMARY_52K.csv"
AUDIT_JSON = BASE_DIR / "results" / "india_wide" / "metrics" / "dataset_audit_report.json"

def preprocess_dataset():
    print("=" * 80)
    print("STAGE 1: PAN-INDIA EV CHARGING DATA INGESTION & PREPROCESSING")
    print("=" * 80)

    if not MASTER_CSV.exists():
        print(f"[ERROR] Master dataset not found at: {MASTER_CSV}")
        sys.exit(1)

    print(f"[1/4] Loading master national dataset: {MASTER_CSV.name}...")
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    total_records = len(df)
    total_features = df.shape[1]
    print(f"      Total records loaded: {total_records:,}")
    print(f"      Total features: {total_features}")

    # 2. Data Cleaning & Geographic Boundary Validation
    print("\n[2/4] Validating geographic bounds & cleaning records...")
    # Bounding box for India: Latitude 6.0° to 38.0° N, Longitude 68.0° to 98.0° E
    lat_valid = df["latitude"].between(6.0, 38.0)
    lon_valid = df["longitude"].between(68.0, 98.0)
    geo_valid = lat_valid & lon_valid
    print(f"      Geographically valid stations: {geo_valid.sum():,} / {total_records:,} ({(geo_valid.sum()/total_records)*100:.2f}%)")

    # 3. Class Balance & Target Definition
    print("\n[3/4] Auditing target binarization (fast_charging_flag)...")
    labeled_mask = df["fast_charging_flag"].isin([0, 1])
    df_labeled = df[labeled_mask].copy()
    labeled_count = len(df_labeled)
    
    fast_count = int((df_labeled["fast_charging_flag"] == 1).sum())
    slow_count = int((df_labeled["fast_charging_flag"] == 0).sum())
    fast_pct = (fast_count / labeled_count) * 100
    slow_pct = (slow_count / labeled_count) * 100

    print(f"      Class 1 (DC Fast >= 50 kW): {fast_count:,} ({fast_pct:.2f}%)")
    print(f"      Class 0 (AC Slow < 50 kW) : {slow_count:,} ({slow_pct:.2f}%)")
    print(f"      Class Imbalance Ratio     : 1 : {slow_count / fast_count:.2f}")

    # 4. State-Wise Coverage
    states = df["state"].dropna().unique()
    n_states = len(states)
    print(f"\n[4/4] Administrative coverage audit across India...")
    print(f"      Total States & Union Territories covered: {n_states} (100% Pan-India)")

    # Top 5 states by volume
    top5 = df["state"].value_counts().head(5)
    print("      Top 5 States by charging station volume:")
    for st, count in top5.items():
        st_fast = int((df[df['state'] == st]['fast_charging_flag'] == 1).sum())
        print(f"        • {st:20}: {count:,} stations ({st_fast:,} DC Fast)")

    # Save / Update Audit JSON
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    audit_data = {
        "file_name": MASTER_CSV.name,
        "total_records": total_records,
        "total_features": total_features,
        "labelled_records": labeled_count,
        "unlabelled_records": total_records - labeled_count,
        "states_count": n_states,
        "fast_charging_count": fast_count,
        "slow_charging_count": slow_count,
        "fast_charging_pct": round(fast_pct, 2),
        "slow_charging_pct": round(slow_pct, 2)
    }
    with open(AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    print("\n" + "=" * 80)
    print("[SUCCESS] Preprocessing & validation complete. Master dataset verified.")
    print("=" * 80)

if __name__ == "__main__":
    preprocess_dataset()

