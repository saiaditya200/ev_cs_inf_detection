# Phase 10: Split Reproducibility Specification

**Project Title:** Explainable AI Framework for Smart EV Charging Infrastructure Planning  
**Phase Status:** COMPLETE & VERIFIED  
**Date:** 2026-09-26  

---

## 1. Environment & Software Versions

- **Python Version:** `3.8.10`
- **scikit-learn Version:** `1.3.0`
- **pandas Version:** `2.0.3`
- **numpy Version:** `1.24.3`

## 2. Cross-Validation Configuration

- **Algorithm:** `sklearn.model_selection.StratifiedGroupKFold`
- **Number of Folds (`n_splits`):** `5`
- **Shuffle:** `True`
- **Random State (`random_state`):** `42`
- **Grouping Variable:** `district` (33 clusters)
- **Target Variable:** `fast_charging_flag` (binary: 131 Class 1, 23 Class 0)

## 3. Outer Test Benchmark Definition

- **Designated Test Fold:** Fold 3
- **Test Sample Size:** 35 stations (22.7% of labeled sample)
- **Test Class Breakdown:** 9 Class 0 (25.7%), 26 Class 1 (74.3%)
- **Test Districts (8):** Bhadradri Kothagudem, Jagitial, Karimnagar, Medak, Mulugu, Vikarabad, Warangal Rural, Warangal Urban
- **Train Districts (25):** Exactly 0 overlap with test districts

## 4. Artifact Checksums

- `fast_charging_modeling_matrix.csv`: `fc32c75635fe2e763ec174e491a897adc691539fba0be655f06b9d58bca79783`
- `phase10_district_class_distribution.csv`: `af1a21035d2862a34cddcf3f8bec4ac2bd57cd0f7f4ea02831b032fcfffbd686`
- `phase10_spatial_splits.csv`: `27423b8a5d084fe09e4dcd037f376b11c5c7acf904ccd243d0e85dfcafe624e5`
