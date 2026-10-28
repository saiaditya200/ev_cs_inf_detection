# Spatial Validation Protocol

1. **Development Cross-Validation**:
   - 4-fold `GroupKFold` strictly grouped by `district`.
   - Stations from the same administrative district never appear in both training and validation folds simultaneously.
   - Prevents spatial autocorrelation leakage across local geographic clusters.

2. **Quarantined Outer-Test Evaluation**:
   - 35 stations across 8 independent districts completely held out from all model selection, tuning, and threshold calibration.
   - Evaluated strictly in a single pass after candidate models are frozen.
   - Outer-test performance is never used for hyperparameter optimization or decision threshold selection.
