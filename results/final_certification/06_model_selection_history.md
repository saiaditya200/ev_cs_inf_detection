# Model Selection History

The progression of research stages documented in `results/final_certification/model_selection_history.csv`:

- **Stage 1: Baseline Comparison**: RF, XGBoost, LightGBM, Logistic Regression on default parameters under 4-fold GroupKFold. Random Forest established superior balanced accuracy.
- **Stage 2: Joint Hyperparameter & Threshold Optimization**: 30,628 evaluations across hyperparameter grids and thresholds. Identified `JOINT_RF_0257` (depth=6, threshold=0.76).
- **Stage 3: Ensemble Bake-off**: Soft-voting and stacking ensembles vs best individual models. Single Random Forest (depth=8, threshold=0.755) outperformed complex ensembles while preserving TreeSHAP compatibility.
- **Stage 4: AC-Focused Optimization**: Minority cost weighting evaluated; moderate weights improved dev recall but introduced fold-level instability.
- **Stage 5: Pareto AC/DC Optimization**: Complete Pareto frontier across 97,188 operating points. Non-dominated leaders `RF_0002` (Band 2) and `RF_0039` (Band 4) established as secondary research benchmarks.
- **Stage 6: AC Signal Enrichment**: Investigated site-context features; revealed severe spatial covariate shift (100% outer AC on trunk roads vs 35.7% dev AC in residential zones).
- **Stage 7: Provenance Reconciliation (Phase 15.1)**: Reconciled all computational provenance; confirmed 100% concordance between prediction vectors and metrics.
