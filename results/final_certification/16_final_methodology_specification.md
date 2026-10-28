# Final Methodology Specification

Canonical Architecture:

```
MULTI-SOURCE REAL DATA
        ↓
DATA QUALITY AUDIT
        ↓
TARGET INTEGRITY
        ↓
PREPROCESSING
        ↓
FEATURE ENGINEERING
        ↓
LEAKAGE CONTROL
        ↓
21 CERTIFIED EXOGENOUS PREDICTORS
        ↓
DISTRICT-GROUPED VALIDATION
        ↓
RF / XGBOOST / LIGHTGBM RESEARCH COMPARISON
        ↓
HYPERPARAMETER + THRESHOLD OPTIMIZATION
        ↓
DEVELOPMENT-ONLY SELECTION
        ↓
ONE FROZEN RANDOM FOREST DEPLOYMENT MODEL
        ↓
INDEPENDENT OUTER TEST
        ↓
TreeSHAP
        ↓
COUNTERFACTUALS
        ↓
EV ADOPTION CONTEXT
        ↓
INFRASTRUCTURE GAP
        ↓
GRID CONTEXT
        ↓
SPATIAL SUITABILITY
        ↓
AHP
        ↓
TOPSIS
        ↓
EIPS
        ↓
GIS
        ↓
FLASK
        ↓
INTERACTIVE DECISION SUPPORT
```

## Architectural Principles

1. **Separation of Concerns**: EIPS is NOT fed into the station classifier. Station ML probabilities are NOT aggregated into EIPS.
2. **Exogenous Predictor Purity**: Predictors are restricted to spatial, demographic, and road topology indicators.
3. **Quarantined Evaluation**: Outer-test data is reserved exclusively for final single-pass benchmark reporting.
4. **Diagnostic Integrity**: AC Signal Enrichment is documented as a research diagnostic, not a modified production contract.
