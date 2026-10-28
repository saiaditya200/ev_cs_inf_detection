# Explainable AI (TreeSHAP) Manifest

- **Algorithm**: `shap.TreeExplainer` applied to the certified Random Forest probability outputs.
- **Predictor Space**: Exactly 21 certified exogenous predictors.
- **Certified SHAP Interpretation**:
  > "SHAP values represent model-based feature attributions to the Random Forest's DC Fast probability output. They indicate how features contribute to the model prediction and do not establish causality, physical mechanisms, or actual charging demand/utilization."
- **Leakage Verification**: Zero target fields, equipment power ratings, or operator identifiers enter the SHAP explainer.
- **Explanations Provided**:
  - Global feature attributions: Summary beeswarm plots and mean absolute SHAP importances.
  - Local site explanations: Waterfall and force plots decomposing individual site scores.
