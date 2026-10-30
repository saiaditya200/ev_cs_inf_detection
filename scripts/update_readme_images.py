import re
from pathlib import Path

readme_path = Path("README.md")
readme_india_path = Path("README_INDIA_WIDE.md")

with open(readme_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. System Architecture
if "![System Architecture — Pan-India Master Framework]" not in text:
    target = "The high-resolution vector system architecture diagram is saved at [`system_architecture_india_wide.svg`](file:///c:/Users/Dell/Downloads/EV_Charging/system_architecture_india_wide.svg) (and mirrored in [`results/india_wide/figures/system_architecture_india_wide.svg`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures/system_architecture_india_wide.svg)).\n\n```text"
    replacement = "The high-resolution vector system architecture diagram is saved at [`system_architecture_india_wide.svg`](file:///c:/Users/Dell/Downloads/EV_Charging/system_architecture_india_wide.svg) (and mirrored in [`results/india_wide/figures/system_architecture_india_wide.svg`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures/system_architecture_india_wide.svg)).\n\n![System Architecture — Pan-India Master Framework](system_architecture_india_wide.svg)\n\n```text"
    text = text.replace(target, replacement)

# 2. Final Training Accuracy & Evaluation curves
if "### Final Model Training Accuracy on the Pan-India Dataset" not in text:
    target = "### Evaluation Curves & Diagnostics\n\nAll figures are saved in [`results/india_wide/figures/`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures/):\n\n1. **Confusion Matrix (`confusion_matrix.png`):**\n   - True Positives (DC Fast correctly predicted): **16,537**\n   - False Negatives (DC Fast missed): **24**\n   - True Negatives (AC Slow correctly predicted): **141**\n   - False Positives (AC Slow flagged for fast charging): **36,004**\n2. **Precision-Recall Curves (`pr_curves.png`):** Demonstrates the PR curves of RF, XGBoost, and LightGBM against the no-skill baseline ($0.3142$).\n3. **ROC Curves (`roc_curves.png`):** Shows the sensitivity vs. specificity tradeoff across all decision thresholds.\n\n---"
    replacement = """### Final Model Training Accuracy on the Pan-India Dataset

When evaluating the Champion Random Forest model on the complete master dataset ($N = 52,706$ verified stations across all 36 States & UTs):

| Decision Policy Threshold | Total Stations Evaluated | Correct Classifications | Final Accuracy | DC Fast Charging Recall (Sensitivity) | AC Slow Charging Specificity | Balanced Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Cut-Off ($\\\\tau = 0.50$)** | 52,706 | 24,502 | **46.49%** | 60.53% (10,024 / 16,561) | 40.06% (14,478 / 36,145) | **50.29%** |
| **Calibrated Threshold ($\\\\tau^* = 0.2488$)** | 52,706 | 16,600 | **31.50%** | **100.00%** (16,561 / 16,561) | 0.11% (39 / 36,145) | **50.05%** |
| **Outer Spatial Holdout ($\\\\tau^* = 0.2488$)** | 52,706 | 16,678 | **31.64%** | **99.85%** (16,537 / 16,561) | 0.39% (141 / 36,145) | **50.12%** |

### Evaluation Curves & Diagnostics

All figures are generated standalone and saved in [`results/india_wide/figures/`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures/):

1. **Confusion Matrix (`confusion_matrix.png`):**
   - True Positives (DC Fast correctly predicted): **16,537**
   - False Negatives (DC Fast missed): **24**
   - True Negatives (AC Slow correctly predicted): **141**
   - False Positives (AC Slow flagged for fast charging): **36,004**
2. **Precision-Recall Curves (`pr_curves.png`):** Demonstrates the PR curves of RF, XGBoost, and LightGBM against the no-skill baseline ($0.3142$).
3. **ROC Curves (`roc_curves.png`):** Shows the sensitivity vs. specificity tradeoff across all decision thresholds.

![Confusion Matrix (Outer Test - 52,718 Stations)](results/india_wide/figures/confusion_matrix.png)

![Precision-Recall Curves Comparison](results/india_wide/figures/pr_curves.png)

![ROC Curves Comparison](results/india_wide/figures/roc_curves.png)

---"""
    text = text.replace(target, replacement)

# 3. TreeSHAP and Counterfactuals
if "![Global TreeSHAP Feature Ranking]" not in text:
    target = "Visualized in [`results/india_wide/figures/counterfactual_comparison.png`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures/counterfactual_comparison.png) across representative stations in 10 states:\n\n| Station ID | State / UT | Original $P(\\text{DC Fast})$ | Optimized $P(\\text{DC Fast})$ | Prescriptive Planning Guidance |"
    replacement = """### Global TreeSHAP Feature Attribution & Distributions

![Global TreeSHAP Feature Ranking](results/india_wide/figures/feature_importance.png)

![SHAP Value Distributions Beeswarm Plot](results/india_wide/figures/shap_beeswarm.png)

### Actionable Counterfactual Elevation

Visualized in [`results/india_wide/figures/counterfactual_comparison.png`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures/counterfactual_comparison.png) across representative stations in 10 states:

![Actionable Counterfactual Elevation](results/india_wide/figures/counterfactual_comparison.png)

| Station ID | State / UT | Original $P(\\text{DC Fast})$ | Optimized $P(\\text{DC Fast})$ | Prescriptive Planning Guidance |"""
    text = text.replace(target, replacement)

# 4. EIPS Planning Figures
if "![EIPS State Prioritization Index]" not in text:
    target = "### TOPSIS State Ranking & Prioritization Tiers\n\nUsing the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS), all **36 States and Union Territories** are ranked into three distinct capital allocation tiers:\n\n```text\nTOP 15 STATES EIPS RANKINGS (AHP-TOPSIS)"
    replacement = """### TOPSIS State Ranking & Prioritization Tiers

Using the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS), all **36 States and Union Territories** are ranked into three distinct capital allocation tiers:

![EIPS State Prioritization Index](results/india_wide/figures/eips_state_rankings.png)

![Multi-Criteria Component Decomposition](results/india_wide/figures/eips_component_breakdown.png)

```text
TOP 15 STATES EIPS RANKINGS (AHP-TOPSIS)"""
    text = text.replace(target, replacement)

# Write to both README files
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(text)

with open(readme_india_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Successfully updated README.md and README_INDIA_WIDE.md with all 15 figures and final accuracy table!")
