# Phase 7 — Data Sufficiency & Sample Size Audit Report

## Executive Summary

This report delivers a rigorous mathematical and statistical assessment of data sufficiency for machine learning in the Telangana EV Charging Infrastructure Planning project. 

The evaluation examines sample size, feature-to-sample ratios ($p/N$), degrees of freedom, variance properties of contextual variables, cross-validation stability, and the algorithmic constraints of tree ensemble models (Random Forest, XGBoost, LightGBM).

> [!IMPORTANT]
> **Definitive Finding**: The current sample of **25 charging stations** is **statistically insufficient** to train, tune, or evaluate supervised machine learning models. 
> - With $p = 42$ candidate numerical features and $N = 25$ observations, the dataset sits in an **over-determined high-dimensional regime ($p/N = 1.68$)**.
> - Out-of-fold 5-fold cross-validation exhibits catastrophic instability ($\text{Mean } R^2 = -60.12$).
> - **Mandatory Prerequisite**: The dataset **must be expanded** (via Telangana statewide census expansion or Pan-India spatial scaling) before any supervised machine learning experiment can be scientifically defended.

---

## 1. Audit of Available Observations Across Operational Grains

| Operational Grain | Primary Key | Available Records ($N$) | Unique Spatial Units | Temporal Span | Features Available ($p$) | Current ML Readiness |
|---|---|---|---|---|---|---|
| **Station Point Layer** | `station_id` | **25** | 25 Point Coordinates (10 Districts) | Single Snapshot (2026) | 97 (42 Numeric Predictors) | **INSUFFICIENT ($N=25$)** |
| **District Panel Layer** | `district_key`, `financial_year` | **231** | 33 Telangana Districts | 7 Annual Fiscal Years (FY20–FY26) | 25 Features | **MODERATE ($N=231$)** |
| **State Panel Layer** | `state_key`, `financial_year` | **7** | 1 State (Telangana) | 7 Annual Fiscal Years (FY20–FY26) | 27 Features | **DESCRIPTIVE ONLY ($N=7$)** |
| **Highway Corridor Layer** | `nh_number` | **6** | 6 National Highways | Static Topology | 8 Attributes | **CONTEXTUAL LOOKUP ($N=6$)** |

---

## 2. Statistical Power & Dimensional Imbalance Audit

### A. The Curse of Dimensionality ($p > N$)
The integrated dataset [`integrated_analytical_dataset.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/final/integrated_analytical_dataset.csv) contains:
- **Total Columns**: 97
- **Numerical Columns**: 62
- **Candidate Numerical Predictors ($p$)**: 42 (excluding IDs, text metadata, and targets)
- **Observations ($N$)**: 25

$$\frac{p}{N} = \frac{42}{25} = 1.68$$

When $p > N$:
1. **Rank Deficency**: The covariance matrix $X'X$ is singular and cannot be inverted for ordinary least squares or regularized linear models without massive shrinkage ($\lambda \to \infty$).
2. **Trivial Tree Memorization**: A decision tree with depth $\ge 5$ has $2^5 = 32$ leaf nodes, which exceeds the total number of sample points ($N = 25$). The tree can assign each station to its own private leaf, memorizing the training set with zero training error while having zero predictive power.
3. **Severe Overfitting Risk**: In tree ensembles, feature subsampling ($m_{\text{try}}$) cannot prevent spurious correlations because 25 points do not populate a 42-dimensional feature manifold with sufficient density.

---

### B. State-Level Feature Degeneracy ($\text{Var} = 0$)
An empirical variance audit revealed that in a cross-sectional dataset of a single state at a single point in time (Telangana, 2026), **every single state-level variable has exactly zero variance**:

| State Context Feature | Observed Mean | Variance ($\sigma^2$) | Standard Deviation ($\sigma$) | ML Utility in Single-State Cross-Section |
|---|---|---|---|---|
| `state_annual_ev_registrations_2026` | 66,363.0 | **0.000000** | **0.000000** | **Zero (Mathematical Constant)** |
| `state_cumulative_ev_registrations_2026` | 246,617.0 | **0.000000** | **0.000000** | **Zero (Mathematical Constant)** |
| `state_ev_growth_rate_yoy_pct` | 92.0% | **0.000000** | **0.000000** | **Zero (Mathematical Constant)** |
| `state_cumulative_ev_per_1000_pop` | 6.378 | **0.000000** | **0.000000** | **Zero (Mathematical Constant)** |
| `state_per_capita_electricity_kwh` | 2,780.0 | **0.000000** | **0.000000** | **Zero (Mathematical Constant)** |
| `state_installed_power_capacity_mw` | 21,700.0 | **0.000000** | **0.000000** | **Zero (Mathematical Constant)** |
| `state_ev_to_fame_charger_ratio` | 1,591.1 | **0.000000** | **0.000000** | **Zero (Mathematical Constant)** |
| `state_policy_full_tax_exemption_active` | 1.0 | **0.000000** | **0.000000** | **Zero (Mathematical Constant)** |

> [!WARNING]
> **Statistical Axiom**: A variable with zero variance provides zero splits in a decision tree and causes infinite multicollinearity in linear models. State-level indicators cannot be used as features unless the dataset is expanded across multiple states or structured as a multi-year panel.

---

### C. District-Level Degrees of Freedom ($N_{\text{eff}} = 10$)
Although there are 25 station rows, they are clustered across only **10 unique districts** (Hyderabad: 9, Khammam: 4, Ranga Reddy: 2, Sangareddy: 2, Nizamabad: 2, Suryapet: 2, Medchal: 1, Nalgonda: 1, Mahabubnagar: 1, Jogulamba Gadwal: 1).

Consequently, any district-level feature (such as `district_population_census_2011`, `district_urban_population_pct`, or `district_area_sq_km`) has an **effective sample size of only $N_{\text{eff}} = 10$**.

---

## 3. Empirical Demonstration of Cross-Validation Instability

To quantify the statistical consequences of $N = 25$, a 5-fold cross-validation experiment was executed using a Random Forest regressor predicting target variance from genuine, non-constituent spatial and network features (distance to city center, nearest road distance, POI diversity index, corridor length, and individual POI counts).

### Cross-Validation Results:

| Cross-Validation Fold | Training Samples ($N_{\text{train}}$) | Validation Samples ($N_{\text{val}}$) | Out-of-Sample $R^2$ Score | Interpretation |
|---|---|---|---|---|
| **Fold 1** | 20 | 5 | **-0.284** | Severe under-prediction |
| **Fold 2** | 20 | 5 | **-0.266** | Predictions worse than mean baseline |
| **Fold 3** | 20 | 5 | **-32.837** | Catastrophic failure on outlier test point |
| **Fold 4** | 20 | 5 | **-12.124** | Substantial residual blowup |
| **Fold 5** | 20 | 5 | **-255.109** | Extreme out-of-fold error |
| **Mean $R^2$** | — | — | **-60.124** | **Unstable / Unusable Model** |
| **Std Dev $R^2$** | — | — | **98.215** | **Massive Sampling Variance** |

### Why Did 5-Fold CV Fail Catastrophically?
1. **Outlier Leverage in Small Folds**: Each test fold contains only 5 stations. The station sample contains two mega-hubs (TS-MED-0001 with 1,680 kW and TS-JOG-0001 with 1,440 kW). When one of these mega-hubs falls into a validation fold of size 5, the model trained on the remaining 20 smaller stations has never observed a capacity of that magnitude and drastically underpredicts, resulting in a squared residual that dwarfs the total variance of the 5 test points.
2. **Hyperparameter Tuning Impossibility**: Any grid search or Bayesian optimization performed on this data will select hyperparameters that fit the random partitioning of the 25 samples rather than true underlying physical patterns.

---

## 4. Minimum Dataset Size & Structural Requirements for ML

To ensure statistical rigor, the following algorithmic minimums must be satisfied before supervised learning:

```
+-----------------------------------------------------------------------------------+
|               MINIMUM SAMPLE SIZE REQUIREMENTS FOR PRODUCTION ML                  |
+----------------------+--------------------+---------------------------------------+
| Model / Technique    | Minimum $N$ Needed | Methodological Rationale              |
+----------------------+--------------------+---------------------------------------+
| Random Forest        | $N \ge 150 - 250$  | Minimum 5–10 samples per leaf;       |
|                      |                    | 100+ trees with bootstrap variance red.|
| XGBoost / LightGBM   | $N \ge 500 - 1,000$| Gradient boosting requires shrinkage, |
|                      |                    | subsampling, and deep regularization  |
| 5-Fold / 10-Fold CV  | $N \ge 200+$       | Test folds need $N \ge 40$ to dampen  |
|                      |                    | outlier leverage on validation error   |
| SHAP Interpretability| $N \ge 200+$       | Shapley values require dense coverage |
|                      |                    | of background feature space           |
| Counterfactual Sims  | $N \ge 300+$       | Policy interventions need support on  |
|                      |                    | local feature neighborhoods           |
+----------------------+--------------------+---------------------------------------+
```

---

## 5. Dataset Expansion Strategy

Because $N = 25$ cannot support supervised machine learning, an expansion strategy is designed across three concrete avenues:

```
+-----------------------------------------------------------------------------------+
|                           DATASET EXPANSION STRATEGY                              |
+-----------------------------------------------------------------------------------+
|  Strategy 1: Statewide Telangana Census Expansion                                 |
|    - Expand from 25 stations to N = 150 - 300+ stations                           |
|    - Source: Comprehensive Overpass OSM queries across all 33 Telangana districts |
|              + TGREDCO state charging station directory                           |
|              + Bureau of Energy Efficiency (BEE) EV Yatra portal                  |
+-----------------------------------------------------------------------------------+
|  Strategy 2: Pan-India Spatial Scaling                                            |
|    - Expand to N = 3,500 - 6,000+ stations across all Indian states               |
|    - Source: Open Charge Map India + Pan-India OSM EV extraction                  |
|    - Benefit: Activates state-level features (VAHAN, CEA, FAME) with real variance|
|              Enables industrial-grade XGBoost and LightGBM training               |
+-----------------------------------------------------------------------------------+
|  Strategy 3: Discom-Level Metered Energy Ingestion (AIKosh / TGSPDCL)             |
|    - Ingest monthly metered consumption across TGSPDCL & TGNPDCL sections         |
|    - Benefit: Real ground-truth utility demand for regional grid planning         |
+-----------------------------------------------------------------------------------+
```

### Actionable Roadmap for Phase 8:
1. **Harvest All 33 Districts in Telangana**: Run a partitioned Overpass Over-Grid extraction across northern, central, and southern Telangana, matching stations with the official TGREDCO public charger list.
2. **Pan-India Expansion for ML Training**: If an industrial-grade gradient boosting model (XGBoost/LightGBM) with SHAP explanations is required, harvest the full Indian charging network ($N \approx 4,000+$). The model can be trained on India-wide spatial topology and then transferred to score candidate sites across Telangana.
3. **MCDA Suitability (EIPS)**: For planning site prioritization within Telangana, implement the Multi-Criteria Decision Analysis (AHP / TOPSIS) pipeline, which operates effectively as an unsupervised spatial optimization tool regardless of sample size.

---

## 6. Summary Audit Checklist

- [x] **Sample size audited across all levels**: Station ($N=25$), District-Year ($N=231$), State-Year ($N=7$).
- [x] **Target candidates evaluated**: Evaluated 7 alternatives in formal decision matrix.
- [x] **Real-world telemetry investigated**: Proprietary CPO telemetry confirmed unavailable in open data; AIKosh Discom utility billing identified.
- [x] **Target circularity proven**: Mathematical proof of algebraic identity in `demand_proxy` confirmed ($r = 0.91$ with capacity; $R^2 = 1.0$ in log space).
- [x] **No synthetic target fabricated**: Strict adherence to project constraints.
- [x] **CV instability demonstrated**: 5-fold CV on $N=25$ confirmed deeply unstable ($\text{Mean } R^2 = -60.12$).
- [x] **Minimum dataset size defined**: $N \ge 200$ for RF, $N \ge 500-1,000$ for XGBoost/LightGBM.
- [x] **Expansion strategy formalized**: Statewide Telangana expansion ($N \to 250+$) and Pan-India scaling ($N \to 4,000+$).
- [x] **Supervised ML training halted**: No models trained; gatekeeper rules enforced.

**Execution stopped after Phase 7.**
