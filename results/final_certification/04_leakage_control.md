# Leakage Control & Variable Exclusions

To ensure zero target leakage and strict prospective validity, the following safeguards are audited and certified:

1. **Target Formulation Exclusion**: Variables directly measuring or determining charging speed (`charging_power_kw`, `charger_type`) are removed prior to feature processing.
2. **Hardware Specification Exclusion**: Site equipment attributes (`station_capacity`, `connector_type`, `number_of_chargers`, `operator`) are stripped from training and inference pipelines.
3. **Regional Aggregate Leakage Control**: District-level aggregations of fast chargers (`district_fast_stations`, `district_fast_charger_ratio`) are excluded from station predictors.
4. **Synthetic Variable Prohibition**: Unmeasured variables such as `traffic_density` are strictly omitted rather than filled with fabricated numbers.
5. **Inference Guardrails**: In `08_predict.py` and `app.py`, any target or equipment fields passed in existing station records are stripped prior to vector validation.
