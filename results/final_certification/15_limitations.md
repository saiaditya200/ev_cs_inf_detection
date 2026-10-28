# Research Limitations

1. **Station-Level Ground Truth**: Ground-truth labels are limited to the available 154 labeled stations across Telangana.
2. **Minority Class Imbalance**: AC Slow charging is the minority class (14.94% of labeled stations).
3. **Cross-District Transferability**: Outer-test AC Slow transfer remains limited due to observed spatial covariate shift.
4. **Spatial Covariate Shift**: Outer-test AC stations along highways physically mimic DC Fast installations in commercial density and road type.
5. **Traffic Density Unavailability**: Empirical traffic sensor measurements are unavailable across all 33 districts and were not synthetically fabricated.
6. **Electrical Site-Level Data Absence**: Transformer ratings and sanctioned feeder capacities are unavailable as certified predictors.
7. **EIPS Planning Boundary**: EIPS is a relative macro prioritization ranking, not an exact site location or station count prescription.
8. **Demand Forecasting Distinction**: Machine learning classification categorizes infrastructure suitability, not continuous future kWh energy demand.
9. **Non-Causality**: TreeSHAP values represent statistical feature attributions and do not establish causal relationships.
10. **Error Floor Boundaries**: The study does not mathematically bound Bayes error or prove non-separability.
