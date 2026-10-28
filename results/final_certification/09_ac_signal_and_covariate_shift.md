# AC Signal & Spatial Covariate Shift

## Empirical Findings

1. **Strongest Development Discriminator**: `road_type_residential` exhibits Cohen's $d = -1.6074$ ($p = 5.20 \times 10^{-7}$). In development, 35.71% of AC stations were situated on residential roads, compared to only 1.90% of DC stations.
2. **Scientific Interpretation**: "Empirical feature distributions show that road_type_residential has the largest observed effect size separating AC Slow from DC Fast in the development data." (Not described as causal, universally predictive, or a deployment rule).
3. **Spatial Covariate Shift**: In the quarantined outer test, **100.0% of actual AC stations (9/9) were situated on trunk highways**, and **88.9% (8/9) were located on Government Office campuses** with high commercial POI densities (mean 32.6), physically mimicking DC Fast charging hubs.
4. **Scientific Finding**: "Spatial covariate shift was observed between the development and held-out districts. The tested exogenous spatial feature space was insufficient to reliably distinguish AC Slow from DC Fast under the observed held-out distribution."
5. **Site-Context Enrichment Conclusion**: "Site-context enrichment did not demonstrate improved cross-district AC Slow transfer under the independent outer-test protocol."
6. **Future Data-Enrichment Direction**: "Electrical/site-level infrastructure variables (transformer sizing, sanctioned feeder capacity) may provide additional discriminatory information and represent a logical future data-enrichment direction."
