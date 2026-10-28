# Certified 21-Feature Contract

The station classification pipeline uses strictly and exclusively the following 21 exogenous spatial and demographic predictors:

1. `latitude` (GIS Coordinate)
2. `longitude` (GIS Coordinate)
3. `distance_to_highway_km` (Proximity to nearest National/State Highway)
4. `distance_to_city_center_km` (Proximity to district administrative center)
5. `nearby_poi_count` (Total commercial and amenity POIs within catchment)
6. `highway_proximity_score` ($1 / (1 + \text{distance\_km})$)
7. `city_center_proximity_score` ($1 / (1 + \text{distance\_km})$)
8. `poi_diversity_index` (Normalized Shannon entropy across POI categories)
9. `road_type_motorway` (One-hot highway tag)
10. `road_type_primary` (One-hot highway tag)
11. `road_type_residential` (One-hot highway tag)
12. `road_type_secondary` (One-hot highway tag)
13. `road_type_service` (One-hot highway tag)
14. `road_type_trunk` (One-hot highway tag)
15. `road_type_unclassified` (One-hot highway tag)
16. `district_population_density_2011` (Census 2011 density)
17. `district_urban_population_pct` (Census 2011 urbanization rate)
18. `district_projected_density_fy24` (Projected demographic density)
19. `district_nh_crossing_count` (Highway intersection density)
20. `district_ev_penetration_rate` (Registered EV adoption context)
21. `discom_utility_data_available_flag` (Grid administrative reporting indicator)

## Explicitly Excluded Variables

The following variables are strictly excluded from the predictor space:

- `charger_type`: Direct target proxy.
- `charging_power_kw`: Direct mathematical formulation of target.
- `station_capacity`: Equipment specification.
- `connector_type`: Hardware specification.
- `number_of_chargers`: Facility hardware count.
- `operator`: Non-spatial commercial identity.
- `district_fast_stations`: Target aggregate leakage.
- `district_fast_charger_ratio`: Target aggregate leakage.
- `traffic_density`: Excluded because empirical sensor measurements are unavailable across all 33 districts and synthetic data fabrication is strictly prohibited.
