# Geographic Information System (GIS) Manifest

- **Station GIS Layer**: `results/gis/station_gis_layer.geojson` (166 verified station features).
- **District EIPS Layer**: `results/gis/district_eips_layer.geojson` (33 verified Telangana district polygons).
- **Interactive Planning Map**: `results/gis/interactive_planning_map.html`.
- **Integrity Verification**:
  - Station coordinates preserved from OpenStreetMap and survey records.
  - District boundaries verified without fabricated centroid fallbacks.
  - Infrastructure-gap visualization accurately reflects verified district deficits.
  - District EIPS rankings are rendered independently from station-level machine learning classifications.
