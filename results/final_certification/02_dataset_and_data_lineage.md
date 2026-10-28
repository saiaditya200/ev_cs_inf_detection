# Dataset & Data Lineage

- **Master Dataset File**: `data/final/FINAL_PROJECT_DATASET.csv`
- **Total Station Population**: 166 stations across 78 attributes.
- **Supervised Training Population**: 154 labeled stations across all 33 Telangana districts.
  - Class 1 (DC Fast $\ge 50$ kW): 131 stations (85.06%).
  - Class 0 (AC Slow $< 50$ kW): 23 stations (14.94%).
- **Held-Out Unlabeled Stations**: 12 stations reserved for descriptive visualization.
- **Geographic Coverage**: All 33 administrative districts of Telangana are represented.
- **Partitioning Protocol (Frozen Phase-10)**:
  - Development Set: 119 stations across 25 districts (14 AC / 105 DC).
  - Quarantined Outer Test: 35 stations across 8 held-out districts (9 AC / 26 DC).
  - District Overlap: Exactly ZERO districts shared between development and outer test.
