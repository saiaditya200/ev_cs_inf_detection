#!/usr/bin/env python3
"""
Generate high-resolution Pan-India EV Infrastructure Map with pure WHITE background.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path("C:/Users/Dell/Downloads/EV_Charging")
DATA_DIR = BASE_DIR / "data" / "india_wide"
FIG_DIR = BASE_DIR / "results" / "india_wide" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df_30 = pd.read_csv(DATA_DIR / "INDIA_WIDE_EV_CHARGING_MASTER_DATASET_52K_30FEAT.csv")

fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")

# Plot AC Slow stations
slow = df_30[df_30["fast_charging_flag"] == 0]
fast = df_30[df_30["fast_charging_flag"] == 1]

ax.scatter(
    slow["longitude"], slow["latitude"],
    c="#3b82f6", s=4, alpha=0.45, label=f"AC Slow / Destination (<50 kW, N={len(slow):,})",
    edgecolors="none", zorder=3
)
ax.scatter(
    fast["longitude"], fast["latitude"],
    c="#dc2626", s=6, alpha=0.75, label=f"DC Fast / Rapid Corridor (≥50 kW, N={len(fast):,})",
    edgecolors="none", zorder=4
)

ax.set_title("Geospatial Distribution of EV Charging Infrastructure\n52,718 Audited Stations Across 36 States & UTs (100% Pan-India)", 
             fontsize=11.5, fontweight="bold", pad=12, color="#0f172a")
ax.set_xlabel("Longitude (°E)", fontsize=10, fontweight="bold", color="#334155")
ax.set_ylabel("Latitude (°N)", fontsize=10, fontweight="bold", color="#334155")
ax.set_xlim(68.0, 98.0)
ax.set_ylim(6.0, 38.0)
ax.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1", zorder=1)
ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)

for spine in ax.spines.values():
    spine.set_color("#94a3b8")
    spine.set_linewidth(1.0)

plt.tight_layout()
out_white = FIG_DIR / "gis_spatial_distribution_white.png"
fig.savefig(out_white, dpi=300, facecolor="#ffffff")
# Also overwrite gis_spatial_distribution_india.png so everything is consistently white-bg
fig.savefig(FIG_DIR / "gis_spatial_distribution_india.png", dpi=300, facecolor="#ffffff")
plt.close(fig)

print(f"[OK] Generated white-background GIS map: {out_white}")

