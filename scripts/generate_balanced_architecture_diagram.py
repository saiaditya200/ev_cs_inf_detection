import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

BASE_DIR = Path("C:/Users/Dell/Downloads/EV_Charging")
FIG_DIR = BASE_DIR / "results" / "india_wide" / "figures"

# Initialize 16:9 canvas (24 x 13.5 inches at 300 DPI = 7200 x 4050 px)
fig, ax = plt.subplots(figsize=(24, 13.5), dpi=300)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Helper function to draw rounded cards
def draw_card(ax, x, y, w, h, bg_color="#ffffff", border_color="#cbd5e1", lw=1.2, radius=0.8, zorder=2):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=lw,
        zorder=zorder
    )
    ax.add_patch(rect)
    return rect

# Helper function to place an image cleanly with a crisp white frame
def place_image(ax, img_path, extent, border_color="#cbd5e1", lw=1.0):
    img = Image.open(img_path)
    x0, x1, y0, y1 = extent
    draw_card(ax, x0-0.2, y0-0.2, (x1-x0)+0.4, (y1-y0)+0.4, bg_color="#ffffff", border_color=border_color, lw=lw, radius=0.4, zorder=3)
    ax.imshow(img, extent=extent, aspect="auto", zorder=4)

# ==============================================================================
# HEADER BANNER (Top: Y=92.5 to 99)
# ==============================================================================
draw_card(ax, 2, 92.5, 96, 6.2, bg_color="#ffffff", border_color="#3b82f6", lw=1.8, radius=0.8)
ax.text(50, 96.5, "EXPLAINABLE AI FRAMEWORK FOR EV CHARGING INFRASTRUCTURE PLANNING", 
        fontsize=17, fontweight="bold", color="#1e3a8a", ha="center", va="center", zorder=5)
ax.text(50, 94.0, "Pan-India Siting Optimization, Anti-Leakage Spatial Ensembles, TreeSHAP Diagnostics & Regional Prioritization", 
        fontsize=11, fontweight="medium", color="#475569", ha="center", va="center", zorder=5)

# ==============================================================================
# COLUMN 1: INPUT LAYER (x = 2 to 24.5)
# ==============================================================================
draw_card(ax, 2, 2.5, 22.5, 88.5, bg_color="#f8fafc", border_color="#93c5fd", lw=1.8, radius=1.0)

# Header
draw_card(ax, 4.5, 86.8, 17.5, 3.4, bg_color="#dbeafe", border_color="#3b82f6", lw=1.2, radius=0.5)
ax.text(13.25, 88.5, "INPUT LAYER", fontsize=12, fontweight="bold", color="#1e40af", ha="center", va="center", zorder=5)

# Input Card 1: National Spatial Registry
draw_card(ax, 3.2, 46.5, 20.1, 39.0, bg_color="#ffffff", border_color="#cbd5e1", lw=1.2, radius=0.8)
ax.text(13.25, 83.2, "National Master EV Registry", fontsize=11, fontweight="bold", color="#0f172a", ha="center", zorder=5)
ax.text(13.25, 81.2, "52,718 Stations Across 36 States & UTs (100% Pan-India)", fontsize=8.2, color="#475569", ha="center", zorder=5)

# Embed Image 1: White-Background GIS Map
place_image(ax, FIG_DIR / "gis_spatial_distribution_white.png", [4.2, 22.3, 49.5, 79.5], border_color="#93c5fd")
ax.text(13.25, 47.8, "Audited Bounding: [8.08°N–34.55°N, 69.80°E–93.85°E] (Zero Synthetic)", fontsize=7.6, fontweight="bold", color="#1e3a8a", ha="center", zorder=5)

# Flow arrow inside Input
ax.annotate("", xy=(13.25, 43.0), xytext=(13.25, 46.0),
            arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=2.0), zorder=5)

# Input Card 2: Anti-Leakage Gate
draw_card(ax, 3.2, 3.5, 20.1, 39.0, bg_color="#ffffff", border_color="#cbd5e1", lw=1.2, radius=0.8)
ax.text(13.25, 40.2, "Anti-Leakage Defensive Gate", fontsize=11, fontweight="bold", color="#b45309", ha="center", zorder=5)
ax.text(13.25, 38.2, "Separation of Post-Build Specs from Greenfield Predictors", fontsize=8.0, color="#78350f", ha="center", zorder=5)

# Quarantined Box
draw_card(ax, 4.2, 28.5, 18.1, 8.5, bg_color="#fef2f2", border_color="#f87171", lw=1.2, radius=0.5)
ax.text(13.25, 34.8, "9 Quarantined Hardware Attributes", fontsize=8.8, fontweight="bold", color="#991b1b", ha="center", zorder=5)
ax.text(13.25, 31.0, "Rated kW • Port Counts • Supply Voltage • Connector Types\n(Eliminates Artificial 99.8% Target Leakage)", fontsize=7.4, color="#b91c1c", ha="center", zorder=5)

# Arrow across Gate
ax.annotate("", xy=(13.25, 25.5), xytext=(13.25, 28.0),
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.8), zorder=5)

# Retained Predictors Box
draw_card(ax, 4.2, 14.5, 18.1, 10.5, bg_color="#f0fdf4", border_color="#4ade80", lw=1.2, radius=0.5)
ax.text(13.25, 22.8, "21 Exogenous Spatial & Grid Predictors", fontsize=8.8, fontweight="bold", color="#166534", ha="center", zorder=5)
ax.text(13.25, 18.0, "• National Highway Distance • Road Network Hierarchy\n• POI Agglomeration Count • Shannon POI Entropy\n• Urban Tier Classification • 33kV Substation Proximity", fontsize=7.4, color="#15803d", ha="center", zorder=5)

# Target Binarization Bar
draw_card(ax, 4.2, 4.8, 18.1, 8.8, bg_color="#ffffff", border_color="#cbd5e1", lw=1.0, radius=0.4)
ax.text(13.25, 11.5, "CEA Guideline Target Binarization", fontsize=8.2, fontweight="bold", color="#334155", ha="center", zorder=5)

w_total = 16.0
w_slow = w_total * 0.6858
w_fast = w_total * 0.3142
x_start = 5.25

ax.barh([8.5], [w_slow], left=[x_start], height=2.8, color="#2563eb", edgecolor="none", zorder=5)
ax.barh([8.5], [w_fast], left=[x_start + w_slow], height=2.8, color="#059669", edgecolor="none", zorder=5)

ax.text(x_start + w_slow / 2, 8.5, "AC Slow: 68.6%", fontsize=7.5, fontweight="bold", color="#ffffff", ha="center", va="center", zorder=6)
ax.text(x_start + w_slow + w_fast / 2, 8.5, "DC: 31.4%", fontsize=7.2, fontweight="bold", color="#ffffff", ha="center", va="center", zorder=6)
ax.text(13.25, 6.0, "36,157 AC Slow (<50kW)  |  16,561 DC Fast (≥50kW)", fontsize=6.8, color="#64748b", ha="center", va="center", zorder=5)

# Main Connecting Arrow: INPUT -> PIPELINE
ax.annotate("", xy=(26.2, 47.0), xytext=(24.6, 47.0),
            arrowprops=dict(arrowstyle="->", color="#2563eb", lw=3.0), zorder=6)


# ==============================================================================
# COLUMN 2: CORE PIPELINE (x = 26.5 to 73.5)
# ==============================================================================
draw_card(ax, 26.5, 2.5, 47.0, 88.5, bg_color="#f8fafc", border_color="#93c5fd", lw=1.8, radius=1.0)

# Header
draw_card(ax, 38.5, 86.8, 23.0, 3.4, bg_color="#dbeafe", border_color="#3b82f6", lw=1.2, radius=0.5)
ax.text(50.0, 88.5, "SPATIAL XAI PLANNING PIPELINE", fontsize=12, fontweight="bold", color="#1e40af", ha="center", va="center", zorder=5)

# --- STAGE 1: SPATIAL DATA PARTITIONING ---
draw_card(ax, 27.5, 75.5, 45.0, 10.5, bg_color="#ffffff", border_color="#cbd5e1", lw=1.2, radius=0.8)
ax.text(29.0, 83.8, "1. Spatially Grouped Stratified Partitioning (Zero Spatial Overlap)", fontsize=10, fontweight="bold", color="#0f172a", zorder=5)

draw_card(ax, 28.5, 76.8, 13.5, 5.8, bg_color="#eff6ff", border_color="#93c5fd", lw=1.0, radius=0.4)
ax.text(35.25, 80.5, "Training Partition (70%)", fontsize=8.2, fontweight="bold", color="#1e40af", ha="center", zorder=5)
ax.text(35.25, 78.0, "36,902 Stations | 25 States", fontsize=7.4, color="#475569", ha="center", zorder=5)

draw_card(ax, 43.2, 76.8, 13.5, 5.8, bg_color="#f0fdf4", border_color="#86efac", lw=1.0, radius=0.4)
ax.text(49.95, 80.5, "Validation Partition (15%)", fontsize=8.2, fontweight="bold", color="#166534", ha="center", zorder=5)
ax.text(49.95, 78.0, "7,908 Stations | 5 States", fontsize=7.4, color="#475569", ha="center", zorder=5)

draw_card(ax, 58.0, 76.8, 13.5, 5.8, bg_color="#faf5ff", border_color="#d8b4fe", lw=1.0, radius=0.4)
ax.text(64.75, 80.5, "Independent Test (15%)", fontsize=8.2, fontweight="bold", color="#6b21a8", ha="center", zorder=5)
ax.text(64.75, 78.0, "7,908 Stations | 6 Unseen States", fontsize=7.4, color="#475569", ha="center", zorder=5)

# Down Arrow Stage 1 -> 2
ax.annotate("", xy=(50.0, 71.5), xytext=(50.0, 75.0),
            arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.8), zorder=5)

# --- STAGE 2: MACHINE LEARNING SPATIAL ENGINE ---
draw_card(ax, 27.5, 51.5, 45.0, 19.5, bg_color="#ffffff", border_color="#cbd5e1", lw=1.2, radius=0.8)
ax.text(29.0, 68.8, "2. Spatial Machine Learning Engine & Candidate Architecture Bake-Off", fontsize=10, fontweight="bold", color="#0f172a", zorder=5)

# Image: Architecture Bake-off
place_image(ax, FIG_DIR / "model_architecture_accuracy_comparison.jpg", [28.5, 48.5, 52.8, 67.0], border_color="#93c5fd")

# Right side of Stage 2: Champion Model multi-metric profile image
place_image(ax, FIG_DIR / "model_champion_metrics_profile.jpg", [49.5, 71.5, 52.8, 67.0], border_color="#f87171")

# Down Arrow Stage 2 -> 3
ax.annotate("", xy=(50.0, 47.5), xytext=(50.0, 51.0),
            arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.8), zorder=5)

# --- STAGE 3: EXPLAINABLE AI & COUNTERFACTUAL OPTIMIZATION ---
draw_card(ax, 27.5, 27.5, 45.0, 19.5, bg_color="#ffffff", border_color="#cbd5e1", lw=1.2, radius=0.8)
ax.text(29.0, 44.8, "3. Explainable AI (TreeSHAP) & Nelder-Mead L1 Counterfactual Optimization", fontsize=10, fontweight="bold", color="#0f172a", zorder=5)

# Left Image: TreeSHAP Feature Attributions
place_image(ax, FIG_DIR / "shap_summary_bar.png", [28.5, 48.5, 28.8, 43.0], border_color="#c084fc")

# Right Image: Counterfactual Optimization
place_image(ax, FIG_DIR / "counterfactual_comparison_india.png", [49.5, 71.5, 28.8, 43.0], border_color="#f87171")

# Down Arrow Stage 3 -> 4
ax.annotate("", xy=(50.0, 23.5), xytext=(50.0, 27.0),
            arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.8), zorder=5)

# --- STAGE 4: REGIONAL PRIORITIZATION (EIPS VIA AHP) ---
draw_card(ax, 27.5, 3.5, 45.0, 19.5, bg_color="#ffffff", border_color="#cbd5e1", lw=1.2, radius=0.8)
ax.text(29.0, 20.8, "4. Multi-Criteria Regional Prioritization (EIPS via Analytic Hierarchy Process)", fontsize=10, fontweight="bold", color="#0f172a", zorder=5)

# Left Image: EIPS Rankings
place_image(ax, FIG_DIR / "eips_state_rankings_india.png", [28.5, 50.5, 4.8, 19.0], border_color="#34d399")

# Right Side: AHP Weighting & Stability Breakdown
draw_card(ax, 51.5, 4.8, 20.0, 14.2, bg_color="#ffffff", border_color="#cbd5e1", lw=1.0, radius=0.5)
ax.text(61.5, 17.2, "AHP Multi-Sectoral Weight Matrix", fontsize=8.8, fontweight="bold", color="#1e293b", ha="center", zorder=5)

criteria = [
    ("• EV Fleet Penetration Rate:", "35% (Demand-Side Pressure)"),
    ("• Infrastructure Gap Index:", "25% (EV/Dispenser Deficit)"),
    ("• National Highway Corridors:", "20% (Arterial Transit Flux)"),
    ("• DISCOM Substation Capacity:", "20% (Grid Hosting Ability)")
]
for i, (c_name, c_val) in enumerate(criteria):
    ax.text(52.5, 14.6 - i * 2.2, c_name, fontsize=7.4, fontweight="bold", color="#334155", zorder=5)
    ax.text(61.5, 14.6 - i * 2.2, c_val, fontsize=7.4, color="#0f766e", zorder=5)

ax.text(61.5, 6.2, "Mathematical Consistency: CR = 0.042 < 0.10 (Stable)", fontsize=7.8, fontweight="bold", color="#166534", ha="center", zorder=5)

# Main Connecting Arrow: PIPELINE -> OUTPUT
ax.annotate("", xy=(75.2, 47.0), xytext=(73.6, 47.0),
            arrowprops=dict(arrowstyle="->", color="#2563eb", lw=3.0), zorder=6)


# ==============================================================================
# COLUMN 3: OUTPUT LAYER (x = 75.5 to 98)
# ==============================================================================
draw_card(ax, 75.5, 2.5, 22.5, 88.5, bg_color="#f8fafc", border_color="#93c5fd", lw=1.8, radius=1.0)

# Header
draw_card(ax, 78.0, 86.8, 17.5, 3.4, bg_color="#dbeafe", border_color="#3b82f6", lw=1.2, radius=0.5)
ax.text(86.75, 88.5, "OUTPUT & DECISION SYSTEM", fontsize=12, fontweight="bold", color="#1e40af", ha="center", va="center", zorder=5)

# Output Card 1: Optimal Siting Decision (Probability Density)
draw_card(ax, 76.5, 62.0, 20.5, 23.5, bg_color="#ffffff", border_color="#cbd5e1", lw=1.2, radius=0.8)
ax.text(86.75, 83.2, "Optimal Siting & Tech Decision", fontsize=10.5, fontweight="bold", color="#0f172a", ha="center", zorder=5)
ax.text(86.75, 81.2, "Bimodal Discrimination at τ* = 0.63 (DC Fast ≥ 50kW)", fontsize=7.8, color="#475569", ha="center", zorder=5)
place_image(ax, FIG_DIR / "model_probability_distribution.jpg", [77.5, 96.0, 63.0, 79.5], border_color="#60a5fa")

# Down Arrow in Output
ax.annotate("", xy=(86.75, 59.2), xytext=(86.75, 61.5),
            arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.8), zorder=5)

# Output Card 2: Holdout Test Confusion Matrix (95.31% Accuracy)
draw_card(ax, 76.5, 33.5, 20.5, 25.0, bg_color="#ffffff", border_color="#cbd5e1", lw=1.2, radius=0.8)
ax.text(86.75, 56.2, "Empirical Validation (N = 7,906)", fontsize=10.5, fontweight="bold", color="#15803d", ha="center", zorder=5)
ax.text(86.75, 54.2, "Accuracy: 95.31% | PR-AUC: 0.8794 | ROC-AUC: 0.9505", fontsize=7.8, fontweight="bold", color="#166534", ha="center", zorder=5)
place_image(ax, FIG_DIR / "confusion_matrix.jpg", [78.2, 95.3, 34.5, 52.5], border_color="#4ade80")

# Down Arrow in Output
ax.annotate("", xy=(86.75, 30.5), xytext=(86.75, 33.0),
            arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.8), zorder=5)

# Output Card 3: Interactive Full-Stack Web GIS Interface
draw_card(ax, 76.5, 3.5, 20.5, 26.5, bg_color="#ffffff", border_color="#cbd5e1", lw=1.2, radius=0.8)
ax.text(86.75, 27.8, "Interactive Pan-India GIS Platform", fontsize=10.5, fontweight="bold", color="#581c87", ha="center", zorder=5)
ax.text(86.75, 25.8, "Real-Time Flask REST API (<150ms) + Leaflet Cartography", fontsize=7.8, color="#6b21a8", ha="center", zorder=5)
place_image(ax, FIG_DIR / "dashboard_overview.jpg", [77.5, 96.0, 4.5, 24.2], border_color="#c084fc")

plt.tight_layout()

# Save in high resolution (300 DPI) as both JPG and PNG
out_jpg = FIG_DIR / "system_architecture_balanced_visual.jpg"
out_png = FIG_DIR / "system_architecture_balanced_visual.png"
fig.savefig(out_jpg, dpi=300, bbox_inches="tight", facecolor="#ffffff", pil_kwargs={"quality": 95})
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="#ffffff")
plt.close(fig)

print(f"[SUCCESS] Saved system_architecture_balanced_visual.jpg: {out_jpg}")

