import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR = Path("C:/Users/Dell/Downloads/EV_Charging")
FIG_DIR = BASE_DIR / "results" / "india_wide" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Universal font and style settings matching the reference aesthetic
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#1e293b"
plt.rcParams["axes.linewidth"] = 1.0

# ==============================================================================
# FIGURE 1: Model Comparison (PR-AUC & Brier Score)
# Matching media_1794154687957_973e3d0b.png
# ==============================================================================
def generate_figure_1():
    print("[1/4] Generating Figure 1: Model Comparison (PR-AUC & Brier Score)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#ffffff")
    ax2.set_facecolor("#ffffff")

    models = ["Random Forest", "XGBoost", "LightGBM"]
    pr_auc_vals = [0.8794, 0.9094, 0.8703]
    brier_vals = [0.0597, 0.0431, 0.0456]

    # Colors: Champion highlighted, others muted
    pr_colors = ["#2b5c9e", "#8faed3", "#8faed3"]
    brier_colors = ["#d95f2e", "#f0aa86", "#f0aa86"]

    # Subplot 1: PR-AUC
    bars1 = ax1.bar(models, pr_auc_vals, color=pr_colors, width=0.72, edgecolor="none", zorder=3)
    ax1.set_title("Outer Test PR-AUC (higher is better)", fontsize=12.5, fontweight="bold", pad=12, color="#000000")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, fontsize=10.5, color="#000000")
    ax1.tick_params(axis='y', labelsize=10)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#000000")
    ax1.spines["bottom"].set_color("#000000")

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.4f}",
                 ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#000000")

    # Subplot 2: Brier Score
    bars2 = ax2.bar(models, brier_vals, color=brier_colors, width=0.72, edgecolor="none", zorder=3)
    ax2.set_title("Brier Score (lower is better)", fontsize=12.5, fontweight="bold", pad=12, color="#000000")
    ax2.set_ylim(0, 0.08)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, fontsize=10.5, color="#000000")
    ax2.tick_params(axis='y', labelsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color("#000000")
    ax2.spines["bottom"].set_color("#000000")

    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.0015, f"{h:.4f}",
                 ha="center", va="bottom", fontsize=11.5, fontweight="bold", color="#000000")

    plt.tight_layout()
    p_png = FIG_DIR / "model_comparison_pr_brier.png"
    p_jpg = FIG_DIR / "model_comparison_pr_brier.jpg"
    p_alt_jpg = FIG_DIR / "model_architecture_accuracy_comparison.jpg"
    p_alt_png = FIG_DIR / "model_architecture_accuracy_comparison.png"
    
    fig.savefig(p_png, dpi=300, bbox_inches="tight")
    fig.savefig(p_alt_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    im = Image.open(p_png).convert("RGB")
    im.save(p_jpg, "JPEG", quality=95)
    im.save(p_alt_jpg, "JPEG", quality=95)
    print(f"      Saved {p_png.name} & {p_jpg.name}")

# ==============================================================================
# FIGURE 2: Confusion Matrix (Independent Spatial Holdout - 7,906 stations)
# Matching media_1794154692407_5d1f3755.png
# ==============================================================================
def generate_figure_2():
    print("[2/4] Generating Figure 2: Confusion Matrix Heatmap...")
    fig, ax = plt.subplots(figsize=(6.6, 6.0), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Exact numbers from 7,906 independent test split with tau* = 0.63
    # TN=5,174, FP=244, FN=127, TP=2,361
    cm_matrix = np.array([[5174, 244],
                          [127, 2361]])

    # Heatmap color background: 
    # [0,0] and [1,1] are high correct classifications (darker blues)
    # [0,1] and [1,0] are errors (very light blues)
    color_grid = np.array([[0.82, 0.15],
                           [0.10, 0.65]])

    cax = ax.imshow(color_grid, cmap="Blues", vmin=0, vmax=1.0, aspect="auto")

    # Titles & labels matching exact styling
    ax.set_title("Confusion Matrix – Independent Spatial Test Set\n(τ* = 0.63, 7,906 stations)",
                 fontsize=12.5, fontweight="bold", pad=12, color="#000000")
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["AC Slow (0)", "DC Fast (1)"], fontsize=11, color="#000000")
    ax.set_yticklabels(["AC Slow (0)", "DC Fast (1)"], fontsize=11, color="#000000", rotation=90, va="center")
    
    ax.set_xlabel("Predicted", fontsize=12, labelpad=8, color="#000000")
    ax.set_ylabel("Actual", fontsize=12, labelpad=8, color="#000000")

    # Add text inside cells
    # [0,0] TN = 5,174 (white text on dark blue)
    ax.text(0, 0, "5,174", ha="center", va="center", fontsize=23, fontweight="bold", color="#ffffff")
    # [0,1] FP = 244 (black text on light cell)
    ax.text(1, 0, "244", ha="center", va="center", fontsize=23, fontweight="bold", color="#000000")
    # [1,0] FN = 127 (black text on light cell)
    ax.text(0, 1, "127", ha="center", va="center", fontsize=23, fontweight="bold", color="#000000")
    # [1,1] TP = 2,361 (white text on medium blue)
    ax.text(1, 1, "2,361", ha="center", va="center", fontsize=23, fontweight="bold", color="#ffffff")

    # Border
    for spine in ax.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    p_png = FIG_DIR / "confusion_matrix.png"
    p_jpg = FIG_DIR / "confusion_matrix.jpg"
    p_alt_jpg = FIG_DIR / "model_confusion_matrix_test.jpg"
    p_alt_png = FIG_DIR / "model_confusion_matrix_test.png"

    fig.savefig(p_png, dpi=300, bbox_inches="tight")
    fig.savefig(p_alt_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    im = Image.open(p_png).convert("RGB")
    im.save(p_jpg, "JPEG", quality=95)
    im.save(p_alt_jpg, "JPEG", quality=95)
    print(f"      Saved {p_png.name} & {p_jpg.name}")

# ==============================================================================
# FIGURE 3: Global TreeSHAP Feature Ranking (Top 10)
# Matching media_1794154700567_fd7fde87.png
# ==============================================================================
def generate_figure_3():
    print("[3/4] Generating Figure 3: Global TreeSHAP Feature Ranking...")
    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    features = [
        "distance_to_highway_km",
        "nearby_poi_count",
        "poi_diversity_index",
        "distance_to_city_center_km",
        "district_ev_penetration_rate",
        "highway_proximity_score",
        "district_nh_crossing_count",
        "grid_capacity_33kv_indicator",
        "traffic_corridor_high_density",
        "urban_tier_metro_or_tier1"
    ]
    shap_vals = [0.284, 0.215, 0.162, 0.128, 0.114, 0.098, 0.086, 0.072, 0.058, 0.045]

    # Invert so highest is at the top
    features = features[::-1]
    shap_vals = shap_vals[::-1]

    bars = ax.barh(features, shap_vals, color="#2b5c9e", height=0.72, edgecolor="none", zorder=3)
    ax.set_title("Global TreeSHAP Feature Ranking (Top 10)", fontsize=13, fontweight="bold", pad=12, color="#000000")
    ax.set_xlabel("Mean |SHAP value|", fontsize=11, color="#000000")
    ax.set_xlim(0, 0.33)
    ax.tick_params(axis='y', labelsize=10, colors="#000000")
    ax.tick_params(axis='x', labelsize=10, colors="#000000")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.3f}",
                va="center", ha="left", fontsize=10.5, color="#000000")

    plt.tight_layout()
    p_png = FIG_DIR / "shap_summary_bar.png"
    p_jpg = FIG_DIR / "shap_summary_bar.jpg"
    p_alt_png = FIG_DIR / "treeshap_feature_importance_top10.png"
    p_alt_jpg = FIG_DIR / "treeshap_feature_importance_top10.jpg"

    fig.savefig(p_png, dpi=300, bbox_inches="tight")
    fig.savefig(p_alt_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    im = Image.open(p_png).convert("RGB")
    im.save(p_jpg, "JPEG", quality=95)
    im.save(p_alt_jpg, "JPEG", quality=95)
    print(f"      Saved {p_png.name} & {p_jpg.name}")

# ==============================================================================
# FIGURE 4: Tier 1 States - EIPS Prioritization
# Matching media_1794154707198_fb540e4a.png
# ==============================================================================
def generate_figure_4():
    print("[4/4] Generating Figure 4: Tier 1 States - EIPS Prioritization...")
    fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    states = [
        "#1 Chandigarh",
        "#2 Delhi",
        "#3 Lakshadweep",
        "#4 Maharashtra",
        "#5 Karnataka",
        "#6 Gujarat",
        "#7 Telangana",
        "#8 Mizoram",
        "#9 Goa",
        "#10 Uttar Pradesh"
    ]
    eips_scores = [50.3, 48.6, 47.0, 44.5, 44.1, 40.8, 40.1, 34.3, 34.0, 32.7]

    # Invert so #1 is at top
    states = states[::-1]
    eips_scores = eips_scores[::-1]

    bars = ax.barh(states, eips_scores, color="#2b5c9e", height=0.72, edgecolor="none", zorder=3)
    ax.set_title("Tier 1 States – EIPS Prioritization", fontsize=13, fontweight="bold", pad=12, color="#000000")
    ax.set_xlabel("EIPS Score (AHP Multi-Sectoral)", fontsize=11, color="#000000")
    ax.set_xlim(0, 58)
    ax.tick_params(axis='y', labelsize=10, colors="#000000")
    ax.tick_params(axis='x', labelsize=10, colors="#000000")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{w:.1f}",
                va="center", ha="left", fontsize=10.5, color="#000000")

    plt.tight_layout()
    p_png = FIG_DIR / "eips_state_rankings_india.png"
    p_jpg = FIG_DIR / "eips_state_rankings_india.jpg"
    p_alt_png = FIG_DIR / "tier1_states_eips_prioritization.png"
    p_alt_jpg = FIG_DIR / "tier1_states_eips_prioritization.jpg"

    fig.savefig(p_png, dpi=300, bbox_inches="tight")
    fig.savefig(p_alt_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    im = Image.open(p_png).convert("RGB")
    im.save(p_jpg, "JPEG", quality=95)
    im.save(p_alt_jpg, "JPEG", quality=95)
    print(f"      Saved {p_png.name} & {p_jpg.name}")

if __name__ == "__main__":
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    print("\n[SUCCESS] All 4 updated result figures generated successfully!")
