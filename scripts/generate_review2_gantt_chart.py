import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

FIG_DIR = Path("C:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures")

fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")

# Tasks, timelines (start_week, duration_weeks), status
tasks = [
    ("1. Problem Definition & 52k National Dataset Synthesis", 1.0, 2.5, "#2563eb", "Completed"),
    ("2. Data Preprocessing & Anti-Leakage Feature Gate (21 Predictors)", 2.5, 2.5, "#2563eb", "Completed"),
    ("3. GroupKFold Spatial Partitioning (70% Train, 15% Val, 15% Test)", 4.5, 2.5, "#059669", "Completed (Review-1)"),
    ("4. Spatial ML Model Retraining & Candidate Bake-Off (RF / XGB / LGBM)", 6.5, 2.5, "#059669", "Completed"),
    ("5. Threshold Calibration (tau*=0.63) & Validation (95.31% Accuracy)", 8.0, 2.5, "#d97706", "Completed (Review-2)"),
    ("6. TreeSHAP Explainability & Nelder-Mead L1 Counterfactuals", 9.5, 2.5, "#7c3aed", "Completed"),
    ("7. Multi-Criteria AHP Regional Prioritization (EIPS Ranking)", 10.5, 2.0, "#7c3aed", "Completed"),
    ("8. Full-Stack Flask Web GIS Decision Dashboard & Final Phase-I Report", 11.5, 2.5, "#0284c7", "In Progress"),
    ("9. Major Project Phase-II Advanced Research Roadmap (Months 1-6)", 13.5, 2.5, "#64748b", "Phase-II Gateway")
]

y_pos = np.arange(len(tasks))[::-1]

for i, (task, start, dur, color, status) in enumerate(tasks):
    y = y_pos[i]
    # Draw task bar
    ax.barh(y, dur, left=start, height=0.52, color=color, edgecolor="#0f172a", linewidth=1.0, zorder=3)
    # Status label at end of bar
    ax.text(start + dur + 0.2, y, f"[{status}]", va="center", ha="left", fontsize=9.5, fontweight="bold", color=color)

# Review milestones vertical lines
ax.axvline(6.0, color="#059669", linestyle="--", linewidth=1.8, zorder=4)
ax.text(6.0, 9.1, "REVIEW - 1 (W6)\n[Verified]", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#059669",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdf4", edgecolor="#059669", lw=1.2))

ax.axvline(10.0, color="#d97706", linestyle="--", linewidth=2.0, zorder=4)
ax.text(10.0, 9.1, "MAJOR REVIEW - 2 (W10)\n[Current Milestone]", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#d97706",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7", edgecolor="#d97706", lw=1.4))

ax.axvline(14.0, color="#2563eb", linestyle="--", linewidth=1.8, zorder=4)
ax.text(14.0, 9.1, "PHASE-I FINAL VIVA (W14)\n[Submission]", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#2563eb",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eff6ff", edgecolor="#2563eb", lw=1.2))

# Styling
ax.set_yticks(y_pos)
ax.set_yticklabels([t[0] for t in tasks], fontsize=10.5, fontweight="bold", color="#0f172a")
ax.set_xlabel("Project Timeline (Weeks 1 to 16) — Academic Year 2026–27", fontsize=11.5, fontweight="bold", color="#0f172a", labelpad=10)
ax.set_xlim(0.5, 17.0)
ax.set_ylim(-0.8, 10.5)
ax.set_xticks(np.arange(1, 17))
ax.set_xticklabels([f"W{w}" for w in range(1, 17)], fontsize=10, fontweight="bold")
ax.grid(axis="x", linestyle=":", color="#cbd5e1", alpha=0.8, zorder=1)

ax.set_title("Project Work Plan & Milestone Schedule — IV B.Tech Main Project Phase–I (Major Review - 2)", 
             fontsize=13.5, fontweight="bold", pad=28, color="#1e3a8a")

for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#64748b")
ax.spines["bottom"].set_linewidth(1.2)

plt.tight_layout()
out_p = FIG_DIR / "project_gantt_chart_review_2_final.png"
fig.savefig(out_p, dpi=300, facecolor="#ffffff")
fig.savefig(FIG_DIR / "project_gantt_chart_review_2_final.jpg", dpi=300, facecolor="#ffffff")
plt.close(fig)
print(f"[SUCCESS] Saved final Gantt chart: {out_p}")

