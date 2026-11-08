import os
import shutil
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

BASE_DIR = Path("C:/Users/Dell/Downloads/EV_Charging")
FIG_DIR = BASE_DIR / "results" / "india_wide" / "figures"
SRC_PPTX = Path(r"C:\Users\Dell\Downloads\Major_Project_Phase-1_PPT_India_Wide_EV_Charging.pptx")
BACKUP_PPTX = Path(r"C:\Users\Dell\Downloads\Major_Project_Phase-1_PPT_India_Wide_EV_Charging_BACKUP.pptx")
LOCAL_MODIFIED_PPTX = BASE_DIR / "Major_Project_Phase-1_PPT_India_Wide_EV_Charging_Modified.pptx"

# 1. Always restore from the clean backup first if it exists, or create it
if BACKUP_PPTX.exists():
    shutil.copyfile(BACKUP_PPTX, SRC_PPTX)
    print(f"[OK] Restored master PPTX from clean backup: {BACKUP_PPTX}")
else:
    shutil.copyfile(SRC_PPTX, BACKUP_PPTX)
    print(f"[OK] Created backup at: {BACKUP_PPTX}")

prs = Presentation(SRC_PPTX)
print(f"[INFO] Loaded presentation with {len(prs.slides)} slides.")

# Helper to replace image in a picture shape or blip fill
def replace_shape_image(shape, image_path):
    blip = shape._element.xpath('.//a:blip')
    if blip:
        rId = blip[0].attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed']
        part = shape.part.related_part(rId)
        with open(image_path, 'rb') as f:
            part._blob = f.read()
        return True
    return False

# ==============================================================================
# COMMENT 1: In first slide change to Major Review-2
# ==============================================================================
slide1 = prs.slides[0]
found_c1 = False
for s in slide1.shapes:
    if s.shape_type == 6: # Group
        for sub in s.shapes:
            if sub.has_text_frame and "MAJOR PROJECT (PHASE-1) PROJECT REVIEW" in sub.text_frame.text:
                sub.text_frame.text = "MAJOR PROJECT (PHASE-1) MAJOR REVIEW - 2 A.Y 2026-27"
                found_c1 = True
                print("  [OK] Comment 1 applied: Subtitle updated to 'MAJOR REVIEW - 2'.")
    elif s.has_text_frame and "MAJOR PROJECT (PHASE-1) PROJECT REVIEW" in s.text_frame.text:
        s.text_frame.text = "MAJOR PROJECT (PHASE-1) MAJOR REVIEW - 2 A.Y 2026-27"
        found_c1 = True
        print("  [OK] Comment 1 applied: Subtitle updated to 'MAJOR REVIEW - 2'.")

# ==============================================================================
# COMMENT 2: Change the slides from Introduction to Abstract and Abstract to Introduction (swap)
# ==============================================================================
# Current: Slide 3 is INTRODUCTION, Slide 4 is ABSTRACT.
# Swap them so Slide 3 is ABSTRACT and Slide 4 is INTRODUCTION (matching the Agenda).
sldIdLst = prs.slides._sldIdLst
elem3 = sldIdLst[3] # Slide 4 (0-indexed: index 3)
sldIdLst.remove(elem3)
sldIdLst.insert(2, elem3) # Insert before index 2, making Slide 4 the new Slide 3
print("  [OK] Comment 2 applied: Swapped Slide 3 and Slide 4 (Abstract is now Slide 3, Introduction is Slide 4).")

# Update Slide 3 (Abstract) metrics to authoritative champion results
slide3 = prs.slides[2]
for s in slide3.shapes:
    if s.has_text_frame and "This project presents a spatial-aware" in s.text_frame.text:
        s.text_frame.text = (
            "This project presents a spatial-aware, explainable machine learning framework for planning EV charging infrastructure across India. "
            "Trained on an audited national master dataset of 52,718 public charging stations (30 features across all 36 States and UTs), it enforces a strict Anti-Leakage Gate isolating 21 pre-construction spatial, demographic, and grid predictors. "
            "Using 5-fold state-grouped spatial cross-validation (70% Training [36,873 stations], 15% Validation [7,927 stations], 15% Independent Test [7,906 stations]), Random Forest is established as Champion with 95.31% Accuracy, 0.8794 PR-AUC, 0.9505 ROC-AUC, and calibrated threshold tau* = 0.63. "
            "TreeSHAP and counterfactual optimization provide transparent engineering upgrade pathways, while an AHP-based EIPS framework (CR = 0.042) prioritizes all 36 States/UTs into phased investment tiers, deployed via a high-performance Flask-Leaflet Web GIS platform."
        )
        s.text_frame.paragraphs[0].font.size = Pt(13)
        s.text_frame.paragraphs[0].font.color.rgb = RGBColor(15, 23, 42)
        print("  [OK] Updated Slide 3 (Abstract) with authoritative 95.31% metrics.")

# ==============================================================================
# COMMENT 3: Keep the objective points minimum four (4)
# ==============================================================================
# Objectives is Slide 9 (index 8)
slide9 = prs.slides[8]
for s in slide9.shapes:
    if s.has_text_frame and "To integrate 52,718 charging stations" in s.text_frame.text:
        tf = s.text_frame
        tf.clear()
        
        objectives = [
            ("Objective 1 — National Dataset Synthesis & Anti-Leakage Feature Gate",
             "Synthesize, clean, and validate an audited national master registry of 52,718 public charging stations across all 36 States & UTs with 30 attributes, enforcing an Anti-Leakage Gate that quarantines 9 hardware-intrinsic attributes and isolates 21 exogenous spatial, transit, demographic, and power grid predictors."),
            ("Objective 2 — Spatially-Grouped Machine Learning Classification",
             "Implement a state-grouped spatial data division (70% Training [36,873 stations], 15% Validation [7,927 stations], 15% Independent Holdout Testing [7,906 stations] across 36 States/UTs) and benchmark Random Forest, XGBoost, and LightGBM models with calibrated decision thresholds to achieve 95.31% Test Accuracy for DC Fast (≥50 kW) vs. AC Slow (<50 kW) siting."),
            ("Objective 3 — Explainable AI (XAI) & Actionable Counterfactual Optimization",
             "Deploy TreeSHAP to decode global and local spatial feature attributions (identifying highway proximity and commercial POI density as key drivers), and implement constrained L1 Nelder-Mead optimization to prescribe actionable engineering upgrade pathways that elevate marginal sites (P < 0.80) to ≥ 80.0% viability."),
            ("Objective 4 — Multi-Criteria Regional Prioritization & Web GIS Platform",
             "Formulate the Explainable Infrastructure Priority Score (EIPS) using Saaty's Analytic Hierarchy Process (AHP) matrix (CR = 0.042 < 0.10) to rank all 36 States and UTs for phased public investment, delivering all models, counterfactual guidance, and spatial telemetry through an interactive Flask-Leaflet GIS dashboard.")
        ]
        
        for idx, (title, desc) in enumerate(objectives):
            p_title = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
            p_title.text = f"• {title}:"
            p_title.font.bold = True
            p_title.font.size = Pt(13.5)
            p_title.font.color.rgb = RGBColor(30, 58, 138)
            p_title.space_after = Pt(2)
            
            p_desc = tf.add_paragraph()
            p_desc.text = f"   {desc}"
            p_desc.font.size = Pt(11.5)
            p_desc.font.color.rgb = RGBColor(15, 23, 42)
            p_desc.space_after = Pt(8)
            
        print("  [OK] Comment 3 applied: Objectives condensed to exactly 4 comprehensive, robust points.")

# ==============================================================================
# COMMENT 7 (Part A): Mention training & testing percentages in Dataset slides
# ==============================================================================
# Slide 11: DATASETS
slide11 = prs.slides[10]
for s in slide11.shapes:
    if s.has_text_frame and "Dataset Description" in s.text_frame.text:
        tf = s.text_frame
        tf.clear()
        
        lines = [
            ("Dataset Description & Study Area", True, Pt(14), RGBColor(30, 58, 138)),
            ("• Study Area: Complete national coverage across all 36 States & Union Territories of India.", False, Pt(12), RGBColor(15, 23, 42)),
            ("• Authoritative Sources: BEE/OpenChargeMap, OpenStreetMap (OSM), MoRTH VAHAN 4.0, CEA/DISCOM, Census.", False, Pt(12), RGBColor(15, 23, 42)),
            ("• Total Processed Scale: 52,718 individual charging stations × 30 raw spatial, electrical, and demographic columns.", False, Pt(12), RGBColor(15, 23, 42)),
            ("• Feature Partitioning: 21 pre-construction exogenous predictors retained; 9 internal hardware columns quarantined.", False, Pt(12), RGBColor(15, 23, 42)),
            ("• Binary Target Definition: fast_charging_flag — DC Fast ≥ 50 kW (16,561 stations; 31.42%) vs. AC Slow < 50 kW (36,157 stations; 68.58%).", False, Pt(12), RGBColor(15, 23, 42)),
            ("Spatial Data Division (Zero Overlap):", True, Pt(13), RGBColor(180, 83, 9)),
            ("• Training Set (70.0%): 36,873 stations across 25 States & UTs (learning regional spatial patterns)", False, Pt(12), RGBColor(15, 23, 42)),
            ("• Validation Set (15.0%): 7,927 stations across 5 Quarantined States (hyperparameter & threshold calibration)", False, Pt(12), RGBColor(15, 23, 42)),
            ("• Independent Test Set (15.0%): 7,906 stations across 6 Unseen States/UTs (gold-standard generalization test)", False, Pt(12), RGBColor(15, 23, 42))
        ]
        
        for idx, (txt, bold, size, col) in enumerate(lines):
            p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
            p.text = txt
            p.font.bold = bold
            p.font.size = size
            p.font.color.rgb = col
            p.space_after = Pt(3) if not bold else Pt(1)
        print("  [OK] Comment 7 applied: Mentioned 70% / 15% / 15% training & testing percentages in Datasets slide.")

# ==============================================================================
# COMMENT 5 & COMMENT 7 (Part B): Proposed approach rewrite (Mention each clearly)
# ==============================================================================
# Slide 13: METHODOLOGY
slide13 = prs.slides[12]
for s in slide13.shapes:
    if s.has_text_frame and ("Methodology" in s.text_frame.text or "Data Preprocessing:" in s.text_frame.text):
        tf = s.text_frame
        tf.clear()
        
        stages = [
            ("1. Data Ingestion & Anti-Leakage Gate",
             "Ingested 52,718 national stations across 36 States & UTs (30 raw features). Enforced an Anti-Leakage Gate that quarantined 9 hardware attributes (ports, rated kW, supply voltage) to prevent target leakage, isolating 21 pre-construction spatial, transit, and grid predictors."),
            ("2. Spatially-Grouped Stratified Partitioning",
             "Divided the national dataset using 5-fold GroupKFold by State/UT: 70% Training (36,873 stations / 25 states), 15% Validation (7,927 stations / 5 states), and 15% Independent Testing (7,906 stations / 6 unseen states), guaranteeing zero spatial autocorrelation leakage."),
            ("3. Spatial ML Classification Engine",
             "Trained and benchmarked Random Forest, XGBoost, and LightGBM. Established Random Forest as Champion with calibrated threshold (tau* = 0.63), achieving 95.31% Overall Test Accuracy, 0.8794 PR-AUC, and 0.9505 ROC-AUC on unseen states."),
            ("4. Explainable AI & Counterfactual Optimization",
             "Extracted global and local feature attributions via TreeSHAP (highway proximity, POI count, and entropy lead). Implemented constrained L1 Nelder-Mead optimization to prescribe actionable physical site modifications, elevating low-viability sites (P < 0.80) to ≥ 80.0%."),
            ("5. Regional Prioritization (EIPS) & Web GIS Platform",
             "Formulated the Explainable Infrastructure Priority Score (EIPS) via Saaty's AHP multi-sectoral weighting (EV Demand 35%, Gap 25%, Corridors 20%, Grid 20%; CR = 0.042 < 0.10) to rank all 36 States/UTs. Deployed an interactive Flask-Leaflet GIS platform (<150 ms latency).")
        ]
        
        for idx, (st_title, st_desc) in enumerate(stages):
            p_t = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
            p_t.text = f"• {st_title}:"
            p_t.font.bold = True
            p_t.font.size = Pt(13)
            p_t.font.color.rgb = RGBColor(30, 58, 138)
            p_t.space_after = Pt(1)
            
            p_d = tf.add_paragraph()
            p_d.text = f"   {st_desc}"
            p_d.font.size = Pt(11)
            p_d.font.color.rgb = RGBColor(15, 23, 42)
            p_d.space_after = Pt(6)
        print("  [OK] Comment 5 applied: Rewrote Proposed Approach / Methodology mentioning all 5 phases clearly.")

# ==============================================================================
# COMMENT 4: System Architecture:- There should be connection and rewrite the architecture.
# ==============================================================================
# Slide 14: PROPOSED APPROACH (System Architecture)
slide14 = prs.slides[13]
# Update title text
for s in slide14.shapes:
    if s.has_text_frame and "PROPOSED APPROACH" in s.text_frame.text:
        s.text_frame.text = "SYSTEM ARCHITECTURE & END-TO-END PIPELINE"
        s.text_frame.paragraphs[0].font.size = Pt(20)
        s.text_frame.paragraphs[0].font.bold = True

# Replace the architecture image inside Freeform 5 with system_architecture_balanced_visual.jpg
arch_img_path = FIG_DIR / "system_architecture_balanced_visual.jpg"
if arch_img_path.exists():
    s_freeform = slide14.shapes[2]
    if replace_shape_image(s_freeform, arch_img_path):
        print("  [OK] Comment 4 applied: Replaced System Architecture image with connected 3-column architecture diagram.")
    else:
        print("  [WARN] Failed to replace Freeform 5 image fill.")

# ==============================================================================
# COMMENT 6: Change Gantt Chart
# ==============================================================================
# Slide 15: WORK PLAN WITH TIMELINE (GANTT CHART)
slide15 = prs.slides[14]
gantt_img_path = FIG_DIR / "project_gantt_chart_review_2_final.png"
if gantt_img_path.exists():
    for s in slide15.shapes:
        if s.name == "Gantt Chart" or s.shape_type == 13:
            replace_shape_image(s, gantt_img_path)
            print("  [OK] Comment 6 applied: Replaced Gantt Chart image with Major Review-2 milestone schedule.")
            break

# ==============================================================================
# RESULTS SLIDES: Update to 95.31% Final Authority & Individual Matplotlib Charts
# ==============================================================================
# Slide 16: PROPOSED OUTCOME
slide16 = prs.slides[15]
for s in slide16.shapes:
    if s.has_text_frame and "A Random Forest champion classifies" in s.text_frame.text:
        tf = s.text_frame
        tf.clear()
        outcomes = [
            ("Champion Spatial Classifier:", "Random Forest achieves 95.31% Overall Test Accuracy, 0.8794 PR-AUC, and 0.9505 ROC-AUC on 15% unseen holdout stations (N = 7,906) across 8 states/UTs with calibrated threshold tau* = 0.63."),
            ("High DC Fast Detection Rate:", "Captures 94.90% (2,361 / 2,488) of DC Fast charging hubs and 95.50% (5,174 / 5,418) of AC Slow stations on completely unseen regional topologies."),
            ("Explainable AI Insights:", "TreeSHAP isolates primary spatial drivers: National Highway distance (0.284), commercial POI density (0.215), and Shannon POI diversity index (0.162)."),
            ("Actionable Counterfactual Guidance:", "Constrained L1 Nelder-Mead optimization provides feasible physical modification pathways, elevating low-probability candidate sites (P < 0.80) to ≥ 80.0%."),
            ("Objective Regional Prioritization:", "AHP-based EIPS framework (CR = 0.042 < 0.10) ranks all 36 States/UTs: Chandigarh, Delhi, Lakshadweep, Maharashtra, and Karnataka lead Tier 1."),
            ("Full-Stack Planning Deployment:", "Interactive Flask REST API (<150 ms inference) and Leaflet.js cartography visualize spatial predictions, SHAP attributions, and state rankings.")
        ]
        for idx, (ot_title, ot_desc) in enumerate(outcomes):
            p = tf.add_paragraph() if idx > 0 else tf.paragraphs[0]
            p.text = f"• {ot_title} {ot_desc}"
            p.font.size = Pt(11.5)
            p.font.color.rgb = RGBColor(15, 23, 42)
            p.space_after = Pt(5)
        print("  [OK] Updated Slide 16 (Proposed Outcomes) to authoritative 95.31% metrics.")

# Slide 17: RESULTS: MODEL PERFORMANCE
slide17 = prs.slides[16]
for s in slide17.shapes:
    if s.name == "Model Comparison":
        replace_shape_image(s, FIG_DIR / "model_comparison_pr_brier.jpg")
    elif s.name == "Confusion Matrix":
        replace_shape_image(s, FIG_DIR / "confusion_matrix.jpg")
    elif s.name == "Results" and s.has_text_frame:
        tf = s.text_frame
        tf.clear()
        p1 = tf.paragraphs[0]
        p1.text = "• Model Bake-Off on 15% Independent Test Set (N = 7,906 Stations across 8 Unseen States/UTs): Random Forest achieves 95.31% Accuracy, 95.20% Balanced Accuracy, 0.8794 PR-AUC, and 0.9505 ROC-AUC; XGBoost achieves 95.38% (PR-AUC 0.8805) and LightGBM achieves 95.38%."
        p1.font.size = Pt(11)
        p1.font.bold = True
        p1.font.color.rgb = RGBColor(15, 23, 42)
        p1.space_after = Pt(4)
        
        p2 = tf.add_paragraph()
        p2.text = "• Calibrated Decision Threshold (tau* = 0.63): Yields 94.90% DC Fast Recall (2,361/2,488 detected) with 90.63% Precision, and 95.50% AC Slow Recall (5,174/5,418 detected) with 97.60% Precision. Demonstrates robust spatial generalizability with zero overfitting."
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = RGBColor(30, 58, 138)
        print("  [OK] Updated Slide 17 with individual Matplotlib JPG charts & 95.31% metrics.")

# Slide 18: RESULTS: EXPLAINABILITY & EIPS RANKING
slide18 = prs.slides[17]
for s in slide18.shapes:
    if s.name == "SHAP Importance":
        replace_shape_image(s, FIG_DIR / "shap_summary_bar.png")
    elif s.name == "EIPS Rankings":
        replace_shape_image(s, FIG_DIR / "eips_state_rankings_india.png")
    elif s.name == "Results" and s.has_text_frame:
        tf = s.text_frame
        tf.clear()
        p1 = tf.paragraphs[0]
        p1.text = "• TreeSHAP Global Feature Hierarchy: National Highway distance (|SHAP| = 0.284), commercial POI amenity count (0.215), and Shannon POI diversity index (0.162) are the primary drivers of fast-charging viability. Suburban arterial intersections exhibit superior hosting potential over congested city centers."
        p1.font.size = Pt(10.5)
        p1.font.color.rgb = RGBColor(15, 23, 42)
        p1.space_after = Pt(4)
        
        p2 = tf.add_paragraph()
        p2.text = "• AHP-Weighted Regional Prioritization (EIPS): Derived via Saaty's pairwise comparison matrix (CR = 0.042 < 0.10, proving mathematical stability). Criteria weights: EV Demand 35%, Infrastructure Gap 25%, Highway Transit 20%, Grid Spare Capacity 20%. Ranks all 36 States/UTs into 3 action tiers."
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = RGBColor(30, 58, 138)
        print("  [OK] Updated Slide 18 with individual TreeSHAP and EIPS ranking charts.")

# Slide 19: CONCLUSION
slide19 = prs.slides[18]
for s in slide19.shapes:
    if s.has_text_frame and "This project presents a spatial-aware" in s.text_frame.text:
        tf = s.text_frame
        tf.clear()
        conc_text = (
            "This project successfully designs and validates an Explainable AI and multi-criteria spatial decision framework for public EV charging infrastructure planning across India.\n\n"
            "• National Scale & Zero Leakage: Audited 52,718 charging stations across all 36 States & UTs with 30 attributes, enforcing an Anti-Leakage Gate that quarantined 9 hardware attributes and retained 21 pre-construction predictors.\n"
            "• Spatially-Grouped Validation: Enforced strict state-level boundary isolation (70% Train [36,873], 15% Val [7,927], 15% Holdout Test [7,906]) to eliminate spatial autocorrelation overfitting.\n"
            "• High-Fidelity Classification: Random Forest champion model achieved 95.31% Overall Test Accuracy, 0.8794 PR-AUC, and 0.9505 ROC-AUC on completely unseen states with calibrated threshold tau* = 0.63.\n"
            "• Transparent Decision Support: TreeSHAP attribution and Nelder-Mead L1 counterfactuals provide transparent, actionable engineering directives for site upgrades, while AHP-based EIPS ranks all 36 States/UTs.\n"
            "• Operational GIS Tool: Deployed an interactive Flask REST API (<150 ms) and Leaflet.js cartographic dashboard for real-time municipal, utility, and CPO investment planning."
        )
        p = tf.paragraphs[0]
        p.text = conc_text
        p.font.size = Pt(11.5)
        p.font.color.rgb = RGBColor(15, 23, 42)
        print("  [OK] Updated Slide 19 (Conclusion) with definitive Phase-I conclusions.")

# 4. Save both the updated file in Downloads and the local workspace copy
prs.save(SRC_PPTX)
prs.save(LOCAL_MODIFIED_PPTX)
print("\n" + "=" * 80)
print(f"[SUCCESS] Modified presentation saved successfully!")
print(f"  Direct Download File: {SRC_PPTX}")
print(f"  Local Workspace File: {LOCAL_MODIFIED_PPTX}")
print("=" * 80)

