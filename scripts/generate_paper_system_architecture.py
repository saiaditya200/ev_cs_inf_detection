"""
Python generator for publication-grade SYSTEM ARCHITECTURE diagram.
Modeled exactly on the user's reference minor-project composition:
- Canvas: 1800 x 1800 px, pale cyan gradient, thin border.
- Header: Bold serif 'SYSTEM ARCHITECTURE'.
- Left Column: DATA INPUTS with 4 real project input images.
- Upper Central Panel: Data Engineering with 3D slabs, 21-D skip connection assembly, 3D feature cube.
- Middle Central Panel: Spatial-Aware ML Classification with 3 3D blocks (Spatial Partitioning, Optimization, Champion RF).
- Lower Central Panel: Explainable AI with TreeSHAP beeswarm image and Counterfactual comparison image.
- Right Column: EVALUATION & PLANNING OUTPUTS with 4 real project output figures.
- Zero emojis, zero cartoon doodles. Ready for paper inclusion.
"""

import os
import math
from PIL import Image, ImageDraw, ImageFont

def get_font(name, size, bold=False):
    font_path = f"C:/Windows/Fonts/{name}"
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()

font_title_main = get_font("timesbd.ttf", 44)
font_panel_h1 = get_font("arialbd.ttf", 19)
font_panel_sub = get_font("arial.ttf", 13)
font_block_h1 = get_font("arialbd.ttf", 16)
font_block_body = get_font("arial.ttf", 12)
font_block_bold = get_font("arialbd.ttf", 12)
font_slab_text = get_font("arialbd.ttf", 13)
font_item_title = get_font("arialbd.ttf", 14)
font_item_sub = get_font("arial.ttf", 11.5)
font_badge = get_font("arialbd.ttf", 11)
font_arrow_tag = get_font("arialbd.ttf", 12)

def adjust_color(color, factor):
    r, g, b = color
    if factor > 0:
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
    else:
        factor = abs(factor)
        r = int(r * (1 - factor))
        g = int(g * (1 - factor))
        b = int(b * (1 - factor))
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

def draw_3d_slab(draw, x, y, w, h, d, color, outline="#334155"):
    c_top = adjust_color(color, 0.28)
    c_side = adjust_color(color, -0.28)
    
    # Top face
    draw.polygon([
        (x, y),
        (x + d, y - d),
        (x + w + d, y - d),
        (x + w, y)
    ], fill=c_top, outline=outline, width=1)
    
    # Side face
    draw.polygon([
        (x + w, y),
        (x + w + d, y - d),
        (x + w + d, y + h - d),
        (x + w, y + h)
    ], fill=c_side, outline=outline, width=1)
    
    # Front face
    draw.rectangle([x, y, x + w, y + h], fill=color, outline=outline, width=1)

def draw_vertical_text(img, text, font, font_color, center_x, center_y):
    dummy_draw = ImageDraw.Draw(img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    txt_img = Image.new('RGBA', (tw + 12, th + 12), (0, 0, 0, 0))
    t_draw = ImageDraw.Draw(txt_img)
    t_draw.text((6, 6), text, font=font, fill=font_color)
    
    rotated = txt_img.rotate(90, expand=True, resample=Image.BICUBIC)
    rw, rh = rotated.size
    img.paste(rotated, (int(center_x - rw / 2), int(center_y - rh / 2)), rotated)

def draw_3d_block(draw, x, y, w, h, d, color, title, subtitle_lines=None, badge=None, outline="#334155"):
    c_top = adjust_color(color, 0.25)
    c_side = adjust_color(color, -0.28)
    
    draw.polygon([(x, y), (x + d, y - d), (x + w + d, y - d), (x + w, y)], fill=c_top, outline=outline, width=1)
    draw.polygon([(x + w, y), (x + w + d, y - d), (x + w + d, y + h - d), (x + w, y + h)], fill=c_side, outline=outline, width=1)
    draw.rectangle([x, y, x + w, y + h], fill=color, outline=outline, width=1)
    
    # Title
    t_bbox = draw.textbbox((0, 0), title, font=font_block_h1)
    tw = t_bbox[2] - t_bbox[0]
    draw.text((x + (w - tw) / 2, y + 16), title, font=font_block_h1, fill="#0f172a")
    
    # Separator line
    draw.line([(x + 16, y + 42), (x + w - 16, y + 42)], fill=outline, width=1)
    
    # Subtitles
    curr_y = y + 56
    if subtitle_lines:
        for line, is_bold in subtitle_lines:
            f = font_block_bold if is_bold else font_block_body
            fc = "#c2410c" if is_bold else "#334155"
            s_bbox = draw.textbbox((0, 0), line, font=f)
            sw = s_bbox[2] - s_bbox[0]
            draw.text((x + (w - sw) / 2, curr_y), line, font=f, fill=fc)
            curr_y += 24
            
    # Bottom Badge
    if badge:
        bw = w - 32
        bh = 26
        bx = x + 16
        by = y + h - 36
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=4, fill="#ffffff", outline=outline, width=1)
        b_bbox = draw.textbbox((0, 0), badge, font=font_badge)
        b_tw = b_bbox[2] - b_bbox[0]
        draw.text((bx + (bw - b_tw) / 2, by + 5), badge, font=font_badge, fill=outline)

def draw_3d_cube(draw, x, y, size, d, color, outline="#334155"):
    c_top = adjust_color(color, 0.30)
    c_side = adjust_color(color, -0.30)
    s = size
    draw.polygon([(x, y), (x + d, y - d), (x + s + d, y - d), (x + s, y)], fill=c_top, outline=outline, width=1)
    draw.polygon([(x + s, y), (x + s + d, y - d), (x + s + d, y + s - d), (x + s, y + s)], fill=c_side, outline=outline, width=1)
    draw.rectangle([x, y, x + s, y + s], fill=color, outline=outline, width=1)

def draw_arrow(draw, start, end, color="#334155", width=2, arrow_size=8):
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    angle = math.atan2(y2 - y1, x2 - x1)
    p1 = (x2 - arrow_size * math.cos(angle - math.pi / 6), y2 - arrow_size * math.sin(angle - math.pi / 6))
    p2 = (x2 - arrow_size * math.cos(angle + math.pi / 6), y2 - arrow_size * math.sin(angle + math.pi / 6))
    draw.polygon([end, p1, p2], fill=color)

def draw_curved_skip_arrow(draw, start_x, start_y, end_x, end_y, height=35, color="#334155", width=2, arrow_size=8):
    draw.line([(start_x, start_y), (start_x, start_y - height)], fill=color, width=width)
    draw.line([(start_x, start_y - height), (end_x, start_y - height)], fill=color, width=width)
    draw.line([(end_x, start_y - height), (end_x, end_y)], fill=color, width=width)
    p1 = (end_x - arrow_size * 0.7, end_y - arrow_size)
    p2 = (end_x + arrow_size * 0.7, end_y - arrow_size)
    draw.polygon([(end_x, end_y), p1, p2], fill=color)

def paste_rounded_image(canvas, img_path, box, radius=8, outline="#334155", outline_width=2):
    x, y, w, h = [int(v) for v in box]
    radius = int(radius)
    outline_width = int(outline_width)
    if not os.path.exists(img_path):
        return
    src = Image.open(img_path).convert('RGB')
    src_w, src_h = src.size
    target_aspect = w / h
    src_aspect = src_w / src_h
    
    if src_aspect > target_aspect:
        new_w = int(src_h * target_aspect)
        left = int((src_w - new_w) / 2)
        src = src.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_aspect)
        top = int((src_h - new_h) / 2)
        src = src.crop((0, top, src_w, top + new_h))
        
    src = src.resize((w, h), Image.Resampling.LANCZOS)
    mask = Image.new('L', (w, h), 0)
    m_draw = ImageDraw.Draw(mask)
    m_draw.rounded_rectangle([0, 0, w, h], radius=radius, fill=255)
    canvas.paste(src, (x, y), mask)
    c_draw = ImageDraw.Draw(canvas)
    c_draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, outline=outline, width=outline_width)

def generate_system_architecture_diagram(output_path):
    width = 1800
    height = 1800
    
    # 1. Background
    img = Image.new('RGB', (width, height), (218, 235, 248))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        r = int(216 + (244 - 216) * (y / height))
        g = int(234 + (250 - 234) * (y / height))
        b = int(248 + (253 - 248) * (y / height))
        draw.line([(0, y), (width, y)], fill=(r, g, b))
        
    # Outer subtle border
    draw.rectangle([12, 12, width - 12, height - 12], outline="#b0c8e0", width=3)
    
    # 2. Main Title
    title_text = "SYSTEM ARCHITECTURE"
    t_bbox = draw.textbbox((0, 0), title_text, font=font_title_main)
    tw = t_bbox[2] - t_bbox[0]
    draw.text(((width - tw) / 2, 45), title_text, font=font_title_main, fill="#0f172a")
    
    # =========================================================================
    # 3. LEFT COLUMN: DATA INPUTS (x=35, y=105, w=275, h=1650)
    # =========================================================================
    lc_x, lc_y, lc_w, lc_h = 35, 105, 275, 1650
    draw.rounded_rectangle([lc_x, lc_y, lc_x + lc_w, lc_y + lc_h], radius=14, fill="#ffffff", outline="#334155", width=2)
    
    h_text = "DATA INPUTS"
    h_bbox = draw.textbbox((0, 0), h_text, font=font_panel_h1)
    draw.text((lc_x + (lc_w - (h_bbox[2] - h_bbox[0])) / 2, lc_y + 16), h_text, font=font_panel_h1, fill="#0f172a")
    draw.line([(lc_x + 16, lc_y + 46), (lc_x + lc_w - 16, lc_y + 46)], fill="#cbd5e1", width=1)
    
    inputs_data = [
        ("EV_Charging/reports/figures/phase10_spatial_split_map.png", "Charging Stations", "OpenChargeMap API (166 Sites)"),
        ("EV_Charging/reports/phase4_gis_validation_map.png", "Road Networks & POIs", "OSM Highways & Corridors"),
        ("EV_Charging/results/figures/infrastructure_gap_vs_adoption.png", "EV Fleet Registry", "VAHAN Panel (FY18–FY24)"),
        ("EV_Charging/reports/figures/phase10_district_station_counts.png", "Grid Utility Context", "TGSPDCL DISCOM Billing")
    ]
    
    item_y = lc_y + 58
    img_w, img_h = 245, 270
    for img_p, tit, sub in inputs_data:
        ix = lc_x + int((lc_w - img_w) / 2)
        paste_rounded_image(img, img_p, (ix, item_y, img_w, img_h), radius=8, outline="#334155", outline_width=2)
        
        # Labels below image
        t_b = draw.textbbox((0, 0), tit, font=font_item_title)
        draw.text((lc_x + (lc_w - (t_b[2] - t_b[0])) / 2, item_y + img_h + 8), tit, font=font_item_title, fill="#0f172a")
        s_b = draw.textbbox((0, 0), sub, font=font_item_sub)
        draw.text((lc_x + (lc_w - (s_b[2] - s_b[0])) / 2, item_y + img_h + 28), sub, font=font_item_sub, fill="#475569")
        
        item_y += 395
        
    # Arrows from left column
    draw_arrow(draw, (lc_x + lc_w, 360), (345, 360), color="#334155", width=2, arrow_size=10)
    draw_arrow(draw, (lc_x + lc_w, 950), (345, 950), color="#334155", width=2, arrow_size=10)

    # =========================================================================
    # 4. UPPER CENTRAL PANEL: DATA ENGINEERING & SPATIAL ENRICHMENT
    # (x=345, y=105, w=1420, h=565)
    # =========================================================================
    ue_x, ue_y, ue_w, ue_h = 345, 105, 1420, 565
    draw.rounded_rectangle([ue_x, ue_y, ue_x + ue_w, ue_y + ue_h], radius=14, fill="#ffffff", outline="#334155", width=2)
    
    ue_title = "Data Engineering & Spatial Enrichment (Strict Anti-Leakage Gate — Exogenous Predictors)"
    draw.text((ue_x + 24, ue_y + 16), ue_title, font=font_panel_h1, fill="#0f172a")
    ue_sub = "Extracts 21 certified non-leaky spatial, road topology, and demographic predictors from multi-source registries"
    draw.text((ue_x + 24, ue_y + 44), ue_sub, font=font_panel_sub, fill="#475569")
    draw.line([(ue_x + 20, ue_y + 68), (ue_x + ue_w - 20, ue_y + 68)], fill="#e2e8f0", width=1)
    
    # 4.1 Feature Extraction Slabs (4 Slabs)
    slab_y = ue_y + 160
    slab_h = 310
    slab_w = 46
    slab_d = 16
    
    slabs1 = [
        (ue_x + 35, (147, 197, 253), "Haversine Dist (km)"),
        (ue_x + 85, (134, 239, 172), "OSM Road Network"),
        (ue_x + 135, (253, 186, 116), "Shannon POI Entropy"),
        (ue_x + 185, (103, 232, 249), "One-Hot Encodings (7)")
    ]
    for sx, sc, stext in slabs1:
        draw_3d_slab(draw, sx, slab_y, slab_w, slab_h, slab_d, sc)
        draw_vertical_text(img, stext, font_slab_text, "#0f172a", sx + slab_w / 2, slab_y + slab_h / 2)
        
    draw_arrow(draw, (ue_x + 245, slab_y + 155), (ue_x + 295, slab_y + 155), color="#334155", width=2, arrow_size=9)
    
    # 4.2 Inverted Residual Feature Assembly (3 Slabs with arched skip connection)
    assembly_x = ue_x + 300
    draw.text((assembly_x + 15, slab_y - 65), "Feature Assembly Block", font=font_block_h1, fill="#0f172a")
    draw_curved_skip_arrow(draw, assembly_x + 10, slab_y - 20, assembly_x + 185, slab_y - 5, height=35, color="#334155", width=2, arrow_size=8)
    draw.text((assembly_x + 70, slab_y - 75), "x 21 Feats", font=font_arrow_tag, fill="#7e22ce")
    
    slabs2 = [
        (assembly_x + 10, (244, 114, 182), "Road Feats (7)"),
        (assembly_x + 65, (253, 224, 71), "Context Feats (6)"),
        (assembly_x + 120, (216, 180, 254), "Spatial Dist (8)")
    ]
    for sx, sc, stext in slabs2:
        draw_3d_slab(draw, sx, slab_y + 30, slab_w, slab_h - 60, slab_d, sc)
        draw_vertical_text(img, stext, font_slab_text, "#0f172a", sx + slab_w / 2, slab_y + slab_h / 2)
        
    draw.text((assembly_x - 10, slab_y + slab_h + 15), "{ 21 Certified Exogenous Features }", font=font_block_bold, fill="#0f766e")
    
    draw_arrow(draw, (assembly_x + 185, slab_y + 155), (assembly_x + 235, slab_y + 155), color="#334155", width=2, arrow_size=9)
    
    # 4.3 Preprocessing & Leakage Gate Slabs
    gate_x = assembly_x + 240
    # Tall green slab: StandardScaler
    draw_3d_slab(draw, gate_x, slab_y - 20, 54, slab_h + 40, 18, (74, 222, 128))
    draw_vertical_text(img, "StandardScaler (Zero Mean, Unit Var)", font_slab_text, "#0f172a", gate_x + 27, slab_y + slab_h / 2)
    
    # Orange slab: Anti-Leakage Gate
    draw_3d_slab(draw, gate_x + 64, slab_y + 40, 54, slab_h - 70, 18, (251, 146, 60))
    draw_vertical_text(img, "Anti-Leakage Gate (Specs Blocked)", font_slab_text, "#0f172a", gate_x + 91, slab_y + slab_h / 2)
    
    draw_arrow(draw, (gate_x + 135, slab_y + 155), (gate_x + 205, slab_y + 155), color="#334155", width=2, arrow_size=9)
    
    # 4.4 3D Isometric Feature Vector Cube
    cube_x = gate_x + 215
    cube_size = 135
    cube_d = 40
    draw.text((cube_x - 15, slab_y + 25), "Global Feature Vector", font=font_block_h1, fill="#0f172a")
    draw_3d_cube(draw, cube_x + 10, slab_y + 60, cube_size, cube_d, (147, 51, 234))
    draw.text((cube_x + 5, slab_y + cube_size + 115), "21-D Feature Matrix", font=font_panel_h1, fill="#7e22ce")
    draw.text((cube_x + 12, slab_y + cube_size + 140), "154 Stations x 21 Predictors", font=font_panel_sub, fill="#475569")
    
    # Polyline routing from Cube down to Middle ML Panel
    c_out_x = cube_x + 80
    c_out_y = slab_y + cube_size + 80
    draw.line([(c_out_x, c_out_y), (c_out_x, ue_y + ue_h + 20)], fill="#334155", width=2)
    draw.line([(c_out_x, ue_y + ue_h + 20), (510, ue_y + ue_h + 20)], fill="#334155", width=2)
    draw_arrow(draw, (510, ue_y + ue_h + 20), (510, 698), color="#334155", width=2, arrow_size=9)
    draw.text((800, ue_y + ue_h + 8), "21-D Exogenous Matrix Feed", font=font_arrow_tag, fill="#0f172a")

    # =========================================================================
    # 5. MIDDLE CENTRAL PANEL: SPATIAL-AWARE ML CLASSIFICATION PIPELINE
    # (x=345, y=700, w=1035, h=495)
    # =========================================================================
    mp_x, mp_y, mp_w, mp_h = 345, 700, 1035, 495
    draw.rounded_rectangle([mp_x, mp_y, mp_x + mp_w, mp_y + mp_h], radius=14, fill="#ffffff", outline="#334155", width=2)
    
    mp_title = "Spatial-Aware ML Classification Pipeline"
    draw.text((mp_x + 24, mp_y + 16), mp_title, font=font_panel_h1, fill="#0f172a")
    mp_sub = "Zero Spatial Overlap — District GroupKFold Cross-Validation (Cost-Sensitive τ* = 0.76)"
    draw.text((mp_x + 24, mp_y + 44), mp_sub, font=font_panel_sub, fill="#475569")
    draw.line([(mp_x + 20, mp_y + 68), (mp_x + mp_w - 20, mp_y + 68)], fill="#e2e8f0", width=1)
    
    # 3 Connected 3D Blocks
    b_y = mp_y + 105
    b_h = 345
    b_d = 16
    
    # Block 1: Spatial Partitioning (Cyan)
    b1_x = mp_x + 25
    b1_w = 285
    b1_lines = [
        ("District GroupKFold (4 Folds)", False),
        ("Zero District Overlap", True),
        ("119 Dev Stations (25 Dists)", False),
        ("35 Outer Test (8 Dists)", False),
        ("Preserves Spatial Autonomy", False)
    ]
    draw_3d_block(draw, b1_x, b_y, b1_w, b_h, b_d, (153, 246, 228), "Spatial Partitioning", b1_lines, badge="Strict Spatial Independence")
    
    draw_arrow(draw, (b1_x + b1_w + b_d, b_y + b_h / 2), (b1_x + b1_w + 45, b_y + b_h / 2), color="#334155", width=2, arrow_size=9)
    draw.text((b1_x + b1_w + 10, b_y + b_h / 2 - 20), "21-D", font=font_arrow_tag, fill="#0f766e")
    
    # Block 2: Model Optimization (Purple)
    b2_x = b1_x + b1_w + 48
    b2_w = 310
    b2_lines = [
        ("RF | XGBoost | LightGBM", True),
        ("112 Hyperparam Grids", False),
        ("PR-AUC & Recall Tiebreaks", False),
        ("Cost-Sensitive Thresholds", False),
        ("Optimal Cut: tau* = 0.76", True)
    ]
    draw_3d_block(draw, b2_x, b_y, b2_w, b_h, b_d, (233, 213, 255), "Model Optimization", b2_lines, badge="Development Selection Only")
    
    draw_arrow(draw, (b2_x + b2_w + b_d, b_y + b_h / 2), (b2_x + b2_w + 45, b_y + b_h / 2), color="#334155", width=2, arrow_size=9)
    draw.text((b2_x + b2_w + 10, b_y + b_h / 2 - 20), "RF", font=font_arrow_tag, fill="#7e22ce")
    
    # Block 3: Champion Random Forest (Coral)
    b3_x = b2_x + b2_w + 48
    b3_w = 310
    b3_lines = [
        ("Random Forest Champion", True),
        ("n_estimators=200, depth=3", False),
        ("Dev PR-AUC: 0.9209", False),
        ("Outer Test Acc: 57.14%", True),
        ("Balanced Acc: 60.26%", True)
    ]
    draw_3d_block(draw, b3_x, b_y, b3_w, b_h, b_d, (254, 215, 170), "Champion Random Forest", b3_lines, badge="Outer Test Generalization")
    
    # Arrow from Champion RF to Right Column
    draw_arrow(draw, (b3_x + b3_w + b_d, b_y + b_h / 2), (1408, b_y + b_h / 2), color="#334155", width=2, arrow_size=10)
    
    # Downward connector to Lower Explainability Panel
    draw.line([(b3_x + b3_w / 2, b_y + b_h + b_d), (b3_x + b3_w / 2, mp_y + mp_h + 16)], fill="#334155", width=2)
    draw_arrow(draw, (b3_x + b3_w / 2, mp_y + mp_h + 16), (b3_x + b3_w / 2, 1228), color="#334155", width=2, arrow_size=9)

    # =========================================================================
    # 6. LOWER CENTRAL PANEL: EXPLAINABLE AI & COUNTERFACTUAL PLANNING
    # (x=345, y=1225, w=1035, h=530)
    # =========================================================================
    lp_x, lp_y, lp_w, lp_h = 345, 1225, 1035, 530
    draw.rounded_rectangle([lp_x, lp_y, lp_x + lp_w, lp_y + lp_h], radius=14, fill="#ffffff", outline="#334155", width=2)
    
    lp_title = "Explainable AI (TreeSHAP) & Actionable Counterfactuals"
    draw.text((lp_x + 24, lp_y + 16), lp_title, font=font_panel_h1, fill="#0f172a")
    lp_sub = "Micro-Level Attributions (Exact Shapley Values) + Prescriptive What-If Spatial Planning Levers"
    draw.text((lp_x + 24, lp_y + 44), lp_sub, font=font_panel_sub, fill="#475569")
    draw.line([(lp_x + 20, lp_y + 68), (lp_x + lp_w - 20, lp_y + 68)], fill="#e2e8f0", width=1)
    
    # Algorithmic workflow boxes (4 boxes in sequence)
    flow_y = lp_y + 85
    f_boxes = [
        ("Trained RF Model", "Frozen Estimator", 130),
        ("TreeSHAP Engine", "Game Theory Values", 140),
        ("Feature Attribution", "Local Waterfall Cuts", 145),
        ("Counterfactual Search", "Minimal Feasible Delta", 155)
    ]
    fx = lp_x + 25
    for btit, bsub, bw in f_boxes:
        draw.rounded_rectangle([fx, flow_y, fx + bw, flow_y + 50], radius=6, fill="#f8fafc", outline="#334155", width=2)
        bt_b = draw.textbbox((0, 0), btit, font=font_item_title)
        draw.text((fx + (bw - (bt_b[2] - bt_b[0])) / 2, flow_y + 8), btit, font=font_item_title, fill="#0f172a")
        bs_b = draw.textbbox((0, 0), bsub, font=font_item_sub)
        draw.text((fx + (bw - (bs_b[2] - bs_b[0])) / 2, flow_y + 28), bsub, font=font_item_sub, fill="#475569")
        
        # Arrow to next box if not last
        if btit != f_boxes[-1][0]:
            draw_arrow(draw, (fx + bw, flow_y + 25), (fx + bw + 32, flow_y + 25), color="#334155", width=2, arrow_size=7)
            fx += bw + 35
        else:
            fx += bw
            
    # Arrow pointing to the real images below
    draw_arrow(draw, (lp_x + 280, flow_y + 50), (lp_x + 280, flow_y + 90), color="#334155", width=2, arrow_size=8)
    draw_arrow(draw, (lp_x + 760, flow_y + 50), (lp_x + 760, flow_y + 90), color="#334155", width=2, arrow_size=8)
    
    # Real Project Image 1: TreeSHAP Beeswarm Plot
    sh_w, sh_h = 470, 310
    sh_x, sh_y = lp_x + 25, flow_y + 95
    paste_rounded_image(img, "EV_Charging/results/figures/shap_beeswarm.png", (sh_x, sh_y, sh_w, sh_h), radius=8, outline="#334155", outline_width=2)
    draw.text((sh_x + 10, sh_y - 20), "TreeSHAP Summary Attribution (Beeswarm)", font=font_block_bold, fill="#7e22ce")
    
    # Real Project Image 2: Counterfactual Comparison Plot
    cf_w, cf_h = 490, 310
    cf_x, cf_y = lp_x + 515, flow_y + 95
    paste_rounded_image(img, "EV_Charging/results/figures/counterfactual_comparison.png", (cf_x, cf_y, cf_w, cf_h), radius=8, outline="#334155", outline_width=2)
    draw.text((cf_x + 10, cf_y - 20), "Actionable Counterfactual Levers (Feasible Delta)", font=font_block_bold, fill="#0f766e")

    # =========================================================================
    # 7. RIGHT COLUMN: EVALUATION & PLANNING OUTPUTS
    # (x=1410, y=700, w=355, h=1055)
    # =========================================================================
    rc_x, rc_y, rc_w, rc_h = 1410, 700, 355, 1055
    draw.rounded_rectangle([rc_x, rc_y, rc_x + rc_w, rc_y + rc_h], radius=14, fill="#ffffff", outline="#334155", width=2)
    
    rc_h_text = "EVALUATION & PLANNING OUTPUTS"
    rc_bbox = draw.textbbox((0, 0), rc_h_text, font=font_block_h1)
    draw.text((rc_x + (rc_w - (rc_bbox[2] - rc_bbox[0])) / 2, rc_y + 16), rc_h_text, font=font_block_h1, fill="#0f172a")
    draw.line([(rc_x + 16, rc_y + 46), (rc_x + rc_w - 16, rc_y + 46)], fill="#cbd5e1", width=1)
    
    outputs_data = [
        ("EV_Charging/results/figures/confusion_matrix.png", "Outer Test Confusion Matrix", "57.1% Acc | 60.3% Balanced Acc"),
        ("EV_Charging/results/figures/pr_curves.png", "Precision-Recall Curves", "Champion PR-AUC: 0.8211"),
        ("EV_Charging/results/figures/feature_importance.png", "SHAP Feature Importance", "City Center & Highway Proximity Top"),
        ("EV_Charging/results/figures/eips_district_rankings.png", "EIPS District Priority (33 Dists)", "AHP-TOPSIS (Medchal #1: 57.57)")
    ]
    
    out_y = rc_y + 58
    out_img_w, out_img_h = 320, 185
    for img_p, tit, sub in outputs_data:
        ox = rc_x + int((rc_w - out_img_w) / 2)
        paste_rounded_image(img, img_p, (ox, out_y, out_img_w, out_img_h), radius=8, outline="#334155", outline_width=2)
        
        t_b = draw.textbbox((0, 0), tit, font=font_item_title)
        draw.text((rc_x + (rc_w - (t_b[2] - t_b[0])) / 2, out_y + out_img_h + 6), tit, font=font_item_title, fill="#0f172a")
        s_b = draw.textbbox((0, 0), sub, font=font_item_sub)
        draw.text((rc_x + (rc_w - (s_b[2] - s_b[0])) / 2, out_y + out_img_h + 24), sub, font=font_item_sub, fill="#475569")
        
        out_y += 245

    # 8. Save High-Resolution Diagram
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "PNG", dpi=(300, 300))
    print(f"Successfully generated publication-grade system architecture diagram at: {output_path}")

if __name__ == "__main__":
    out_file = "EV_Charging/reports/figures/system_architecture_paper.png"
    generate_system_architecture_diagram(out_file)
