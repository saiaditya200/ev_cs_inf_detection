import os
import base64
import io
import subprocess
from PIL import Image

def get_base64_img(path, max_dim=(700, 700)):
    if not os.path.exists(path):
        print(f"Warning: path {path} not found")
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAAElFTkSuQmCC"
    im = Image.open(path)
    im.thumbnail(max_dim, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_svg():
    print("Loading and encoding images to Base64...")
    
    # Left column images
    b64_in_stations = get_base64_img("EV_Charging/reports/figures/phase10_spatial_split_map.png", (450, 450))
    b64_in_osm = get_base64_img("EV_Charging/results/figures/gis_spatial_distribution.png", (450, 450))
    b64_in_vahan = get_base64_img("EV_Charging/reports/figures/phase10_district_station_counts.png", (450, 450))
    b64_in_grid = get_base64_img("EV_Charging/results/figures/infrastructure_gap_vs_adoption.png", (450, 450))
    
    # Right column images
    b64_out_cm = get_base64_img("EV_Charging/results/figures/confusion_matrix.png", (450, 450))
    b64_out_pr = get_base64_img("EV_Charging/results/figures/pr_curves.png", (450, 450))
    b64_out_shap = get_base64_img("EV_Charging/results/figures/shap_summary_bar.png", (450, 450))
    b64_out_eips = get_base64_img("EV_Charging/results/figures/eips_district_rankings.png", (450, 450))
    
    # Lower panel images
    b64_xai_beeswarm = get_base64_img("EV_Charging/results/figures/shap_beeswarm.png", (800, 600))
    b64_xai_cf = get_base64_img("EV_Charging/results/figures/counterfactual_comparison.png", (800, 600))
    
    print("Assembling perfected SVG XML...")
    
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1600 1600" width="1600" height="1600">
  <defs>
    <!-- Background Gradient -->
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#dbeaf8"/>
      <stop offset="50%" stop-color="#edf4fb"/>
      <stop offset="100%" stop-color="#f8fafc"/>
    </linearGradient>

    <!-- 3D Slab Gradients: Upper Panel -->
    <!-- Blue Slab -->
    <linearGradient id="slabBlueFront" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="100%" stop-color="#0284c7"/>
    </linearGradient>
    <linearGradient id="slabBlueTop" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#bae6fd"/>
      <stop offset="100%" stop-color="#7dd3fc"/>
    </linearGradient>
    <linearGradient id="slabBlueSide" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#0369a1"/>
      <stop offset="100%" stop-color="#075985"/>
    </linearGradient>

    <!-- Green Slab -->
    <linearGradient id="slabGreenFront" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#34d399"/>
      <stop offset="100%" stop-color="#059669"/>
    </linearGradient>
    <linearGradient id="slabGreenTop" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#a7f3d0"/>
      <stop offset="100%" stop-color="#6ee7b7"/>
    </linearGradient>
    <linearGradient id="slabGreenSide" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#047857"/>
      <stop offset="100%" stop-color="#065f46"/>
    </linearGradient>

    <!-- Orange Slab -->
    <linearGradient id="slabOrangeFront" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fb923c"/>
      <stop offset="100%" stop-color="#ea580c"/>
    </linearGradient>
    <linearGradient id="slabOrangeTop" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#fed7aa"/>
      <stop offset="100%" stop-color="#fdba74"/>
    </linearGradient>
    <linearGradient id="slabOrangeSide" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#c2410c"/>
      <stop offset="100%" stop-color="#9a3412"/>
    </linearGradient>

    <!-- Teal Slab -->
    <linearGradient id="slabTealFront" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#2dd4bf"/>
      <stop offset="100%" stop-color="#0d9488"/>
    </linearGradient>
    <linearGradient id="slabTealTop" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#99f6e4"/>
      <stop offset="100%" stop-color="#5eead4"/>
    </linearGradient>
    <linearGradient id="slabTealSide" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#0f766e"/>
      <stop offset="100%" stop-color="#115e59"/>
    </linearGradient>

    <!-- Coral / Pink Slab (Residual) -->
    <linearGradient id="slabPinkFront" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#f87171"/>
      <stop offset="100%" stop-color="#dc2626"/>
    </linearGradient>
    <linearGradient id="slabPinkTop" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#fecaca"/>
      <stop offset="100%" stop-color="#fca5a5"/>
    </linearGradient>
    <linearGradient id="slabPinkSide" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#b91c1c"/>
      <stop offset="100%" stop-color="#991b1b"/>
    </linearGradient>

    <!-- Yellow / Gold Slab (Residual) -->
    <linearGradient id="slabYellowFront" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fde047"/>
      <stop offset="100%" stop-color="#ca8a04"/>
    </linearGradient>
    <linearGradient id="slabYellowTop" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#fef08a"/>
      <stop offset="100%" stop-color="#facc15"/>
    </linearGradient>
    <linearGradient id="slabYellowSide" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#a16207"/>
      <stop offset="100%" stop-color="#854d0e"/>
    </linearGradient>

    <!-- Purple Slab (Residual) -->
    <linearGradient id="slabPurpleFront" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#c084fc"/>
      <stop offset="100%" stop-color="#9333ea"/>
    </linearGradient>
    <linearGradient id="slabPurpleTop" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#e9d5ff"/>
      <stop offset="100%" stop-color="#d8b4fe"/>
    </linearGradient>
    <linearGradient id="slabPurpleSide" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#7e22ce"/>
      <stop offset="100%" stop-color="#6b21a8"/>
    </linearGradient>

    <!-- 3D Cube Gradients -->
    <linearGradient id="cubeTop" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#d8b4fe"/>
      <stop offset="100%" stop-color="#a855f7"/>
    </linearGradient>
    <linearGradient id="cubeFront" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#9333ea"/>
      <stop offset="100%" stop-color="#7e22ce"/>
    </linearGradient>
    <linearGradient id="cubeSide" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#6b21a8"/>
      <stop offset="100%" stop-color="#581c87"/>
    </linearGradient>

    <!-- Middle Panel 3D Block Gradients -->
    <!-- Block 1: Teal -->
    <linearGradient id="blkTealFront" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#14b8a6"/>
      <stop offset="100%" stop-color="#0f766e"/>
    </linearGradient>
    <linearGradient id="blkTealTop" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#5eead4"/>
      <stop offset="100%" stop-color="#2dd4bf"/>
    </linearGradient>
    <linearGradient id="blkTealSide" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#115e59"/>
      <stop offset="100%" stop-color="#134e4a"/>
    </linearGradient>

    <!-- Block 2: Purple -->
    <linearGradient id="blkPurpFront" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#a855f7"/>
      <stop offset="100%" stop-color="#7e22ce"/>
    </linearGradient>
    <linearGradient id="blkPurpTop" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#d8b4fe"/>
      <stop offset="100%" stop-color="#c084fc"/>
    </linearGradient>
    <linearGradient id="blkPurpSide" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#6b21a8"/>
      <stop offset="100%" stop-color="#581c87"/>
    </linearGradient>

    <!-- Block 3: Red/Coral -->
    <linearGradient id="blkRedFront" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#ef4444"/>
      <stop offset="100%" stop-color="#b91c1c"/>
    </linearGradient>
    <linearGradient id="blkRedTop" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#fca5a5"/>
      <stop offset="100%" stop-color="#f87171"/>
    </linearGradient>
    <linearGradient id="blkRedSide" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#991b1b"/>
      <stop offset="100%" stop-color="#7f1d1d"/>
    </linearGradient>

    <!-- Arrow Markers -->
    <marker id="arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1e293b"/>
    </marker>
    <marker id="arrowBlue" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563eb"/>
    </marker>

    <!-- Drop Shadows -->
    <filter id="shadowSm" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>
    <filter id="shadowMd" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="2" dy="4" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.12"/>
    </filter>
  </defs>

  <!-- ================= CANVAS BACKGROUND & BORDER ================= -->
  <rect x="8" y="8" width="1584" height="1584" rx="14" fill="url(#bgGrad)" stroke="#b8d3eb" stroke-width="2"/>

  <!-- ================= HEADER SECTION ================= -->
  <text x="800" y="48" text-anchor="middle" font-family="'Times New Roman', Times, Georgia, serif" font-size="34" font-weight="bold" fill="#0f172a" letter-spacing="2">SYSTEM ARCHITECTURE</text>
  <text x="800" y="74" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="14.5" font-weight="600" fill="#475569" letter-spacing="0.5">Explainable AI Framework for Smart EV Charging Infrastructure Planning — Spatial-Temporal Pipeline</text>

  <!-- ================= LEFT COLUMN: DATA INPUTS ================= -->
  <g id="left-column">
    <rect x="24" y="95" width="232" height="1480" rx="10" fill="#ffffff" stroke="#1e293b" stroke-width="1.8" filter="url(#shadowSm)"/>
    <path d="M 24 105 A 10 10 0 0 1 34 95 L 246 95 A 10 10 0 0 1 256 105 L 256 135 L 24 135 Z" fill="#f1f5f9" stroke="#1e293b" stroke-width="1.8"/>
    <text x="140" y="122" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="14" font-weight="bold" fill="#0f172a" letter-spacing="1">DATA INPUTS</text>

    <!-- Card 1: BEE Stations -->
    <g transform="translate(32, 145)">
      <rect x="0" y="0" width="216" height="345" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <clipPath id="clipIn1"><rect x="8" y="8" width="200" height="210" rx="6"/></clipPath>
      <image href="data:image/png;base64,{b64_in_stations}" x="8" y="8" width="200" height="210" preserveAspectRatio="xMidYMid slice" clip-path="url(#clipIn1)"/>
      <rect x="8" y="8" width="200" height="210" rx="6" fill="none" stroke="#94a3b8" stroke-width="1"/>
      <text x="12" y="238" font-family="-apple-system, sans-serif" font-size="13.5" font-weight="bold" fill="#0f172a">Charging Stations</text>
      <text x="12" y="258" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="500" fill="#475569">BEE Ground Truth (N=300)</text>
      <rect x="12" y="270" width="192" height="26" rx="5" fill="#e0f2fe" stroke="#7dd3fc" stroke-width="0.8"/>
      <text x="108" y="287" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0369a1">Observable AC / DC Status</text>
      <text x="12" y="318" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">Geo-tagged Telangana Sites</text>
      <text x="12" y="334" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">Binary Target Variable y_i</text>
    </g>

    <!-- Card 2: OSM Roads & POIs -->
    <g transform="translate(32, 510)">
      <rect x="0" y="0" width="216" height="345" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <clipPath id="clipIn2"><rect x="8" y="8" width="200" height="210" rx="6"/></clipPath>
      <image href="data:image/png;base64,{b64_in_osm}" x="8" y="8" width="200" height="210" preserveAspectRatio="xMidYMid slice" clip-path="url(#clipIn2)"/>
      <rect x="8" y="8" width="200" height="210" rx="6" fill="none" stroke="#94a3b8" stroke-width="1"/>
      <text x="12" y="238" font-family="-apple-system, sans-serif" font-size="13.5" font-weight="bold" fill="#0f172a">Road &amp; POI Density</text>
      <text x="12" y="258" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="500" fill="#475569">OpenStreetMap Highway GIS</text>
      <rect x="12" y="270" width="192" height="26" rx="5" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="0.8"/>
      <text x="108" y="287" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#047857">500m / 1km / 2km Isochrones</text>
      <text x="12" y="318" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">Primary Highway Intersections</text>
      <text x="12" y="334" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">Commercial Hub Proximity</text>
    </g>

    <!-- Card 3: VAHAN EV Registry -->
    <g transform="translate(32, 875)">
      <rect x="0" y="0" width="216" height="345" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <clipPath id="clipIn3"><rect x="8" y="8" width="200" height="210" rx="6"/></clipPath>
      <image href="data:image/png;base64,{b64_in_vahan}" x="8" y="8" width="200" height="210" preserveAspectRatio="xMidYMid slice" clip-path="url(#clipIn3)"/>
      <rect x="8" y="8" width="200" height="210" rx="6" fill="none" stroke="#94a3b8" stroke-width="1"/>
      <text x="12" y="238" font-family="-apple-system, sans-serif" font-size="13.5" font-weight="bold" fill="#0f172a">EV Fleet Registry</text>
      <text x="12" y="258" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="500" fill="#475569">VAHAN Government Portal</text>
      <rect x="12" y="270" width="192" height="26" rx="5" fill="#fef3c7" stroke="#fcd34d" stroke-width="0.8"/>
      <text x="108" y="287" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#b45309">33 Districts EV Adoption</text>
      <text x="12" y="318" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">2W / 3W / 4W Fleet Volumes</text>
      <text x="12" y="334" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">Per-Capita EV Penetration</text>
    </g>

    <!-- Card 4: TGSPDCL Grid Utility -->
    <g transform="translate(32, 1240)">
      <rect x="0" y="0" width="216" height="325" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <clipPath id="clipIn4"><rect x="8" y="8" width="200" height="195" rx="6"/></clipPath>
      <image href="data:image/png;base64,{b64_in_grid}" x="8" y="8" width="200" height="195" preserveAspectRatio="xMidYMid slice" clip-path="url(#clipIn4)"/>
      <rect x="8" y="8" width="200" height="195" rx="6" fill="none" stroke="#94a3b8" stroke-width="1"/>
      <text x="12" y="222" font-family="-apple-system, sans-serif" font-size="13.5" font-weight="bold" fill="#0f172a">Power Grid Utility</text>
      <text x="12" y="240" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="500" fill="#475569">TGSPDCL Feeder Network</text>
      <rect x="12" y="250" width="192" height="26" rx="5" fill="#ccfbf1" stroke="#5eead4" stroke-width="0.8"/>
      <text x="108" y="267" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0f766e">11kV / 33kV Substation Proximity</text>
      <text x="12" y="295" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">Feeder Capacity &amp; Headroom</text>
      <text x="12" y="310" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">Transformer Loading Margin</text>
    </g>
  </g>

  <!-- ================= UPPER PANEL: DATA ENGINEERING & SPATIAL ENRICHMENT ================= -->
  <g id="upper-panel">
    <rect x="272" y="95" width="1056" height="540" rx="10" fill="#ffffff" stroke="#1e293b" stroke-width="1.8" filter="url(#shadowSm)"/>
    <text x="800" y="118" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="15" font-weight="bold" fill="#0f172a">Data Engineering &amp; Spatial Enrichment (Strict Anti-Leakage Gate — Exogenous Predictors)</text>

    <!-- Vector Arrow from Left Column into Upper Panel -->
    <path d="M 256 360 L 302 360" fill="none" stroke="#1e293b" stroke-width="2.2" marker-end="url(#arrow)"/>

    <!-- 4 INGESTION SLABS (3D Tilted Isometric Slabs) -->
    <!-- Slab 1: Blue (BEE Spatial) -->
    <g transform="translate(306, 225)">
      <polygon points="0,0 12,-12 42,-12 30,0" fill="url(#slabBlueTop)" stroke="#0369a1" stroke-width="0.8"/>
      <polygon points="30,0 42,-12 42,248 30,260" fill="url(#slabBlueSide)" stroke="#0369a1" stroke-width="0.8"/>
      <rect x="0" y="0" width="30" height="260" fill="url(#slabBlueFront)" stroke="#0369a1" stroke-width="0.8"/>
      <text x="-130" y="20" transform="rotate(-90)" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#ffffff">BEE Query</text>
    </g>

    <!-- Slab 2: Green (OSM Buffers) -->
    <g transform="translate(352, 225)">
      <polygon points="0,0 12,-12 42,-12 30,0" fill="url(#slabGreenTop)" stroke="#047857" stroke-width="0.8"/>
      <polygon points="30,0 42,-12 42,248 30,260" fill="url(#slabGreenSide)" stroke="#047857" stroke-width="0.8"/>
      <rect x="0" y="0" width="30" height="260" fill="url(#slabGreenFront)" stroke="#047857" stroke-width="0.8"/>
      <text x="-130" y="20" transform="rotate(-90)" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#ffffff">OSM POIs</text>
    </g>

    <!-- Slab 3: Orange (VAHAN Fleet) -->
    <g transform="translate(398, 225)">
      <polygon points="0,0 12,-12 42,-12 30,0" fill="url(#slabOrangeTop)" stroke="#c2410c" stroke-width="0.8"/>
      <polygon points="30,0 42,-12 42,248 30,260" fill="url(#slabOrangeSide)" stroke="#c2410c" stroke-width="0.8"/>
      <rect x="0" y="0" width="30" height="260" fill="url(#slabOrangeFront)" stroke="#c2410c" stroke-width="0.8"/>
      <text x="-130" y="20" transform="rotate(-90)" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#ffffff">VAHAN EV</text>
    </g>

    <!-- Slab 4: Teal (TGSPDCL Grid) -->
    <g transform="translate(444, 225)">
      <polygon points="0,0 12,-12 42,-12 30,0" fill="url(#slabTealTop)" stroke="#0f766e" stroke-width="0.8"/>
      <polygon points="30,0 42,-12 42,248 30,260" fill="url(#slabTealSide)" stroke="#0f766e" stroke-width="0.8"/>
      <rect x="0" y="0" width="30" height="260" fill="url(#slabTealFront)" stroke="#0f766e" stroke-width="0.8"/>
      <text x="-130" y="20" transform="rotate(-90)" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#ffffff">TGSPDCL</text>
    </g>

    <!-- Underneath Label for 4 Ingestion Slabs -->
    <path d="M 306 502 L 306 510 L 390 510 L 390 515 L 390 510 L 474 510 L 474 502" fill="none" stroke="#1e293b" stroke-width="1.2"/>
    <text x="390" y="530" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="bold" fill="#0f172a">Spatial Ingestion Slabs (N=300)</text>

    <!-- Connecting Arrow to Feature Assembly -->
    <path d="M 496 360 L 536 360" fill="none" stroke="#1e293b" stroke-width="2.2" marker-end="url(#arrow)"/>

    <!-- INVERTED RESIDUAL FEATURE ASSEMBLY -->
    <g id="residual-assembly">
      <!-- Title text above bracket -->
      <text x="705" y="142" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12.5" font-weight="bold" fill="#0f172a">Inverted Residual Feature Assembly (×21)</text>
      <!-- Overhead Bracket -->
      <path d="M 548 156 L 548 150 L 705 150 L 705 146 L 705 150 L 853 150 L 853 156" fill="none" stroke="#1e293b" stroke-width="1.2"/>

      <!-- Curved Skip Connection Vector Arrow -->
      <path d="M 545 210 C 545 162, 856 162, 856 210" fill="none" stroke="#2563eb" stroke-width="2.2" stroke-dasharray="5 3" marker-end="url(#arrowBlue)"/>
      <rect x="630" y="156" width="150" height="20" rx="4" fill="#ffffff" stroke="#2563eb" stroke-width="1"/>
      <text x="705" y="170" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#2563eb">+ Skip Identity (21 Feats)</text>

      <!-- Slab 1: Pink (1x1 Expand / Spatial Lag) -->
      <g transform="translate(548, 225)">
        <polygon points="0,0 12,-12 77,-12 65,0" fill="url(#slabPinkTop)" stroke="#b91c1c" stroke-width="0.8"/>
        <polygon points="65,0 77,-12 77,248 65,260" fill="url(#slabPinkSide)" stroke="#b91c1c" stroke-width="0.8"/>
        <rect x="0" y="0" width="65" height="260" fill="url(#slabPinkFront)" stroke="#b91c1c" stroke-width="0.8"/>
        <text x="-130" y="38" transform="rotate(-90)" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="13" font-weight="bold" fill="#ffffff">Spatial Lag 1×1</text>
      </g>

      <!-- Intermediate Arrow -->
      <path d="M 632 360 L 655 360" fill="none" stroke="#1e293b" stroke-width="2" marker-end="url(#arrow)"/>

      <!-- Slab 2: Yellow (3x3 DWConv / Density) -->
      <g transform="translate(662, 225)">
        <polygon points="0,0 12,-12 77,-12 65,0" fill="url(#slabYellowTop)" stroke="#a16207" stroke-width="0.8"/>
        <polygon points="65,0 77,-12 77,248 65,260" fill="url(#slabYellowSide)" stroke="#a16207" stroke-width="0.8"/>
        <rect x="0" y="0" width="65" height="260" fill="url(#slabYellowFront)" stroke="#a16207" stroke-width="0.8"/>
        <text x="-130" y="38" transform="rotate(-90)" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="13" font-weight="bold" fill="#713f12">Density Norm 3×3</text>
      </g>

      <!-- Intermediate Arrow -->
      <path d="M 746 360 L 769 360" fill="none" stroke="#1e293b" stroke-width="2" marker-end="url(#arrow)"/>

      <!-- Slab 3: Purple (1x1 Project / StandardScaler) -->
      <g transform="translate(776, 225)">
        <polygon points="0,0 12,-12 77,-12 65,0" fill="url(#slabPurpleTop)" stroke="#7e22ce" stroke-width="0.8"/>
        <polygon points="65,0 77,-12 77,248 65,260" fill="url(#slabPurpleSide)" stroke="#7e22ce" stroke-width="0.8"/>
        <rect x="0" y="0" width="65" height="260" fill="url(#slabPurpleFront)" stroke="#7e22ce" stroke-width="0.8"/>
        <text x="-130" y="38" transform="rotate(-90)" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="13" font-weight="bold" fill="#ffffff">Robust Scale 1×1</text>
      </g>

      <!-- Underneath Bracket & Label -->
      <path d="M 548 502 L 548 510 L 705 510 L 705 515 L 705 510 L 853 510 L 853 502" fill="none" stroke="#1e293b" stroke-width="1.3"/>
      <text x="705" y="530" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#0f172a">×21 Exogenous Feature Vector</text>
    </g>

    <!-- Connecting Arrow to Anti-Leakage Gate -->
    <path d="M 870 360 L 908 360" fill="none" stroke="#1e293b" stroke-width="2.2" marker-end="url(#arrow)"/>

    <!-- STRICT ANTI-LEAKAGE ISOLATION GATE -->
    <g id="leakage-gate">
      <!-- Slab 1: Green Guard -->
      <g transform="translate(916, 225)">
        <polygon points="0,0 12,-12 52,-12 40,0" fill="url(#slabGreenTop)" stroke="#047857" stroke-width="0.8"/>
        <polygon points="40,0 52,-12 52,248 40,260" fill="url(#slabGreenSide)" stroke="#047857" stroke-width="0.8"/>
        <rect x="0" y="0" width="40" height="260" fill="url(#slabGreenFront)" stroke="#047857" stroke-width="0.8"/>
        <text x="-130" y="25" transform="rotate(-90)" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#ffffff">Temporal Guard</text>
      </g>

      <!-- Slab 2: Orange Gate -->
      <g transform="translate(976, 225)">
        <polygon points="0,0 12,-12 52,-12 40,0" fill="url(#slabOrangeTop)" stroke="#c2410c" stroke-width="0.8"/>
        <polygon points="40,0 52,-12 52,248 40,260" fill="url(#slabOrangeSide)" stroke="#c2410c" stroke-width="0.8"/>
        <rect x="0" y="0" width="40" height="260" fill="url(#slabOrangeFront)" stroke="#c2410c" stroke-width="0.8"/>
        <text x="-130" y="25" transform="rotate(-90)" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#ffffff">Spatial Gate</text>
      </g>

      <!-- Underneath Label -->
      <path d="M 916 502 L 916 510 L 966 510 L 966 515 L 966 510 L 1016 510 L 1016 502" fill="none" stroke="#1e293b" stroke-width="1.2"/>
      <text x="966" y="530" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="bold" fill="#0f172a">Zero Leakage Certified</text>
    </g>

    <!-- Connecting Arrow to 3D Cube -->
    <path d="M 1038 360 L 1092 360" fill="none" stroke="#1e293b" stroke-width="2.2" marker-end="url(#arrow)"/>

    <!-- 3D ISOMETRIC PURPLE CUBE (Feature Matrix Assembly) -->
    <g transform="translate(1105, 305)" id="feature-cube">
      <!-- Top Face -->
      <polygon points="0,0 35,-35 125,-35 90,0" fill="url(#cubeTop)" stroke="#581c87" stroke-width="1.2"/>
      <!-- Right Side Face -->
      <polygon points="90,0 125,-35 125,55 90,90" fill="url(#cubeSide)" stroke="#581c87" stroke-width="1.2"/>
      <!-- Front Face -->
      <rect x="0" y="0" width="90" height="90" fill="url(#cubeFront)" stroke="#581c87" stroke-width="1.2"/>
      <!-- Center Text -->
      <text x="45" y="54" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="24" font-weight="bold" fill="#ffffff">21-D</text>
      
      <!-- Labels Above & Below -->
      <text x="62" y="-45" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="13" font-weight="bold" fill="#0f172a">Feature Matrix Assembly</text>
      <text x="50" y="122" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="13" font-weight="bold" fill="#7e22ce">21-D Feature Vector</text>
    </g>
  </g>

  <!-- ================= CHANNEL ARROW: UPPER TO MIDDLE ================= -->
  <!-- Routes strictly in the channel between panels (y=650) without crossing inside panels! -->
  <path d="M 1165 435 L 1165 650 L 305 650 L 305 855 L 325 855" fill="none" stroke="#1e293b" stroke-width="2.2" marker-end="url(#arrow)"/>
  <rect x="670" y="638" width="160" height="24" rx="5" fill="#ffffff" stroke="#1e293b" stroke-width="1.2"/>
  <text x="750" y="654" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="bold" fill="#0f172a">21-D Exogenous Tensor</text>

  <!-- ================= MIDDLE PANEL: SPATIAL-AWARE ML CLASSIFICATION PIPELINE ================= -->
  <g id="middle-panel">
    <rect x="272" y="665" width="1056" height="400" rx="10" fill="#ffffff" stroke="#1e293b" stroke-width="1.8" filter="url(#shadowSm)"/>
    <text x="800" y="696" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="16" font-weight="bold" fill="#0f172a">Spatial-Aware ML Classification Pipeline</text>

    <!-- BLOCK 1: Spatial Partitioning (Teal 3D Block) -->
    <g transform="translate(332, 740)" filter="url(#shadowMd)">
      <!-- Top Face -->
      <polygon points="0,0 14,-14 274,-14 260,0" fill="url(#blkTealTop)" stroke="#115e59" stroke-width="1"/>
      <!-- Right Side Face -->
      <polygon points="260,0 274,-14 274,236 260,250" fill="url(#blkTealSide)" stroke="#115e59" stroke-width="1"/>
      <!-- Front Face -->
      <rect x="0" y="0" width="260" height="250" rx="4" fill="url(#blkTealFront)" stroke="#115e59" stroke-width="1.2"/>
      <!-- Content -->
      <text x="130" y="34" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="17" font-weight="bold" fill="#ffffff">Spatial Partitioning</text>
      <line x1="20" y1="46" x2="240" y2="46" stroke="#5eead4" stroke-width="1.5"/>
      <rect x="15" y="58" width="230" height="175" rx="5" fill="#ffffff" fill-opacity="0.95"/>
      <text x="28" y="85" font-family="-apple-system, sans-serif" font-size="13" font-weight="bold" fill="#0f172a">Spatial Block CV (5-Fold)</text>
      <text x="28" y="112" font-family="-apple-system, sans-serif" font-size="12" font-weight="500" fill="#334155">• GroupKFold by District</text>
      <text x="28" y="136" font-family="-apple-system, sans-serif" font-size="12" font-weight="500" fill="#334155">• Moran's I Autocorr. Guard</text>
      <text x="28" y="160" font-family="-apple-system, sans-serif" font-size="12" font-weight="500" fill="#334155">• Prevents Spatial Cheating</text>
      <rect x="25" y="178" width="210" height="42" rx="4" fill="#ccfbf1" stroke="#2dd4bf" stroke-width="0.8"/>
      <text x="130" y="196" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0f766e">Zero Geographic Leakage</text>
      <text x="130" y="212" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10.5" fill="#115e59">Strict Training Isolation</text>
    </g>

    <!-- Connecting Arrow to Optuna -->
    <path d="M 606 855 L 662 855" fill="none" stroke="#1e293b" stroke-width="2.2" marker-end="url(#arrow)"/>
    <rect x="608" y="830" width="50" height="18" rx="3" fill="#ffffff" stroke="#94a3b8" stroke-width="0.8"/>
    <text x="633" y="843" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#475569">Folds</text>

    <!-- BLOCK 2: Optuna Tuning (Purple 3D Block) -->
    <g transform="translate(670, 740)" filter="url(#shadowMd)">
      <polygon points="0,0 14,-14 274,-14 260,0" fill="url(#blkPurpTop)" stroke="#6b21a8" stroke-width="1"/>
      <polygon points="260,0 274,-14 274,236 260,250" fill="url(#blkPurpSide)" stroke="#6b21a8" stroke-width="1"/>
      <rect x="0" y="0" width="260" height="250" rx="4" fill="url(#blkPurpFront)" stroke="#6b21a8" stroke-width="1.2"/>
      <text x="130" y="34" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="17" font-weight="bold" fill="#ffffff">Optuna Tuning</text>
      <line x1="20" y1="46" x2="240" y2="46" stroke="#d8b4fe" stroke-width="1.5"/>
      <rect x="15" y="58" width="230" height="175" rx="5" fill="#ffffff" fill-opacity="0.95"/>
      <text x="28" y="85" font-family="-apple-system, sans-serif" font-size="13" font-weight="bold" fill="#0f172a">Bayesian TPE Search</text>
      <text x="28" y="112" font-family="-apple-system, sans-serif" font-size="12" font-weight="500" fill="#334155">• 50 Iterations Optimization</text>
      <text x="28" y="136" font-family="-apple-system, sans-serif" font-size="12" font-weight="500" fill="#334155">• Balanced Log-Loss Obj.</text>
      <text x="28" y="160" font-family="-apple-system, sans-serif" font-size="12" font-weight="500" fill="#334155">• Dynamic Threshold Tuning</text>
      <rect x="25" y="178" width="210" height="42" rx="4" fill="#f3e8ff" stroke="#c084fc" stroke-width="0.8"/>
      <text x="130" y="196" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#7e22ce">Calibrated Threshold τ* = 0.76</text>
      <text x="130" y="212" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10.5" fill="#6b21a8">Prevents Class Imbalance Collapse</text>
    </g>

    <!-- Connecting Arrow to Champion RF -->
    <path d="M 944 855 L 1000 855" fill="none" stroke="#1e293b" stroke-width="2.2" marker-end="url(#arrow)"/>
    <rect x="946" y="830" width="50" height="18" rx="3" fill="#ffffff" stroke="#94a3b8" stroke-width="0.8"/>
    <text x="971" y="843" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#475569">Params</text>

    <!-- BLOCK 3: Champion Random Forest (Coral/Red 3D Block) -->
    <g transform="translate(1008, 740)" filter="url(#shadowMd)">
      <polygon points="0,0 14,-14 274,-14 260,0" fill="url(#blkRedTop)" stroke="#991b1b" stroke-width="1"/>
      <polygon points="260,0 274,-14 274,236 260,250" fill="url(#blkRedSide)" stroke="#991b1b" stroke-width="1"/>
      <rect x="0" y="0" width="260" height="250" rx="4" fill="url(#blkRedFront)" stroke="#991b1b" stroke-width="1.2"/>
      <text x="130" y="34" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="17" font-weight="bold" fill="#ffffff">Champion Random Forest</text>
      <line x1="20" y1="46" x2="240" y2="46" stroke="#fca5a5" stroke-width="1.5"/>
      <rect x="15" y="58" width="230" height="175" rx="5" fill="#ffffff" fill-opacity="0.95"/>
      <text x="28" y="83" font-family="-apple-system, sans-serif" font-size="12.5" font-weight="bold" fill="#0f172a">n_estimators=200, depth=3</text>
      <text x="28" y="106" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="bold" fill="#b91c1c">Outer Test Acc: 57.14%</text>
      <text x="28" y="126" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="bold" fill="#b91c1c">Balanced Acc: 60.26%</text>
      <text x="28" y="148" font-family="-apple-system, sans-serif" font-size="11" font-weight="500" fill="#334155">AC Recall: 46.15% (Target: &gt;45%)</text>
      <text x="28" y="166" font-family="-apple-system, sans-serif" font-size="11" font-weight="500" fill="#334155">DC Recall: 74.36% (High-Speed)</text>
      <rect x="25" y="180" width="210" height="42" rx="4" fill="#fee2e2" stroke="#f87171" stroke-width="0.8"/>
      <text x="130" y="198" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#b91c1c">Rigorous Spatial-Block Validation</text>
      <text x="130" y="214" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10.5" fill="#991b1b">Zero Overfitting Confirmed</text>
    </g>

    <!-- Arrow Exiting to Right Column -->
    <path d="M 1282 855 L 1340 855" fill="none" stroke="#1e293b" stroke-width="2.2" marker-end="url(#arrow)"/>
    <rect x="1280" y="832" width="58" height="18" rx="3" fill="#ffffff" stroke="#94a3b8" stroke-width="0.8"/>
    <text x="1309" y="845" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#0f172a">Outputs</text>
  </g>

  <!-- ================= CHANNEL ARROW: MIDDLE TO LOWER ================= -->
  <!-- Routes strictly in the channel between panels (y=1078) without crossing inside panels! -->
  <path d="M 1138 995 L 1138 1078 L 305 1078 L 305 1162 L 325 1162" fill="none" stroke="#1e293b" stroke-width="2.2" marker-end="url(#arrow)"/>
  <rect x="670" y="1066" width="160" height="24" rx="5" fill="#ffffff" stroke="#1e293b" stroke-width="1.2"/>
  <text x="750" y="1082" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="bold" fill="#0f172a">Fitted Random Forest Model</text>

  <!-- ================= LOWER PANEL: EXPLAINABLE AI & ACTIONABLE COUNTERFACTUALS ================= -->
  <g id="lower-panel">
    <rect x="272" y="1090" width="1056" height="485" rx="10" fill="#ffffff" stroke="#1e293b" stroke-width="1.8" filter="url(#shadowSm)"/>
    <text x="800" y="1120" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="16" font-weight="bold" fill="#0f172a">Explainable AI (TreeSHAP) &amp; Actionable Counterfactual Optimization</text>

    <!-- Workflow Box 1 -->
    <g transform="translate(332, 1140)">
      <rect x="0" y="0" width="220" height="54" rx="6" fill="#f0f9ff" stroke="#38bdf8" stroke-width="1.2"/>
      <text x="110" y="22" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12.5" font-weight="bold" fill="#0369a1">TreeSHAP Explainer</text>
      <text x="110" y="42" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="500" fill="#475569">Log-odds marginalization</text>
    </g>

    <!-- Arrow 1 -> 2 -->
    <path d="M 552 1167 L 588 1167" fill="none" stroke="#1e293b" stroke-width="2" marker-end="url(#arrow)"/>

    <!-- Workflow Box 2 -->
    <g transform="translate(594, 1140)">
      <rect x="0" y="0" width="220" height="54" rx="6" fill="#f0fdf4" stroke="#4ade80" stroke-width="1.2"/>
      <text x="110" y="22" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12.5" font-weight="bold" fill="#15803d">Additive Attributions</text>
      <text x="110" y="42" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="500" fill="#475569">Exact ϕ_i feature impacts</text>
    </g>

    <!-- Arrow 2 -> 3 -->
    <path d="M 814 1167 L 850 1167" fill="none" stroke="#1e293b" stroke-width="2" marker-end="url(#arrow)"/>

    <!-- Workflow Box 3 -->
    <g transform="translate(856, 1140)">
      <rect x="0" y="0" width="220" height="54" rx="6" fill="#fffbeb" stroke="#fcd34d" stroke-width="1.2"/>
      <text x="110" y="22" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12.5" font-weight="bold" fill="#b45309">Nelder-Mead Solver</text>
      <text x="110" y="42" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="500" fill="#475569">Minimizing L1 actionable cost</text>
    </g>

    <!-- Arrow 3 -> 4 -->
    <path d="M 1076 1167 L 1112 1167" fill="none" stroke="#1e293b" stroke-width="2" marker-end="url(#arrow)"/>

    <!-- Workflow Box 4 -->
    <g transform="translate(1118, 1140)">
      <rect x="0" y="0" width="180" height="54" rx="6" fill="#faf5ff" stroke="#c084fc" stroke-width="1.2"/>
      <text x="90" y="22" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12.5" font-weight="bold" fill="#7e22ce">Prescriptive Deltas</text>
      <text x="90" y="42" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="500" fill="#475569">Feasible site upgrades</text>
    </g>

    <!-- TWO REAL PROJECT EMBEDDED VISUALIZATIONS -->
    <!-- Visual 1: SHAP Beeswarm -->
    <g transform="translate(300, 1215)">
      <rect x="0" y="0" width="485" height="340" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="485" height="32" rx="8" fill="#f1f5f9"/>
      <text x="242" y="22" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="13" font-weight="bold" fill="#0f172a">Global Feature Attribution (TreeSHAP Beeswarm)</text>
      <clipPath id="clipXai1"><rect x="10" y="40" width="465" height="260" rx="6"/></clipPath>
      <image href="data:image/png;base64,{b64_xai_beeswarm}" x="10" y="40" width="465" height="260" preserveAspectRatio="xMidYMid meet" clip-path="url(#clipXai1)"/>
      <text x="242" y="324" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="500" fill="#475569">Distribution of SHAP impacts across test sites (Top: 11kV Feeder Dist, POI Density)</text>
    </g>

    <!-- Visual 2: Counterfactual What-If -->
    <g transform="translate(815, 1215)">
      <rect x="0" y="0" width="485" height="340" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="485" height="32" rx="8" fill="#f1f5f9"/>
      <text x="242" y="22" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="13" font-weight="bold" fill="#0f172a">Actionable What-If Counterfactual Roadmap</text>
      <clipPath id="clipXai2"><rect x="10" y="40" width="465" height="260" rx="6"/></clipPath>
      <image href="data:image/png;base64,{b64_xai_cf}" x="10" y="40" width="465" height="260" preserveAspectRatio="xMidYMid meet" clip-path="url(#clipXai2)"/>
      <text x="242" y="324" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="500" fill="#475569">Minimum-effort intervention deltas required to transition low-speed AC sites to DC Fast hubs</text>
    </g>
  </g>

  <!-- ================= RIGHT COLUMN: EVALUATION & OUTPUTS ================= -->
  <g id="right-column">
    <rect x="1344" y="95" width="232" height="1480" rx="10" fill="#ffffff" stroke="#1e293b" stroke-width="1.8" filter="url(#shadowSm)"/>
    <path d="M 1344 105 A 10 10 0 0 1 1354 95 L 1566 95 A 10 10 0 0 1 1576 105 L 1576 135 L 1344 135 Z" fill="#f1f5f9" stroke="#1e293b" stroke-width="1.8"/>
    <text x="1460" y="122" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="13.5" font-weight="bold" fill="#0f172a" letter-spacing="1">EVALUATION &amp; OUTPUTS</text>

    <!-- Card 1: Confusion Matrix -->
    <g transform="translate(1352, 145)">
      <rect x="0" y="0" width="216" height="345" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <clipPath id="clipOut1"><rect x="8" y="8" width="200" height="210" rx="6"/></clipPath>
      <image href="data:image/png;base64,{b64_out_cm}" x="8" y="8" width="200" height="210" preserveAspectRatio="xMidYMid slice" clip-path="url(#clipOut1)"/>
      <rect x="8" y="8" width="200" height="210" rx="6" fill="none" stroke="#94a3b8" stroke-width="1"/>
      <text x="12" y="238" font-family="-apple-system, sans-serif" font-size="13.5" font-weight="bold" fill="#0f172a">Confusion Matrix</text>
      <text x="12" y="258" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="500" fill="#475569">Spatial Block Test Split</text>
      <rect x="12" y="270" width="192" height="26" rx="5" fill="#fee2e2" stroke="#fca5a5" stroke-width="0.8"/>
      <text x="108" y="287" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#b91c1c">Balanced Acc: 60.26%</text>
      <text x="12" y="318" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">AC Recall: 46.15%</text>
      <text x="12" y="334" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">DC Fast Recall: 74.36%</text>
    </g>

    <!-- Card 2: PR Curves -->
    <g transform="translate(1352, 510)">
      <rect x="0" y="0" width="216" height="345" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <clipPath id="clipOut2"><rect x="8" y="8" width="200" height="210" rx="6"/></clipPath>
      <image href="data:image/png;base64,{b64_out_pr}" x="8" y="8" width="200" height="210" preserveAspectRatio="xMidYMid slice" clip-path="url(#clipOut2)"/>
      <rect x="8" y="8" width="200" height="210" rx="6" fill="none" stroke="#94a3b8" stroke-width="1"/>
      <text x="12" y="238" font-family="-apple-system, sans-serif" font-size="13.5" font-weight="bold" fill="#0f172a">PR &amp; ROC Analysis</text>
      <text x="12" y="258" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="500" fill="#475569">Threshold Calibration</text>
      <rect x="12" y="270" width="192" height="26" rx="5" fill="#fef3c7" stroke="#fcd34d" stroke-width="0.8"/>
      <text x="108" y="287" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#b45309">Macro PR-AUC: 0.58</text>
      <text x="12" y="318" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">Calibrated Cutoff τ* = 0.76</text>
      <text x="12" y="334" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">ROC-AUC Score: 0.65</text>
    </g>

    <!-- Card 3: SHAP Summary Bar -->
    <g transform="translate(1352, 875)">
      <rect x="0" y="0" width="216" height="345" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <clipPath id="clipOut3"><rect x="8" y="8" width="200" height="210" rx="6"/></clipPath>
      <image href="data:image/png;base64,{b64_out_shap}" x="8" y="8" width="200" height="210" preserveAspectRatio="xMidYMid slice" clip-path="url(#clipOut3)"/>
      <rect x="8" y="8" width="200" height="210" rx="6" fill="none" stroke="#94a3b8" stroke-width="1"/>
      <text x="12" y="238" font-family="-apple-system, sans-serif" font-size="13.5" font-weight="bold" fill="#0f172a">SHAP Global Drivers</text>
      <text x="12" y="258" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="500" fill="#475569">Feature Importance Ranking</text>
      <rect x="12" y="270" width="192" height="26" rx="5" fill="#f3e8ff" stroke="#c084fc" stroke-width="0.8"/>
      <text x="108" y="287" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#7e22ce">Mean |SHAP Value| (ϕ_i)</text>
      <text x="12" y="318" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">Top: 11kV Substation Dist</text>
      <text x="12" y="334" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">Rank 2: Commercial POI Dens</text>
    </g>

    <!-- Card 4: EIPS District Rankings -->
    <g transform="translate(1352, 1240)">
      <rect x="0" y="0" width="216" height="325" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <clipPath id="clipOut4"><rect x="8" y="8" width="200" height="195" rx="6"/></clipPath>
      <image href="data:image/png;base64,{b64_out_eips}" x="8" y="8" width="200" height="195" preserveAspectRatio="xMidYMid slice" clip-path="url(#clipOut4)"/>
      <rect x="8" y="8" width="200" height="195" rx="6" fill="none" stroke="#94a3b8" stroke-width="1"/>
      <text x="12" y="222" font-family="-apple-system, sans-serif" font-size="13.5" font-weight="bold" fill="#0f172a">District Allocation</text>
      <text x="12" y="240" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="500" fill="#475569">AHP-TOPSIS Prioritization</text>
      <rect x="12" y="250" width="192" height="26" rx="5" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="0.8"/>
      <text x="108" y="267" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#047857">Tier 1: High Priority (0.65-0.78)</text>
      <text x="12" y="295" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">Hyderabad (0.78), Rangareddy</text>
      <text x="12" y="310" font-family="-apple-system, sans-serif" font-size="10.5" fill="#64748b">33 Districts Prioritized</text>
    </g>
  </g>
</svg>
'''
    
    # Save SVG to multiple project locations
    svg_destinations = [
        "EV_Charging/reports/figures/system_architecture.svg",
        "EV_Charging/results/figures/system_architecture.svg",
        "reports/figures/system_architecture.svg",
        "system_architecture.svg"
    ]
    
    for dest in svg_destinations:
        os.makedirs(os.path.dirname(dest) if os.path.dirname(dest) else ".", exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"Saved SVG to: {dest}")

if __name__ == "__main__":
    generate_svg()
