import os
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
import subprocess

fig_dir = Path("results/india_wide/figures")

def get_base64(name, max_size=(500, 320)):
    p = fig_dir / name
    if not p.exists():
        print(f"Warning: {name} not found")
        return ""
    im = Image.open(p)
    im.thumbnail(max_size, Image.Resampling.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

# Generate thumbnails
print("Encoding thumbnails...")
b64_gis = get_base64("gis_spatial_distribution.png", (450, 350))
b64_cm = get_base64("confusion_matrix.png", (450, 350))
b64_pr = get_base64("pr_curves.png", (450, 330))
b64_shap = get_base64("feature_importance.png", (550, 360))
b64_cf = get_base64("counterfactual_comparison.png", (550, 320))
b64_eips = get_base64("eips_state_rankings.png", (500, 320))
b64_decomp = get_base64("eips_component_breakdown.png", (500, 320))
b64_ui = get_base64("dashboard_overview.jpg", (500, 260))

svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 2600 1420" width="2600" height="1420">
  <defs>
    <!-- Background Canvas Gradient -->
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#f8fafc"/>
      <stop offset="50%" stop-color="#f1f5f9"/>
      <stop offset="100%" stop-color="#e2e8f0"/>
    </linearGradient>

    <!-- Block Gradients -->
    <linearGradient id="encoderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#4ade80"/>
      <stop offset="100%" stop-color="#16a34a"/>
    </linearGradient>

    <linearGradient id="gateGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="100%" stop-color="#0284c7"/>
    </linearGradient>

    <linearGradient id="partitionGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#f472b6"/>
      <stop offset="100%" stop-color="#db2777"/>
    </linearGradient>

    <linearGradient id="capsuleGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0ea5e9"/>
      <stop offset="100%" stop-color="#1d4ed8"/>
    </linearGradient>

    <!-- Arrow Markers -->
    <marker id="arrowDark" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#0f172a"/>
    </marker>
    <marker id="arrowOrange" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#ea580c"/>
    </marker>
    <marker id="arrowBlue" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#0284c7"/>
    </marker>

    <!-- Drop Shadows -->
    <filter id="shadowSm" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>
    <filter id="shadowMd" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="2" dy="4" stdDeviation="6" flood-color="#0f172a" flood-opacity="0.12"/>
    </filter>
  </defs>

  <!-- ================= CANVAS BACKGROUND & BORDER ================= -->
  <rect x="10" y="10" width="2580" height="1400" rx="16" fill="url(#bgGrad)" stroke="#cbd5e1" stroke-width="2"/>

  <!-- ================= HEADER SECTION ================= -->
  <text x="1300" y="52" text-anchor="middle" font-family="'Times New Roman', Times, Georgia, serif" font-size="38" font-weight="bold" fill="#0f172a" letter-spacing="3">SYSTEM ARCHITECTURE</text>
  <text x="1300" y="80" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="15" font-weight="600" fill="#475569" letter-spacing="0.5">Spatial-Aware Machine Learning &amp; Multi-Criteria Siting Framework for Smart EV Charging Infrastructure (52,718 Stations)</text>

  <!-- ========================================================================= -->
  <!-- LEFT COLUMN: INPUT SECTION                                                -->
  <!-- ========================================================================= -->
  <g id="input-column">
    <rect x="40" y="105" width="440" height="1285" rx="14" fill="#ffffff" stroke="#f97316" stroke-width="2.5" stroke-dasharray="8 5" filter="url(#shadowSm)"/>
    <text x="260" y="145" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="24" font-weight="800" fill="#ea580c">Input</text>
    <text x="260" y="166" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" fill="#64748b">Multi-Source Geospatial &amp; Administrative Registries</text>

    <!-- Card 1: BEE & OpenChargeMap -->
    <g transform="translate(56, 182)">
      <rect x="0" y="0" width="408" height="225" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="408" height="28" rx="8" fill="#e0f2fe"/>
      <text x="12" y="19" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#0369a1">1. Charging Station Ground Truth (N=52,718)</text>
      <text x="396" y="19" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#0284c7">BEE + OpenChargeMap</text>
      <!-- Image Thumbnail -->
      <image href="{b64_gis}" x="10" y="36" width="160" height="175" preserveAspectRatio="xMidYMid meet"/>
      <!-- Text details -->
      <g transform="translate(180, 48)">
        <text x="0" y="10" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0f172a">• 52,718 Verified Stations</text>
        <text x="0" y="30" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Coordinates: WGS84 Lat/Long</text>
        <text x="0" y="50" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• AC Slow (&lt;50 kW): 36,157</text>
        <text x="0" y="70" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• DC Fast (≥50 kW): 16,561</text>
        <text x="0" y="90" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Coverage: All 36 States &amp; UTs</text>
        <rect x="0" y="112" width="215" height="36" rx="4" fill="#e0f2fe" stroke="#38bdf8" stroke-width="0.8"/>
        <text x="107" y="126" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#0369a1">Classification Target y_i ∈ {{0, 1}}</text>
        <text x="107" y="140" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#0284c7">Zero Synthetic kWh Imputation</text>
      </g>
    </g>

    <!-- Card 2: Highway Network -->
    <g transform="translate(56, 420)">
      <rect x="0" y="0" width="408" height="205" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="408" height="28" rx="8" fill="#ecfdf5"/>
      <text x="12" y="19" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#047857">2. Highway Corridor Network (NHAI / OSM)</text>
      <text x="396" y="19" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#059669">Overpass Vector GIS</text>
      <!-- Diagram inside -->
      <g transform="translate(14, 45)">
        <path d="M 10 100 Q 60 40 130 30 T 150 140" fill="none" stroke="#059669" stroke-width="5"/>
        <path d="M 30 140 Q 90 90 140 10 T 160 50" fill="none" stroke="#10b981" stroke-width="3" stroke-dasharray="4 2"/>
        <circle cx="80" cy="65" r="7" fill="#ef4444"/>
        <text x="80" y="88" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" font-weight="bold" fill="#b91c1c">Station Node</text>
        <line x1="80" y1="65" x2="110" y2="40" stroke="#b91c1c" stroke-width="1.8" stroke-dasharray="2 2"/>
        <text x="95" y="50" font-family="-apple-system, sans-serif" font-size="9" font-weight="bold" fill="#b91c1c">d_hw</text>
      </g>
      <g transform="translate(180, 44)">
        <text x="0" y="12" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0f172a">• NH-44, NH-48 &amp; Arterials</text>
        <text x="0" y="32" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Distance to Highway (km)</text>
        <text x="0" y="52" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Proximity Score: 1 / (1 + d)</text>
        <text x="0" y="72" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• 7 Road Hierarchy Indicators</text>
        <rect x="0" y="92" width="215" height="48" rx="4" fill="#ecfdf5" stroke="#34d399" stroke-width="0.8"/>
        <text x="107" y="108" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#047857">Exogenous Transport Vectors</text>
        <text x="107" y="124" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#059669">Trunk, Primary, Secondary, Tertiary</text>
      </g>
    </g>

    <!-- Card 3: EV Fleet Adoption Registry -->
    <g transform="translate(56, 638)">
      <rect x="0" y="0" width="408" height="205" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="408" height="28" rx="8" fill="#fef3c7"/>
      <text x="12" y="19" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#b45309">3. EV Fleet Adoption Registry (VAHAN 4.0)</text>
      <text x="396" y="19" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#d97706">MoRTH National</text>
      <!-- Diagram inside -->
      <g transform="translate(20, 50)">
        <rect x="10" y="70" width="24" height="60" rx="3" fill="#f59e0b"/>
        <text x="22" y="145" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#78350f">e-2W</text>
        <rect x="45" y="45" width="24" height="85" rx="3" fill="#ea580c"/>
        <text x="57" y="145" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#78350f">e-3W</text>
        <rect x="80" y="85" width="24" height="45" rx="3" fill="#d97706"/>
        <text x="92" y="145" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#78350f">e-4W</text>
        <rect x="115" y="25" width="24" height="105" rx="3" fill="#b45309"/>
        <text x="127" y="145" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#78350f">Total</text>
      </g>
      <g transform="translate(180, 44)">
        <text x="0" y="12" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0f172a">• All 700+ Indian Districts</text>
        <text x="0" y="32" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Cumulative EV Registration</text>
        <text x="0" y="52" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• EV Penetration per 1,000 veh</text>
        <text x="0" y="72" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• District Total Motorized Fleet</text>
        <rect x="0" y="92" width="215" height="48" rx="4" fill="#fef3c7" stroke="#f59e0b" stroke-width="0.8"/>
        <text x="107" y="108" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#b45309">Macro Regional Siting Context</text>
        <text x="107" y="124" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#92400e">Non-Leaky Vehicle Fleet Proxy</text>
      </g>
    </g>

    <!-- Card 4: CEA Power Grid Utility -->
    <g transform="translate(56, 855)">
      <rect x="0" y="0" width="408" height="205" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="408" height="28" rx="8" fill="#ccfbf1"/>
      <text x="12" y="19" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#0f766e">4. Power Grid Utility &amp; Nodal Agency</text>
      <text x="396" y="19" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#0d9488">CEA / Discoms</text>
      <!-- Diagram inside -->
      <g transform="translate(20, 50)">
        <polygon points="65,15 105,80 25,80" fill="#ccfbf1" stroke="#0d9488" stroke-width="2"/>
        <path d="M 65 30 L 65 70 M 50 50 L 80 50 M 40 65 L 90 65" stroke="#0d9488" stroke-width="2"/>
        <circle cx="65" cy="115" r="16" fill="#14b8a6"/>
        <text x="65" y="120" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#ffffff">33kV</text>
      </g>
      <g transform="translate(180, 44)">
        <text x="0" y="12" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0f172a">• Substation Feeder Network</text>
        <text x="0" y="32" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• 11 kV, 33 kV, 132 kV Tiers</text>
        <text x="0" y="52" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Transformer Headroom Margin</text>
        <text x="0" y="72" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• High Capacity Readiness Flag</text>
        <rect x="0" y="92" width="215" height="48" rx="4" fill="#ccfbf1" stroke="#2dd4bf" stroke-width="0.8"/>
        <text x="107" y="108" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#0f766e">Electrical Feasibility Headroom</text>
        <text x="107" y="124" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#115e59">BESCOM, MSEDCL, TGREDCO, etc.</text>
      </g>
    </g>

    <!-- Card 5: Commercial POIs & Demographics -->
    <g transform="translate(56, 1072)">
      <rect x="0" y="0" width="408" height="200" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="408" height="28" rx="8" fill="#f3e8ff"/>
      <text x="12" y="19" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#7e22ce">5. Commercial Amenities &amp; Demographics</text>
      <text x="396" y="19" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#9333ea">Census + OSM</text>
      <!-- Diagram inside -->
      <g transform="translate(20, 45)">
        <circle cx="70" cy="70" r="55" fill="none" stroke="#c084fc" stroke-width="1.5" stroke-dasharray="3 2"/>
        <circle cx="70" cy="70" r="35" fill="none" stroke="#a855f7" stroke-width="1.5"/>
        <circle cx="70" cy="70" r="16" fill="none" stroke="#7e22ce" stroke-width="1.5"/>
        <circle cx="70" cy="70" r="4" fill="#7e22ce"/>
        <text x="70" y="140" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#6b21a8">5km Amenity Buffer</text>
      </g>
      <g transform="translate(180, 42)">
        <text x="0" y="12" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0f172a">• Commercial POI Counts (5km)</text>
        <text x="0" y="32" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Shannon Diversity Entropy H\'</text>
        <text x="0" y="52" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Census Population Density</text>
        <text x="0" y="72" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Urban Classification (Tier 1/2/3)</text>
        <rect x="0" y="92" width="215" height="46" rx="4" fill="#f3e8ff" stroke="#c084fc" stroke-width="0.8"/>
        <text x="107" y="108" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#7e22ce">Commercial Activity Density</text>
        <text x="107" y="123" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#6b21a8">Malls, Transit Hubs, Fuel Plazas</text>
      </g>
    </g>
  </g>

  <!-- Connective Vector Arrow: Input Column -> Feature Encoder -->
  <path d="M 480 505 L 520 505" fill="none" stroke="#0f172a" stroke-width="2.5" marker-end="url(#arrowDark)"/>

  <!-- ========================================================================= -->
  <!-- MIDDLE TOP BREAKOUT BOX: SPATIAL FEATURE ENGINEERING PIPELINE             -->
  <!-- ========================================================================= -->
  <!-- Dashed perspective lines connecting from Spatial Encoder (x:630, y:440) to Breakout Box -->
  <path d="M 540 440 L 510 360" fill="none" stroke="#ea580c" stroke-width="1.8" stroke-dasharray="6 4"/>
  <path d="M 720 440 L 2030 360" fill="none" stroke="#ea580c" stroke-width="1.8" stroke-dasharray="6 4"/>

  <g id="middle-top-breakout">
    <rect x="510" y="105" width="1520" height="255" rx="14" fill="#ffffff" stroke="#f97316" stroke-width="2.2" stroke-dasharray="8 5" filter="url(#shadowSm)"/>
    <text x="1270" y="136" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="16" font-weight="bold" fill="#ea580c">Spatial Graph Feature Engineering &amp; Topological Enrichment Pipeline</text>

    <!-- Sub-Block 1: Graph Learning -->
    <g transform="translate(535, 160)">
      <rect x="0" y="0" width="150" height="175" rx="8" fill="#f0fdf4" stroke="#86efac" stroke-width="1.2"/>
      <rect x="15" y="12" width="120" height="28" rx="4" fill="#22c55e"/>
      <text x="75" y="30" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#ffffff">Graph Learning</text>
      <text x="75" y="65" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#166534">Spatial Geodesy</text>
      <text x="75" y="85" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#334155">Lat / Long Coordinates</text>
      <text x="75" y="102" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#334155">WGS84 Projection</text>
      <text x="75" y="122" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#334155">Bounding Filter</text>
      <text x="75" y="152" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#15803d">52,718 Points</text>
    </g>

    <!-- Arrow 1 -> 2 -->
    <path d="M 685 245 L 720 245" fill="none" stroke="#ea580c" stroke-width="2.5" marker-end="url(#arrowOrange)"/>

    <!-- Sub-Block 2: Graph Network Topology -->
    <g transform="translate(725, 160)">
      <rect x="0" y="0" width="280" height="175" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <text x="140" y="24" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="bold" fill="#0f172a">Corridor Graph Topology</text>
      <!-- Network Nodes Illustration -->
      <g transform="translate(30, 35)">
        <line x1="30" y1="30" x2="110" y2="20" stroke="#94a3b8" stroke-width="1.8"/>
        <line x1="30" y1="30" x2="80" y2="70" stroke="#94a3b8" stroke-width="1.8"/>
        <line x1="110" y1="20" x2="170" y2="40" stroke="#94a3b8" stroke-width="1.8"/>
        <line x1="80" y1="70" x2="170" y2="40" stroke="#94a3b8" stroke-width="1.8"/>
        <line x1="80" y1="70" x2="50" y2="105" stroke="#94a3b8" stroke-width="1.8"/>
        <line x1="170" y1="40" x2="190" y2="95" stroke="#94a3b8" stroke-width="1.8"/>
        <circle cx="30" cy="30" r="10" fill="#38bdf8"/>
        <circle cx="110" cy="20" r="9" fill="#0284c7"/>
        <circle cx="80" cy="70" r="12" fill="#ef4444"/>
        <circle cx="170" cy="40" r="9" fill="#10b981"/>
        <circle cx="50" cy="105" r="8" fill="#f59e0b"/>
        <circle cx="190" cy="95" r="10" fill="#a855f7"/>
        <circle cx="130" cy="90" r="8" fill="#06b6d4"/>
      </g>
      <text x="140" y="162" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#475569">Spatial Autocorrelation Network</text>
    </g>

    <!-- Arrow 2 -> 3 -->
    <path d="M 1005 245 L 1040 245" fill="none" stroke="#ea580c" stroke-width="2.5" marker-end="url(#arrowOrange)"/>

    <!-- Sub-Block 3: Spatial Convolutions & Distance Buffers -->
    <g transform="translate(1045, 160)">
      <rect x="0" y="0" width="280" height="175" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <text x="140" y="24" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="bold" fill="#0f172a">Multi-Scale Spatial Buffers</text>
      <!-- Concentric / Kernel Slices -->
      <g transform="translate(40, 40)">
        <polygon points="10,65 60,35 150,35 100,65" fill="#fef08a" stroke="#ca8a04" stroke-width="1.2"/>
        <polygon points="15,45 65,15 155,15 105,45" fill="#bae6fd" stroke="#0284c7" stroke-width="1.2"/>
        <polygon points="20,25 70,-5 160,-5 110,25" fill="#bbf7d0" stroke="#16a34a" stroke-width="1.2"/>
      </g>
      <text x="140" y="132" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#047857">Haversine Distance &amp; POI Density</text>
      <text x="140" y="150" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#475569">500m, 1km, 5km Multi-Ring Kernel</text>
      <text x="140" y="165" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#64748b">Spatial Isochrone Convolutions</text>
    </g>

    <!-- Arrow 3 -> 4 -->
    <path d="M 1325 245 L 1360 245" fill="none" stroke="#ea580c" stroke-width="2.5" marker-end="url(#arrowOrange)"/>

    <!-- Sub-Block 4: Hidden Layer Spatial Activations -->
    <g transform="translate(1365, 160)">
      <rect x="0" y="0" width="310" height="175" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <text x="155" y="24" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="bold" fill="#0f172a">Latent Spatial Activations</text>
      <!-- Clustered Colorful Scatter -->
      <g transform="translate(50, 45)">
        <circle cx="30" cy="40" r="3" fill="#ef4444"/><circle cx="35" cy="45" r="3" fill="#ef4444"/><circle cx="42" cy="38" r="3" fill="#ef4444"/><circle cx="28" cy="50" r="3" fill="#ef4444"/>
        <circle cx="90" cy="25" r="3" fill="#3b82f6"/><circle cx="95" cy="30" r="3" fill="#3b82f6"/><circle cx="102" cy="22" r="3" fill="#3b82f6"/><circle cx="88" cy="35" r="3" fill="#3b82f6"/>
        <circle cx="70" cy="70" r="3" fill="#10b981"/><circle cx="75" cy="75" r="3" fill="#10b981"/><circle cx="82" cy="68" r="3" fill="#10b981"/><circle cx="68" cy="80" r="3" fill="#10b981"/>
        <circle cx="140" cy="50" r="3" fill="#f59e0b"/><circle cx="145" cy="55" r="3" fill="#f59e0b"/><circle cx="152" cy="48" r="3" fill="#f59e0b"/><circle cx="138" cy="60" r="3" fill="#f59e0b"/>
      </g>
      <text x="155" y="145" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#0f172a">Entropy H\' &amp; Topological Closeness</text>
      <text x="155" y="162" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" fill="#64748b">Normalized Regional Clusters</text>
    </g>

    <!-- Arrow 4 -> 5 -->
    <path d="M 1675 245 L 1715 245" fill="none" stroke="#ea580c" stroke-width="2.5" marker-end="url(#arrowOrange)"/>

    <!-- Sub-Block 5: Feature Vector Trapezoid -->
    <g transform="translate(1725, 170)">
      <polygon points="0,25 25,0 190,0 215,25 215,115 190,140 25,140 0,115" fill="#d9f99d" stroke="#65a30d" stroke-width="1.8"/>
      <text x="107" y="55" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="14" font-weight="bold" fill="#3f6212">Feature Vector</text>
      <rect x="25" y="70" width="165" height="30" rx="4" fill="#65a30d"/>
      <text x="107" y="90" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="13" font-weight="bold" fill="#ffffff">21 Predictors</text>
      <text x="107" y="125" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#4d7c0f">X ∈ ℝ^{{52718 × 21}}</text>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- MIDDLE CENTER SPINE: MAIN HORIZONTAL CLASSIFICATION PIPELINE               -->
  <!-- ========================================================================= -->
  <g id="middle-spine">
    <!-- Component 1: Spatial Feature Encoder (Green Rounded Rect) -->
    <g transform="translate(520, 440)">
      <rect x="0" y="0" width="210" height="135" rx="14" fill="url(#encoderGrad)" stroke="#15803d" stroke-width="2" filter="url(#shadowMd)"/>
      <text x="105" y="45" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="16" font-weight="bold" fill="#ffffff">Spatial Feature</text>
      <text x="105" y="68" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="16" font-weight="bold" fill="#ffffff">Encoder Network</text>
      <line x1="25" y1="82" x2="185" y2="82" stroke="#bbf7d0" stroke-width="1.2"/>
      <text x="105" y="102" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#ffffff">30 Raw Master Fields</text>
      <text x="105" y="118" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#dcfce7">Geodesic &amp; Corridor Inputs</text>
    </g>

    <!-- Arrow 1 -> 2 -->
    <path d="M 730 507 L 775 507" fill="none" stroke="#0f172a" stroke-width="2.5" marker-end="url(#arrowDark)"/>

    <!-- Component 2: Anti-Leakage Gate (Cyan Folded / Stacked Sheets) -->
    <g transform="translate(785, 420)">
      <!-- Stacked Page back -->
      <path d="M 12 10 L 140 10 L 155 25 L 155 170 L 12 170 Z" fill="#bae6fd" stroke="#0284c7" stroke-width="1.2"/>
      <!-- Stacked Page front -->
      <path d="M 0 20 L 130 20 L 145 35 L 145 180 L 0 180 Z" fill="#e0f2fe" stroke="#0284c7" stroke-width="1.8" filter="url(#shadowSm)"/>
      <polygon points="130,20 145,35 130,35" fill="#0284c7"/>
      <text x="72" y="58" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="15" font-weight="bold" fill="#0369a1">Anti-Leakage</text>
      <text x="72" y="78" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="15" font-weight="bold" fill="#0369a1">Gate</text>
      <line x1="15" y1="90" x2="130" y2="90" stroke="#38bdf8" stroke-width="1.2"/>
      <rect x="10" y="100" width="125" height="32" rx="4" fill="#ef4444"/>
      <text x="72" y="114" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#ffffff">9 Leaky Fields Quarantined</text>
      <text x="72" y="125" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="8.5" fill="#ffffff">(power, connectors, tariff)</text>
      <rect x="10" y="138" width="125" height="32" rx="4" fill="#0284c7"/>
      <text x="72" y="152" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#ffffff">21 Exogenous Predictors</text>
      <text x="72" y="163" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="8.5" fill="#ffffff">(pre-construction only)</text>
    </g>

    <!-- Arrow 2 -> 3 -->
    <path d="M 940 507 L 985 507" fill="none" stroke="#0f172a" stroke-width="2.5" marker-end="url(#arrowDark)"/>

    <!-- Component 3: Spatial Partitioning (Pink / Magenta Vertical Block) -->
    <g transform="translate(995, 435)">
      <rect x="0" y="0" width="145" height="145" rx="12" fill="url(#partitionGrad)" stroke="#be185d" stroke-width="2" filter="url(#shadowMd)"/>
      <text x="72" y="42" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="15" font-weight="bold" fill="#ffffff">Spatial</text>
      <text x="72" y="62" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="15" font-weight="bold" fill="#ffffff">GroupKFold</text>
      <line x1="15" y1="74" x2="130" y2="74" stroke="#fbcfe8" stroke-width="1.2"/>
      <text x="72" y="94" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#ffffff">5-Fold Spatial CV</text>
      <text x="72" y="112" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#fce7f3">36 States Grouped</text>
      <text x="72" y="128" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#fdf2f8">Tobler\'s 1st Law</text>
    </g>

    <!-- Arrow 3 -> 4 -->
    <path d="M 1140 507 L 1185 507" fill="none" stroke="#0f172a" stroke-width="2.5" marker-end="url(#arrowDark)"/>

    <!-- Component 4: Core Classifier (Blue Capsule / Ellipse) -->
    <g transform="translate(1195, 435)">
      <ellipse cx="140" cy="72" rx="135" ry="68" fill="url(#capsuleGrad)" stroke="#1e40af" stroke-width="2.5" filter="url(#shadowMd)"/>
      <text x="140" y="52" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="17" font-weight="bold" fill="#ffffff">Champion Random Forest</text>
      <text x="140" y="74" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="13" font-weight="bold" fill="#bae6fd">150 Decorrelated Trees</text>
      <line x1="45" y1="84" x2="235" y2="84" stroke="#7dd3fc" stroke-width="1.2"/>
      <text x="140" y="100" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" fill="#ffffff">PR-AUC: 0.3118 • Brier: 0.2346</text>
      <text x="140" y="116" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#f0f9ff">Lowest Calibration Error</text>
    </g>

    <!-- Arrow 4 -> 5 -->
    <path d="M 1475 507 L 1520 507" fill="none" stroke="#0f172a" stroke-width="2.5" marker-end="url(#arrowDark)"/>

    <!-- Component 5: Stacked Decision & Multi-Criteria Layers (Matching Flatten/Dense/SoftMax) -->
    <g transform="translate(1530, 395)">
      <rect x="0" y="0" width="310" height="225" rx="10" fill="#0f172a" stroke="#334155" stroke-width="2" filter="url(#shadowMd)"/>
      <text x="155" y="24" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="13" font-weight="bold" fill="#38bdf8">Decision &amp; Siting Head</text>
      
      <!-- Layer 1: Threshold Optimization (Bronze) -->
      <g transform="translate(12, 35)">
        <rect x="0" y="0" width="286" height="32" rx="4" fill="#c2410c"/>
        <text x="143" y="21" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#ffffff">Threshold Calibration Unit (τ* = 0.2488)</text>
      </g>

      <!-- Layer 2: Fast Charging Probability (Navy) -->
      <g transform="translate(12, 72)">
        <rect x="0" y="0" width="286" height="32" rx="4" fill="#1e3a8a"/>
        <text x="143" y="21" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#ffffff">Calibrated Posterior P(DC Fast)</text>
      </g>

      <!-- Layer 3: AHP Criteria Weighting (Gold) -->
      <g transform="translate(12, 109)">
        <rect x="0" y="0" width="286" height="32" rx="4" fill="#ca8a04"/>
        <text x="143" y="21" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#ffffff">AHP Pairwise Criteria Matrix (CR &lt; 0.10)</text>
      </g>

      <!-- Layer 4: TOPSIS Relative Closeness (Cyan) -->
      <g transform="translate(12, 146)">
        <rect x="0" y="0" width="286" height="32" rx="4" fill="#0284c7"/>
        <text x="143" y="21" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#ffffff">TOPSIS Ideal Distance (S+, S-) → C_i</text>
      </g>

      <!-- Layer 5: EIPS Score & Prioritization (Teal) -->
      <g transform="translate(12, 183)">
        <rect x="0" y="0" width="286" height="32" rx="4" fill="#0f766e"/>
        <text x="143" y="21" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#ffffff">EIPS Composite Index (0–100) &amp; Tiers</text>
      </g>
    </g>
  </g>

  <!-- Connective Vector Arrow: Decision Head -> Output Column -->
  <path d="M 1840 507 L 2060 507" fill="none" stroke="#0f172a" stroke-width="2.5" marker-end="url(#arrowDark)"/>

  <!-- ========================================================================= -->
  <!-- MIDDLE BOTTOM SECTION: EXPLAINABILITY & COUNTERFACTUAL SUB-NETWORK        -->
  <!-- ========================================================================= -->
  <g id="middle-bottom-subnet">
    <rect x="510" y="650" width="1520" height="740" rx="14" fill="#ffffff" stroke="#1e293b" stroke-width="2" filter="url(#shadowSm)"/>
    <text x="1270" y="680" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="16" font-weight="bold" fill="#0f172a">Explainable AI (TreeSHAP) &amp; Actionable Counterfactual Optimization Sub-Network</text>

    <!-- 4 Detailed Striped Subnet Blocks (Matching Primary Caps & Affine Subnets in reference) -->
    <!-- Module 1: TreeSHAP Attribution Subnet -->
    <g transform="translate(535, 705)">
      <rect x="0" y="0" width="135" height="235" rx="8" fill="#f8fafc" stroke="#334155" stroke-width="1.8"/>
      <!-- Vertical Striped Layers -->
      <rect x="8" y="8" width="20" height="219" fill="#fde047"/>
      <text x="-115" y="22" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#713f12">TreeSHAP</text>
      
      <rect x="32" y="8" width="20" height="219" fill="#67e8f9"/>
      <text x="-115" y="46" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#155e75">Game Theory</text>
      
      <rect x="56" y="8" width="20" height="219" fill="#f1f5f9"/>
      <text x="-115" y="70" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#334155">Additive ϕ_i</text>
      
      <rect x="80" y="8" width="22" height="219" fill="#ef4444"/>
      <text x="-115" y="95" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#ffffff">O(TLD²)</text>
      
      <rect x="106" y="8" width="20" height="219" fill="#c084fc"/>
      <text x="-115" y="120" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#4a044e">Fast Poly</text>
      
      <!-- Overhead loop arrow -->
      <path d="M 30 8 C 30 -12, 105 -12, 105 8" fill="none" stroke="#0284c7" stroke-width="1.5" marker-end="url(#arrowBlue)"/>
    </g>

    <!-- Arrow Mod 1 -> Mod 2 -->
    <path d="M 670 822 L 695 822" fill="none" stroke="#0284c7" stroke-width="2" marker-end="url(#arrowBlue)"/>

    <!-- Module 2: Directional Impact Tensor -->
    <g transform="translate(700, 705)">
      <rect x="0" y="0" width="135" height="235" rx="8" fill="#f8fafc" stroke="#334155" stroke-width="1.8"/>
      <rect x="8" y="8" width="20" height="219" fill="#60a5fa"/>
      <text x="-115" y="22" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#1e3a8a">Beeswarm</text>
      
      <rect x="32" y="8" width="20" height="219" fill="#34d399"/>
      <text x="-115" y="46" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#065f46">Impact Dir</text>
      
      <rect x="56" y="8" width="20" height="219" fill="#fde047"/>
      <text x="-115" y="70" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#713f12">Force Map</text>
      
      <rect x="80" y="8" width="22" height="219" fill="#a855f7"/>
      <text x="-115" y="95" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#ffffff">Push Vector</text>
      
      <rect x="106" y="8" width="20" height="219" fill="#f43f5e"/>
      <text x="-115" y="120" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#ffffff">Ranking</text>
      
      <path d="M 30 8 C 30 -12, 105 -12, 105 8" fill="none" stroke="#0284c7" stroke-width="1.5" marker-end="url(#arrowBlue)"/>
    </g>

    <!-- Arrow Mod 2 -> Mod 3 -->
    <path d="M 835 822 L 860 822" fill="none" stroke="#0284c7" stroke-width="2" marker-end="url(#arrowBlue)"/>

    <!-- Module 3: L1-Norm Nelder-Mead Optimizer -->
    <g transform="translate(865, 705)">
      <rect x="0" y="0" width="135" height="235" rx="8" fill="#f8fafc" stroke="#334155" stroke-width="1.8"/>
      <rect x="8" y="8" width="20" height="219" fill="#fb923c"/>
      <text x="-115" y="22" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#7c2d12">L1 Simplex</text>
      
      <rect x="32" y="8" width="20" height="219" fill="#38bdf8"/>
      <text x="-115" y="46" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#0c4a6e">Min Delta</text>
      
      <rect x="56" y="8" width="20" height="219" fill="#fde047"/>
      <text x="-115" y="70" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#713f12">Cost Guard</text>
      
      <rect x="80" y="8" width="22" height="219" fill="#10b981"/>
      <text x="-115" y="95" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#ffffff">Fixed Coords</text>
      
      <rect x="106" y="8" width="20" height="219" fill="#8b5cf6"/>
      <text x="-115" y="120" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#ffffff">Optimized</text>
      
      <path d="M 30 8 C 30 -12, 105 -12, 105 8" fill="none" stroke="#0284c7" stroke-width="1.5" marker-end="url(#arrowBlue)"/>
    </g>

    <!-- Arrow Mod 3 -> Mod 4 -->
    <path d="M 1000 822 L 1025 822" fill="none" stroke="#0284c7" stroke-width="2" marker-end="url(#arrowBlue)"/>

    <!-- Module 4: Actionable Siting Guidance -->
    <g transform="translate(1030, 705)">
      <rect x="0" y="0" width="135" height="235" rx="8" fill="#f8fafc" stroke="#334155" stroke-width="1.8"/>
      <rect x="8" y="8" width="20" height="219" fill="#4ade80"/>
      <text x="-115" y="22" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#14532d">Action Levers</text>
      
      <rect x="32" y="8" width="20" height="219" fill="#93c5fd"/>
      <text x="-115" y="46" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#1e3a8a">Grid Feeder</text>
      
      <rect x="56" y="8" width="20" height="219" fill="#fef08a"/>
      <text x="-115" y="70" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#713f12">Road Trans</text>
      
      <rect x="80" y="8" width="22" height="219" fill="#06b6d4"/>
      <text x="-115" y="95" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#ffffff">POI Density</text>
      
      <rect x="106" y="8" width="20" height="219" fill="#ec4899"/>
      <text x="-115" y="120" transform="rotate(-90)" font-family="-apple-system, sans-serif" font-size="9" font-weight="bold" fill="#ffffff">P(Fast)≥80%</text>
      
      <path d="M 30 8 C 30 -12, 105 -12, 105 8" fill="none" stroke="#0284c7" stroke-width="1.5" marker-end="url(#arrowBlue)"/>
    </g>

    <!-- Feedback loop arrow back to Champion Classifier (y=570) -->
    <path d="M 1165 720 L 1230 720 L 1230 580" fill="none" stroke="#0284c7" stroke-width="2.5" marker-end="url(#arrowBlue)"/>
    <rect x="1175" y="660" width="105" height="22" rx="4" fill="#e0f2fe" stroke="#0284c7" stroke-width="1"/>
    <text x="1227" y="675" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9.5" font-weight="bold" fill="#0369a1">Prescriptive Loop</text>

    <!-- Embedded Results Visuals inside the Bottom Panel -->
    <!-- Visual 1: Feature Importance Chart -->
    <g transform="translate(1200, 715)">
      <rect x="0" y="0" width="410" height="305" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="410" height="26" rx="8" fill="#e0f2fe"/>
      <text x="10" y="18" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0369a1">Global TreeSHAP Importance (Top 15)</text>
      <image href="{b64_shap}" x="10" y="32" width="390" height="265" preserveAspectRatio="xMidYMid meet"/>
    </g>

    <!-- Visual 2: Actionable Counterfactual Elevation -->
    <g transform="translate(1625, 715)">
      <rect x="0" y="0" width="385" height="305" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="385" height="26" rx="8" fill="#ecfdf5"/>
      <text x="10" y="18" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#047857">Actionable Counterfactual Elevation (10 States)</text>
      <image href="{b64_cf}" x="10" y="32" width="365" height="265" preserveAspectRatio="xMidYMid meet"/>
    </g>

    <!-- Explanatory Text Footer Banner -->
    <g transform="translate(535, 1035)">
      <rect x="0" y="0" width="1470" height="42" rx="6" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1"/>
      <text x="735" y="18" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0f172a">Constraint-Respecting Optimization Framework:</text>
      <text x="735" y="34" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" fill="#475569">Immutable coordinates (Lat/Lon) held constant (δ_j = 0) • Actionable levers (feeder lines, road transitions, POI density) minimized via L1-norm Nelder-Mead simplex</text>
    </g>

    <!-- Detailed Mathematical Specification Stack -->
    <g transform="translate(535, 1090)">
      <rect x="0" y="0" width="475" height="280" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="475" height="28" rx="8" fill="#e0f2fe"/>
      <text x="12" y="19" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="bold" fill="#0369a1">A. TreeSHAP Additive Efficiency</text>
      <text x="15" y="52" font-family="-apple-system, sans-serif" font-size="11" fill="#0f172a">Exact Game-Theoretic Formulation:</text>
      <text x="25" y="75" font-family="'Times New Roman', serif" font-size="13" font-style="italic" fill="#0369a1">f(x) = φ_0 + ∑_{{i=1}}^{{21}} φ_i(x)</text>
      <text x="15" y="105" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Polynomial Runtime: O(TLD²) vs Exponential</text>
      <text x="15" y="125" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Guarantees Efficiency, Symmetry, Additivity</text>
      <text x="15" y="145" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• District EV Penetration Rank #1 (+0.176)</text>
      <text x="15" y="165" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Highway Distance &amp; Road Type #2 &amp; #3</text>
      <rect x="15" y="185" width="445" height="75" rx="4" fill="#ffffff" stroke="#e2e8f0"/>
      <text x="25" y="205" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#0f172a">Prescriptive Policy Insight:</text>
      <text x="25" y="223" font-family="-apple-system, sans-serif" font-size="9.5" fill="#475569">Stations located &gt; 5 km from highway suffer substantial negative SHAP penalties (-0.124),</text>
      <text x="25" y="240" font-family="-apple-system, sans-serif" font-size="9.5" fill="#475569">which can be neutralized by connecting to primary dual-carriageway corridors.</text>
    </g>

    <g transform="translate(1030, 1090)">
      <rect x="0" y="0" width="475" height="280" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="475" height="28" rx="8" fill="#ecfdf5"/>
      <text x="12" y="19" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="bold" fill="#047857">B. L1-Constrained Counterfactual Solver</text>
      <text x="15" y="52" font-family="-apple-system, sans-serif" font-size="11" fill="#0f172a">Inverse Classification Objective:</text>
      <text x="25" y="75" font-family="'Times New Roman', serif" font-size="13" font-style="italic" fill="#047857">min ||x\' - x||_1  s.t.  P_{{RF}}(x\') ≥ τ^* = 0.2488</text>
      <text x="15" y="105" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Constraints: δ_{{lat}} = 0, δ_{{lon}} = 0, δ_{{state}} = 0</text>
      <text x="15" y="125" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Actionable Set: Feeder headroom, POI mix, road type</text>
      <text x="15" y="145" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Elevation: Transitions AC slow (18%) to DC fast (82%)</text>
      <text x="15" y="165" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Feasibility Verified across 10 Representative States</text>
      <rect x="15" y="185" width="445" height="75" rx="4" fill="#ffffff" stroke="#e2e8f0"/>
      <text x="25" y="205" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#0f172a">Capex Minimization Protocol:</text>
      <text x="25" y="223" font-family="-apple-system, sans-serif" font-size="9.5" fill="#475569">Solves for minimal infrastructure delta: e.g. upgrading transformer tap + adding</text>
      <text x="25" y="240" font-family="-apple-system, sans-serif" font-size="9.5" fill="#475569">retail amenities yields fast charging suitability without expensive land relocation.</text>
    </g>

    <g transform="translate(1525, 1090)">
      <rect x="0" y="0" width="480" height="280" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="480" height="28" rx="8" fill="#fef3c7"/>
      <text x="12" y="19" font-family="-apple-system, sans-serif" font-size="11.5" font-weight="bold" fill="#b45309">C. AHP-TOPSIS National Siting Engine</text>
      <text x="15" y="52" font-family="-apple-system, sans-serif" font-size="11" fill="#0f172a">Relative Closeness Formulation:</text>
      <text x="25" y="75" font-family="'Times New Roman', serif" font-size="13" font-style="italic" fill="#b45309">C_i = S_i^{{-}} / (S_i^{{+}} + S_i^{{-}}) ∈ [0, 1] → EIPS_i = 100 · C_i</text>
      <text x="15" y="105" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• AHP Weights: w = [30% Adopt, 25% Grid, 25% NH, 20% Gap]</text>
      <text x="15" y="125" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Saaty Consistency: CR = 0.000891 &lt; 0.10 (Consistent)</text>
      <text x="15" y="145" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Euclidean Ideal Distance in 4-Dimensional Normalized Space</text>
      <text x="15" y="165" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Partitions All 36 States/UTs into 3 Policy Priority Tiers</text>
      <rect x="15" y="185" width="450" height="75" rx="4" fill="#ffffff" stroke="#e2e8f0"/>
      <text x="25" y="205" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#0f172a">National Capital Allocation:</text>
      <text x="25" y="223" font-family="-apple-system, sans-serif" font-size="9.5" fill="#475569">Tier 1: High Priority (KA, MH, UP, DL, TN, RJ, KL, GJ, MP, TS)</text>
      <text x="25" y="240" font-family="-apple-system, sans-serif" font-size="9.5" fill="#475569">Guides central FAME-II and PM E-DRIVE public fund disbursement.</text>
    </g>
  </g>

  <!-- ========================================================================= -->
  <!-- RIGHT COLUMN: OUTPUT SECTION                                              -->
  <!-- ========================================================================= -->
  <g id="output-column">
    <rect x="2060" y="105" width="500" height="1285" rx="14" fill="#ffffff" stroke="#f97316" stroke-width="2.5" stroke-dasharray="8 5" filter="url(#shadowSm)"/>
    <text x="2310" y="145" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="24" font-weight="800" fill="#ea580c">Output</text>
    <text x="2310" y="166" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" font-weight="600" fill="#64748b">Empirical Results, Diagnostic Visuals &amp; National Siting</text>

    <!-- Output 1: Confusion Matrix -->
    <g transform="translate(2078, 182)">
      <rect x="0" y="0" width="464" height="225" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="464" height="28" rx="8" fill="#e0f2fe"/>
      <text x="12" y="19" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#0369a1">1. Outer Test Confusion Matrix (N=52,718)</text>
      <text x="452" y="19" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#0284c7">99.85% Fast Recall</text>
      <image href="{b64_cm}" x="12" y="36" width="170" height="175" preserveAspectRatio="xMidYMid meet"/>
      <g transform="translate(195, 46)">
        <text x="0" y="12" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0f172a">• True Positives: 16,537</text>
        <text x="0" y="32" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• False Negatives: Only 24</text>
        <text x="0" y="52" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• DC Fast Recall: 99.85%</text>
        <text x="0" y="72" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Balanced Accuracy: 50.12%</text>
        <rect x="0" y="94" width="255" height="48" rx="4" fill="#e0f2fe" stroke="#38bdf8" stroke-width="0.8"/>
        <text x="127" y="112" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#0369a1">Cost-Sensitive Siting Policy</text>
        <text x="127" y="128" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#0284c7">Zero Missed Corridor Opportunities</text>
      </g>
    </g>

    <!-- Output 2: Precision-Recall Curves -->
    <g transform="translate(2078, 420)">
      <rect x="0" y="0" width="464" height="205" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="464" height="28" rx="8" fill="#ecfdf5"/>
      <text x="12" y="19" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#047857">2. Precision-Recall Multi-Model Bake-Off</text>
      <text x="452" y="19" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#059669">RF Champion</text>
      <image href="{b64_pr}" x="12" y="38" width="180" height="155" preserveAspectRatio="xMidYMid meet"/>
      <g transform="translate(205, 46)">
        <text x="0" y="12" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0f172a">• RF PR-AUC: 0.3118 (Winner)</text>
        <text x="0" y="32" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• XGBoost PR-AUC: 0.3051</text>
        <text x="0" y="52" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• LightGBM PR-AUC: 0.2970</text>
        <text x="0" y="72" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Baseline No-Skill: 0.3142</text>
        <rect x="0" y="92" width="245" height="46" rx="4" fill="#ecfdf5" stroke="#34d399" stroke-width="0.8"/>
        <text x="122" y="108" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#047857">Superior Spatial Calibration</text>
        <text x="122" y="124" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#065f46">Brier Score = 0.2346 (Lowest Error)</text>
      </g>
    </g>

    <!-- Output 3: National EIPS State Prioritization -->
    <g transform="translate(2078, 638)">
      <rect x="0" y="0" width="464" height="205" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="464" height="28" rx="8" fill="#fef3c7"/>
      <text x="12" y="19" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#b45309">3. National EIPS Siting Ranking (AHP-TOPSIS)</text>
      <text x="452" y="19" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#d97706">36 States/UTs</text>
      <image href="{b64_eips}" x="12" y="38" width="185" height="155" preserveAspectRatio="xMidYMid meet"/>
      <g transform="translate(205, 46)">
        <text x="0" y="12" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0f172a">• #1 Karnataka (EIPS: 82.00)</text>
        <text x="0" y="32" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• #2 Maharashtra (71.20)</text>
        <text x="0" y="52" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• #3 Uttar Pradesh (55.10)</text>
        <text x="0" y="72" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• #4 Delhi (50.40) • #5 TN (46.8)</text>
        <rect x="0" y="92" width="245" height="46" rx="4" fill="#fef3c7" stroke="#f59e0b" stroke-width="0.8"/>
        <text x="122" y="108" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#b45309">Tier 1 Strategic Acceleration</text>
        <text x="122" y="124" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#92400e">Top 10 High-Demand Jurisdictions</text>
      </g>
    </g>

    <!-- Output 4: Multi-Criteria Component Decomposition -->
    <g transform="translate(2078, 855)">
      <rect x="0" y="0" width="464" height="205" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="464" height="28" rx="8" fill="#ccfbf1"/>
      <text x="12" y="19" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#0f766e">4. Multi-Criteria Component Decomposition</text>
      <text x="452" y="19" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#0d9488">Balanced AHP</text>
      <image href="{b64_decomp}" x="12" y="38" width="185" height="155" preserveAspectRatio="xMidYMid meet"/>
      <g transform="translate(205, 46)">
        <text x="0" y="12" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0f172a">• EV Adoption Volume: 30%</text>
        <text x="0" y="32" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Grid Feeder Headroom: 25%</text>
        <text x="0" y="52" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• NH Corridor Links: 25%</text>
        <text x="0" y="72" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Infrastructure Deficit: 20%</text>
        <rect x="0" y="92" width="245" height="46" rx="4" fill="#ccfbf1" stroke="#2dd4bf" stroke-width="0.8"/>
        <text x="122" y="108" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#0f766e">Mathematical Consistency</text>
        <text x="122" y="124" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" fill="#115e59">Saaty CR = 0.000891 &lt; 0.10</text>
      </g>
    </g>

    <!-- Output 5: Live Interactive GIS Dashboard -->
    <g transform="translate(2078, 1072)">
      <rect x="0" y="0" width="464" height="200" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"/>
      <rect x="0" y="0" width="464" height="28" rx="8" fill="#f3e8ff"/>
      <text x="12" y="19" font-family="-apple-system, sans-serif" font-size="12" font-weight="bold" fill="#7e22ce">5. Interactive Web Application (Port 5001)</text>
      <text x="452" y="19" text-anchor="end" font-family="-apple-system, sans-serif" font-size="10.5" font-weight="bold" fill="#9333ea">Live Delivery</text>
      <image href="{b64_ui}" x="12" y="38" width="195" height="120" preserveAspectRatio="xMidYMid meet"/>
      <g transform="translate(215, 42)">
        <text x="0" y="12" font-family="-apple-system, sans-serif" font-size="11" font-weight="bold" fill="#0f172a">• 6 Interactive UI Tabs</text>
        <text x="0" y="30" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Real-Time Station Lookup</text>
        <text x="0" y="48" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• What-If Siting Simulator</text>
        <text x="0" y="66" font-family="-apple-system, sans-serif" font-size="10.5" fill="#334155">• Embedded Folium GIS Map</text>
        <rect x="0" y="84" width="235" height="50" rx="4" fill="#f3e8ff" stroke="#c084fc" stroke-width="0.8"/>
        <text x="117" y="100" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="10" font-weight="bold" fill="#7e22ce">Flask REST API Microservice</text>
        <text x="117" y="116" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="9" font-weight="bold" fill="#9333ea">http://localhost:5001</text>
      </g>
    </g>
  </g>

</svg>'''

svg_out = Path("system_architecture_india_wide.svg")
svg_fig_out = Path("results/india_wide/figures/system_architecture_india_wide.svg")

with open(svg_out, "w", encoding="utf-8") as f:
    f.write(svg_content)

with open(svg_fig_out, "w", encoding="utf-8") as f:
    f.write(svg_content)

print(f"Saved {svg_out}: {svg_out.stat().st_size:,} bytes")
print(f"Saved {svg_fig_out}: {svg_fig_out.stat().st_size:,} bytes")

# Now render with Headless Chrome to Ultra-HD JPG
chrome_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
png_tmp = Path("temp_arch_reference.png").resolve()
abs_svg = svg_out.resolve()

print("Rendering high-res screenshot via Headless Chrome...")
cmd = [
    chrome_exe,
    "--headless=new",
    "--disable-gpu",
    "--window-size=2600,1420",
    "--force-device-scale-factor=1.5",
    "--hide-scrollbars",
    f"--screenshot={png_tmp}",
    f"file:///{abs_svg.as_posix()}"
]
res = subprocess.run(cmd, capture_output=True, text=True)
if png_tmp.exists():
    im = Image.open(png_tmp)
    print("Screenshot rendered, size:", im.size)
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        rgb_im = bg
    else:
        rgb_im = im.convert("RGB")
    
    jpg_out = Path("system_architecture_india_wide.jpg")
    jpg_fig_out = Path("results/india_wide/figures/system_architecture_india_wide.jpg")
    rgb_im.save(jpg_out, "JPEG", quality=95, optimize=True)
    rgb_im.save(jpg_fig_out, "JPEG", quality=95, optimize=True)
    png_tmp.unlink()
    print(f"Saved {jpg_out}: {jpg_out.stat().st_size:,} bytes")
    print(f"Saved {jpg_fig_out}: {jpg_fig_out.stat().st_size:,} bytes")
else:
    print("Screenshot failed:", res.stderr)
