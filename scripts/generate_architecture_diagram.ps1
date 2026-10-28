# PowerShell script to generate high-resolution SYSTEM ARCHITECTURE diagram
# Pure academic methodology architecture strictly matching the visual grammar, structure, and proportions of the reference
# Canvas: Square 1600 x 1600 px (1:1 Aspect Ratio)
# Flow: Inputs (Left) -> Exogenous Feature Extractor (Top, Preserved) -> Spatial-Aware Classification (Middle) -> XAI & Planning Pipelines (Bottom) -> Outputs (Right)

Add-Type -AssemblyName System.Drawing

$width = 1600
$height = 1600
$bmp = New-Object System.Drawing.Bitmap $width, $height
$g = [System.Drawing.Graphics]::FromImage($bmp)

# Ultra-high quality rendering settings
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit

# -------------------------------------------------------------
# COLOR PALETTE (Restrained Academic Pastels matching Reference)
# -------------------------------------------------------------
$cBgTop = [System.Drawing.Color]::FromArgb(236, 245, 253)
$cBgBottom = [System.Drawing.Color]::FromArgb(248, 252, 255)
$cWhite = [System.Drawing.Color]::White
$cDarkText = [System.Drawing.Color]::FromArgb(15, 23, 42)
$cMutedText = [System.Drawing.Color]::FromArgb(71, 85, 105)
$cBorderOuter = [System.Drawing.Color]::FromArgb(51, 65, 85)
$cBorderLight = [System.Drawing.Color]::FromArgb(203, 213, 225)

# Pipeline Colors (Teal, Purple, Coral/Orange, Green, Blue, Amber)
$cTealFill = [System.Drawing.Color]::FromArgb(204, 251, 241)
$cTealBorder = [System.Drawing.Color]::FromArgb(13, 148, 136)
$cTealDark = [System.Drawing.Color]::FromArgb(15, 118, 110)

$cPurpleFill = [System.Drawing.Color]::FromArgb(243, 232, 255)
$cPurpleBorder = [System.Drawing.Color]::FromArgb(147, 51, 234)
$cPurpleDark = [System.Drawing.Color]::FromArgb(126, 34, 206)

$cCoralFill = [System.Drawing.Color]::FromArgb(254, 215, 170)
$cCoralBorder = [System.Drawing.Color]::FromArgb(234, 88, 12)
$cCoralDark = [System.Drawing.Color]::FromArgb(194, 65, 12)

$cGreenFill = [System.Drawing.Color]::FromArgb(220, 252, 231)
$cGreenBorder = [System.Drawing.Color]::FromArgb(22, 163, 74)
$cGreenDark = [System.Drawing.Color]::FromArgb(21, 128, 61)

$cBlueFill = [System.Drawing.Color]::FromArgb(224, 242, 254)
$cBlueBorder = [System.Drawing.Color]::FromArgb(2, 132, 199)
$cBlueDark = [System.Drawing.Color]::FromArgb(3, 105, 161)

$cAmberFill = [System.Drawing.Color]::FromArgb(254, 243, 199)
$cAmberBorder = [System.Drawing.Color]::FromArgb(217, 119, 6)
$cAmberDark = [System.Drawing.Color]::FromArgb(180, 83, 9)

# Persistent Solid Brushes
$bDark = New-Object System.Drawing.SolidBrush($cDarkText)
$bMuted = New-Object System.Drawing.SolidBrush($cMutedText)
$bWhite = New-Object System.Drawing.SolidBrush($cWhite)
$bTealDark = New-Object System.Drawing.SolidBrush($cTealDark)
$bPurpleDark = New-Object System.Drawing.SolidBrush($cPurpleDark)
$bCoralDark = New-Object System.Drawing.SolidBrush($cCoralDark)
$bGreenDark = New-Object System.Drawing.SolidBrush($cGreenDark)
$bBlueDark = New-Object System.Drawing.SolidBrush($cBlueDark)
$bAmberDark = New-Object System.Drawing.SolidBrush($cAmberDark)

# Persistent Pens
$pBorderOuter = [System.Drawing.Pen]::new($cBorderOuter, [float]1.8)
$pBorderLight = [System.Drawing.Pen]::new($cBorderLight, [float]1.2)
$pBorderThin = [System.Drawing.Pen]::new($cBorderOuter, [float]1.2)

# -------------------------------------------------------------
# FONTS (Clear academic hierarchy)
# -------------------------------------------------------------
$fontHeader = New-Object System.Drawing.Font('Times New Roman', 34, [System.Drawing.FontStyle]::Bold)
$fontBoxTitle = New-Object System.Drawing.Font('Arial', 14.5, [System.Drawing.FontStyle]::Bold)
$fontBoxSub = New-Object System.Drawing.Font('Arial', 11, [System.Drawing.FontStyle]::Regular)
$fontBlockTitle = New-Object System.Drawing.Font('Arial', 12.5, [System.Drawing.FontStyle]::Bold)
$fontBlockSub = New-Object System.Drawing.Font('Arial', 10.5, [System.Drawing.FontStyle]::Regular)
$fontCardTitle = New-Object System.Drawing.Font('Arial', 11, [System.Drawing.FontStyle]::Bold)
$fontCardText = New-Object System.Drawing.Font('Arial', 10, [System.Drawing.FontStyle]::Regular)
$fontBadge = New-Object System.Drawing.Font('Arial', 9.5, [System.Drawing.FontStyle]::Bold)

# String formats
$sfCenter = New-Object System.Drawing.StringFormat
$sfCenter.Alignment = [System.Drawing.StringAlignment]::Center
$sfCenter.LineAlignment = [System.Drawing.StringAlignment]::Center

# -------------------------------------------------------------
# DRAW BACKGROUND
# -------------------------------------------------------------
$bgRect = New-Object System.Drawing.Rectangle 0, 0, $width, $height
$bgBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush($bgRect, $cBgTop, $cBgBottom, [System.Drawing.Drawing2D.LinearGradientMode]::Vertical)
$g.FillRectangle($bgBrush, $bgRect)
$bgBrush.Dispose()

# Outer subtle border
$penOuter = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(180, 205, 230), [float]3.0)
$g.DrawRectangle($penOuter, 10, 10, ($width - 20), ($height - 20))
$penOuter.Dispose()

# -------------------------------------------------------------
# TITLE
# -------------------------------------------------------------
$g.DrawString('SYSTEM ARCHITECTURE', $fontHeader, $bDark, [float]($width / 2), 42.0, $sfCenter)

# Helper function for rounded rectangles
function Get-RoundPath($x, $y, $w, $h, $r) {
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $d = $r * 2
    $path.AddArc($x, $y, $d, $d, 180, 90)
    $path.AddArc(($x + $w - $d), $y, $d, $d, 270, 90)
    $path.AddArc(($x + $w - $d), ($y + $h - $d), $d, $d, 0, 90)
    $path.AddArc($x, ($y + $h - $d), $d, $d, 90, 90)
    $path.CloseFigure()
    return $path
}

# Helper to draw a main white container
function Show-MainContainer($x, $y, $w, $h, $title, $subTitle) {
    $path = Get-RoundPath $x $y $w $h 12
    $g.FillPath($bWhite, $path)
    $g.DrawPath($pBorderOuter, $path)
    $path.Dispose()

    if ($title -ne '') {
        $g.DrawString($title, $fontBoxTitle, $bDark, [float]($x + 22), [float]($y + 14))
        if ($subTitle -ne '') {
            $sz = $g.MeasureString($title, $fontBoxTitle)
            $g.DrawString($subTitle, $fontBoxSub, $bMuted, [float]($x + 22 + $sz.Width + 10), [float]($y + 17))
        }
    }
}

# Helper to draw technical connecting arrow
function Show-ArrowLine($x1, $y1, $x2, $y2, $lbl) {
    $p = [System.Drawing.Pen]::new($cBorderOuter, [float]1.8)
    $cap = [System.Drawing.Drawing2D.AdjustableArrowCap]::new(4.5, 4.5, $true)
    $p.CustomEndCap = $cap
    $g.DrawLine($p, [float]$x1, [float]$y1, [float]$x2, [float]$y2)
    $cap.Dispose()
    $p.Dispose()

    if ($lbl -ne '') {
        $mx = ($x1 + $x2) / 2
        $my = ($y1 + $y2) / 2 - 12
        $g.DrawString($lbl, $fontBadge, $bDark, [float]$mx, [float]$my, $sfCenter)
    }
}

# Helper to draw 3D Slab with label
function Show-3DSlabRGB($x, $y, $w, $h, $d, $r, $gVal, $bVal, $label) {
    $cFront = [System.Drawing.Color]::FromArgb($r, $gVal, $bVal)
    $rT = [Math]::Min(255, ($r + 35))
    $gT = [Math]::Min(255, ($gVal + 35))
    $bT = [Math]::Min(255, ($bVal + 35))
    $cTop = [System.Drawing.Color]::FromArgb($rT, $gT, $bT)

    $rS = [Math]::Max(0, ($r - 35))
    $gS = [Math]::Max(0, ($gVal - 35))
    $bS = [Math]::Max(0, ($bVal - 35))
    $cSide = [System.Drawing.Color]::FromArgb($rS, $gS, $bS)

    $pFront = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(80, 0, 0, 0), [float]1.0)
    
    # Front face
    $bF = New-Object System.Drawing.SolidBrush($cFront)
    $g.FillRectangle($bF, [float]$x, [float]$y, [float]$w, [float]$h)
    $g.DrawRectangle($pFront, [float]$x, [float]$y, [float]$w, [float]$h)
    $bF.Dispose()

    # Top face
    $topPts = @(
        (New-Object System.Drawing.PointF($x, $y)),
        (New-Object System.Drawing.PointF(($x + $d), ($y - $d))),
        (New-Object System.Drawing.PointF(($x + $w + $d), ($y - $d))),
        (New-Object System.Drawing.PointF(($x + $w), $y))
    )
    $bTop = New-Object System.Drawing.SolidBrush($cTop)
    $g.FillPolygon($bTop, $topPts)
    $g.DrawPolygon($pFront, $topPts)
    $bTop.Dispose()

    # Side face
    $sidePts = @(
        (New-Object System.Drawing.PointF(($x + $w), $y)),
        (New-Object System.Drawing.PointF(($x + $w + $d), ($y - $d))),
        (New-Object System.Drawing.PointF(($x + $w + $d), ($y + $h - $d))),
        (New-Object System.Drawing.PointF(($x + $w), ($y + $h)))
    )
    $bSide = New-Object System.Drawing.SolidBrush($cSide)
    $g.FillPolygon($bSide, $sidePts)
    $g.DrawPolygon($pFront, $sidePts)
    $bSide.Dispose()
    $pFront.Dispose()

    # Vertical text inside slab
    if ($label -ne '') {
        $state = $g.Save()
        $g.TranslateTransform(($x + $w/2 + 1), ($y + $h/2))
        $g.RotateTransform(-90)
        $tBrush = New-Object System.Drawing.SolidBrush($cDarkText)
        $g.DrawString($label, $fontCardText, $tBrush, 0.0, 0.0, $sfCenter)
        $tBrush.Dispose()
        $g.Restore($state)
    }
}

function Show-3DCubeRGB($x, $y, $size, $r, $gVal, $bVal) {
    $d = [float]($size * 0.45)
    $cFront = [System.Drawing.Color]::FromArgb($r, $gVal, $bVal)
    $rT = [Math]::Min(255, ($r + 40))
    $gT = [Math]::Min(255, ($gVal + 40))
    $bT = [Math]::Min(255, ($bVal + 40))
    $cTop = [System.Drawing.Color]::FromArgb($rT, $gT, $bT)

    $rS = [Math]::Max(0, ($r - 40))
    $gS = [Math]::Max(0, ($gVal - 40))
    $bS = [Math]::Max(0, ($bVal - 40))
    $cSide = [System.Drawing.Color]::FromArgb($rS, $gS, $bS)

    $p = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(70, 0, 0, 0), [float]1.2)

    # Front
    $bF = New-Object System.Drawing.SolidBrush($cFront)
    $g.FillRectangle($bF, [float]$x, [float]$y, [float]$size, [float]$size)
    $g.DrawRectangle($p, [float]$x, [float]$y, [float]$size, [float]$size)
    $bF.Dispose()

    # Top
    $topPts = @(
        (New-Object System.Drawing.PointF($x, $y)),
        (New-Object System.Drawing.PointF(($x + $d), ($y - $d))),
        (New-Object System.Drawing.PointF(($x + $size + $d), ($y - $d))),
        (New-Object System.Drawing.PointF(($x + $size), $y))
    )
    $bT = New-Object System.Drawing.SolidBrush($cTop)
    $g.FillPolygon($bT, $topPts)
    $g.DrawPolygon($p, $topPts)
    $bT.Dispose()

    # Side
    $sidePts = @(
        (New-Object System.Drawing.PointF(($x + $size), $y)),
        (New-Object System.Drawing.PointF(($x + $size + $d), ($y - $d))),
        (New-Object System.Drawing.PointF(($x + $size + $d), ($y + $size - $d))),
        (New-Object System.Drawing.PointF(($x + $size), ($y + $size)))
    )
    $bS = New-Object System.Drawing.SolidBrush($cSide)
    $g.FillPolygon($bS, $sidePts)
    $g.DrawPolygon($p, $sidePts)
    $bS.Dispose()
    $p.Dispose()
}

# =============================================================
# 1. LEFT COLUMN: INPUTS (4 Clean Academic Input Cards)
# =============================================================
$inX = 25
$inY = 80
$inW = 230
$inH = 1495
Show-MainContainer $inX $inY $inW $inH '' ''

# Inner Header
$g.DrawString('INPUTS', $fontBoxTitle, $bDark, [float]($inX + $inW/2), [float]($inY + 22), $sfCenter)

# Card 1: Charging Station Data
$c1Y = 135
$c1H = 320
$pathC1 = Get-RoundPath ($inX + 14) $c1Y ($inW - 28) $c1H 10
$bCard1 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(250, 252, 255))
$g.FillPath($bCard1, $pathC1)
$g.DrawPath($pBorderLight, $pathC1)
$pathC1.Dispose()

$imgStationsPath = 'results/figures/gis_spatial_distribution.png'
if (Test-Path $imgStationsPath) {
    $imgStations = [System.Drawing.Image]::FromFile((Resolve-Path $imgStationsPath))
    $g.DrawImage($imgStations, ($inX + 22), ($c1Y + 12), ($inW - 44), 215)
    $imgStations.Dispose()
} else {
    $bPh1 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(241, 245, 249))
    $g.FillRectangle($bPh1, ($inX + 22), ($c1Y + 12), ($inW - 44), 215)
    $bPh1.Dispose()
}
$g.DrawRectangle($pBorderLight, ($inX + 22), ($c1Y + 12), ($inW - 44), 215)
$g.DrawString('Charging Station Data', $fontCardTitle, $bDark, [float]($inX + $inW/2), [float]($c1Y + 248), $sfCenter)
$g.DrawString('166 Verified EV Stations', $fontCardText, $bMuted, [float]($inX + $inW/2), [float]($c1Y + 274), $sfCenter)
$g.DrawString('33 Telangana Districts', $fontCardText, $bMuted, [float]($inX + $inW/2), [float]($c1Y + 296), $sfCenter)

# Card 2: GIS / OSM Data
$c2Y = 485
$c2H = 320
$pathC2 = Get-RoundPath ($inX + 14) $c2Y ($inW - 28) $c2H 10
$g.FillPath($bCard1, $pathC2)
$g.DrawPath($pBorderLight, $pathC2)
$pathC2.Dispose()

$g.FillRectangle($bWhite, ($inX + 22), ($c2Y + 12), ($inW - 44), 215)
$pRoadHwy = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(234, 88, 12), [float]3.0)
$pRoadSec = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(14, 165, 233), [float]2.0)
$g.DrawLine($pRoadHwy, ($inX + 35), ($c2Y + 50), ($inX + $inW - 35), ($c2Y + 185))
$g.DrawLine($pRoadSec, ($inX + 35), ($c2Y + 175), ($inX + $inW - 35), ($c2Y + 60))
$g.DrawLine($pRoadSec, ($inX + $inW/2), ($c2Y + 25), ($inX + $inW/2), ($c2Y + 210))
$pRoadHwy.Dispose()
$pRoadSec.Dispose()

$bPoi1 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(13, 148, 136))
$bPoi2 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(234, 88, 12))
$g.FillEllipse($bPoi1, [float]($inX + 90), [float]($c2Y + 75), 14.0, 14.0)
$g.FillEllipse($bPoi2, [float]($inX + 145), [float]($c2Y + 150), 14.0, 14.0)
$g.FillEllipse($bPoi1, [float]($inX + 130), [float]($c2Y + 65), 12.0, 12.0)
$bPoi1.Dispose()
$bPoi2.Dispose()
$g.DrawRectangle($pBorderLight, ($inX + 22), ($c2Y + 12), ($inW - 44), 215)

$g.DrawString('GIS / OSM Road Network', $fontCardTitle, $bDark, [float]($inX + $inW/2), [float]($c2Y + 248), $sfCenter)
$g.DrawString('Roads, Highways & POIs', $fontCardText, $bMuted, [float]($inX + $inW/2), [float]($c2Y + 274), $sfCenter)
$g.DrawString('OpenStreetMap Topology', $fontCardText, $bMuted, [float]($inX + $inW/2), [float]($c2Y + 296), $sfCenter)

# Card 3: EV Adoption Data
$c3Y = 835
$c3H = 320
$pathC3 = Get-RoundPath ($inX + 14) $c3Y ($inW - 28) $c3H 10
$g.FillPath($bCard1, $pathC3)
$g.DrawPath($pBorderLight, $pathC3)
$pathC3.Dispose()

$g.FillRectangle($bWhite, ($inX + 22), ($c3Y + 12), ($inW - 44), 215)
$bBar = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(168, 85, 247))
$pBar = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(126, 34, 206), [float]1.0)
$barHeights = @(28, 48, 75, 110, 150, 190)
for ($bi = 0; $bi -lt 6; $bi++) {
    $bx = $inX + 40 + ($bi * 26)
    $bh = $barHeights[$bi]
    $by = $c3Y + 215 - $bh
    $g.FillRectangle($bBar, $bx, $by, 18, $bh)
    $g.DrawRectangle($pBar, $bx, $by, 18, $bh)
}
$bBar.Dispose()
$pBar.Dispose()
$g.DrawRectangle($pBorderLight, ($inX + 22), ($c3Y + 12), ($inW - 44), 215)

$g.DrawString('EV Adoption Data', $fontCardTitle, $bDark, [float]($inX + $inW/2), [float]($c3Y + 248), $sfCenter)
$g.DrawString('District x FY2018-FY2024', $fontCardText, $bMuted, [float]($inX + $inW/2), [float]($c3Y + 274), $sfCenter)
$g.DrawString('VAHAN National Registry', $fontCardText, $bMuted, [float]($inX + $inW/2), [float]($c3Y + 296), $sfCenter)

# Card 4: Grid / Utility Data
$c4Y = 1185
$c4H = 320
$pathC4 = Get-RoundPath ($inX + 14) $c4Y ($inW - 28) $c4H 10
$g.FillPath($bCard1, $pathC4)
$g.DrawPath($pBorderLight, $pathC4)
$pathC4.Dispose()
$bCard1.Dispose()

$g.FillRectangle($bWhite, ($inX + 22), ($c4Y + 12), ($inW - 44), 215)
$pGridLine = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(217, 119, 6), [float]2.5)
$gridPts = @(
    (New-Object System.Drawing.PointF(($inX + 30), ($c4Y + 150))),
    (New-Object System.Drawing.PointF(($inX + 60), ($c4Y + 70))),
    (New-Object System.Drawing.PointF(($inX + 90), ($c4Y + 180))),
    (New-Object System.Drawing.PointF(($inX + 120), ($c4Y + 60))),
    (New-Object System.Drawing.PointF(($inX + 150), ($c4Y + 160))),
    (New-Object System.Drawing.PointF(($inX + 180), ($c4Y + 80))),
    (New-Object System.Drawing.PointF(($inX + 200), ($c4Y + 130)))
)
$g.DrawLines($pGridLine, $gridPts)
$pGridLine.Dispose()
$g.DrawRectangle($pBorderLight, ($inX + 22), ($c4Y + 12), ($inW - 44), 215)

$g.DrawString('Grid / Utility Data', $fontCardTitle, $bDark, [float]($inX + $inW/2), [float]($c4Y + 248), $sfCenter)
$g.DrawString('TGSPDCL / DISCOM Load', $fontCardText, $bMuted, [float]($inX + $inW/2), [float]($c4Y + 274), $sfCenter)
$g.DrawString('288 Section Billing Records', $fontCardText, $bMuted, [float]($inX + $inW/2), [float]($c4Y + 296), $sfCenter)

# Arrow from Left Inputs to Top Feature Extractor
Show-ArrowLine ($inX + $inW) 275 280 275 ''

# =============================================================
# 2. TOP CONTAINER: EXOGENOUS FEATURE EXTRACTOR (PRESERVED)
# =============================================================
$topX = 280
$topY = 80
$topW = 1295
$topH = 390
Show-MainContainer $topX $topY $topW $topH 'Exogenous Feature Extractor' '(Strict Anti-Leakage Gate - Hardware Specs Blocked)'

# Group 1: 3D Layer Slabs for Spatial & GIS (Blue, Green, Orange, Teal)
$slY = 155
$slH = 195
$slD = 16

# Spatial Coords
Show-3DSlabRGB 330 $slY 32 $slH $slD 125 211 252 'Spatial Coords (8)'
# Distance to Highway
Show-3DSlabRGB 376 $slY 32 $slH $slD 134 239 172 'Distance to Highway'
# Distance to City Center
Show-3DSlabRGB 422 $slY 32 $slH $slD 253 186 116 'Distance to City Center'
# Nearby Amenities & POIs
Show-3DSlabRGB 468 $slY 32 $slH $slD 94 234 212 'POI Density & Diversity'

Show-ArrowLine 525 ($slY + $slH/2) 565 ($slY + $slH/2) ''

# Group 2: INVERTED RESIDUAL / ANTI-LEAKAGE GATE (Preserved block)
$gateX = 570
$gateY = 132
$gateW = 350
$gateH = 240
$pathGate = Get-RoundPath $gateX $gateY $gateW $gateH 10
$bGate = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(248, 250, 252))
$g.FillPath($bGate, $pathGate)
$g.DrawPath($pBorderThin, $pathGate)
$bGate.Dispose()
$pathGate.Dispose()

# Label above the gate
$g.DrawString('Anti-Leakage Quarantine Gate', $fontCardTitle, $bDark, [float]($gateX + $gateW/2), [float]($gateY + 16), $sfCenter)
$g.DrawString('[Blocks: power_kw, charger_type, connector, capacity]', $fontCardText, [System.Drawing.Brushes]::Firebrick, [float]($gateX + $gateW/2), [float]($gateY + 36), $sfCenter)

# 3 Inner Blocks with curved bypass
$b1X = $gateX + 24
$b2X = $gateX + 128
$b3X = $gateX + 232
$bY = $gateY + 58
$bW = 86
$bH = 135

# Block 1: Pink (Leakage Gate)
Show-3DSlabRGB $b1X $bY $bW $bH 10 253 164 175 'Leakage Gate'
Show-ArrowLine ($b1X + $bW + 2) ($bY + $bH/2) ($b2X - 2) ($bY + $bH/2) ''

# Block 2: Yellow (Shannon POI)
Show-3DSlabRGB $b2X $bY $bW $bH 10 253 224 71 'Shannon POI'
Show-ArrowLine ($b2X + $bW + 2) ($bY + $bH/2) ($b3X - 2) ($bY + $bH/2) ''

# Block 3: Purple (One-Hot 7D)
Show-3DSlabRGB $b3X $bY $bW $bH 10 216 180 254 'One-Hot (7)'

# Curved bypass line above blocks (residual style)
$pBypass = [System.Drawing.Pen]::new($cBorderOuter, [float]1.8)
$capBypass = [System.Drawing.Drawing2D.AdjustableArrowCap]::new(4.0, 4.0, $true)
$pBypass.CustomEndCap = $capBypass
$bypassPts = @(
    (New-Object System.Drawing.PointF(($b1X + 22), ($bY - 4))),
    (New-Object System.Drawing.PointF(($b1X + 22), ($bY - 14))),
    (New-Object System.Drawing.PointF(($b3X + $bW - 6), ($bY - 14))),
    (New-Object System.Drawing.PointF(($b3X + $bW - 6), ($bY - 4)))
)
$g.DrawLines($pBypass, $bypassPts)
$capBypass.Dispose()
$pBypass.Dispose()

# Bracket label underneath
$g.DrawString('21 Certified Exogenous Features', $fontBadge, $bDark, [float]($gateX + $gateW/2), [float]($gateY + $gateH - 14), $sfCenter)

Show-ArrowLine ($gateX + $gateW) ($slY + $slH/2) 960 ($slY + $slH/2) ''

# Group 3: Right Slabs (Road types & Demographics)
Show-3DSlabRGB 965 $slY 42 $slH $slD 134 239 172 'Road Features (7)'
Show-3DSlabRGB 1022 $slY 42 $slH $slD 253 186 116 'District Context (6)'

Show-ArrowLine 1082 ($slY + $slH/2) 1125 ($slY + $slH/2) ''

# Group 4: 3D Purple Isometric Cube (Global Feature Vector)
$cubeX = 1135
$cubeY = 195
$cubeSize = 88
Show-3DCubeRGB $cubeX $cubeY $cubeSize 147 51 234

$g.DrawString('Global Feature Vector', $fontCardTitle, $bDark, [float]($cubeX + $cubeSize/2 + 10), [float]($cubeY - 26), $sfCenter)
$g.DrawString('21-D Feature Vector', $fontBlockTitle, $bDark, [float]($cubeX + $cubeSize/2 + 10), [float]($cubeY + $cubeSize + 26), $sfCenter)

# Arrow from 21-D cube down into whitespace gap, turning left, and down into Stage 2.1
$arrowPen = [System.Drawing.Pen]::new($cBorderOuter, [float]2.2)
$capArrow = [System.Drawing.Drawing2D.AdjustableArrowCap]::new(4.5, 4.5, $true)
$arrowPen.CustomEndCap = $capArrow
$downPts = @(
    (New-Object System.Drawing.PointF(($cubeX + $cubeSize/2 + 10), ($cubeY + $cubeSize + 38))),
    (New-Object System.Drawing.PointF(($cubeX + $cubeSize/2 + 10), 492)),
    (New-Object System.Drawing.PointF(432, 492)),
    (New-Object System.Drawing.PointF(432, 510))
)
$g.DrawLines($arrowPen, $downPts)
$capArrow.Dispose()
$arrowPen.Dispose()
$g.DrawString('21-D Feature Matrix', $fontBadge, $bDark, [float]770, 482.0, $sfCenter)

# =============================================================
# 3. MIDDLE CONTAINER: SECTION 2 — SPATIAL-AWARE ML CLASSIFICATION PIPELINE
# 3 Solid Modular Architecture Blocks (Mirroring StandardScaler, PCA, SVM in Reference)
# =============================================================
$midX = 280
$midY = 510
$midW = 970
$midH = 375
Show-MainContainer $midX $midY $midW $midH 'Spatial-Aware ML Classification Pipeline' ''

$bHMid = 295
$bYMid = 565

# Stage 2.1: INPUT TO MODELING (Teal Box — NO StandardScaler, spacious & clean)
$mb1X = 300
$mb1W = 265
$pathMb1 = Get-RoundPath $mb1X $bYMid $mb1W $bHMid 10
$bMb1 = New-Object System.Drawing.SolidBrush($cTealFill)
$pMb1 = [System.Drawing.Pen]::new($cTealBorder, [float]1.8)
$g.FillPath($bMb1, $pathMb1)
$g.DrawPath($pMb1, $pathMb1)
$pathMb1.Dispose()
$bMb1.Dispose()

$g.DrawString('Spatial Partitioning', $fontBlockTitle, $bTealDark, [float]($mb1X + $mb1W/2), [float]($bYMid + 32), $sfCenter)
$g.DrawString('District GroupKFold (4-Fold)', $fontBlockSub, $bDark, [float]($mb1X + $mb1W/2), [float]($bYMid + 60), $sfCenter)

$g.DrawLine($pMb1, ($mb1X + 25), ($bYMid + 90), ($mb1X + $mb1W - 25), ($bYMid + 90))

$g.DrawString('21-D Feature Matrix (Raw)', $fontCardTitle, $bDark, [float]($mb1X + $mb1W/2), [float]($bYMid + 125), $sfCenter)
$g.DrawString('Zero District Overlap Audit', $fontBlockSub, $bMuted, [float]($mb1X + $mb1W/2), [float]($bYMid + 155), $sfCenter)
$g.DrawString('119 Development  |  25 Districts', $fontBlockSub, $bDark, [float]($mb1X + $mb1W/2), [float]($bYMid + 195), $sfCenter)
$g.DrawString('35 Outer Test  |  8 Districts', $fontBlockSub, $bDark, [float]($mb1X + $mb1W/2), [float]($bYMid + 225), $sfCenter)
$g.DrawString('Strict Spatial Independence', $fontBadge, $bTealDark, [float]($mb1X + $mb1W/2), [float]($bYMid + 262), $sfCenter)
$pMb1.Dispose()

# Arrow between Stage 2.1 and Stage 2.2
Show-ArrowLine ($mb1X + $mb1W) ($bYMid + $bHMid/2) 615 ($bYMid + $bHMid/2) '21-D'

# Stage 2.2: MODEL RESEARCH & OPTIMIZATION (Purple Box — RF | XGBoost | LightGBM converging)
$mb2X = 615
$mb2W = 270
$pathMb2 = Get-RoundPath $mb2X $bYMid $mb2W $bHMid 10
$bMb2 = New-Object System.Drawing.SolidBrush($cPurpleFill)
$pMb2 = [System.Drawing.Pen]::new($cPurpleBorder, [float]1.8)
$g.FillPath($bMb2, $pathMb2)
$g.DrawPath($pMb2, $pathMb2)
$pathMb2.Dispose()
$bMb2.Dispose()

$g.DrawString('Model Optimization', $fontBlockTitle, $bPurpleDark, [float]($mb2X + $mb2W/2), [float]($bYMid + 32), $sfCenter)
$g.DrawString('Research Candidates (Dev Only)', $fontBlockSub, $bMuted, [float]($mb2X + $mb2W/2), [float]($bYMid + 58), $sfCenter)

$g.DrawLine($pMb2, ($mb2X + 25), ($bYMid + 85), ($mb2X + $mb2W - 25), ($bYMid + 85))

# 3 Candidate Models pill row
$g.DrawString('RF   |   XGBoost   |   LightGBM', $fontCardTitle, $bPurpleDark, [float]($mb2X + $mb2W/2), [float]($bYMid + 115), $sfCenter)
$g.DrawString('Hyperparameter Tuning', $fontBlockSub, $bDark, [float]($mb2X + $mb2W/2), [float]($bYMid + 150), $sfCenter)
$g.DrawString('4-Fold Spatial Cross-Validation', $fontCardText, $bMuted, [float]($mb2X + $mb2W/2), [float]($bYMid + 175), $sfCenter)
$g.DrawString('Threshold Optimization (tau)', $fontBlockSub, $bDark, [float]($mb2X + $mb2W/2), [float]($bYMid + 210), $sfCenter)
$g.DrawString('Development-Only Selection', $fontBadge, $bPurpleDark, [float]($mb2X + $mb2W/2), [float]($bYMid + 255), $sfCenter)
$pMb2.Dispose()

# Arrow between Stage 2.2 and Stage 2.3
Show-ArrowLine ($mb2X + $mb2W) ($bYMid + $bHMid/2) 955 ($bYMid + $bHMid/2) 'Tuned RF'

# Stage 2.3: FINAL MODEL & EVALUATION (Coral Box — Random Forest Champion)
$mb3X = 955
$mb3W = 270
$pathMb3 = Get-RoundPath $mb3X $bYMid $mb3W $bHMid 10
$bMb3 = New-Object System.Drawing.SolidBrush($cCoralFill)
$pMb3 = [System.Drawing.Pen]::new($cCoralBorder, [float]1.8)
$g.FillPath($bMb3, $pathMb3)
$g.DrawPath($pMb3, $pathMb3)
$pathMb3.Dispose()
$bMb3.Dispose()

$g.DrawString('Random Forest Champion', $fontBlockTitle, $bCoralDark, [float]($mb3X + $mb3W/2), [float]($bYMid + 32), $sfCenter)
$g.DrawString('Single Deployed Model (tau = 0.76)', $fontBlockSub, $bMuted, [float]($mb3X + $mb3W/2), [float]($bYMid + 58), $sfCenter)

$g.DrawLine($pMb3, ($mb3X + 20), ($bYMid + 85), ($mb3X + $mb3W - 20), ($bYMid + 85))
$g.DrawString('n_estimators = 250,  depth = 3', $fontBlockSub, $bDark, [float]($mb3X + $mb3W/2), [float]($bYMid + 115), $sfCenter)
$g.DrawString('min_samples_split=2, leaf=2, feats=0.5', $fontCardText, $bMuted, [float]($mb3X + $mb3W/2), [float]($bYMid + 140), $sfCenter)

$g.DrawLine($pMb3, ($mb3X + 35), ($bYMid + 165), ($mb3X + $mb3W - 35), ($bYMid + 165))
$g.DrawString('Development CV: 80.06%', $fontBlockSub, $bDark, [float]($mb3X + $mb3W/2), [float]($bYMid + 192), $sfCenter)
$g.DrawString('Outer Test Accuracy: 71.43%', $fontCardTitle, $bCoralDark, [float]($mb3X + $mb3W/2), [float]($bYMid + 222), $sfCenter)
$g.DrawString('154 Labeled Stations Evaluated', $fontCardText, $bMuted, [float]($mb3X + $mb3W/2), [float]($bYMid + 252), $sfCenter)
$pMb3.Dispose()

# Arrow from Stage 2.3 to Right Outputs
Show-ArrowLine ($mb3X + $mb3W) ($bYMid + $bHMid/2) 1270 ($bYMid + $bHMid/2) ''

# Branching connector down to Explainability container
$branchPen = [System.Drawing.Pen]::new($cBorderOuter, [float]2.0)
$capBranch = [System.Drawing.Drawing2D.AdjustableArrowCap]::new(4.5, 4.5, $true)
$branchPen.CustomEndCap = $capBranch
$branchPts = @(
    (New-Object System.Drawing.PointF(($mb3X + $mb3W/2), ($bYMid + $bHMid))),
    (New-Object System.Drawing.PointF(($mb3X + $mb3W/2), 905)),
    (New-Object System.Drawing.PointF(580, 905)),
    (New-Object System.Drawing.PointF(580, 925))
)
$g.DrawLines($branchPen, $branchPts)
$capBranch.Dispose()
$branchPen.Dispose()

# =============================================================
# 4. BOTTOM CONTAINER: SECTION 3 — TREE SHAP & COUNTERFACTUAL EXPLAINABILITY + STRATEGIC PLANNING
# Two Clean Parallel Architectural Methodology Streams (Matching Grad-CAM Pipeline in Reference)
# =============================================================
$botX = 280
$botY = 925
$botW = 970
$botH = 650
Show-MainContainer $botX $botY $botW $botH 'TreeSHAP & Counterfactual Explainability + Strategic Planning' ''

# -------------------------------------------------------------
# STREAM 3.1: EXPLAINABLE AI ENGINE (TreeSHAP & Counterfactuals: y = 975 to 1250)
# -------------------------------------------------------------
$g.DrawString('3.1 EXPLAINABLE AI (XAI) PIPELINE', $fontBlockTitle, $bDark, [float]($botX + 25), 975.0)

# Path A: TreeSHAP Pipeline (Horizontal Flow of White Blocks)
$g.DrawString('PATH A: TreeSHAP Feature Attribution', $fontCardTitle, $bPurpleDark, [float]($botX + 25), 1010.0)

$flY = 1040
$flH = 50
$flW = 125

# Block 1: Trained RF
$fx1 = $botX + 25
$pathF1 = Get-RoundPath $fx1 $flY $flW $flH 6
$g.FillPath($bWhite, $pathF1)
$g.DrawPath($pBorderThin, $pathF1)
$g.DrawString('Trained Model', $fontBadge, $bDark, [float]($fx1 + $flW/2), [float]($flY + 16), $sfCenter)
$g.DrawString('Random Forest', $fontCardText, $bMuted, [float]($fx1 + $flW/2), [float]($flY + 33), $sfCenter)
$pathF1.Dispose()
Show-ArrowLine ($fx1 + $flW) ($flY + $flH/2) ($fx1 + $flW + 35) ($flY + $flH/2) ''

# Block 2: TreeSHAP Algorithm
$fx2 = $fx1 + $flW + 35
$pathF2 = Get-RoundPath $fx2 $flY ($flW + 15) $flH 6
$g.FillPath($bWhite, $pathF2)
$g.DrawPath($pBorderThin, $pathF2)
$g.DrawString('TreeSHAP Algorithm', $fontBadge, $bDark, [float]($fx2 + ($flW + 15)/2), [float]($flY + 16), $sfCenter)
$g.DrawString('Exact Tree Attribution', $fontCardText, $bMuted, [float]($fx2 + ($flW + 15)/2), [float]($flY + 33), $sfCenter)
$pathF2.Dispose()
Show-ArrowLine ($fx2 + $flW + 15) ($flY + $flH/2) ($fx2 + $flW + 50) ($flY + $flH/2) ''

# Block 3: SHAP Values
$fx3 = $fx2 + $flW + 50
$pathF3 = Get-RoundPath $fx3 $flY $flW $flH 6
$g.FillPath($bWhite, $pathF3)
$g.DrawPath($pBorderThin, $pathF3)
$g.DrawString('Shapley Values', $fontBadge, $bDark, [float]($fx3 + $flW/2), [float]($flY + 16), $sfCenter)
$g.DrawString('Additive phi_j Matrix', $fontCardText, $bMuted, [float]($fx3 + $flW/2), [float]($flY + 33), $sfCenter)
$pathF3.Dispose()
Show-ArrowLine ($fx3 + $flW) ($flY + $flH/2) ($fx3 + $flW + 35) ($flY + $flH/2) ''

# Block 4: Feature Attribution
$fx4 = $fx3 + $flW + 35
$pathF4 = Get-RoundPath $fx4 $flY ($flW + 15) $flH 6
$g.FillPath($bWhite, $pathF4)
$g.DrawPath($pBorderThin, $pathF4)
$g.DrawString('Feature Attribution', $fontBadge, $bDark, [float]($fx4 + ($flW + 15)/2), [float]($flY + 16), $sfCenter)
$g.DrawString('Global & Local Ranking', $fontCardText, $bMuted, [float]($fx4 + ($flW + 15)/2), [float]($flY + 33), $sfCenter)
$pathF4.Dispose()
Show-ArrowLine ($fx4 + $flW + 15) ($flY + $flH/2) ($fx4 + $flW + 45) ($flY + $flH/2) ''

# Inline Summary Thumbnail at the end of Path A (Matching Grad-CAM Overlay thumbnail in Reference)
$fx5 = $fx4 + $flW + 45
$imgShapPath = 'results/figures/shap_beeswarm.png'
if (Test-Path $imgShapPath) {
    $imgShap = [System.Drawing.Image]::FromFile((Resolve-Path $imgShapPath))
    $g.DrawImage($imgShap, [float]$fx5, [float]($flY - 20), 250.0, 90.0)
    $imgShap.Dispose()
}
$g.DrawRectangle($pBorderThin, [float]$fx5, [float]($flY - 20), 250.0, 90.0)
$g.DrawString('SHAP Beeswarm Summary', $fontBadge, $bDark, [float]($fx5 + 125), [float]($flY + 76), $sfCenter)

# Path B: Counterfactual Analysis (Horizontal Flow)
$g.DrawString('PATH B: Constrained Counterfactual Reasoning', $fontCardTitle, $bTealDark, [float]($botX + 25), 1130.0)

$cfY = 1160
$cfH = 50
$cfW = 160

# CF 1
$cx1 = $botX + 25
$pathC1b = Get-RoundPath $cx1 $cfY $cfW $cfH 6
$g.FillPath($bWhite, $pathC1b)
$g.DrawPath($pBorderThin, $pathC1b)
$g.DrawString('Candidate Site', $fontBadge, $bDark, [float]($cx1 + $cfW/2), [float]($cfY + 16), $sfCenter)
$g.DrawString('P(DC Fast) < 0.76', $fontCardTitle, [System.Drawing.Brushes]::Firebrick, [float]($cx1 + $cfW/2), [float]($cfY + 33), $sfCenter)
$pathC1b.Dispose()
Show-ArrowLine ($cx1 + $cfW) ($cfY + $cfH/2) ($cx1 + $cfW + 40) ($cfY + $cfH/2) ''

# CF 2
$cx2 = $cx1 + $cfW + 40
$pathC2b = Get-RoundPath $cx2 $cfY ($cfW + 20) $cfH 6
$g.FillPath($bWhite, $pathC2b)
$g.DrawPath($pBorderThin, $pathC2b)
$g.DrawString('Constrained What-If', $fontBadge, $bDark, [float]($cx2 + ($cfW + 20)/2), [float]($cfY + 16), $sfCenter)
$g.DrawString('Minimal Feasible Perturbation', $fontCardText, $bMuted, [float]($cx2 + ($cfW + 20)/2), [float]($cfY + 33), $sfCenter)
$pathC2b.Dispose()
Show-ArrowLine ($cx2 + $cfW + 20) ($cfY + $cfH/2) ($cx2 + $cfW + 60) ($cfY + $cfH/2) ''

# CF 3
$cx3 = $cx2 + $cfW + 60
$pathC3b = Get-RoundPath $cx3 $cfY $cfW $cfH 6
$g.FillPath($bWhite, $pathC3b)
$g.DrawPath($pBorderThin, $pathC3b)
$g.DrawString('Feasible Target', $fontBadge, $bDark, [float]($cx3 + $cfW/2), [float]($cfY + 16), $sfCenter)
$g.DrawString('P(DC Fast) >= 0.76', $fontCardTitle, [System.Drawing.Brushes]::ForestGreen, [float]($cx3 + $cfW/2), [float]($cfY + 33), $sfCenter)
$pathC3b.Dispose()
Show-ArrowLine ($cx3 + $cfW) ($cfY + $cfH/2) ($cx3 + $cfW + 40) ($cfY + $cfH/2) ''

# CF 4: Actionable Levers Pill
$cx4 = $cx3 + $cfW + 40
$cx4W = $botX + $botW - 25 - $cx4
$pathC4b = Get-RoundPath $cx4 $cfY $cx4W $cfH 6
$bLevBg = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(240, 253, 250))
$pLevBorder = [System.Drawing.Pen]::new($cTealBorder, [float]1.2)
$g.FillPath($bLevBg, $pathC4b)
$g.DrawPath($pLevBorder, $pathC4b)
$g.DrawString('Actionable Levers', $fontBadge, $bTealDark, [float]($cx4 + $cx4W/2), [float]($cfY + 16), $sfCenter)
$g.DrawString('Hwy Dist  |  City Dist  |  POIs', $fontCardText, $bDark, [float]($cx4 + $cx4W/2), [float]($cfY + 33), $sfCenter)
$pathC4b.Dispose()
$bLevBg.Dispose()
$pLevBorder.Dispose()

# Dividing line between XAI and EIPS
$pDiv = [System.Drawing.Pen]::new($cBorderLight, [float]1.2)
$g.DrawLine($pDiv, ($botX + 25), 1245, ($botX + $botW - 25), 1245)

# -------------------------------------------------------------
# STREAM 3.2: STRATEGIC INFRASTRUCTURE PLANNING LAYER (EIPS)
# Pure MCDA Architecture Pipeline (NOT an ML Model)
# -------------------------------------------------------------
$g.DrawString('3.2 STRATEGIC INFRASTRUCTURE PLANNING: EIPS (Multi-Criteria Decision Analysis)', $fontBlockTitle, $bPurpleDark, [float]($botX + 25), 1265.0)

# EIPS Horizontal Pipeline
$epY = 1300
$epH = 50

# Step 1: 5 Planning Criteria
$epX1 = $botX + 25
$epW1 = 200
$pathEp1 = Get-RoundPath $epX1 $epY $epW1 $epH 6
$g.FillPath($bWhite, $pathEp1)
$g.DrawPath($pBorderThin, $pathEp1)
$g.DrawString('5 Macro Criteria', $fontBadge, $bDark, [float]($epX1 + $epW1/2), [float]($epY + 16), $sfCenter)
$g.DrawString('EV, Gap, Grid, Corridors', $fontCardText, $bMuted, [float]($epX1 + $epW1/2), [float]($epY + 33), $sfCenter)
$pathEp1.Dispose()
Show-ArrowLine ($epX1 + $epW1) ($epY + $epH/2) ($epX1 + $epW1 + 30) ($epY + $epH/2) ''

# Step 2: AHP
$epX2 = $epX1 + $epW1 + 30
$epW2 = 145
$pathEp2 = Get-RoundPath $epX2 $epY $epW2 $epH 6
$g.FillPath($bWhite, $pathEp2)
$g.DrawPath($pBorderThin, $pathEp2)
$g.DrawString('AHP Weighting', $fontBadge, $bPurpleDark, [float]($epX2 + $epW2/2), [float]($epY + 16), $sfCenter)
$g.DrawString('Consistency CR = 0.00089', $fontCardText, $bMuted, [float]($epX2 + $epW2/2), [float]($epY + 33), $sfCenter)
$pathEp2.Dispose()
Show-ArrowLine ($epX2 + $epW2) ($epY + $epH/2) ($epX2 + $epW2 + 30) ($epY + $epH/2) ''

# Step 3: TOPSIS
$epX3 = $epX2 + $epW2 + 30
$epW3 = 145
$pathEp3 = Get-RoundPath $epX3 $epY $epW3 $epH 6
$g.FillPath($bWhite, $pathEp3)
$g.DrawPath($pBorderThin, $pathEp3)
$g.DrawString('TOPSIS Ranking', $fontBadge, $bPurpleDark, [float]($epX3 + $epW3/2), [float]($epY + 16), $sfCenter)
$g.DrawString('Ideal Closeness C_i*', $fontCardText, $bMuted, [float]($epX3 + $epW3/2), [float]($epY + 33), $sfCenter)
$pathEp3.Dispose()
Show-ArrowLine ($epX3 + $epW3) ($epY + $epH/2) ($epX3 + $epW3 + 30) ($epY + $epH/2) ''

# Step 4: EIPS Score
$epX4 = $epX3 + $epW3 + 30
$epW4 = 140
$pathEp4 = Get-RoundPath $epX4 $epY $epW4 $epH 6
$g.FillPath($bWhite, $pathEp4)
$g.DrawPath($pBorderThin, $pathEp4)
$g.DrawString('EIPS Metric', $fontBadge, $bDark, [float]($epX4 + $epW4/2), [float]($epY + 16), $sfCenter)
$g.DrawString('Composite Score [0, 1]', $fontCardText, $bMuted, [float]($epX4 + $epW4/2), [float]($epY + 33), $sfCenter)
$pathEp4.Dispose()
Show-ArrowLine ($epX4 + $epW4) ($epY + $epH/2) ($epX4 + $epW4 + 30) ($epY + $epH/2) ''

# Step 5: District Priorities
$epX5 = $epX4 + $epW4 + 30
$epW5 = $botX + $botW - 25 - $epX5
$pathEp5 = Get-RoundPath $epX5 $epY $epW5 $epH 6
$g.FillPath($bWhite, $pathEp5)
$g.DrawPath($pBorderThin, $pathEp5)
$g.DrawString('District Priority Ranking', $fontBadge, $bPurpleDark, [float]($epX5 + $epW5/2), [float]($epY + 16), $sfCenter)
$g.DrawString('33 Telangana Districts', $fontCardText, $bMuted, [float]($epX5 + $epW5/2), [float]($epY + 33), $sfCenter)
$pathEp5.Dispose()

# Spacious Strategic Tiers Synthesis Bar
$tierY = 1380
$tierH = 160
$pathTier = Get-RoundPath ($botX + 25) $tierY ($botW - 50) $tierH 10
$bTierBg = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(254, 252, 245))
$pTierBorder = [System.Drawing.Pen]::new($cAmberBorder, [float]1.2)
$g.FillPath($bTierBg, $pathTier)
$g.DrawPath($pTierBorder, $pathTier)
$pathTier.Dispose()
$bTierBg.Dispose()

$g.DrawString('STRATEGIC INFRASTRUCTURE DEPLOYMENT TIERS (33 TELANGANA DISTRICTS)', $fontCardTitle, $bAmberDark, [float]($botX + $botW/2), [float]($tierY + 18), $sfCenter)

# 3 Pillars inside the deployment matrix
$pW = ($botW - 100) / 3
$pY = $tierY + 45
$pH = 95

# Pillar 1: Tier 1
$pX1 = $botX + 45
$pathP1 = Get-RoundPath $pX1 $pY $pW $pH 6
$bWhiteBox = New-Object System.Drawing.SolidBrush($cWhite)
$g.FillPath($bWhiteBox, $pathP1)
$g.DrawPath($pBorderLight, $pathP1)
$pathP1.Dispose()
$g.DrawString('Tier 1: High Priority (Metro Core)', $fontBadge, $bPurpleDark, [float]($pX1 + $pW/2), [float]($pY + 16), $sfCenter)
$g.DrawString('Hyderabad, Rangareddy, Medchal', $fontBlockSub, $bDark, [float]($pX1 + $pW/2), [float]($pY + 40), $sfCenter)
$g.DrawString('Immediate Fast-Charging Corridor Deployment', $fontCardText, $bMuted, [float]($pX1 + $pW/2), [float]($pY + 66), $sfCenter)

# Pillar 2: Tier 2
$pX2 = $pX1 + $pW + 15
$pathP2 = Get-RoundPath $pX2 $pY $pW $pH 6
$g.FillPath($bWhiteBox, $pathP2)
$g.DrawPath($pBorderLight, $pathP2)
$pathP2.Dispose()
$g.DrawString('Tier 2: Medium Priority (Transit Nodes)', $fontBadge, $bTealDark, [float]($pX2 + $pW/2), [float]($pY + 16), $sfCenter)
$g.DrawString('Sangareddy, Warangal, Nalgonda (9 Dists)', $fontBlockSub, $bDark, [float]($pX2 + $pW/2), [float]($pY + 40), $sfCenter)
$g.DrawString('Highway Intercity Corridors & Regional Hubs', $fontCardText, $bMuted, [float]($pX2 + $pW/2), [float]($pY + 66), $sfCenter)

# Pillar 3: Tier 3
$pX3 = $pX2 + $pW + 15
$pathP3 = Get-RoundPath $pX3 $pY $pW $pH 6
$g.FillPath($bWhiteBox, $pathP3)
$g.DrawPath($pBorderLight, $pathP3)
$pathP3.Dispose()
$bWhiteBox.Dispose()
$g.DrawString('Tier 3: Baseline Priority (Feeder Dists)', $fontBadge, $bAmberDark, [float]($pX3 + $pW/2), [float]($pY + 16), $sfCenter)
$g.DrawString('21 Peripheral & Agricultural Districts', $fontBlockSub, $bDark, [float]($pX3 + $pW/2), [float]($pY + 40), $sfCenter)
$g.DrawString('Grid Reinforcement & Destination AC Hubs', $fontCardText, $bMuted, [float]($pX3 + $pW/2), [float]($pY + 66), $sfCenter)

$pTierBorder.Dispose()
$pDiv.Dispose()

# =============================================================
# 5. RIGHT COLUMN: OUTPUTS (Matches 4-Class Output Area)
# 5 Visual Output Cards with Schematic Glyphs & Concise Specifications
# =============================================================
$outX = 1270
$outY = 510
$outW = 305
$outH = 1065
Show-MainContainer $outX $outY $outW $outH '' ''

# Inner Header for Outputs
$g.DrawString('OUTPUTS', $fontBoxTitle, $bDark, [float]($outX + $outW/2), [float]($outY + 22), $sfCenter)

# Card 1: ML CLASSIFICATION (Green Card)
$oc1Y = $outY + 48
$oc1H = 185
$pathOc1 = Get-RoundPath ($outX + 14) $oc1Y ($outW - 28) $oc1H 10
$bOc1 = New-Object System.Drawing.SolidBrush($cGreenFill)
$pOc1 = [System.Drawing.Pen]::new($cGreenBorder, [float]1.8)
$g.FillPath($bOc1, $pathOc1)
$g.DrawPath($pOc1, $pathOc1)
$pathOc1.Dispose()

$g.DrawString('1. ML CLASSIFICATION', $fontCardTitle, $bGreenDark, [float]($outX + 24), [float]($oc1Y + 16))
$g.DrawLine($pOc1, ($outX + 24), ($oc1Y + 34), ($outX + $outW - 24), ($oc1Y + 34))

# Visual schematic: Dual Toggle / Decision Needle
$v1X = $outX + 35
$v1Y = $oc1Y + 48
$v1W = $outW - 70
$v1H = 34
$pathV1 = Get-RoundPath $v1X $v1Y $v1W $v1H 6
$g.FillPath($bWhite, $pathV1)
$g.DrawPath($pBorderLight, $pathV1)
$pathV1.Dispose()

$bFast = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(34, 197, 94))
$g.FillRectangle($bFast, [float]($v1X + 4), [float]($v1Y + 4), [float]($v1W/2 - 6), [float]($v1H - 8))
$bFast.Dispose()
$g.DrawString('DC FAST', $fontBadge, $bWhite, [float]($v1X + $v1W/4), [float]($v1Y + 17), $sfCenter)
$g.DrawString('AC SLOW', $fontBadge, $bMuted, [float]($v1X + 3*$v1W/4), [float]($v1Y + 17), $sfCenter)

$g.DrawString('Viability Decision Cut', $fontBadge, $bDark, [float]($outX + $outW/2), [float]($oc1Y + 105), $sfCenter)
$g.DrawString('P(DC Fast | x) >= 0.76', $fontCardTitle, $bGreenDark, [float]($outX + $outW/2), [float]($oc1Y + 130), $sfCenter)
$g.DrawString('131 Fast  |  23 Non-Fast Stations', $fontCardText, $bMuted, [float]($outX + $outW/2), [float]($oc1Y + 155), $sfCenter)
$bOc1.Dispose()
$pOc1.Dispose()

# Card 2: XAI EXPLANATION (Purple Card)
$oc2Y = $oc1Y + $oc1H + 16
$oc2H = 185
$pathOc2 = Get-RoundPath ($outX + 14) $oc2Y ($outW - 28) $oc2H 10
$bOc2 = New-Object System.Drawing.SolidBrush($cPurpleFill)
$pOc2 = [System.Drawing.Pen]::new($cPurpleBorder, [float]1.8)
$g.FillPath($bOc2, $pathOc2)
$g.DrawPath($pOc2, $pathOc2)
$pathOc2.Dispose()

$g.DrawString('2. XAI EXPLANATION', $fontCardTitle, $bPurpleDark, [float]($outX + 24), [float]($oc2Y + 16))
$g.DrawLine($pOc2, ($outX + 24), ($oc2Y + 34), ($outX + $outW - 24), ($oc2Y + 34))

# Visual schematic: 3-bar attribution mini-chart
$v2X = $outX + 35
$v2Y = $oc2Y + 48
$v2W = $outW - 70
$v2H = 34
$pathV2 = Get-RoundPath $v2X $v2Y $v2W $v2H 6
$g.FillPath($bWhite, $pathV2)
$g.DrawPath($pBorderLight, $pathV2)
$pathV2.Dispose()

$bSh1 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(147, 51, 234))
$bSh2 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(192, 132, 252))
$g.FillRectangle($bSh1, [float]($v2X + 10), [float]($v2Y + 6), 140.0, 6.0)
$g.FillRectangle($bSh2, [float]($v2X + 10), [float]($v2Y + 14), 105.0, 6.0)
$g.FillRectangle($bSh1, [float]($v2X + 10), [float]($v2Y + 22), 70.0, 6.0)
$bSh1.Dispose()
$bSh2.Dispose()

$g.DrawString('SHAP Feature Attribution', $fontBadge, $bDark, [float]($outX + $outW/2), [float]($oc2Y + 105), $sfCenter)
$g.DrawString('Global Importance & Local Waterfall', $fontCardText, $bMuted, [float]($outX + $outW/2), [float]($oc2Y + 130), $sfCenter)
$g.DrawString('Exact Additive Shapley Form', $fontBadge, $bPurpleDark, [float]($outX + $outW/2), [float]($oc2Y + 155), $sfCenter)
$bOc2.Dispose()
$pOc2.Dispose()

# Card 3: COUNTERFACTUAL ANALYSIS (Teal Card)
$oc3Y = $oc2Y + $oc2H + 16
$oc3H = 185
$pathOc3 = Get-RoundPath ($outX + 14) $oc3Y ($outW - 28) $oc3H 10
$bOc3 = New-Object System.Drawing.SolidBrush($cTealFill)
$pOc3 = [System.Drawing.Pen]::new($cTealBorder, [float]1.8)
$g.FillPath($bOc3, $pathOc3)
$g.DrawPath($pOc3, $pathOc3)
$pathOc3.Dispose()

$g.DrawString('3. COUNTERFACTUALS', $fontCardTitle, $bTealDark, [float]($outX + 24), [float]($oc3Y + 16))
$g.DrawLine($pOc3, ($outX + 24), ($oc3Y + 34), ($outX + $outW - 24), ($oc3Y + 34))

# Visual schematic: Step Target Crossing
$v3X = $outX + 35
$v3Y = $oc3Y + 48
$v3W = $outW - 70
$v3H = 34
$pathV3 = Get-RoundPath $v3X $v3Y $v3W $v3H 6
$g.FillPath($bWhite, $pathV3)
$g.DrawPath($pBorderLight, $pathV3)
$pathV3.Dispose()

$pStep = [System.Drawing.Pen]::new($cTealBorder, [float]2.0)
$capStep = [System.Drawing.Drawing2D.AdjustableArrowCap]::new(3.5, 3.5, $true)
$pStep.CustomEndCap = $capStep
$g.DrawLine($pStep, [float]($v3X + 15), [float]($v3Y + 24), [float]($v3X + 80), [float]($v3Y + 24))
$g.DrawLine($pStep, [float]($v3X + 80), [float]($v3Y + 24), [float]($v3X + 80), [float]($v3Y + 10))
$g.DrawLine($pStep, [float]($v3X + 80), [float]($v3Y + 10), [float]($v3X + 180), [float]($v3Y + 10))
$capStep.Dispose()
$pStep.Dispose()

$g.DrawString('Actionable What-If Guidance', $fontBadge, $bDark, [float]($outX + $outW/2), [float]($oc3Y + 105), $sfCenter)
$g.DrawString('Threshold Crossing to >= 0.76', $fontCardText, $bMuted, [float]($outX + $outW/2), [float]($oc3Y + 130), $sfCenter)
$g.DrawString('Highway, City-Centre & POI Levers', $fontBadge, $bTealDark, [float]($outX + $outW/2), [float]($oc3Y + 155), $sfCenter)
$bOc3.Dispose()
$pOc3.Dispose()

# Card 4: EIPS PRIORITIZATION (Amber Card)
$oc4Y = $oc3Y + $oc3H + 16
$oc4H = 185
$pathOc4 = Get-RoundPath ($outX + 14) $oc4Y ($outW - 28) $oc4H 10
$bOc4 = New-Object System.Drawing.SolidBrush($cAmberFill)
$pOc4 = [System.Drawing.Pen]::new($cAmberBorder, [float]1.8)
$g.FillPath($bOc4, $pathOc4)
$g.DrawPath($pOc4, $pathOc4)
$pathOc4.Dispose()

$g.DrawString('4. EIPS PRIORITIZATION', $fontCardTitle, $bAmberDark, [float]($outX + 24), [float]($oc4Y + 16))
$g.DrawLine($pOc4, ($outX + 24), ($oc4Y + 34), ($outX + $outW - 24), ($oc4Y + 34))

# Visual schematic: 3-Tier Hierarchy Icon
$v4X = $outX + 35
$v4Y = $oc4Y + 48
$v4W = $outW - 70
$v4H = 34
$pathV4 = Get-RoundPath $v4X $v4Y $v4W $v4H 6
$g.FillPath($bWhite, $pathV4)
$g.DrawPath($pBorderLight, $pathV4)
$pathV4.Dispose()

$bT1 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(217, 119, 6))
$bT2 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(245, 158, 11))
$bT3 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(252, 211, 77))
$g.FillRectangle($bT1, [float]($v4X + 15), [float]($v4Y + 6), 50.0, 22.0)
$g.FillRectangle($bT2, [float]($v4X + 75), [float]($v4Y + 10), 55.0, 18.0)
$g.FillRectangle($bT3, [float]($v4X + 140), [float]($v4Y + 14), 55.0, 14.0)
$bT1.Dispose()
$bT2.Dispose()
$bT3.Dispose()

$g.DrawString('District Strategic Priority Ranks', $fontBadge, $bDark, [float]($outX + $outW/2), [float]($oc4Y + 105), $sfCenter)
$g.DrawString('33 Telangana Districts Ranked', $fontCardText, $bMuted, [float]($outX + $outW/2), [float]($oc4Y + 130), $sfCenter)
$g.DrawString('AHP + TOPSIS Decision Score', $fontBadge, $bAmberDark, [float]($outX + $outW/2), [float]($oc4Y + 155), $sfCenter)
$bOc4.Dispose()
$pOc4.Dispose()

# Card 5: GIS DECISION SUPPORT (Blue Card)
$oc5Y = $oc4Y + $oc4H + 16
$oc5H = 215
$pathOc5 = Get-RoundPath ($outX + 14) $oc5Y ($outW - 28) $oc5H 10
$bOc5 = New-Object System.Drawing.SolidBrush($cBlueFill)
$pOc5 = [System.Drawing.Pen]::new($cBlueBorder, [float]1.8)
$g.FillPath($bOc5, $pathOc5)
$g.DrawPath($pOc5, $pathOc5)
$pathOc5.Dispose()

$g.DrawString('5. GIS DECISION SUPPORT', $fontCardTitle, $bBlueDark, [float]($outX + 24), [float]($oc5Y + 16))
$g.DrawLine($pOc5, ($outX + 24), ($oc5Y + 34), ($outX + $outW - 24), ($oc5Y + 34))

# Visual schematic: Map silhouette with node pins
$v5X = $outX + 35
$v5Y = $oc5Y + 48
$v5W = $outW - 70
$v5H = 46
$pathV5 = Get-RoundPath $v5X $v5Y $v5W $v5H 6
$g.FillPath($bWhite, $pathV5)
$g.DrawPath($pBorderLight, $pathV5)
$pathV5.Dispose()

$pPin = [System.Drawing.Pen]::new($cBlueBorder, [float]1.5)
$bPin1 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(2, 132, 199))
$bPin2 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(234, 88, 12))
$g.DrawLine($pPin, [float]($v5X + 30), [float]($v5Y + 32), [float]($v5X + 100), [float]($v5Y + 18))
$g.DrawLine($pPin, [float]($v5X + 100), [float]($v5Y + 18), [float]($v5X + 170), [float]($v5Y + 28))
$g.FillEllipse($bPin1, [float]($v5X + 24), [float]($v5Y + 26), 12.0, 12.0)
$g.FillEllipse($bPin2, [float]($v5X + 94), [float]($v5Y + 12), 12.0, 12.0)
$g.FillEllipse($bPin1, [float]($v5X + 164), [float]($v5Y + 22), 12.0, 12.0)
$bPin1.Dispose()
$bPin2.Dispose()
$pPin.Dispose()

$g.DrawString('Interactive Planning Dashboard', $fontBadge, $bDark, [float]($outX + $outW/2), [float]($oc5Y + 115), $sfCenter)
$g.DrawString('District Choropleths & Stations', $fontCardText, $bMuted, [float]($outX + $outW/2), [float]($oc5Y + 140), $sfCenter)
$g.DrawString('Production GIS Web Platform', $fontCardTitle, $bBlueDark, [float]($outX + $outW/2), [float]($oc5Y + 172), $sfCenter)
$bOc5.Dispose()
$pOc5.Dispose()

# =============================================================
# SAVE ARTIFACTS
# =============================================================
$destMaster = 'results/figures/system_architecture.png'
$destReport = 'reports/figures/system_architecture.png'
$destBrain = 'C:\Users\Dell\.gemini\antigravity-ide\brain\968d70ae-4fca-407c-85e5-eb5f78da18e4\system_architecture.png'

# Save master PNG
$bmp.Save($destMaster, [System.Drawing.Imaging.ImageFormat]::Png)
Write-Host "Successfully saved master diagram to $destMaster"

# Copy to report directory
if (Test-Path 'reports/figures') {
    Copy-Item -Path $destMaster -Destination $destReport -Force
    Write-Host "Successfully copied to $destReport"
}

# Copy to brain artifact directory
Copy-Item -Path $destMaster -Destination $destBrain -Force
Write-Host "Successfully copied to brain artifact $destBrain"

# Clean up GDI+ resources
$pBorderOuter.Dispose()
$pBorderLight.Dispose()
$pBorderThin.Dispose()
$bDark.Dispose()
$bMuted.Dispose()
$bWhite.Dispose()
$bTealDark.Dispose()
$bPurpleDark.Dispose()
$bCoralDark.Dispose()
$bGreenDark.Dispose()
$bBlueDark.Dispose()
$bAmberDark.Dispose()
$g.Dispose()
$bmp.Dispose()
Write-Host "Square 1:1 Architecture diagram generation complete!"
