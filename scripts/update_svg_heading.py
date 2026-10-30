from pathlib import Path
import subprocess
from PIL import Image

svg_path = Path("system_architecture_india_wide.svg")
svg_fig_path = Path("results/india_wide/figures/system_architecture_india_wide.svg")

with open(svg_path, "r", encoding="utf-8") as f:
    svg_text = f.read()

# Replace header text
old_header = """<text x="800" y="48" text-anchor="middle" font-family="'Times New Roman', Times, Georgia, serif" font-size="34" font-weight="bold" fill="#0f172a" letter-spacing="2">SYSTEM ARCHITECTURE — PAN-INDIA MASTER FRAMEWORK</text>"""
new_header = """<text x="800" y="48" text-anchor="middle" font-family="'Times New Roman', Times, Georgia, serif" font-size="34" font-weight="bold" fill="#0f172a" letter-spacing="2">SYSTEM ARCHITECTURE</text>"""

assert old_header in svg_text, "old_header string not found in SVG!"
svg_text = svg_text.replace(old_header, new_header)

with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg_text)

with open(svg_fig_path, "w", encoding="utf-8") as f:
    f.write(svg_text)

print("Updated both SVGs successfully with ONLY SYSTEM ARCHITECTURE heading.")

# Re-render JPG using headless Chrome
chrome_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
png_out = Path("temp_arch.png").resolve()
abs_svg = svg_path.resolve()

cmd = [
    chrome_exe,
    "--headless=new",
    "--disable-gpu",
    "--window-size=1600,1600",
    "--force-device-scale-factor=2",
    "--hide-scrollbars",
    f"--screenshot={png_out}",
    f"file:///{abs_svg.as_posix()}"
]
res = subprocess.run(cmd, capture_output=True, text=True)
if png_out.exists():
    im = Image.open(png_out)
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        rgb_im = bg
    else:
        rgb_im = im.convert("RGB")
    
    rgb_im.save("system_architecture_india_wide.jpg", "JPEG", quality=95, optimize=True)
    rgb_im.save("results/india_wide/figures/system_architecture_india_wide.jpg", "JPEG", quality=95, optimize=True)
    png_out.unlink()
    print("Re-generated high-resolution JPGs successfully!")
else:
    print("Failed to generate PNG screenshot:", res.stderr)
