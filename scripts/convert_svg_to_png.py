import os
import subprocess
from PIL import Image

def render_svg():
    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    browser = chrome if os.path.exists(chrome) else edge

    svg_path = os.path.abspath("system_architecture.svg")
    out_png = os.path.abspath("EV_Charging/reports/figures/system_architecture_paper.png")
    
    url = "file:///" + svg_path.replace("\\", "/")
    
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=1.25",
        "--window-size=1600,1600",
        f"--screenshot={out_png}",
        url
    ]
    
    print("Converting SVG to PNG using:", browser)
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Done. Return code:", res.returncode)
    
    if os.path.exists(out_png):
        im = Image.open(out_png)
        print("Generated image size:", im.size, "File size:", os.path.getsize(out_png), "bytes")
        
        # Crop or verify exactly 1600x1600 or scaled
        # Copy to other figure locations
        copies = [
            "EV_Charging/reports/figures/system_architecture.png",
            "EV_Charging/results/figures/system_architecture.png",
            "reports/figures/system_architecture_paper.png",
            "reports/figures/system_architecture.png"
        ]
        for c in copies:
            os.makedirs(os.path.dirname(c), exist_ok=True)
            im.save(c, "PNG")
            print(f"Mirrored to: {c}")

if __name__ == "__main__":
    render_svg()

