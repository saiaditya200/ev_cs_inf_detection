from pathlib import Path

readme_path = Path("README.md")
readme_india_path = Path("README_INDIA_WIDE.md")

with open(readme_path, "r", encoding="utf-8") as f:
    text = f.read()

target = "The high-resolution vector system architecture diagram is saved at [`system_architecture_india_wide.svg`](file:///c:/Users/Dell/Downloads/EV_Charging/system_architecture_india_wide.svg) (and mirrored in [`results/india_wide/figures/system_architecture_india_wide.svg`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures/system_architecture_india_wide.svg))."
replacement = "The high-resolution vector system architecture diagram is saved at [`system_architecture_india_wide.svg`](file:///c:/Users/Dell/Downloads/EV_Charging/system_architecture_india_wide.svg) (mirrored in [`results/india_wide/figures/system_architecture_india_wide.svg`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures/system_architecture_india_wide.svg)). An ultra-high-resolution JPEG version optimized for WhatsApp, presentations, and social media sharing is available at [`system_architecture_india_wide.jpg`](file:///c:/Users/Dell/Downloads/EV_Charging/system_architecture_india_wide.jpg) (mirrored in [`results/india_wide/figures/system_architecture_india_wide.jpg`](file:///c:/Users/Dell/Downloads/EV_Charging/results/india_wide/figures/system_architecture_india_wide.jpg))."

text = text.replace(target, replacement)

# In directory tree
dir_target = "├── system_architecture_india_wide.svg                             # Pan-India System Architecture SVG"
dir_replacement = """├── system_architecture_india_wide.svg                             # Pan-India System Architecture SVG (Light Theme)
├── system_architecture_india_wide.jpg                             # High-Res Architecture JPG (Social Media / WhatsApp)"""

text = text.replace(dir_target, dir_replacement)

# Also in results/india_wide/figures in directory tree
fig_target = "│       ├── system_architecture_india_wide.svg                 # High-Res Architecture Diagram"
fig_replacement = """│       ├── system_architecture_india_wide.svg                 # High-Res Architecture Diagram (Vector SVG)
│       ├── system_architecture_india_wide.jpg                 # Ultra-High-Res Architecture Diagram (JPEG)"""

text = text.replace(fig_target, fig_replacement)

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(text)

with open(readme_india_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Updated both README files with JPG references!")

