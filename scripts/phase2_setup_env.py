"""
Phase 2: Environment Setup
===========================
Creates .env file with the API key and ensures .gitignore protects secrets.
Usage: python scripts/phase2_setup_env.py <API_KEY>
"""

import sys
import os
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
GITIGNORE_PATH = os.path.join(PROJECT_ROOT, ".gitignore")


def create_env(api_key: str) -> None:
    """Create or update .env file with the API key."""
    env_content = f"""# ============================================================
# EV Charging Infrastructure Planning - Environment Variables
# ============================================================
# SECURITY: This file is in .gitignore — NEVER commit to version control
# Created: Phase 2 automated setup

# Google Maps Platform
GOOGLE_MAPS_API_KEY={api_key}

# Project settings
PROJECT_FOCUS_STATE=Telangana
PROJECT_FOCUS_COUNTRY=India
"""
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write(env_content)
    print(f"[OK] .env file created at: {ENV_PATH}")
    print(f"     Key length: {len(api_key)} chars, starts with: {api_key[:8]}...")


def create_gitignore() -> None:
    """Create or update .gitignore to protect secrets and data files."""
    gitignore_content = """# ============================================================
# EV Charging Infrastructure Planning - .gitignore
# ============================================================

# Secrets
.env
*.env
.env.*
config/secrets/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/
eggs/
*.egg

# Virtual environments
venv/
.venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
desktop.ini

# Data (large files)
data/raw/*.json
data/raw/*.csv
data/raw/*.xlsx
!data/raw/.gitkeep

# Model outputs
models/*.pkl
models/*.joblib
models/*.h5

# Jupyter
.ipynb_checkpoints/

# Logs
*.log
logs/
"""
    with open(GITIGNORE_PATH, "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    print(f"[OK] .gitignore created at: {GITIGNORE_PATH}")


def validate_api_key_format(key: str) -> bool:
    """Basic format validation for Google API keys."""
    if not key:
        print("[FAIL] API key is empty")
        return False
    if key == "YOUR_API_KEY_HERE":
        print("[FAIL] Placeholder key detected — provide a real key")
        return False
    if len(key) < 20:
        print(f"[WARN] API key seems too short ({len(key)} chars)")
        return False
    if " " in key:
        print("[FAIL] API key contains spaces — check for copy errors")
        return False
    print(f"[OK] API key format looks valid ({len(key)} chars)")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/phase2_setup_env.py <YOUR_GOOGLE_MAPS_API_KEY>")
        print("\nAlternatively, set GOOGLE_MAPS_API_KEY env var first.")
        sys.exit(1)

    api_key = sys.argv[1].strip()

    print("=" * 60)
    print("PHASE 2: ENVIRONMENT SETUP")
    print("=" * 60)

    # Validate
    if not validate_api_key_format(api_key):
        sys.exit(1)

    # Create files
    create_env(api_key)
    create_gitignore()

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("Next: Run phase2_hyderabad_test.py to validate the key")


if __name__ == "__main__":
    main()
