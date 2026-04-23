# organize_project.py — run once from your project root
import os, shutil
from pathlib import Path

root = Path(".")

# Create folders
(root / "scripts").mkdir(exist_ok=True)
(root / "docs").mkdir(exist_ok=True)

# Move one-time utility scripts to scripts/
to_scripts = [
    "collect_data.py", "collect_ddg.py", "cleaning_data.py",
    "augment_data.py", "auto_sort.py", "separation.py",
    "bootstrap_generated_gallery.py"
]
for f in to_scripts:
    if (root / f).exists():
        shutil.move(str(root / f), str(root / "scripts" / f))
        print(f"Moved: {f} → scripts/")

# Move docs
if (root / "PHASE2_RUNBOOK.md").exists():
    shutil.move("PHASE2_RUNBOOK.md", "docs/PHASE2_RUNBOOK.md")
    print("Moved: PHASE2_RUNBOOK.md → docs/")

# Delete pycache
for p in root.rglob("__pycache__"):
    shutil.rmtree(p)
    print(f"Deleted: {p}")

print("\nDone. Now create .gitignore and push to GitHub.")