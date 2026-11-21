"""
LexBrief AI - Academic Submission Zip Creator
Project ID: CSEAIML2026-P13
Packages the entire project into a submission-ready ZIP file.
"""
import os
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ZIP = BASE_DIR / 'LexBrief_AI_Legal_Research_Assistant.zip'

EXCLUDE_DIRS = {'__pycache__', '.git', '.pytest_cache', 'instance', 'uploads', 'food_quality', 'dataset', 'models'}
EXCLUDE_EXTS = {'.pyc', '.zip', '.tmp'}

def create_submission_zip():
    print(f"Creating submission zip: {OUTPUT_ZIP.name}")
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(BASE_DIR):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                p = Path(root) / file
                if p.suffix in EXCLUDE_EXTS or file.startswith('.'):
                    continue
                # Relative path inside zip
                rel_path = p.relative_to(BASE_DIR)
                print(f"  + Adding {rel_path}")
                zf.write(p, rel_path)

    print(f"\n[+] Successfully generated: {OUTPUT_ZIP}")
    print(f"    Size: {round(OUTPUT_ZIP.stat().st_size / 1024, 2)} KB")

if __name__ == '__main__':
    create_submission_zip()
