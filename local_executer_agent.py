# C60.ai Local Auto Executor
# =============================
# This script loads each phase markdown, extracts code blocks, and writes them to files
# without requiring OpenAI API calls

import os
import re
import time
import subprocess
from pathlib import Path

# ======================
# Setup & Config
# ======================
PHASE_DIR = Path("phases")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ======================
# Git Automation
# ======================
def git_commit_phase(phase_num):
    try:
        msg = f"auto-commit: phase_{phase_num:02d} outputs"
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", msg], check=True)
        print(f"✓ Git commit created for phase {phase_num:02d}.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Git commit failed: {e}")

# ======================
# Core Functions
# ======================
def load_markdown(phase_path):
    try:
        # Try reading with UTF-8 first, fall back to other encodings if needed
        try:
            return phase_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try with different encodings if UTF-8 fails
            for encoding in ['utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    return phase_path.read_text(encoding=encoding)
                except UnicodeDecodeError:
                    continue
            raise
    except Exception as e:
        print(f"❌ Error reading {phase_path}: {str(e)}")
        raise

def extract_code_blocks(markdown_text):
    """Extract code blocks from markdown text"""
    # Pattern to match code blocks with language specifier
    pattern = r"```(\w+)\n(.*?)```"
    blocks = re.findall(pattern, markdown_text, re.DOTALL)
    
    # Also look for code blocks without language specifier
    simple_pattern = r"```\n(.*?)```"
    simple_blocks = re.findall(simple_pattern, markdown_text, re.DOTALL)
    
    # Add simple blocks with "txt" as language
    for block in simple_blocks:
        blocks.append(("txt", block))
    
    return blocks

def write_outputs(phase_num, markdown_text):
    phase_label = f"phase_{phase_num:02d}"
    try:
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        base = OUTPUT_DIR / phase_label
        base.mkdir(exist_ok=True, parents=True)
        
        # Extract and process code blocks
        blocks = extract_code_blocks(markdown_text)
        count = 0
        
        for lang, code in blocks:
            count += 1
            lang = lang.lower()
            
            # Map language to file extension
            ext_map = {
                "python": "py",
                "py": "py",
                "bash": "sh",
                "sh": "sh",
                "javascript": "js",
                "js": "js",
                "json": "json",
                "html": "html",
                "css": "css",
                "ipynb": "ipynb",
                "txt": "txt"
            }
            
            ext = ext_map.get(lang, "txt")
            
            # Special handling for Jupyter notebooks
            if lang == "ipynb":
                output_file = base / f"{phase_label}_notebook_{count}.ipynb"
            else:
                output_file = base / f"{phase_label}_{lang}_{count}.{ext}"
            
            output_file.write_text(code.strip(), encoding='utf-8')
            print(f"  - Created {output_file}")
                
        return True
        
    except Exception as e:
        print(f"❌ Error in write_outputs: {str(e)}")
        raise

# ======================
# Main Execution Loop
# ======================
if __name__ == "__main__":
    try:
        print("🚀 Starting C60.ai Local Auto Executor...")
        phase_files = sorted(PHASE_DIR.glob("phase_*.md"))

        if not phase_files:
            print(f"❌ No markdown files found in: {PHASE_DIR}/")
            exit(1)

        for phase_path in phase_files:
            try:
                phase_num = int(phase_path.stem.split('_')[1])
                print(f"\n🧠 Processing {phase_path.name}...")
                markdown = load_markdown(phase_path)
                write_outputs(phase_num, markdown)
                git_commit_phase(phase_num)
                print(f"✅ Finished phase {phase_num}")
            except Exception as e:
                print(f"❌ Error in {phase_path.name}: {e}")
                continue

        print("\n🏁 All phases completed successfully.")

    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        exit(1)