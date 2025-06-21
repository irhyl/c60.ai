# C60.ai Auto Executor Scaffold
# =============================
# This script loads each phase markdown, sends it to GPT-4, and writes code/notebooks accordingly.
# Assumes you have OpenAI API access and 7 markdown files in /phases directory.

import os
import openai
from pathlib import Path

# ======================
# Config
# ======================
openai.api_key = os.getenv("OPENAI_API_KEY")
PHASE_DIR = Path("phases")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL = "gpt-4"

# ======================
# Utilities
# ======================
def load_markdown(phase_path):
    return phase_path.read_text()

def prompt_gpt(instruction_text):
    system_msg = (
        "You are C60 Architect, a senior AGI systems engineer. Execute this markdown AutoML build phase. "
        "Generate real Python files, Jupyter notebooks, or bash commands as needed. Do NOT explain anything. Just output executable code."
    )
    
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": instruction_text}
        ]
    )
    return response['choices'][0]['message']['content']

def write_outputs(phase_num, gpt_output):
    phase_label = f"phase_{phase_num:02d}"
    base = OUTPUT_DIR / phase_label
    base.mkdir(exist_ok=True)

    # Split by language markers
    blocks = gpt_output.split("```")
    count = 0
    for block in blocks:
        if block.startswith("python"):
            count += 1
            code = block.replace("python\n", "")
            (base / f"{phase_label}_code_{count}.py").write_text(code)
        elif block.startswith("json"):
            count += 1
            code = block.replace("json\n", "")
            (base / f"{phase_label}_config_{count}.json").write_text(code)
        elif block.startswith("bash"):
            count += 1
            code = block.replace("bash\n", "")
            (base / f"{phase_label}_cmd_{count}.sh").write_text(code)
        elif block.startswith("markdown") or block.startswith("md"):
            count += 1
            code = block.replace("markdown\n", "").replace("md\n", "")
            (base / f"{phase_label}_doc_{count}.md").write_text(code)
        elif block.startswith("ipynb") or block.startswith("json"):
            count += 1
            code = block.replace("ipynb\n", "")
            (base / f"{phase_label}_notebook_{count}.ipynb").write_text(code)
        else:
            # generic fallback
            count += 1
            (base / f"{phase_label}_raw_{count}.txt").write_text(block)

# ======================
# Execution Loop
# ======================
if __name__ == "__main__":
    phase_files = sorted(PHASE_DIR.glob("phase_*.md"))
    for i, md_file in enumerate(phase_files, 1):
        print(f"\n>>> Executing {md_file.name}")
        markdown = load_markdown(md_file)
        gpt_output = prompt_gpt(markdown)
        write_outputs(i, gpt_output)
        print(f"✓ Phase {i} outputs written.")

    print("\n✅ All phases executed. Check the /output folder.")
