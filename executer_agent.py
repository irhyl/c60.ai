# C60.ai Auto Executor Scaffold
# =============================
# This script loads each phase markdown, sends it to GPT-4, and writes code/notebooks accordingly.

import os
import time
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APITimeoutError

# ======================
# Setup & Config
# ======================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

client = OpenAI(api_key=api_key)

PHASE_DIR = Path("phases")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL = "gpt-3.5-turbo"  # Changed from gpt-4o to gpt-3.5-turbo which often has different quota limits

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

def prompt_gpt(instruction_text, max_retries=5, initial_delay=2):
    system_msg = (
        "You are C60 Architect, a senior AGI systems engineer. Execute this markdown AutoML build phase. "
        "Generate real Python files, Jupyter notebooks, or bash commands as needed. Do NOT explain anything. Just output executable code."
    )

    attempt = 0
    delay = initial_delay

    while attempt < max_retries:
        attempt += 1
        try:
            start_time = time.time()
            print(f"\nAttempt {attempt}/{max_retries} — sending prompt...")
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": instruction_text}
                ]
            )
            full_response = response.choices[0].message.content
            total_time = time.time() - start_time
            print(f"✅ Response received in {total_time:.1f} seconds — {len(full_response)} characters.")
            return full_response
        except (APIConnectionError, APITimeoutError) as e:
            print(f"⚠️ API Error: {e}. Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            error_msg = str(e)
            if "insufficient_quota" in error_msg:
                print(f"⚠️ OpenAI API Quota Error: Your account has reached its quota limit.")
                print(f"Trying with reduced prompt size and waiting {delay:.1f} seconds...")
                # Try with a shorter prompt if possible
                if len(instruction_text) > 2000:
                    instruction_text = instruction_text[:2000] + "...[truncated for quota reasons]"
            else:
                print(f"⚠️ Unexpected error: {e}")
            
            if attempt >= max_retries:
                raise
            print(f"Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            delay *= 2

    raise Exception(f"❌ Failed to get response after {max_retries} attempts")

def write_outputs(phase_num, gpt_output):
    phase_label = f"phase_{phase_num:02d}"
    try:
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        base = OUTPUT_DIR / phase_label
        base.mkdir(exist_ok=True, parents=True)
        
        # Process the GPT output
        blocks = gpt_output.split("```")
        count = 0
        
        for i in range(1, len(blocks), 2):
            if i >= len(blocks):
                break
                
            block = blocks[i].strip()
            if not block:
                continue
                
            try:
                # Check for language specifier (e.g., ```python)
                lines = block.split('\n')
                if len(lines) > 1 and lines[0].lower() in ['python', 'bash', 'sh']:
                    ext = lines[0].lower()
                    if ext == 'bash':
                        ext = 'sh'
                    code = '\n'.join(lines[1:])  # Remove language specifier
                    count += 1
                    output_file = base / f"{phase_label}_{ext}_{count}.{ext}"
                    output_file.write_text(code.strip(), encoding='utf-8')
                    print(f"  - Created {output_file}")
                    
                elif block.lower().startswith('python'):
                    # Handle case where language specifier is on same line as opening backticks
                    code = block[6:].strip()  # Remove 'python' prefix
                    count += 1
                    output_file = base / f"{phase_label}_python_{count}.py"
                    output_file.write_text(code, encoding='utf-8')
                    print(f"  - Created {output_file}")
                    
                elif block.lower().startswith(('bash', 'sh')):
                    # Handle bash/sh scripts
                    code = block[4:].strip()  # Remove 'bash' or 'sh' prefix
                    count += 1
                    output_file = base / f"{phase_label}_shell_{count}.sh"
                    output_file.write_text(code, encoding='utf-8')
                    print(f"  - Created {output_file}")
                    
                elif 'ipynb' in block.lower():
                    # Special handling for Jupyter notebooks
                    count += 1
                    code = block.replace("ipynb\n", "")
                    output_file = base / f"{phase_label}_notebook_{count}.ipynb"
                    output_file.write_text(code, encoding='utf-8')
                    print(f"  - Created {output_file}")
                    
                else:
                    # Fallback for unclassified code blocks
                    count += 1
                    output_file = base / f"{phase_label}_raw_{count}.txt"
                    output_file.write_text(block, encoding='utf-8')
                    print(f"  - Created {output_file}")
                    
            except Exception as e:
                print(f"  ⚠️ Error processing code block: {str(e)}")
                continue
                
        return True
        
    except Exception as e:
        print(f"❌ Error in write_outputs: {str(e)}")
        raise

# ======================
# Main Execution Loop
# ======================
if __name__ == "__main__":
    try:
        print("🚀 Starting C60.ai Auto Executor...")
        print(f"Using OpenAI model: {MODEL}")
        phase_files = sorted(PHASE_DIR.glob("phase_*.md"))

        if not phase_files:
            print(f"❌ No markdown files found in: {PHASE_DIR}/")
            exit(1)

        for phase_path in phase_files:
            try:
                phase_num = int(phase_path.stem.split('_')[1])
                print(f"\n🧠 Executing {phase_path.name}...")
                markdown = load_markdown(phase_path)
                gpt_output = prompt_gpt(markdown)
                write_outputs(phase_num, gpt_output)
                git_commit_phase(phase_num)
                print(f"✅ Finished phase {phase_num}")
            except Exception as e:
                print(f"❌ Error in {phase_path.name}: {e}")
                continue

        print("\n🏁 All phases completed successfully.")

    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        exit(1)
