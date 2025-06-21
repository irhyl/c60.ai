"""
C60.ai Ollama Executor
======================
This script uses Ollama to process phase markdown files and generate code.
It provides robust error handling, progress tracking, and detailed reporting.
"""

import os
import re
import time
import json
import logging
import subprocess
import requests
from pathlib import Path
import argparse
from datetime import datetime

# ======================
# Setup & Config
# ======================
PHASE_DIR = Path("phases")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Progress tracking file
PROGRESS_FILE = Path("progress.json")

# Default model - you can change this to any model available in Ollama
DEFAULT_MODEL = "codellama"

# Configure logging
log_file = LOGS_DIR / f"ollama_executor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================
# Progress Tracking
# ======================
def load_progress():
    """Load progress from progress.json file"""
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except json.JSONDecodeError:
            logger.warning("Invalid progress file. Starting fresh.")
    return {"completed_phases": [], "last_run": None, "model_used": None}

def save_progress(phase_num, model):
    """Save progress to progress.json file"""
    progress = load_progress()
    if phase_num not in progress["completed_phases"]:
        progress["completed_phases"].append(phase_num)
    progress["completed_phases"].sort()
    progress["last_run"] = datetime.now().isoformat()
    progress["model_used"] = model
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))
    logger.info(f"Progress saved. Completed phases: {progress['completed_phases']}")

def get_next_phase():
    """Get the next phase to process based on progress"""
    progress = load_progress()
    completed = set(progress["completed_phases"])
    all_phases = []
    
    for phase_path in sorted(PHASE_DIR.glob("phase_*.md")):
        try:
            phase_num = int(phase_path.stem.split('_')[1])
            all_phases.append(phase_num)
        except (ValueError, IndexError):
            continue
    
    for phase in sorted(all_phases):
        if phase not in completed:
            return phase
    
    return None

# ======================
# Git Automation
# ======================
def git_commit_phase(phase_num):
    try:
        msg = f"auto-commit: phase_{phase_num:02d} outputs"
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", msg], check=True)
        logger.info(f"✓ Git commit created for phase {phase_num:02d}.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ Git commit failed: {e}")

# ======================
# Ollama API Functions
# ======================
def check_ollama_running():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def list_available_models():
    """List available models in Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except requests.exceptions.ConnectionError:
        return []

def pull_model(model_name):
    """Pull a model if it's not already available"""
    available_models = list_available_models()
    if model_name not in available_models:
        logger.info(f"Model {model_name} not found locally. Pulling from Ollama...")
        try:
            subprocess.run(["ollama", "pull", model_name], check=True)
            logger.info(f"Successfully pulled {model_name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull {model_name}: {e}")
            return False
    return True

def prompt_ollama(model, prompt, max_retries=3, initial_delay=1, system_prompt=None):
    """Send a prompt to Ollama and get the response"""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 4096
        }
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retries} — sending prompt to {model}...")
            start_time = time.time()
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            full_response = result.get("response", "")
            total_time = time.time() - start_time
            logger.info(f"✅ Response received in {total_time:.1f} seconds — {len(full_response)} characters.")
            return full_response
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ API Error: {e}. Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            logger.warning(f"⚠️ Unexpected error: {e}")
            if attempt >= max_retries:
                raise
            logger.info(f"Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
            delay *= 2
    
    error_msg = f"❌ Failed to get response after {max_retries} attempts"
    logger.error(error_msg)
    raise Exception(error_msg)

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
        logger.error(f"❌ Error reading {phase_path}: {str(e)}")
        raise

def extract_code_from_response(response_text):
    """Extract code blocks from the model's response"""
    # Pattern to match code blocks with language specifier
    pattern = r"```(\w+)\n(.*?)```"
    blocks = re.findall(pattern, response_text, re.DOTALL)
    
    # Also look for code blocks without language specifier
    simple_pattern = r"```\n(.*?)```"
    simple_blocks = re.findall(simple_pattern, response_text, re.DOTALL)
    
    # Add simple blocks with "txt" as language
    for block in simple_blocks:
        blocks.append(("txt", block))
    
    return blocks

def write_outputs(phase_num, response_text):
    phase_label = f"phase_{phase_num:02d}"
    try:
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        base = OUTPUT_DIR / phase_label
        base.mkdir(exist_ok=True, parents=True)
        
        # Extract and process code blocks
        blocks = extract_code_from_response(response_text)
        count = 0
        file_list = []
        
        if not blocks:
            logger.warning(f"⚠️ No code blocks found in the response for {phase_label}")
            # Save the full response as a text file
            output_file = base / f"{phase_label}_full_response.txt"
            output_file.write_text(response_text, encoding='utf-8')
            logger.info(f"  - Saved full response to {output_file}")
            file_list.append(str(output_file))
            return file_list
        
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
                "txt": "txt",
                "markdown": "md",
                "md": "md"
            }
            
            ext = ext_map.get(lang, "txt")
            
            # Special handling for Jupyter notebooks
            if lang == "ipynb":
                output_file = base / f"{phase_label}_notebook_{count}.ipynb"
            else:
                output_file = base / f"{phase_label}_{lang}_{count}.{ext}"
            
            output_file.write_text(code.strip(), encoding='utf-8')
            logger.info(f"  - Created {output_file}")
            file_list.append(str(output_file))
                
        return file_list
        
    except Exception as e:
        logger.error(f"❌ Error in write_outputs: {str(e)}")
        raise

def create_summary_report(phase_num, files_created, start_time, end_time):
    """Create a summary report for the phase execution"""
    phase_label = f"phase_{phase_num:02d}"
    summary = {
        "phase": phase_num,
        "phase_label": phase_label,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "files_created": files_created
    }
    
    # Save summary to JSON
    summary_file = OUTPUT_DIR / phase_label / f"{phase_label}_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    logger.info(f"Summary report saved to {summary_file}")
    return summary

def process_phase(phase_path, model):
    """Process a single phase markdown file"""
    try:
        phase_num = int(phase_path.stem.split('_')[1])
        logger.info(f"\n🧠 Processing {phase_path.name}...")
        
        start_time = datetime.now()
        
        # Load the markdown content
        markdown = load_markdown(phase_path)
        
        # Create the system prompt
        system_prompt = """You are C60 Architect, a senior AGI systems engineer specialized in AutoML systems. 
Your task is to implement the code for a specific phase of the C60.ai project based on the markdown description.
Generate complete, functional code that follows best practices and is ready to be executed.
Focus on producing high-quality, well-structured code with proper error handling and documentation."""
        
        # Create the prompt for the model
        prompt = f"""I need you to implement the code for the following phase of the C60.ai AutoML project.
For each file or component mentioned in the phase description, generate the actual code.

Here is the phase description:

{markdown}

Important guidelines:
1. Generate complete, executable code for each file mentioned
2. Include proper error handling, logging, and documentation
3. Follow Python best practices (PEP 8)
4. Ensure the code integrates well with the rest of the C60.ai system
5. Wrap each code block in triple backticks with the appropriate language specifier
6. Do NOT include explanations between code blocks - just output the code

For example:

```python
# Python code here
```

```bash
# Bash commands here
```

Generate all the necessary files mentioned in the phase description.
"""
        
        # Send the prompt to Ollama
        response = prompt_ollama(model, prompt, system_prompt=system_prompt)
        
        # Write the outputs
        files_created = write_outputs(phase_num, response)
        
        # Create summary report
        end_time = datetime.now()
        create_summary_report(phase_num, files_created, start_time, end_time)
        
        # Save progress
        save_progress(phase_num, model)
        
        # Commit the changes
        git_commit_phase(phase_num)
        
        logger.info(f"✅ Finished phase {phase_num}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in {phase_path.name}: {e}")
        return False

# ======================
# Main Execution Loop
# ======================
def main():
    parser = argparse.ArgumentParser(description="C60.ai Ollama Executor")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--phase", type=int, help="Process only a specific phase number")
    parser.add_argument("--resume", action="store_true", help="Resume from the last completed phase")
    parser.add_argument("--list-progress", action="store_true", help="List progress and exit")
    parser.add_argument("--reset-progress", action="store_true", help="Reset progress tracking")
    args = parser.parse_args()
    
    try:
        logger.info("🚀 Starting C60.ai Ollama Executor...")
        
        # Handle progress listing
        if args.list_progress:
            progress = load_progress()
            logger.info(f"Completed phases: {progress['completed_phases']}")
            logger.info(f"Last run: {progress['last_run']}")
            logger.info(f"Model used: {progress['model_used']}")
            return 0
        
        # Handle progress reset
        if args.reset_progress:
            if PROGRESS_FILE.exists():
                PROGRESS_FILE.unlink()
            logger.info("Progress tracking reset.")
            return 0
        
        # Check if Ollama is running
        if not check_ollama_running():
            logger.error("❌ Ollama server is not running. Please start Ollama first.")
            return 1
        
        # Pull the model if needed
        if not pull_model(args.model):
            logger.error(f"❌ Failed to pull model {args.model}. Exiting.")
            return 1
        
        logger.info(f"Using Ollama model: {args.model}")
        
        # Determine which phases to process
        if args.phase:
            phase_path = PHASE_DIR / f"phase_{args.phase:02d}.md"
            if not phase_path.exists():
                logger.error(f"❌ Phase file {phase_path} not found.")
                return 1
            phase_files = [phase_path]
        elif args.resume:
            next_phase = get_next_phase()
            if next_phase is None:
                logger.info("All phases have been completed. Nothing to resume.")
                return 0
            
            phase_path = PHASE_DIR / f"phase_{next_phase:02d}.md"
            if not phase_path.exists():
                logger.error(f"❌ Phase file {phase_path} not found.")
                return 1
            
            logger.info(f"Resuming from phase {next_phase}")
            phase_files = [phase_path]
            
            # Also include all subsequent phases
            all_phases = sorted([
                int(p.stem.split('_')[1]) 
                for p in PHASE_DIR.glob("phase_*.md")
                if p.stem.split('_')[1].isdigit()
            ])
            
            for phase in all_phases:
                if phase > next_phase:
                    next_path = PHASE_DIR / f"phase_{phase:02d}.md"
                    if next_path.exists():
                        phase_files.append(next_path)
        else:
            phase_files = sorted(PHASE_DIR.glob("phase_*.md"))
        
        if not phase_files:
            logger.error(f"❌ No markdown files found in: {PHASE_DIR}/")
            return 1
        
        # Process each phase
        for phase_path in phase_files:
            success = process_phase(phase_path, args.model)
            if not success:
                logger.error(f"❌ Failed to process {phase_path}. Stopping execution.")
                return 1
        
        logger.info("\n🏁 All phases completed successfully.")
        return 0
        
    except Exception as e:
        logger.error(f"\n💥 Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
