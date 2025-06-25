"""
C60.ai Executor Agent
Reads markdown build phases, sends them to an LLM (e.g., GPT-4), and saves the generated code to output/phase_XX/.
"""
import os
from pathlib import Path

PHASES_DIR = Path('phases')
OUTPUT_DIR = Path('output')

def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    PHASES_DIR.mkdir(exist_ok=True)

def list_phases():
    return sorted([f for f in PHASES_DIR.glob('phase_*.md')])

def main():
    ensure_dirs()
    phases = list_phases()
    print(f"Found {len(phases)} phase files.")
    print("(LLM integration not implemented in this scaffold.)")
    # TODO: Integrate with OpenAI or other LLM API
    # For each phase: read markdown, call LLM, save code to output/phase_xx/

if __name__ == "__main__":
    main()
