"""
Seed results_automl.csv with C60.ai rows from results_extended.csv,
so the incremental runner can resume without re-running C60.ai.
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

out = Path("benchmark/results")
out.mkdir(parents=True, exist_ok=True)
csv_path = out / "results_automl.csv"

ext = pd.read_csv(out / "results_extended.csv")
ALL_DATASETS = ["iris", "wine", "breast_cancer", "digits", "pendigits", "letter", "waveform"]
c60 = ext[
    (ext["system"] == "C60.ai") &
    (ext["dataset"].isin(ALL_DATASETS))
].copy()

c60.to_csv(csv_path, index=False)
print(f"Seeded {len(c60)} C60.ai rows into {csv_path}")
