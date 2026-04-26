"""
run_hpo_only.py — Run C60.ai-HPO variant on core datasets only.

Writes to benchmark/results/results_ablation_hpo.csv (separate from
results_ablation.csv to avoid file-lock conflicts with any running process).
Uses identical StratifiedKFold(3) splits so results are directly comparable
to C60.ai-Full rows already in results_ablation.csv.
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from c60.evolution.engine import EvolutionEngine

warnings.filterwarnings("ignore")

COMMON_KWARGS = dict(
    population_size=15,
    max_generations=8,
    task="classification",
    cv=3,
    eval_timeout=30,
    complexity_penalty=0.002,
)

N_FOLDS = 3
SEEDS = [0, 1, 2]

DATASETS = {
    "iris":         load_iris(),
    "wine":         load_wine(),
    "breast_cancer": load_breast_cancer(),
    "digits":       load_digits(),
}

out = Path("benchmark/results")
out.mkdir(parents=True, exist_ok=True)
csv_path = out / "results_ablation_hpo.csv"

already_done: set = set()
if csv_path.exists():
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            already_done.add((row["dataset"], int(row["seed"]), int(row["fold"])))
    print(f"Resuming — {len(already_done)} folds already done.")

total = len(DATASETS) * N_FOLDS * len(SEEDS)
done = 0

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["variant", "dataset", "seed", "fold", "accuracy", "fit_time_s"]
    )
    if csv_path.stat().st_size == 0:
        writer.writeheader()

    for ds_name, dataset in DATASETS.items():
        X, y = dataset.data, dataset.target
        for seed in SEEDS:
            for fold in range(N_FOLDS):
                done += 1
                key = (ds_name, seed, fold)
                if key in already_done:
                    print(f"[{done}/{total}] SKIP {ds_name} seed={seed} fold={fold}")
                    continue

                print(
                    f"[{done}/{total}] C60.ai-HPO {ds_name} seed={seed} fold={fold}",
                    flush=True,
                )
                t0 = time.perf_counter()
                try:
                    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                    splits = list(skf.split(X, y))
                    train_idx, test_idx = splits[fold]
                    X_tr, X_te = X[train_idx], X[test_idx]
                    y_tr, y_te = y[train_idx], y[test_idx]

                    engine = EvolutionEngine(**COMMON_KWARGS, structural_search=False, random_seed=seed)
                    best = engine.fit(X_tr, y_tr)
                    best.fit(X_tr, y_tr)
                    acc = float(np.mean(best.predict(X_te) == y_te))
                    elapsed = time.perf_counter() - t0

                    writer.writerow({
                        "variant": "C60.ai-HPO",
                        "dataset": ds_name,
                        "seed": seed,
                        "fold": fold,
                        "accuracy": f"{acc:.6f}",
                        "fit_time_s": f"{elapsed:.2f}",
                    })
                    f.flush()
                    print(f"  acc={acc:.4f}  time={elapsed:.1f}s", flush=True)
                except Exception:
                    elapsed = time.perf_counter() - t0
                    print(f"  FAILED after {elapsed:.1f}s")
                    traceback.print_exc()

print("\nDone. Results saved to", csv_path)
