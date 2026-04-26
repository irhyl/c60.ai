"""
run_real_automl.py — Benchmark FLAML against C60.ai on 4 core datasets.

Uses the same StratifiedKFold protocol as the main benchmark so results are
directly comparable.  FLAML is given a 30-second wall-clock budget per fold
(matches C60.ai's observed median fit time on the 4 core datasets).

Note: TPOT is excluded because it fails to load on Python 3.13 (matplotlib
DLL MemoryError when spawning worker processes).

Output: benchmark/results/results_real_automl.csv
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
warnings.filterwarnings("ignore")

N_FOLDS   = 3
SEEDS     = [0, 1, 2]
TIME_BUDGET_S = 30         # seconds per fold — matches C60.ai's observed median fit time

DATASETS = {
    "iris":          load_iris(),
    "wine":          load_wine(),
    "breast_cancer": load_breast_cancer(),
    "digits":        load_digits(),
}

# ---------------------------------------------------------------------------
# System factories
# ---------------------------------------------------------------------------

def make_flaml(seed: int):
    from flaml import AutoML
    return AutoML()


SYSTEMS = {
    "FLAML": make_flaml,
}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

out = Path("benchmark/results")
out.mkdir(parents=True, exist_ok=True)
csv_path = out / "results_real_automl.csv"

already_done: set = set()
if csv_path.exists():
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            already_done.add((row["system"], row["dataset"], int(row["seed"]), int(row["fold"])))
    print(f"Resuming — {len(already_done)} folds already done.")

total = len(SYSTEMS) * len(DATASETS) * N_FOLDS * len(SEEDS)
done = 0

with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["system", "dataset", "seed", "fold", "accuracy", "fit_time_s"]
    )
    if csv_path.stat().st_size == 0:
        writer.writeheader()

    for sys_name, factory in SYSTEMS.items():
        for ds_name, dataset in DATASETS.items():
            X, y = dataset.data, dataset.target
            for seed in SEEDS:
                for fold in range(N_FOLDS):
                    done += 1
                    key = (sys_name, ds_name, seed, fold)
                    if key in already_done:
                        print(f"[{done}/{total}] SKIP {sys_name} {ds_name} s={seed} f={fold}")
                        continue

                    print(
                        f"[{done}/{total}] {sys_name} {ds_name} seed={seed} fold={fold}",
                        flush=True,
                    )
                    t0 = time.perf_counter()
                    try:
                        skf = StratifiedKFold(
                            n_splits=N_FOLDS, shuffle=True, random_state=seed
                        )
                        splits = list(skf.split(X, y))
                        train_idx, test_idx = splits[fold]
                        X_tr, X_te = X[train_idx], X[test_idx]
                        y_tr, y_te = y[train_idx], y[test_idx]

                        clf = factory(seed)
                        if sys_name == "FLAML":
                            clf.fit(
                                X_tr, y_tr,
                                task="classification",
                                time_budget=TIME_BUDGET_S,
                                verbose=0,
                                n_jobs=1,
                            )
                        else:
                            clf.fit(X_tr, y_tr)

                        acc = float(np.mean(clf.predict(X_te) == y_te))
                        elapsed = time.perf_counter() - t0

                        writer.writerow({
                            "system":     sys_name,
                            "dataset":    ds_name,
                            "seed":       seed,
                            "fold":       fold,
                            "accuracy":   f"{acc:.6f}",
                            "fit_time_s": f"{elapsed:.2f}",
                        })
                        f.flush()
                        print(f"  acc={acc:.4f}  time={elapsed:.1f}s", flush=True)

                    except Exception:
                        elapsed = time.perf_counter() - t0
                        print(f"  FAILED after {elapsed:.1f}s")
                        traceback.print_exc()

print(f"\nDone. Results: {csv_path}")
