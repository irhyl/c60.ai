"""
run_ablation.py — Ablation study: C60.ai-Full vs C60.ai-HPO.

Compares the complete C60.ai system (structural search + HPO) against
a population-based HPO baseline (structural_search=False) that evolves
hyperparameters only, keeping topology fixed.  This isolates the
contribution of graph-structural genetic operators.

Additionally collects real per-generation fitness curves for every
(variant, dataset, seed) run and saves them to:
    benchmark/results/evolution_curves.json

Usage
-----
    python benchmark/run_ablation.py
    python benchmark/run_ablation.py --fast       # 2 folds x 2 seeds
    python benchmark/run_ablation.py --datasets iris digits waveform
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from c60.evolution.engine import EvolutionEngine

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def load_core_datasets():
    datasets = {}
    for name, loader in [
        ("iris", load_iris),
        ("wine", load_wine),
        ("breast_cancer", load_breast_cancer),
        ("digits", load_digits),
    ]:
        d = loader()
        datasets[name] = (d.data, d.target)
    return datasets


def load_extended_datasets():
    try:
        from sklearn.datasets import fetch_openml
        ext = {}
        for name, oid in [("waveform", 60), ("pendigits", 32), ("letter", 6)]:
            try:
                d = fetch_openml(data_id=oid, as_frame=False, parser="auto")
                X = d.data.astype(float)
                y = LabelEncoder().fit_transform(d.target)
                ext[name] = (X[:8000], y[:8000])
                print(f"  Loaded {name}: {X.shape}")
            except Exception as e:
                print(f"  Skipped {name}: {e}")
        return ext
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# C60 estimator wrappers
# ---------------------------------------------------------------------------

COMMON_KWARGS = dict(
    population_size=15,
    max_generations=8,
    task="classification",
    cv=3,
    eval_timeout=30,
    complexity_penalty=0.002,
)


def make_c60_full(seed: int) -> EvolutionEngine:
    return EvolutionEngine(**COMMON_KWARGS, structural_search=True, random_seed=seed)


def make_c60_hpo(seed: int) -> EvolutionEngine:
    return EvolutionEngine(**COMMON_KWARGS, structural_search=False, random_seed=seed)


VARIANTS = {
    "C60.ai-Full": make_c60_full,
    "C60.ai-HPO":  make_c60_hpo,
}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_ablation(
    datasets: dict,
    n_folds: int,
    seeds: list[int],
    csv_path: Path,
    curves_path: Path,
    already_done: set,
):
    existing_rows = []
    if csv_path.exists():
        with open(csv_path) as f:
            existing_rows = list(csv.DictReader(f))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["variant", "dataset", "seed", "fold", "accuracy", "fit_time_s"]
        )
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)

        curves: dict = {}
        if curves_path.exists():
            with open(curves_path) as cf:
                curves = json.load(cf)

        total = len(VARIANTS) * len(datasets) * len(seeds) * n_folds
        done = 0

        for variant_name, factory in VARIANTS.items():
            for ds_name, (X, y) in datasets.items():
                for seed in seeds:
                    for fold in range(n_folds):
                        key = (variant_name, ds_name, seed, fold)
                        if key in already_done:
                            done += 1
                            continue

                        print(
                            f"[{done+1}/{total}] {variant_name},{ds_name},"
                            f"seed={seed},fold={fold}",
                            flush=True,
                        )
                        t0 = time.perf_counter()

                        try:
                            skf = StratifiedKFold(
                                n_splits=n_folds, shuffle=True, random_state=seed
                            )
                            splits = list(skf.split(X, y))
                            train_idx, test_idx = splits[fold]
                            X_tr, X_te = X[train_idx], X[test_idx]
                            y_tr, y_te = y[train_idx], y[test_idx]

                            engine = factory(seed)
                            best = engine.fit(X_tr, y_tr)
                            best.fit(X_tr, y_tr)
                            acc = float(np.mean(best.predict(X_te) == y_te))

                            # Collect evolution curve
                            curve_key = f"{variant_name}|{ds_name}|{seed}|{fold}"
                            curves[curve_key] = engine.history().best_scores()

                            elapsed = time.perf_counter() - t0
                            row = {
                                "variant": variant_name,
                                "dataset": ds_name,
                                "seed": seed,
                                "fold": fold,
                                "accuracy": f"{acc:.6f}",
                                "fit_time_s": f"{elapsed:.2f}",
                            }
                            writer.writerow(row)
                            f.flush()

                            # Persist curves after every fold
                            with open(curves_path, "w") as cf:
                                json.dump(curves, cf)

                            print(
                                f"  acc={acc:.4f}  time={elapsed:.1f}s  "
                                f"cache_hit={engine.cache_stats()['hit_rate']:.1%}",
                                flush=True,
                            )
                        except Exception:
                            elapsed = time.perf_counter() - t0
                            print(f"  FAILED after {elapsed:.1f}s")
                            traceback.print_exc()

                        done += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="2-fold × 2 seeds instead of 3 × 3")
    parser.add_argument("--no-extended", action="store_true",
                        help="Skip pendigits, letter, waveform")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Restrict to specific datasets")
    args = parser.parse_args()

    n_folds = 2 if args.fast else 3
    seeds   = [0, 1] if args.fast else [0, 1, 2]

    out = Path("benchmark/results")
    out.mkdir(parents=True, exist_ok=True)
    csv_path    = out / "results_ablation.csv"
    curves_path = out / "evolution_curves.json"

    print("Loading datasets...")
    datasets = load_core_datasets()
    if not args.no_extended:
        print("Loading extended datasets (OpenML)...")
        datasets.update(load_extended_datasets())

    if args.datasets:
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    # Work out what's already done
    already_done: set = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                already_done.add((
                    row["variant"], row["dataset"],
                    int(row["seed"]), int(row["fold"])
                ))
        print(f"Resuming — {len(already_done)} folds already complete.")

    print(
        f"\nRunning ablation: {list(VARIANTS)} × "
        f"{list(datasets)} × {n_folds} folds × {seeds} seeds\n"
    )

    run_ablation(datasets, n_folds, seeds, csv_path, curves_path, already_done)

    print(f"\nDone. Results: {csv_path}")
    print(f"Evolution curves: {curves_path}")

    # Summary table
    import pandas as pd
    df = pd.read_csv(csv_path)
    print("\n--- Ablation Summary ---")
    summary = (
        df.groupby(["variant", "dataset"])["accuracy"]
        .apply(lambda x: np.mean(x.astype(float)) * 100)
        .unstack("dataset")
        .round(2)
    )
    print(summary.to_string())

    means = df.groupby("variant")["accuracy"].apply(
        lambda x: np.mean(x.astype(float)) * 100
    ).round(2)
    print("\nOverall means:")
    print(means.to_string())


if __name__ == "__main__":
    main()
