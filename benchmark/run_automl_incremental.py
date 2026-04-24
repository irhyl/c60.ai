"""
run_automl_incremental.py — Incremental AutoML comparison benchmark.

Writes one CSV row per fold immediately so a killed run keeps partial data.
Generates report and plots at the end from whatever data was saved.

Usage
-----
  python benchmark/run_automl_incremental.py
  python benchmark/run_automl_incremental.py --fast
  python benchmark/run_automl_incremental.py --no-c60
"""
import sys
import time
import argparse
import csv
import traceback
import warnings
from pathlib import Path
from copy import deepcopy

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.model_selection import StratifiedKFold


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true",
                   help="2 folds x 2 seeds instead of 3 x 3.")
    p.add_argument("--extended", action="store_true",
                   help="Add pendigits, letter, waveform (OpenML datasets).")
    p.add_argument("--no-c60", action="store_true")
    p.add_argument("--time-budget", type=int, default=60)
    p.add_argument("--output", default="benchmark/results")
    return p.parse_args()


def _eval_fold(estimator, X_train, y_train, X_test, y_test,
               sys_name, ds_name, seed, fold_idx):
    try:
        from sklearn.base import clone
        est = clone(estimator)
    except Exception:
        est = deepcopy(estimator)

    if hasattr(est, "random_seed"):
        est.random_seed = seed * 100 + fold_idx

    t0 = time.perf_counter()
    try:
        est.fit(X_train, y_train)
        acc = float(est.score(X_test, y_test))
        elapsed = time.perf_counter() - t0
        return acc, elapsed, None
    except Exception:
        elapsed = time.perf_counter() - t0
        tb = traceback.format_exc(limit=2)
        return None, elapsed, tb


def _report(csv_path: Path, highlight: str = "C60.ai"):
    """Print mean ± std table and ranks from saved CSV."""
    import csv as csvmod
    from collections import defaultdict

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            acc = row["accuracy"]
            if acc and acc != "":
                try:
                    rows.append({
                        "system": row["system"],
                        "dataset": row["dataset"],
                        "accuracy": float(acc),
                    })
                except ValueError:
                    pass

    if not rows:
        print("No valid results to report.")
        return

    # Group by (system, dataset)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["system"], r["dataset"])].append(r["accuracy"])

    datasets = sorted({r["dataset"] for r in rows})
    systems  = sorted({r["system"]  for r in rows})

    # Mean table
    print("\n" + "=" * 80)
    print("AUTOML COMPARISON — Mean Accuracy (±Std)")
    print("=" * 80)
    col_w = 12
    header = f"{'System':<22}" + "".join(f"{d[:col_w]:>{col_w}}" for d in datasets) + f"{'Mean':>{col_w}}"
    print(header)
    print("-" * len(header))

    system_means = {}
    for sys in systems:
        row_means = []
        parts = [f"{sys:<22}"]
        for ds in datasets:
            accs = groups.get((sys, ds), [])
            if accs:
                m, s = np.mean(accs), np.std(accs)
                parts.append(f"{m:.4f}±{s:.3f}"[:col_w].rjust(col_w))
                row_means.append(m)
            else:
                parts.append(f"{'—':>{col_w}}")
        mean_row = np.mean(row_means) if row_means else float("nan")
        system_means[sys] = mean_row
        prefix = "** " if sys == highlight else "   "
        print(prefix + "".join(parts) + f"{mean_row:>{col_w}.4f}")

    # Rank table
    print("\nAverage Rank (lower = better):")
    ranks = {}
    for ds in datasets:
        ds_means = [(sys, np.mean(groups[(sys, ds)])) for sys in systems if groups.get((sys, ds))]
        ds_means.sort(key=lambda x: -x[1])
        for r, (sys, _) in enumerate(ds_means, 1):
            ranks[sys] = ranks.get(sys, []) + [r]
    avg_ranks = {sys: np.mean(v) for sys, v in ranks.items()}
    for sys, ar in sorted(avg_ranks.items(), key=lambda x: x[1]):
        marker = " **" if sys == highlight else ""
        print(f"  {sys:<22} {ar:.2f}{marker}")


def _plots(csv_path: Path, out: Path):
    """Save comparison and rank PNG charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import csv as csvmod
        from collections import defaultdict

        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csvmod.DictReader(f):
                if row["accuracy"]:
                    try:
                        rows.append((row["system"], row["dataset"], float(row["accuracy"])))
                    except ValueError:
                        pass

        groups = defaultdict(list)
        for sys, ds, acc in rows:
            groups[(sys, ds)].append(acc)

        datasets = sorted({ds for _, ds, _ in rows})
        systems  = sorted({sys for sys, _, _ in rows})

        means = {sys: [np.mean(groups.get((sys, ds), [np.nan])) for ds in datasets]
                 for sys in systems}

        # Bar chart
        x = np.arange(len(datasets))
        width = 0.8 / max(len(systems), 1)
        fig, ax = plt.subplots(figsize=(max(14, len(datasets) * 2), 7))
        for i, sys in enumerate(systems):
            vals = means[sys]
            offset = (i - len(systems) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width * 0.9, label=sys)

        ax.set_xlabel("Dataset")
        ax.set_ylabel("Accuracy")
        ax.set_title("AutoML Systems — Accuracy Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
        ax.set_ylim(0.5, 1.02)
        fig.tight_layout()
        fig.savefig(out / "comparison_automl.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}/comparison_automl.png")

        # Rank chart
        ranks = {}
        for ds in datasets:
            ds_means = [(sys, np.mean(groups.get((sys, ds), [np.nan]))) for sys in systems]
            ds_means = [(s, v) for s, v in ds_means if not np.isnan(v)]
            ds_means.sort(key=lambda x: -x[1])
            for r, (sys, _) in enumerate(ds_means, 1):
                ranks.setdefault(sys, []).append(r)
        avg_ranks = sorted(
            [(sys, np.mean(v)) for sys, v in ranks.items()],
            key=lambda x: x[1]
        )
        sys_names = [s for s, _ in avg_ranks]
        rank_vals  = [r for _, r in avg_ranks]

        fig2, ax2 = plt.subplots(figsize=(max(12, len(systems)), 6))
        colors = ["#2196F3" if s == "C60.ai" else "#90CAF9" for s in sys_names]
        ax2.barh(sys_names[::-1], rank_vals[::-1], color=colors[::-1])
        ax2.set_xlabel("Average Rank (lower = better)")
        ax2.set_title("AutoML Systems — Average Rank")
        ax2.axvline(1, color="gray", linestyle="--", linewidth=0.8)
        fig2.tight_layout()
        fig2.savefig(out / "ranks_automl.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {out}/ranks_automl.png")

    except ImportError:
        print("matplotlib not available — skipping plots.")
    except Exception as e:
        print(f"Plot error: {e}")


def main():
    args = _parse()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results_automl.csv"

    from benchmark.runner import load_datasets, load_extended_datasets
    from benchmark.automl_baselines import all_automl_systems

    print(f"Loading systems (time_budget={args.time_budget}s)...")
    systems = all_automl_systems(time_budget=args.time_budget)

    if not args.no_c60:
        from benchmark.baselines import C60Estimator
        systems.append(("C60.ai", C60Estimator(
            population_size=15, max_generations=8, cv=3,
            eval_timeout=30.0, random_seed=42,
        )))

    print(f"Systems ({len(systems)}): {', '.join(n for n, _ in systems)}\n")
    datasets = load_datasets()
    if args.extended:
        print("Fetching extended OpenML datasets (cached after first download)...")
        datasets = datasets + load_extended_datasets(max_samples=8000)
    print(f"Datasets ({len(datasets)}): {[n for n, *_ in datasets]}\n")

    n_folds = 2 if args.fast else 3
    seeds   = [0, 1] if args.fast else [0, 1, 2]
    n_total = len(systems) * len(datasets) * len(seeds) * n_folds
    print(f"CV: {n_folds}-fold × {len(seeds)} seeds = "
          f"{n_folds * len(seeds)} evals per (system, dataset)")
    print(f"Total evaluations: {n_total}\n")

    # Check if we're resuming from an existing CSV
    done_keys = set()
    if csv_path.exists():
        import pandas as pd
        existing = pd.read_csv(csv_path)
        for _, row in existing.iterrows():
            done_keys.add((row["system"], row["dataset"],
                           int(row["seed"]), int(row["fold"])))
        print(f"Resuming: {len(done_keys)} folds already complete.\n")

    # Open CSV in append mode
    csv_exists = csv_path.exists()
    f_out = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=[
        "system", "dataset", "seed", "fold", "accuracy", "fit_time_s"
    ])
    if not csv_exists:
        writer.writeheader()
        f_out.flush()

    done = len(done_keys)
    try:
        for ds_name, X, y in datasets:
            print(f"\n{'='*60}")
            print(f"Dataset: {ds_name}  ({X.shape[0]} × {X.shape[1]}, "
                  f"{len(np.unique(y))} classes)")
            print("="*60)

            for sys_name, estimator in systems:
                sys_accs = []

                for seed in seeds:
                    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                         random_state=seed)
                    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
                        key = (sys_name, ds_name, seed, fold_idx)
                        if key in done_keys:
                            continue

                        X_tr, X_te = X[tr], X[te]
                        y_tr, y_te = y[tr], y[te]

                        acc, elapsed, err = _eval_fold(
                            estimator, X_tr, y_tr, X_te, y_te,
                            sys_name, ds_name, seed, fold_idx
                        )
                        done += 1
                        pct = 100 * done / n_total

                        writer.writerow({
                            "system":     sys_name,
                            "dataset":    ds_name,
                            "seed":       seed,
                            "fold":       fold_idx,
                            "accuracy":   f"{acc:.6f}" if acc is not None else "",
                            "fit_time_s": f"{elapsed:.2f}",
                        })
                        f_out.flush()

                        status = f"{acc:.4f}" if acc is not None else "FAIL"
                        if err:
                            short_err = err.strip().split("\n")[-1][:60]
                            print(f"  [{pct:5.1f}%] {sys_name:<22} | {ds_name:<12} "
                                  f"s{seed}f{fold_idx} | {status} ({elapsed:.1f}s) "
                                  f"ERR: {short_err}")
                        else:
                            print(f"  [{pct:5.1f}%] {sys_name:<22} | {ds_name:<12} "
                                  f"s{seed}f{fold_idx} | {status} ({elapsed:.1f}s)")
                        if acc is not None:
                            sys_accs.append(acc)

                if sys_accs:
                    print(f"  -> {sys_name}: mean={np.mean(sys_accs):.4f} "
                          f"std={np.std(sys_accs):.4f}  (n={len(sys_accs)})")
    finally:
        f_out.close()

    print(f"\nRaw results -> {csv_path}")
    _report(csv_path)
    _plots(csv_path, out)
    print("\nAll done.")


if __name__ == "__main__":
    main()
