"""
run_automl.py — AutoML comparison benchmark runner.

Compares C60.ai against 12 AutoML systems on the core 4 sklearn datasets
using the same protocol as the main benchmark (3-fold × 3 seeds by default).

Usage
-----
  python benchmark/run_automl.py                  # core 4 datasets, 3x3
  python benchmark/run_automl.py --fast           # 2 folds x 2 seeds
  python benchmark/run_automl.py --save-plots     # write PNG charts
  python benchmark/run_automl.py --no-c60         # AutoML systems only
"""
import sys
import argparse
import io
import contextlib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true",
                   help="2 folds x 2 seeds instead of 3 x 3.")
    p.add_argument("--no-c60", action="store_true",
                   help="Skip C60.ai (AutoML baselines only).")
    p.add_argument("--save-plots", action="store_true",
                   help="Save matplotlib PNG charts.")
    p.add_argument("--time-budget", type=int, default=120,
                   help="Per-system time budget in seconds (default 120).")
    p.add_argument("--output", default="benchmark/results",
                   help="Output directory.")
    return p.parse_args()


def main():
    args = _parse()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    from benchmark.runner import BenchmarkRunner, load_datasets
    from benchmark.automl_baselines import all_automl_systems
    from benchmark.baselines import C60Estimator
    from benchmark.report import ResultsReporter

    # ── Systems ──────────────────────────────────────────────────────────────
    print(f"Loading AutoML systems (time_budget={args.time_budget}s)...")
    systems = all_automl_systems(time_budget=args.time_budget)

    if not args.no_c60:
        c60 = C60Estimator(
            population_size=15,
            max_generations=8,
            cv=3,
            eval_timeout=30.0,
            random_seed=42,
        )
        systems = systems + [("C60.ai", c60)]

    print(f"Systems ({len(systems)}): {', '.join(n for n, _ in systems)}\n")

    # ── Datasets ─────────────────────────────────────────────────────────────
    datasets = load_datasets()
    print(f"Datasets ({len(datasets)}): {[n for n, *_ in datasets]}\n")

    # ── CV scheme ─────────────────────────────────────────────────────────────
    n_folds = 2 if args.fast else 3
    seeds = [0, 1] if args.fast else [0, 1, 2]
    runner = BenchmarkRunner(n_folds=n_folds, seeds=seeds, verbose=True)

    print(f"CV: {n_folds}-fold x {len(seeds)} seeds = "
          f"{n_folds * len(seeds)} evals per (system, dataset)\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    csv_path = out / "results_automl.csv"
    df = runner.run(systems, datasets)
    df.to_csv(csv_path, index=False)
    print(f"\nRaw results -> {csv_path}")

    # ── Report ────────────────────────────────────────────────────────────────
    reporter = ResultsReporter(df, highlight_system="C60.ai")
    reporter.print_full_report()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reporter.print_full_report()
    (out / "report_automl.txt").write_text(buf.getvalue(), encoding="utf-8")
    (out / "summary_automl.md").write_text(
        reporter.summary_table(fmt="markdown"), encoding="utf-8"
    )
    print(f"Report -> {out}/report_automl.txt")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.save_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            reporter.render_comparison(
                figsize=(max(14, len(datasets) * 2), 9)
            ).savefig(out / "comparison_automl.png", dpi=150, bbox_inches="tight")
            reporter.render_ranks().savefig(
                out / "ranks_automl.png", dpi=150, bbox_inches="tight"
            )
            try:
                reporter.render_winloss().savefig(
                    out / "winloss_automl.png", dpi=150, bbox_inches="tight"
                )
            except ValueError as e:
                print(f"  (winloss skipped: {e})")
            print(f"Plots -> {out}/comparison_automl.png")
        except ImportError:
            print("matplotlib not available — skipping plots.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
