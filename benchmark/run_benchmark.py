#!/usr/bin/env python3
"""
run_benchmark.py — Execute the full C60.ai comparative benchmark.

Usage
-----
  python benchmark/run_benchmark.py                     # full benchmark
  python benchmark/run_benchmark.py --fast              # reduced folds/seeds
  python benchmark/run_benchmark.py --save-plots        # also write PNG files
  python benchmark/run_benchmark.py --output results/   # custom output dir

Output
------
  benchmark/results/results.csv        raw per-fold records
  benchmark/results/report.txt         full text report
  benchmark/results/comparison.png     grouped bar chart (if --save-plots)
  benchmark/results/ranks.png          rank bar chart   (if --save-plots)
  benchmark/results/winloss.png        win/tie/loss chart (if --save-plots)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── make sure both src/ and repo root are importable ───────────────────────
_repo = Path(__file__).resolve().parent.parent
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the C60.ai benchmark against sklearn baselines."
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Use 2 folds × 2 seeds instead of 5 × 5 (much faster, less stable).",
    )
    p.add_argument(
        "--save-plots",
        action="store_true",
        help="Save matplotlib figures to PNG files.",
    )
    p.add_argument(
        "--no-c60",
        action="store_true",
        help="Skip C60.ai (baseline comparison only — useful for quick sanity check).",
    )
    p.add_argument(
        "--output",
        default=str(Path(__file__).parent / "results"),
        help="Directory to write output files (default: benchmark/results/).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-fold progress output.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Build system list ────────────────────────────────────────────────
    from benchmark.baselines import all_systems, make_baselines

    if args.no_c60:
        systems = make_baselines()
        print("Running baselines only (--no-c60 flag set).")
    else:
        systems = all_systems()
        print(f"Running {len(systems)} systems: "
              f"{', '.join(n for n, _ in systems)}\n")

    # ── 2. Configure runner ─────────────────────────────────────────────────
    from benchmark.runner import BenchmarkRunner, load_datasets

    n_folds = 2 if args.fast else 5
    seeds   = [0, 1] if args.fast else list(range(5))

    runner = BenchmarkRunner(
        n_folds=n_folds,
        seeds=seeds,
        verbose=not args.quiet,
    )

    datasets = load_datasets()
    print(f"Datasets   : {[n for n, *_ in datasets]}")
    print(f"CV scheme  : {n_folds}-fold × {len(seeds)} seeds "
          f"= {n_folds * len(seeds)} evals per (system, dataset)")
    print(f"Output dir : {out_dir}\n")

    # ── 3. Run ──────────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    df = runner.run(systems, datasets)
    elapsed = time.perf_counter() - t_start

    print(f"\nBenchmark complete in {elapsed/60:.1f} min "
          f"({elapsed:.0f}s total).\n")

    # ── 4. Save raw results ─────────────────────────────────────────────────
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Raw results saved -> {csv_path}")

    # ── 5. Generate report ──────────────────────────────────────────────────
    from benchmark.report import ResultsReporter

    reporter = ResultsReporter(df, highlight_system="C60.ai")
    reporter.print_full_report()

    report_path = out_dir / "report.txt"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reporter.print_full_report()
    report_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Text report saved  -> {report_path}")

    # ── 6. Save markdown table ──────────────────────────────────────────────
    md_path = out_dir / "summary.md"
    md_path.write_text(reporter.summary_table(fmt="markdown"), encoding="utf-8")
    print(f"Markdown table     -> {md_path}")

    # ── 7. Plots ────────────────────────────────────────────────────────────
    if args.save_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")

            fig_cmp = reporter.render_comparison()
            fig_cmp.savefig(out_dir / "comparison.png", dpi=150, bbox_inches="tight")
            fig_cmp.clf()

            fig_rnk = reporter.render_ranks()
            fig_rnk.savefig(out_dir / "ranks.png", dpi=150, bbox_inches="tight")
            fig_rnk.clf()

            try:
                fig_wl = reporter.render_winloss()
                fig_wl.savefig(out_dir / "winloss.png", dpi=150, bbox_inches="tight")
                fig_wl.clf()
            except ValueError as e:
                print(f"  (skipping winloss plot: {e})")

            print(f"Plots saved        → {out_dir}/{{comparison,ranks,winloss}}.png")
        except ImportError:
            print("matplotlib not available — skipping plots.")

    print("\nDone.")


if __name__ == "__main__":
    main()
