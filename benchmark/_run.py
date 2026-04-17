"""
Standalone benchmark runner.

Usage
-----
  python benchmark/_run.py                   # core 4 datasets (full)
  python benchmark/_run.py --extended        # +6 large OpenML datasets
  python benchmark/_run.py --fast            # 2 folds x 2 seeds
  python benchmark/_run.py --no-c60         # baselines only
  python benchmark/_run.py --save-plots      # write PNG charts
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
    p.add_argument("--extended", action="store_true",
                   help="Add 6 larger OpenML datasets.")
    p.add_argument("--fast", action="store_true",
                   help="2 folds x 2 seeds instead of 3 x 3.")
    p.add_argument("--no-c60", action="store_true",
                   help="Skip C60.ai (baselines only).")
    p.add_argument("--save-plots", action="store_true",
                   help="Save matplotlib PNG charts.")
    p.add_argument("--output", default="benchmark/results",
                   help="Output directory.")
    return p.parse_args()


def main():
    args = _parse()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    from benchmark.runner import BenchmarkRunner, load_datasets, load_extended_datasets
    from benchmark.baselines import all_systems, make_baselines
    from benchmark.report import ResultsReporter

    # ── Systems ──────────────────────────────────────────────────────────────
    systems = make_baselines() if args.no_c60 else all_systems()
    print(f"Systems ({len(systems)}): {', '.join(n for n,_ in systems)}\n")

    # ── Datasets ─────────────────────────────────────────────────────────────
    datasets = load_datasets()
    if args.extended:
        print("Fetching extended OpenML datasets (cached after first download)...")
        datasets = datasets + load_extended_datasets(max_samples=8000)
    print(f"Datasets ({len(datasets)}): {[n for n,*_ in datasets]}\n")

    # ── CV scheme: use fewer folds/seeds for big datasets ────────────────────
    n_folds = 2 if args.fast else 3
    seeds   = [0, 1] if args.fast else [0, 1, 2]
    runner  = BenchmarkRunner(n_folds=n_folds, seeds=seeds, verbose=True)

    print(f"CV: {n_folds}-fold x {len(seeds)} seeds = "
          f"{n_folds*len(seeds)} evals per (system, dataset)\n")

    # ── Run ───────────────────────────────────────────────────────────────────
    suffix = "_extended" if args.extended else "_full"
    csv_path = out / f"results{suffix}.csv"

    df = runner.run(systems, datasets)
    df.to_csv(csv_path, index=False)
    print(f"\nRaw results -> {csv_path}")

    # ── Report ────────────────────────────────────────────────────────────────
    reporter = ResultsReporter(df, highlight_system="C60.ai")
    reporter.print_full_report()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reporter.print_full_report()
    (out / f"report{suffix}.txt").write_text(buf.getvalue(), encoding="utf-8")
    (out / f"summary{suffix}.md").write_text(
        reporter.summary_table(fmt="markdown"), encoding="utf-8"
    )
    print(f"Report -> {out}/report{suffix}.txt")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.save_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            reporter.render_comparison(
                figsize=(max(14, len(datasets) * 2), 7)
            ).savefig(out / f"comparison{suffix}.png", dpi=150, bbox_inches="tight")
            reporter.render_ranks().savefig(
                out / f"ranks{suffix}.png", dpi=150, bbox_inches="tight"
            )
            try:
                reporter.render_winloss().savefig(
                    out / f"winloss{suffix}.png", dpi=150, bbox_inches="tight"
                )
            except ValueError as e:
                print(f"  (winloss skipped: {e})")
            print(f"Plots -> {out}/comparison{suffix}.png")
        except ImportError:
            print("matplotlib not available — skipping plots.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
