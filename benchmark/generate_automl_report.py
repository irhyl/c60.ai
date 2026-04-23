"""
generate_automl_report.py — Build final AutoML comparison report and docs.

Run this after run_automl_incremental.py completes.
"""
import sys
import io
import contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from collections import defaultdict


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["system", "dataset", "seed", "fold"], keep="first")
    df = df[df["accuracy"].notna() & (df["accuracy"] != "")]
    df["accuracy"] = df["accuracy"].astype(float)
    return df


def mean_std_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return (system × dataset) mean accuracy table."""
    result = df.groupby(["system", "dataset"])["accuracy"].agg(
        mean="mean", std="std", n="count"
    ).reset_index()
    return result


def rank_table(df: pd.DataFrame) -> pd.DataFrame:
    datasets = df["dataset"].unique()
    systems = df["system"].unique()
    ranks = defaultdict(list)
    for ds in datasets:
        sub = df[df["dataset"] == ds].groupby("system")["accuracy"].mean()
        ordered = sub.sort_values(ascending=False)
        for r, sys in enumerate(ordered.index, 1):
            ranks[sys].append(r)
    avg_ranks = {s: np.mean(v) for s, v in ranks.items()}
    return pd.DataFrame(
        sorted(avg_ranks.items(), key=lambda x: x[1]),
        columns=["system", "avg_rank"]
    )


def wilcoxon_table(df: pd.DataFrame, reference: str = "C60.ai") -> pd.DataFrame | None:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None

    ref_rows = df[df["system"] == reference].set_index(["dataset", "seed", "fold"])["accuracy"]
    results = []
    for sys in df["system"].unique():
        if sys == reference:
            continue
        sys_rows = df[df["system"] == sys].set_index(["dataset", "seed", "fold"])["accuracy"]
        common = ref_rows.index.intersection(sys_rows.index)
        if len(common) < 4:
            continue
        r, s = ref_rows[common].values, sys_rows[common].values
        diff = r - s
        mean_diff = diff.mean()
        try:
            _, p = wilcoxon(r, s, alternative="greater")
        except Exception:
            p = float("nan")
        results.append({
            "baseline": sys,
            "mean_diff": mean_diff,
            "p_value": p,
            "significant": "Yes" if p < 0.05 else "No",
        })
    results.sort(key=lambda x: x["p_value"])
    return pd.DataFrame(results)


def markdown_comparison(df: pd.DataFrame, rank_df: pd.DataFrame,
                        wilcoxon_df, ref: str = "C60.ai") -> str:
    datasets = sorted(df["dataset"].unique())
    systems_by_rank = rank_df["system"].tolist()

    lines = []
    lines.append("## AutoML Framework Comparison (10 Systems)\n")
    lines.append("Protocol: 2-fold × 2 seeds = 4 evaluations per (system, dataset).  ")
    lines.append("Datasets: iris (150×4), wine (178×13), breast\\_cancer (569×30), "
                 "digits (1797×64).\n")
    lines.append("> **Note:** TPOT and FLAML excluded due to platform constraints  ")
    lines.append("> (TPOT requires PyTorch which exceeds available virtual memory;  ")
    lines.append("> FLAML ignores `time_budget` on Windows/Python 3.13, taking >298 s  ")
    lines.append("> for a 10 s budget). H2O excluded due to insufficient disk space  ")
    lines.append("> (requires ~500 MB; C: has <256 MB free).\n")

    # Mean accuracy table
    means = df.groupby(["system", "dataset"])["accuracy"].mean().unstack(fill_value=float("nan"))
    mean_overall = means.mean(axis=1)

    header = "| System |" + "".join(f" {d} |" for d in datasets) + " Mean |"
    sep = "| --- |" + " --- |" * (len(datasets) + 1)
    lines.append("### Mean Accuracy\n")
    lines.append(header)
    lines.append(sep)
    for sys in systems_by_rank:
        if sys not in means.index:
            continue
        row_vals = [means.loc[sys, d] if d in means.columns else float("nan") for d in datasets]
        mean_val = mean_overall.get(sys, float("nan"))
        bold_open = "**" if sys == ref else ""
        bold_close = "**" if sys == ref else ""
        cells = "".join(
            f" {bold_open}{v:.4f}{bold_close} |" if not np.isnan(v) else " — |"
            for v in row_vals
        )
        lines.append(f"| {bold_open}{sys}{bold_close} |{cells} {bold_open}{mean_val:.4f}{bold_close} |")

    lines.append("")
    lines.append("### Average Rank (lower = better)\n")
    lines.append("| Rank | System | Avg Rank |")
    lines.append("| --- | --- | --- |")
    for i, row in rank_df.iterrows():
        marker = " **" if row["system"] == ref else ""
        lines.append(f"| {i+1} | {row['system']}{marker} | {row['avg_rank']:.2f} |")

    if wilcoxon_df is not None:
        lines.append("")
        lines.append("### Statistical Significance (Wilcoxon Signed-Rank, C60.ai vs each system)\n")
        lines.append("H₁: C60.ai accuracy > baseline, one-sided, α = 0.05.\n")
        lines.append("| Baseline | Mean diff | p-value | Significant? |")
        lines.append("| --- | --- | --- | --- |")
        for _, r in wilcoxon_df.iterrows():
            sign = "+" if r["mean_diff"] > 0 else ""
            lines.append(f"| {r['baseline']} | {sign}{r['mean_diff']:.4f} | "
                         f"{r['p_value']:.4f} | {r['significant']} |")

    return "\n".join(lines)


def render_plots(df: pd.DataFrame, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        datasets = sorted(df["dataset"].unique())
        rank_df = rank_table(df)
        systems = rank_df["system"].tolist()
        means = df.groupby(["system", "dataset"])["accuracy"].mean().unstack(fill_value=float("nan"))

        # Grouped bar chart
        x = np.arange(len(datasets))
        width = 0.8 / max(len(systems), 1)
        fig, ax = plt.subplots(figsize=(max(14, len(datasets) * 2), 7))
        colors = plt.cm.tab20.colors
        for i, sys in enumerate(systems):
            vals = [means.loc[sys, d] if sys in means.index and d in means.columns
                    else float("nan") for d in datasets]
            offset = (i - len(systems) / 2 + 0.5) * width
            color = "navy" if sys == "C60.ai" else colors[i % len(colors)]
            ax.bar(x + offset, vals, width * 0.9, label=sys, color=color)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Accuracy")
        ax.set_title("C60.ai vs AutoML Systems — Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.set_ylim(0.5, 1.02)
        fig.tight_layout()
        fig.savefig(out / "comparison_automl.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Rank bar chart
        names = [r["system"] for _, r in rank_df.iterrows()][::-1]
        vals  = [r["avg_rank"] for _, r in rank_df.iterrows()][::-1]
        colors_r = ["navy" if n == "C60.ai" else "#90CAF9" for n in names]
        fig2, ax2 = plt.subplots(figsize=(10, max(6, len(systems) * 0.5)))
        ax2.barh(names, vals, color=colors_r)
        ax2.set_xlabel("Average Rank (lower = better)")
        ax2.set_title("AutoML Systems — Average Rank")
        ax2.axvline(1, color="gray", linestyle="--", linewidth=0.8)
        for name, val in zip(names, vals):
            ax2.text(val + 0.05, names.index(name), f"{val:.2f}", va="center", fontsize=9)
        fig2.tight_layout()
        fig2.savefig(out / "ranks_automl.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

        print(f"Plots saved to {out}/")

    except ImportError:
        print("matplotlib not available — skipping plots.")
    except Exception as e:
        print(f"Plot error: {e}")


def main():
    csv_path = Path("benchmark/results/results_automl.csv")
    out = Path("benchmark/results")

    print(f"Loading {csv_path}...")
    df = load_and_clean(csv_path)
    print(f"  {len(df)} valid rows, {df['system'].nunique()} systems, "
          f"{df['dataset'].nunique()} datasets")

    # Check completeness
    expected = df["dataset"].nunique() * 4  # 4 folds per (system, dataset)
    counts = df.groupby("system").size()
    incomplete = counts[counts < expected]
    if len(incomplete):
        print(f"\n  WARNING: incomplete systems (< {expected} folds):")
        for sys, n in incomplete.items():
            print(f"    {sys}: {n}/{expected}")

    rank_df = rank_table(df)
    wilcoxon_df = wilcoxon_table(df)

    # Print report
    print("\n" + "=" * 70)
    print("MEAN ACCURACY TABLE")
    print("=" * 70)
    datasets = sorted(df["dataset"].unique())
    means = df.groupby(["system", "dataset"])["accuracy"].mean().unstack(fill_value=float("nan"))
    stds  = df.groupby(["system", "dataset"])["accuracy"].std().unstack(fill_value=float("nan"))
    mean_all = means.mean(axis=1)

    for sys in rank_df["system"]:
        if sys not in means.index:
            continue
        parts = [f"{sys:<22}"]
        for ds in datasets:
            m = means.loc[sys, ds] if ds in means.columns else float("nan")
            s = stds.loc[sys, ds]  if ds in stds.columns  else float("nan")
            parts.append(f"{m:.4f}±{s:.3f}" if not np.isnan(m) else "—      ")
        parts.append(f"{mean_all.get(sys, float('nan')):.4f}")
        print("  " + "  ".join(parts))

    print("\n" + "=" * 70)
    print("AVERAGE RANK")
    print("=" * 70)
    for _, r in rank_df.iterrows():
        mark = " <--" if r["system"] == "C60.ai" else ""
        print(f"  {r['system']:<22} {r['avg_rank']:.2f}{mark}")

    if wilcoxon_df is not None:
        print("\n" + "=" * 70)
        print("WILCOXON TESTS (C60.ai vs each baseline)")
        print("=" * 70)
        for _, r in wilcoxon_df.iterrows():
            sign = "+" if r["mean_diff"] > 0 else ""
            print(f"  {r['baseline']:<22} diff={sign}{r['mean_diff']:.4f}  "
                  f"p={r['p_value']:.4f}  {r['significant']}")

    # Save markdown
    md = markdown_comparison(df, rank_df, wilcoxon_df)
    md_path = out / "summary_automl.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"\nMarkdown summary -> {md_path}")

    # Save plots
    render_plots(df, out)

    print("\nDone.")
    return df, rank_df, wilcoxon_df


if __name__ == "__main__":
    main()
