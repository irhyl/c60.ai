"""
report.py — ResultsReporter: tables, plots, and statistical tests.

Generates four artefacts from a benchmark results DataFrame:

  1. summary_table()      — Markdown table of mean ± std accuracy per system
  2. rank_table()         — Average rank across datasets (lower is better)
  3. wilcoxon_vs_c60()    — Wilcoxon signed-rank p-values (C60.ai vs each baseline)
  4. render_comparison()  — Grouped bar chart (matplotlib Figure)
  5. render_ranks()       — CD-diagram-style rank plot

All render_* functions return matplotlib Figure objects and do NOT call plt.show().
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw per-fold records to per-(system, dataset) mean ± std.
    Returns a DataFrame with columns: system, dataset, mean_acc, std_acc,
    mean_fit_s, n_evals.
    """
    return (
        df.dropna(subset=["accuracy"])
        .groupby(["system", "dataset"])
        .agg(
            mean_acc  = ("accuracy",   "mean"),
            std_acc   = ("accuracy",   "std"),
            mean_fit_s= ("fit_time_s", "mean"),
            n_evals   = ("accuracy",   "count"),
        )
        .reset_index()
    )


def _pivot(agg: pd.DataFrame, value: str = "mean_acc") -> pd.DataFrame:
    """Pivot to (system × dataset) matrix."""
    return agg.pivot(index="system", columns="dataset", values=value)


# ---------------------------------------------------------------------------
# ResultsReporter
# ---------------------------------------------------------------------------

class ResultsReporter:
    """
    Generate reports and plots from a benchmark results DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of BenchmarkRunner.run().
    highlight_system : str
        Name of the focal system (highlighted in tables and plots).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        highlight_system: str = "C60.ai",
    ) -> None:
        self._df  = df
        self._agg = _agg(df)
        self._highlight = highlight_system

    # ------------------------------------------------------------------
    # Text tables
    # ------------------------------------------------------------------

    def summary_table(self, fmt: str = "markdown") -> str:
        """
        Return a mean ± std accuracy table.

        fmt : "markdown" | "latex" | "plain"
        """
        pivot_mean = _pivot(self._agg, "mean_acc")
        pivot_std  = _pivot(self._agg, "std_acc")

        datasets = list(pivot_mean.columns)
        systems  = list(pivot_mean.index)

        rows = []
        for sys in systems:
            row = {"System": sys}
            for ds in datasets:
                m = pivot_mean.loc[sys, ds] if sys in pivot_mean.index else np.nan
                s = pivot_std.loc[sys, ds]  if sys in pivot_std.index  else np.nan
                row[ds] = f"{m:.4f} ± {s:.4f}" if not np.isnan(m) else "—"
            # Macro-average
            means = [pivot_mean.loc[sys, ds]
                     for ds in datasets
                     if sys in pivot_mean.index and not np.isnan(pivot_mean.loc[sys, ds])]
            row["Mean"] = f"{np.mean(means):.4f}" if means else "—"
            rows.append(row)

        result_df = pd.DataFrame(rows).set_index("System")

        if fmt == "markdown":
            return result_df.to_markdown()
        elif fmt == "latex":
            return result_df.to_latex()
        else:
            return result_df.to_string()

    def rank_table(self) -> pd.DataFrame:
        """
        Return average rank across datasets (rank 1 = best accuracy).
        Lower rank is better.
        """
        pivot = _pivot(self._agg, "mean_acc")
        # Rank within each dataset (ascending=False: higher acc = lower rank)
        ranked = pivot.rank(ascending=False, method="min")
        avg_rank = ranked.mean(axis=1).sort_values()
        return pd.DataFrame({
            "System": avg_rank.index,
            "Avg Rank": avg_rank.values,
        }).reset_index(drop=True)

    def wilcoxon_vs_c60(self) -> pd.DataFrame:
        """
        Wilcoxon signed-rank test: C60.ai vs each other system.

        Uses per-fold accuracy scores (paired across folds × seeds) to test
        whether C60.ai is significantly better (one-sided, H1: C60 > baseline).

        Returns a DataFrame with columns:
            system, mean_diff, p_value, significant (α=0.05)
        """
        from scipy.stats import wilcoxon

        c60_rows = self._df[self._df["system"] == self._highlight].copy()
        other_systems = [s for s in self._df["system"].unique()
                         if s != self._highlight]
        records = []

        for sys in other_systems:
            sys_rows = self._df[self._df["system"] == sys].copy()

            # Align by (dataset, seed, fold)
            merged = pd.merge(
                c60_rows[["dataset", "seed", "fold", "accuracy"]],
                sys_rows[["dataset", "seed", "fold", "accuracy"]],
                on=["dataset", "seed", "fold"],
                suffixes=("_c60", "_sys"),
            ).dropna()

            if len(merged) < 5:
                records.append({
                    "system": sys,
                    "mean_diff": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                })
                continue

            c60_acc = merged["accuracy_c60"].values
            sys_acc = merged["accuracy_sys"].values
            diff = c60_acc - sys_acc

            if np.all(diff == 0):
                p = 1.0
            else:
                try:
                    _, p = wilcoxon(c60_acc, sys_acc, alternative="greater")
                except Exception:
                    p = np.nan

            records.append({
                "system":      sys,
                "mean_diff":   float(np.mean(diff)),
                "p_value":     float(p) if p is not None else np.nan,
                "significant": bool(p < 0.05) if p is not None and not np.isnan(p) else False,
            })

        return pd.DataFrame(records).sort_values("p_value")

    def print_full_report(self) -> None:
        """Print a formatted summary to stdout."""
        print("\n" + "="*70)
        print("  C60.ai BENCHMARK RESULTS")
        print("="*70)

        print("\n-- Mean +/- Std Accuracy (3-fold CV x 3 seeds) --\n")
        print(self.summary_table(fmt="plain"))

        print("\n-- Average Rank Across Datasets (1 = best) --\n")
        rt = self.rank_table()
        for _, row in rt.iterrows():
            marker = " << C60.ai" if row["System"] == self._highlight else ""
            print(f"  {row['Avg Rank']:4.2f}  {row['System']}{marker}")

        print("\n-- Wilcoxon Signed-Rank: C60.ai vs Baselines --")
        print("   (H1: C60.ai > baseline, one-sided, alpha=0.05)\n")
        wt = self.wilcoxon_vs_c60()
        for _, row in wt.iterrows():
            sig = "* significant" if row["significant"] else "  -"
            diff = f"{row['mean_diff']:+.4f}" if not np.isnan(row["mean_diff"]) else "  n/a"
            p    = f"{row['p_value']:.4f}"   if not np.isnan(row['p_value'])   else "  n/a"
            print(f"  {row['system']:<22} diff={diff}  p={p}  {sig}")

        # Best system summary
        rt = self.rank_table()
        best = rt.iloc[0]
        print(f"\n-- Summary --")
        print(f"  Best-ranked system: {best['System']} (avg rank {best['Avg Rank']:.2f})")
        c60_rank_rows = rt[rt["System"] == self._highlight]
        if not c60_rank_rows.empty:
            c60_rank = c60_rank_rows.iloc[0]["Avg Rank"]
            print(f"  C60.ai rank:        {c60_rank:.2f}")
        print()

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def render_comparison(
        self,
        title: str = "Accuracy by System and Dataset",
        figsize: tuple = (14, 6),
    ):
        """
        Grouped bar chart: one group per dataset, one bar per system.
        C60.ai bars are highlighted in a distinct colour.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError("render_comparison() requires matplotlib.")

        pivot = _pivot(self._agg, "mean_acc")
        pivot_std = _pivot(self._agg, "std_acc")

        datasets = list(pivot.columns)
        systems  = list(pivot.index)
        n_sys    = len(systems)
        n_ds     = len(datasets)

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(n_ds)
        width = 0.8 / n_sys
        offsets = np.linspace(-(0.8/2), 0.8/2, n_sys, endpoint=False) + width/2

        cmap = plt.colormaps.get_cmap("tab10").resampled(n_sys)

        for i, sys in enumerate(systems):
            means = [pivot.loc[sys, ds] if ds in pivot.columns else np.nan
                     for ds in datasets]
            stds  = [pivot_std.loc[sys, ds] if ds in pivot_std.columns else 0
                     for ds in datasets]
            color  = "#e53935" if sys == self._highlight else cmap(i)
            zorder = 4 if sys == self._highlight else 2
            lw     = 2.0 if sys == self._highlight else 0.8
            ax.bar(
                x + offsets[i], means, width,
                yerr=stds, capsize=2,
                color=color, alpha=0.85,
                edgecolor="black", linewidth=lw,
                zorder=zorder,
                label=sys,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(
            fontsize=7, ncol=2,
            loc="lower right", framealpha=0.9,
        )

        fig.tight_layout()
        return fig

    def render_ranks(
        self,
        title: str = "Average Rank Across Datasets (lower = better)",
        figsize: tuple = (8, 5),
    ):
        """
        Horizontal bar chart of average ranks.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("render_ranks() requires matplotlib.")

        rt = self.rank_table().sort_values("Avg Rank", ascending=False)

        fig, ax = plt.subplots(figsize=figsize)
        colors = [
            "#e53935" if s == self._highlight else "#90a4ae"
            for s in rt["System"]
        ]
        bars = ax.barh(rt["System"], rt["Avg Rank"], color=colors,
                       edgecolor="black", linewidth=0.7)

        # Annotate bar values
        for bar, val in zip(bars, rt["Avg Rank"]):
            ax.text(
                bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9,
            )

        ax.set_xlabel("Average Rank (1 = best)", fontsize=11)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlim(0, len(rt) + 0.5)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        fig.tight_layout()
        return fig

    def render_winloss(
        self,
        title: str = "C60.ai vs Baselines: Win / Tie / Loss",
        figsize: tuple = (9, 5),
    ):
        """
        Stacked bar showing how often C60.ai beats each baseline per dataset.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("render_winloss() requires matplotlib.")

        pivot = _pivot(self._agg, "mean_acc")
        if self._highlight not in pivot.index:
            raise ValueError(f"'{self._highlight}' not found in results.")

        c60_row = pivot.loc[self._highlight]
        baselines = [s for s in pivot.index if s != self._highlight]

        wins = ties = losses = 0
        detail = []
        for sys in baselines:
            if sys not in pivot.index:
                continue
            sys_row = pivot.loc[sys]
            for ds in pivot.columns:
                c60_v = c60_row.get(ds, np.nan)
                sys_v = sys_row.get(ds, np.nan)
                if np.isnan(c60_v) or np.isnan(sys_v):
                    continue
                diff = c60_v - sys_v
                if diff > 0.001:
                    wins += 1
                    detail.append((sys, ds, "win"))
                elif diff < -0.001:
                    losses += 1
                    detail.append((sys, ds, "loss"))
                else:
                    ties += 1
                    detail.append((sys, ds, "tie"))

        total = wins + ties + losses or 1
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(
            ["Win", "Tie", "Loss"],
            [wins, ties, losses],
            color=["#43a047", "#ffb300", "#e53935"],
            edgecolor="black", linewidth=0.8,
        )
        for bar, val in zip(bars, [wins, ties, losses]):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{val}\n({100*val/total:.0f}%)",
                ha="center", fontsize=11,
            )
        ax.set_ylabel("Count (across datasets × baselines)", fontsize=10)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylim(0, max(wins, ties, losses) * 1.3 + 1)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig
