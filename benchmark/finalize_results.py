"""
Run this after the 7-dataset AutoML benchmark completes.
Updates fig5_automl.png and prints the final AutoML table for the paper.

Usage:
    python benchmark/finalize_results.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

C60_COLOR = "#1565C0"
GRAY_COLOR = "#90A4AE"
PALETTE = ["#1565C0","#E53935","#43A047","#FB8C00","#8E24AA",
           "#00ACC1","#F4511E","#6D4C41","#546E7A","#00897B"]

SYSTEMS_9 = ["HyperoptSearch","OptunaSearch","BayesSearchCV","SuccessiveHalving",
             "BroadRandomSearch","GreedyEnsemble","AutoStack","FeatEngAutoML","OptunaEnsemble"]
SYSTEMS_ALL = SYSTEMS_9 + ["C60.ai"]
DATASETS_7 = ["iris","wine","breast_cancer","digits","pendigits","letter","waveform"]
DS_LABELS  = ["Iris","Wine","Breast Cancer","Digits","Pendigits","Letter","Waveform"]
DS_SHORT   = ["Iris","Wine","BC","Digits","Pend.","Letter","Wave."]


def check_completeness(df):
    done = df[df["system"].isin(SYSTEMS_9)]
    pivot = done.groupby(["system","dataset"]).size().unstack(fill_value=0)
    complete = all(
        pivot.loc[s, ds] >= 4
        for s in SYSTEMS_9
        for ds in DATASETS_7
        if s in pivot.index and ds in pivot.columns
    )
    total = done.shape[0]
    print(f"Folds complete: {total}/252  ({'DONE' if complete else 'INCOMPLETE'})")
    return complete


def compute_means(df):
    groups = defaultdict(list)
    for _, row in df.iterrows():
        if row["accuracy"] and str(row["accuracy"]).strip():
            try:
                groups[(row["system"], row["dataset"])].append(float(row["accuracy"]))
            except ValueError:
                pass
    return groups


def print_table(groups):
    print("\n" + "=" * 90)
    print("AutoML Comparison — Mean Accuracy (%) across 7 datasets")
    print("=" * 90)

    col_w = 8
    header = f"{'System':<22}" + "".join(f"{d[:col_w]:>{col_w}}" for d in DS_SHORT) + f"{'Mean':>{col_w}}  Rank"
    print(header)
    print("-" * len(header))

    system_means = {}
    for sys in SYSTEMS_ALL:
        row_means = []
        for ds in DATASETS_7:
            accs = groups.get((sys, ds), [])
            if accs:
                row_means.append(np.mean(accs) * 100)
        system_means[sys] = np.mean(row_means) if row_means else float("nan")

    sorted_sys = sorted(SYSTEMS_ALL, key=lambda s: system_means.get(s, 0), reverse=True)

    for rank, sys in enumerate(sorted_sys, 1):
        parts = [f"{'** '+sys if sys=='C60.ai' else '   '+sys:<22}"]
        for ds in DATASETS_7:
            accs = groups.get((sys, ds), [])
            if accs:
                m = np.mean(accs) * 100
                parts.append(f"{m:>{col_w}.2f}")
            else:
                parts.append(f"{'—':>{col_w}}")
        mean = system_means[sys]
        print("".join(parts) + f"{mean:>{col_w}.2f}  #{rank}")


def regenerate_fig5(groups, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-dataset grouped bars
    ax = axes[0]
    n_sys = len(SYSTEMS_ALL)
    x = np.arange(len(DATASETS_7))
    width = 0.72 / n_sys

    system_means = {}
    for sys in SYSTEMS_ALL:
        vals, row_m = [], []
        for ds in DATASETS_7:
            accs = groups.get((sys, ds), [])
            m = np.mean(accs) * 100 if accs else np.nan
            vals.append(m)
            if not np.isnan(m):
                row_m.append(m)
        system_means[sys] = np.mean(row_m) if row_m else np.nan

    sorted_sys = sorted(SYSTEMS_ALL, key=lambda s: system_means.get(s, 0), reverse=True)

    for i, sys in enumerate(sorted_sys):
        vals = [np.mean(groups.get((sys, ds), [np.nan])) * 100 for ds in DATASETS_7]
        offset = (i - n_sys / 2 + 0.5) * width
        color = C60_COLOR if sys == "C60.ai" else PALETTE[i % len(PALETTE)]
        alpha = 1.0 if sys == "C60.ai" else 0.6
        zorder = 4 if sys == "C60.ai" else 2
        ax.bar(x + offset, vals, width * 0.88, label=sys,
               color=color, alpha=alpha, zorder=zorder, edgecolor="white", lw=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(DS_SHORT)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(60, 101)
    ax.set_title("AutoML Systems — Per-Dataset Accuracy (7 Datasets)", fontweight="bold")
    ax.legend(fontsize=6, ncol=2, loc="lower right", framealpha=0.85)

    # Mean accuracy bar chart
    ax = axes[1]
    mean_vals = [system_means[s] for s in sorted_sys]
    colors_r  = [C60_COLOR if s == "C60.ai" else GRAY_COLOR for s in sorted_sys]
    bars = ax.barh(sorted_sys[::-1], mean_vals[::-1],
                   color=colors_r[::-1], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, mean_vals[::-1]):
        ax.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}%", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Mean Accuracy over 7 Datasets (%)")
    ax.set_title("AutoML Systems — Mean Accuracy Ranking (7 Datasets)", fontweight="bold")

    fig.suptitle("C60.ai vs. 9 AutoML Frameworks — Full 7-Dataset Comparison",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    path = out / "fig5_automl.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Regenerated {path}")


def main():
    csv_path = Path("benchmark/results/results_automl.csv")
    out      = Path("benchmark/results/paper_figures")

    df = pd.read_csv(csv_path)
    complete = check_completeness(df)
    groups = compute_means(df)

    print_table(groups)

    if complete:
        regenerate_fig5(groups, out)
        print("\nRun generate_automl_report.py for full statistical analysis.")
    else:
        print("\nBenchmark not yet complete — run again when done.")
        print("Partial fig5 regenerated with available data:")
        regenerate_fig5(groups, out)


if __name__ == "__main__":
    main()
