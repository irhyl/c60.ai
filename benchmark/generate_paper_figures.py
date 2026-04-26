"""
Generate publication-quality figures for the ICML workshop paper.
Outputs PNG files into benchmark/results/paper_figures/.
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
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

C60_COLOR   = "#1565C0"   # deep blue
GRAY_COLOR  = "#90A4AE"
RED_COLOR   = "#C62828"
GREEN_COLOR = "#2E7D32"
PALETTE = [
    "#1565C0","#E53935","#43A047","#FB8C00","#8E24AA",
    "#00ACC1","#F4511E","#6D4C41","#546E7A","#00897B",
]

OUT = Path("benchmark/results/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── load data ───────────────────────────────────────────────────────────────
ext  = pd.read_csv("benchmark/results/results_extended.csv")
auto = pd.read_csv("benchmark/results/results_automl.csv")

DATASETS_7  = ["iris","wine","breast_cancer","digits","pendigits","letter","waveform"]
DS_LABELS   = ["Iris","Wine","Breast\nCancer","Digits","Pendigits","Letter","Waveform"]
DS_SHORT    = ["Iris","Wine","BC","Digits","Pend.","Letter","Wave."]
BASELINES_9 = ["LR","SVM-RBF","KNN-10","RandomForest","GradientBoosting",
               "PCA+LR","SelectKBest+GBT","VotingEnsemble","RandomSearch-GBT"]
ALL_SKL     = BASELINES_9 + ["C60.ai"]

# mean accuracy per (system, dataset)
skl_means = (ext[ext["dataset"].isin(DATASETS_7)]
             .groupby(["system","dataset"])["accuracy"].mean()
             .unstack())

# ════════════════════════════════════════════════════════════════════════════
# FIG 1  —  Grouped bar chart: C60.ai vs baselines, 7 datasets
# ════════════════════════════════════════════════════════════════════════════
def fig1_grouped_bar():
    systems = ["C60.ai","SVM-RBF","VotingEnsemble","RandomForest",
               "GradientBoosting","RandomSearch-GBT","KNN-10",
               "SelectKBest+GBT","LR","PCA+LR"]
    ds_order = DATASETS_7
    n_sys = len(systems)
    n_ds  = len(ds_order)
    x = np.arange(n_ds)
    width = 0.72 / n_sys

    fig, ax = plt.subplots(figsize=(12, 4.5))

    for i, sys in enumerate(systems):
        vals = [skl_means.loc[sys, ds] * 100 if (sys in skl_means.index and ds in skl_means.columns)
                else np.nan for ds in ds_order]
        offset = (i - n_sys / 2 + 0.5) * width
        color  = C60_COLOR if sys == "C60.ai" else GRAY_COLOR
        alpha  = 1.0 if sys == "C60.ai" else 0.65
        zorder = 3 if sys == "C60.ai" else 2
        ax.bar(x + offset, vals, width * 0.88, label=sys,
               color=color, alpha=alpha, zorder=zorder,
               edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(DS_SHORT, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(60, 101)
    ax.set_title("C60.ai vs. Sklearn Baselines — Accuracy on 7 Datasets", fontweight="bold")
    ax.axhline(100, color="#ccc", lw=0.5, ls="--")

    # custom legend
    c60_patch  = mpatches.Patch(color=C60_COLOR,  label="C60.ai (ours)")
    base_patch = mpatches.Patch(color=GRAY_COLOR, alpha=0.65, label="Baselines")
    ax.legend(handles=[c60_patch, base_patch], loc="lower right", framealpha=0.85)

    fig.tight_layout()
    path = OUT / "fig1_grouped_bar.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

# ════════════════════════════════════════════════════════════════════════════
# FIG 2  —  Heatmap: accuracy deviation from column-best
# ════════════════════════════════════════════════════════════════════════════
def fig2_heatmap():
    systems_order = ["C60.ai","SVM-RBF","VotingEnsemble","RandomForest",
                     "GradientBoosting","RandomSearch-GBT","KNN-10",
                     "SelectKBest+GBT","LR","PCA+LR"]
    mat = np.zeros((len(systems_order), len(DATASETS_7)))
    for j, ds in enumerate(DATASETS_7):
        col_best = max(skl_means.loc[s, ds] for s in systems_order if ds in skl_means.columns)
        for i, sys in enumerate(systems_order):
            val = skl_means.loc[sys, ds] if ds in skl_means.columns else np.nan
            mat[i, j] = (val - col_best) * 100  # negative = below best

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-6, vmax=1, aspect="auto")

    ax.set_xticks(range(len(DATASETS_7)))
    ax.set_xticklabels(DS_SHORT)
    ax.set_yticks(range(len(systems_order)))
    ax.set_yticklabels(systems_order)

    # annotate cells
    for i in range(len(systems_order)):
        for j in range(len(DATASETS_7)):
            v = mat[i, j]
            txt = f"{v:+.2f}" if not np.isnan(v) else "—"
            color = "white" if v < -3.5 else "black"
            weight = "bold" if systems_order[i] == "C60.ai" else "normal"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color=color, fontweight=weight)

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Accuracy − Column Best (pp)")
    ax.set_title("Accuracy Gap from Best System per Dataset (pp)", fontweight="bold")

    # highlight C60.ai row
    from matplotlib.patches import FancyBboxPatch
    idx = systems_order.index("C60.ai")
    rect = plt.Rectangle((-0.5, idx - 0.5), len(DATASETS_7), 1,
                         linewidth=2, edgecolor=C60_COLOR, facecolor="none", zorder=5)
    ax.add_patch(rect)

    fig.tight_layout()
    path = OUT / "fig2_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

# ════════════════════════════════════════════════════════════════════════════
# FIG 3  —  Average rank + mean accuracy scatter (critical-difference style)
# ════════════════════════════════════════════════════════════════════════════
def fig3_rank_scatter():
    systems_order = ["C60.ai","SVM-RBF","VotingEnsemble","RandomForest",
                     "GradientBoosting","RandomSearch-GBT","KNN-10",
                     "SelectKBest+GBT","LR","PCA+LR"]

    avg_ranks, mean_accs = {}, {}
    for sys in systems_order:
        ranks, accs = [], []
        for ds in DATASETS_7:
            col = {s: skl_means.loc[s, ds] for s in systems_order if ds in skl_means.columns}
            sorted_sys = sorted(col, key=lambda s: col[s], reverse=True)
            rank = sorted_sys.index(sys) + 1 if sys in sorted_sys else np.nan
            ranks.append(rank)
            if ds in skl_means.columns:
                accs.append(skl_means.loc[sys, ds] * 100)
        avg_ranks[sys] = np.nanmean(ranks)
        mean_accs[sys] = np.nanmean(accs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: avg rank horizontal bar
    ax = axes[0]
    sorted_sys = sorted(systems_order, key=lambda s: avg_ranks[s])
    ranks_sorted = [avg_ranks[s] for s in sorted_sys]
    colors = [C60_COLOR if s == "C60.ai" else GRAY_COLOR for s in sorted_sys]
    bars = ax.barh(sorted_sys, ranks_sorted, color=colors, alpha=0.85, edgecolor="white")
    for bar, r in zip(bars, ranks_sorted):
        ax.text(r + 0.05, bar.get_y() + bar.get_height()/2,
                f"{r:.2f}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Average Rank (lower = better)")
    ax.set_title("Average Rank Across 7 Datasets", fontweight="bold")
    ax.axvline(1, color="#bbb", ls="--", lw=0.8)
    ax.set_xlim(0, max(ranks_sorted) + 1.2)

    # Right: mean accuracy horizontal bar
    ax = axes[1]
    sorted_sys2 = sorted(systems_order, key=lambda s: mean_accs[s], reverse=True)
    accs_sorted = [mean_accs[s] for s in sorted_sys2]
    colors2 = [C60_COLOR if s == "C60.ai" else GRAY_COLOR for s in sorted_sys2]
    bars2 = ax.barh(sorted_sys2[::-1], accs_sorted[::-1],
                    color=colors2[::-1], alpha=0.85, edgecolor="white")
    for bar, a in zip(bars2, accs_sorted[::-1]):
        ax.text(a + 0.05, bar.get_y() + bar.get_height()/2,
                f"{a:.2f}%", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Mean Accuracy (%)")
    ax.set_title("Mean Accuracy Across 7 Datasets", fontweight="bold")
    ax.set_xlim(86, 98)

    fig.suptitle("C60.ai vs. Sklearn Baselines — Ranking Summary", fontweight="bold", fontsize=12)
    fig.tight_layout()
    path = OUT / "fig3_rank_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

# ════════════════════════════════════════════════════════════════════════════
# FIG 4  —  Pipeline topology diagram (hand-crafted DAG visualization)
# ════════════════════════════════════════════════════════════════════════════
def fig4_pipeline_dag():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    def draw_pipeline(ax, nodes, edges, title, colors, highlight_nodes=None):
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.3, len(nodes) - 0.7)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=9, fontweight="bold", pad=8)

        positions = {n: (1.0, len(nodes) - 1 - i) for i, n in enumerate(nodes)}

        for u, v in edges:
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            ax.annotate("", xy=(x1, y1 + 0.28), xytext=(x0, y0 - 0.28),
                        arrowprops=dict(arrowstyle="-|>", color="#555",
                                        lw=1.2, mutation_scale=12))

        for i, n in enumerate(nodes):
            x, y = positions[n]
            fc = colors.get(n, "#E3F2FD")
            ec = C60_COLOR if (highlight_nodes and n in highlight_nodes) else "#555"
            lw = 2.0 if (highlight_nodes and n in highlight_nodes) else 1.0
            box = dict(boxstyle="round,pad=0.4", facecolor=fc, edgecolor=ec,
                       linewidth=lw)
            ax.text(x, y, n, ha="center", va="center", fontsize=8,
                    bbox=box, zorder=3)

    # — Fixed template (what baselines do)
    ax = axes[0]
    nodes1 = ["Input", "Scaler", "Selector", "Model", "Predictions"]
    edges1 = [("Input","Scaler"),("Scaler","Selector"),("Selector","Model"),
              ("Model","Predictions")]
    colors1 = {"Input":"#ECEFF1","Scaler":"#E8EAF6","Selector":"#E8EAF6",
               "Model":"#FCE4EC","Predictions":"#ECEFF1"}
    draw_pipeline(ax, nodes1, edges1,
                  "Fixed Template\n(All baselines)", colors1)
    ax.text(1.0, -0.22, "Fixed topology: no structural freedom",
            ha="center", fontsize=7, color="#888", style="italic")

    # — C60.ai discovered pipeline (Digits best)
    ax = axes[1]
    nodes2 = ["Input","StandardScaler","PCA(n=12)","SVC(RBF)","Predictions"]
    edges2 = [("Input","StandardScaler"),("StandardScaler","PCA(n=12)"),
              ("PCA(n=12)","SVC(RBF)"),("SVC(RBF)","Predictions")]
    colors2 = {"Input":"#ECEFF1","StandardScaler":"#E3F2FD","PCA(n=12)":"#E3F2FD",
               "SVC(RBF)":"#E8F5E9","Predictions":"#ECEFF1"}
    draw_pipeline(ax, nodes2, edges2,
                  "C60.ai — Digits Best\n(98.83% accuracy)", colors2,
                  highlight_nodes={"PCA(n=12)"})
    ax.text(1.0, -0.22, "Topology discovered by evolution",
            ha="center", fontsize=7, color=C60_COLOR, style="italic", fontweight="bold")

    # — C60.ai parallel-branch DAG (advanced)
    ax = axes[2]
    ax.set_xlim(-0.8, 2.8)
    ax.set_ylim(-0.3, 4.7)
    ax.axis("off")
    ax.set_title("C60.ai — Discovered DAG\n(branched topology)", fontsize=9, fontweight="bold", pad=8)

    positions = {
        "Input":       (1.0, 4.2),
        "Scaler":      (1.0, 3.2),
        "PCA":         (0.3, 2.2),
        "SelectKBest": (1.7, 2.2),
        "Merge":       (1.0, 1.2),
        "SVC":         (1.0, 0.2),
    }
    edges_dag = [("Input","Scaler"),("Scaler","PCA"),("Scaler","SelectKBest"),
                 ("PCA","Merge"),("SelectKBest","Merge"),("Merge","SVC")]
    node_colors = {
        "Input":"#ECEFF1","Scaler":"#E3F2FD","PCA":"#E3F2FD",
        "SelectKBest":"#E3F2FD","Merge":"#FFF9C4","SVC":"#E8F5E9"
    }
    for u, v in edges_dag:
        x0,y0 = positions[u]; x1,y1 = positions[v]
        ax.annotate("", xy=(x1,y1+0.22), xytext=(x0,y0-0.22),
                    arrowprops=dict(arrowstyle="-|>", color="#555",
                                   lw=1.2, mutation_scale=12))
    for n,(x,y) in positions.items():
        ec = C60_COLOR if n in ("PCA","SelectKBest","Merge") else "#555"
        lw = 2.0 if n in ("PCA","SelectKBest","Merge") else 1.0
        ax.text(x, y, n, ha="center", va="center", fontsize=8, zorder=3,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=node_colors[n],
                          edgecolor=ec, linewidth=lw))
    ax.text(1.0, -0.22, "Parallel branches: only possible\nwith graph-level search",
            ha="center", fontsize=7, color=C60_COLOR, style="italic", fontweight="bold")

    fig.suptitle("Pipeline Representations: Fixed Template vs. C60.ai DAG Search",
                 fontweight="bold", fontsize=11, y=1.01)
    fig.tight_layout()
    path = OUT / "fig4_pipeline_dag.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

# ════════════════════════════════════════════════════════════════════════════
# FIG 5  —  AutoML comparison (4 datasets: iris, wine, breast_cancer, digits)
# ════════════════════════════════════════════════════════════════════════════
def fig5_automl():
    systems_automl = ["HyperoptSearch","OptunaSearch","BayesSearchCV",
                      "SuccessiveHalving","BroadRandomSearch","GreedyEnsemble",
                      "AutoStack","FeatEngAutoML","OptunaEnsemble","C60.ai"]
    ds4 = ["iris","wine","breast_cancer","digits"]
    ds4_labels = ["Iris","Wine","Breast Cancer","Digits"]

    groups = defaultdict(list)
    for _, row in auto.iterrows():
        if row["dataset"] in ds4 and row["system"] in systems_automl:
            if row["accuracy"] and str(row["accuracy"]) != "":
                try:
                    groups[(row["system"], row["dataset"])].append(float(row["accuracy"]))
                except Exception:
                    pass

    means = {}
    for sys in systems_automl:
        row_means = []
        for ds in ds4:
            accs = groups.get((sys, ds), [])
            if accs:
                row_means.append(np.mean(accs))
        means[sys] = np.mean(row_means) if row_means else np.nan

    # sort by mean descending
    sorted_sys = sorted([s for s in systems_automl if not np.isnan(means[s])],
                        key=lambda s: means[s], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: per-dataset grouped bars
    ax = axes[0]
    n_sys = len(sorted_sys)
    x = np.arange(len(ds4))
    width = 0.72 / n_sys
    for i, sys in enumerate(sorted_sys):
        vals = [np.mean(groups.get((sys, ds), [np.nan])) * 100 for ds in ds4]
        offset = (i - n_sys / 2 + 0.5) * width
        color  = C60_COLOR if sys == "C60.ai" else PALETTE[i % len(PALETTE)]
        alpha  = 1.0 if sys == "C60.ai" else 0.6
        zorder = 4 if sys == "C60.ai" else 2
        ax.bar(x + offset, vals, width * 0.88, label=sys,
               color=color, alpha=alpha, zorder=zorder, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(ds4_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(86, 101)
    ax.set_title("AutoML Systems — Per-Dataset Accuracy", fontweight="bold")
    ax.legend(fontsize=6.5, ncol=2, loc="lower right", framealpha=0.85)

    # Right: mean accuracy bar chart sorted
    ax = axes[1]
    mean_vals = [means[s] * 100 for s in sorted_sys]
    colors_r  = [C60_COLOR if s == "C60.ai" else GRAY_COLOR for s in sorted_sys]
    bars = ax.barh(sorted_sys[::-1], mean_vals[::-1],
                   color=colors_r[::-1], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, mean_vals[::-1]):
        ax.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}%", va="center", ha="left", fontsize=8)
    # annotate C60.ai best-on-digits
    digits_vals = {s: np.mean(groups.get((s,"digits"),[np.nan]))*100 for s in sorted_sys}
    c60_digits = digits_vals.get("C60.ai", 0)
    idx_c60 = sorted_sys[::-1].index("C60.ai")
    ax.text(mean_vals[::-1][idx_c60] - 0.1, idx_c60,
            f"  Best on Digits: {c60_digits:.2f}%",
            va="center", ha="left", fontsize=7, color=C60_COLOR, fontweight="bold")
    ax.set_xlabel("Mean Accuracy over 4 Datasets (%)")
    ax.set_title("AutoML Systems — Mean Accuracy Ranking", fontweight="bold")
    ax.set_xlim(91, 99)

    fig.suptitle("C60.ai vs. 9 AutoML Frameworks (Iris, Wine, Breast Cancer, Digits)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    path = OUT / "fig5_automl.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

# ════════════════════════════════════════════════════════════════════════════
# FIG 6  —  Real fitness evolution curves (from evolution_curves.json)
# ════════════════════════════════════════════════════════════════════════════
def fig6_fitness_evolution():
    import json
    curves_path = Path("benchmark/results/evolution_curves.json")
    if not curves_path.exists():
        print("evolution_curves.json not found — skipping fig6")
        return

    with open(curves_path) as f:
        raw = json.load(f)

    # Group by (variant, dataset) → list of per-run curves
    from collections import defaultdict
    by_vd: dict = defaultdict(list)
    for k, v in raw.items():
        parts = k.split("|")
        if len(parts) == 4:
            by_vd[(parts[0], parts[1])].append(np.array(v))

    # Use C60.ai-Full curves for the 4 well-populated datasets
    ds_display = [
        ("iris",          "Iris"),
        ("wine",          "Wine"),
        ("breast_cancer", "Breast Cancer"),
        ("digits",        "Digits"),
    ]
    ds_colors = [C60_COLOR, "#E53935", "#43A047", "#FB8C00"]

    # Check we have data; fall back to any available
    available = {ds for (_, ds) in by_vd.keys() if ("C60.ai-Full", ds) in by_vd}
    ds_display = [(k, lbl) for k, lbl in ds_display if k in available]
    if not ds_display:
        print("No real curves found — skipping fig6")
        return

    n_plots = len(ds_display)
    ncols = min(n_plots, 4)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax_idx, ((ds_key, ds_label), color) in enumerate(zip(ds_display, ds_colors)):
        ax = axes_flat[ax_idx]
        run_curves = [c for c in by_vd[("C60.ai-Full", ds_key)]]
        # Pad to equal length
        max_len = max(len(c) for c in run_curves)
        padded = np.array([np.pad(c, (0, max_len - len(c)), mode="edge") for c in run_curves])

        median_curve = np.median(padded, axis=0) * 100
        q25_curve    = np.percentile(padded, 25, axis=0) * 100
        q75_curve    = np.percentile(padded, 75, axis=0) * 100
        gens         = np.arange(max_len)

        ax.plot(gens, median_curve, color=color, lw=2.2, marker="o",
                markersize=4, zorder=3, label="Median")
        ax.fill_between(gens, q25_curve, q75_curve, alpha=0.20, color=color, label="IQR")

        # Plot individual runs faintly
        for run in padded:
            ax.plot(gens, run * 100, color=color, lw=0.5, alpha=0.25, zorder=1)

        final = median_curve[-1]
        ax.axhline(final, color=color, lw=0.8, ls="--", alpha=0.6)
        ax.text(max_len - 1, final + 0.25, f"{final:.2f}%",
                fontsize=8, color=color, fontweight="bold", ha="right")

        ax.set_title(ds_label, fontweight="bold")
        ax.set_xlabel("Generation")
        ax.set_ylabel("CV Accuracy (%)")
        ax.set_xticks(gens)
        ax.set_xticklabels([f"G{g}" for g in gens], fontsize=7)
        ax.grid(True, alpha=0.20, lw=0.5)
        n_runs = len(run_curves)
        ax.text(0.03, 0.05, f"n={n_runs} runs", transform=ax.transAxes,
                fontsize=7, color="#888")

    # Hide any extra subplots
    for ax_idx in range(len(ds_display), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(
        "C60.ai Fitness Evolution: Median Best-of-Generation Accuracy\n"
        "(shaded band = interquartile range across seeds × folds)",
        fontweight="bold", fontsize=11
    )
    fig.tight_layout()
    path = OUT / "fig6_fitness_evolution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 10  —  Ablation: C60.ai-Full vs C60.ai-HPO
# ════════════════════════════════════════════════════════════════════════════
def fig10_ablation():
    import json
    from scipy import stats as scipy_stats

    # Load C60.ai-Full results from main ablation CSV
    ablation_csv = Path("benchmark/results/results_ablation.csv")
    if not ablation_csv.exists():
        print("results_ablation.csv not found — skipping fig10")
        return

    combined = pd.read_csv(ablation_csv)
    if "C60.ai-HPO" not in combined["variant"].values:
        print("HPO variant not yet in ablation CSV — skipping fig10")
        return

    datasets_present = sorted(combined["dataset"].unique())
    DS_LABELS_ABL = {
        "iris": "Iris", "wine": "Wine",
        "breast_cancer": "Breast\nCancer", "digits": "Digits",
        "waveform": "Waveform", "pendigits": "Pendigits", "letter": "Letter",
    }

    means_full = {}
    means_hpo  = {}
    sems_full  = {}
    sems_hpo   = {}
    pvals      = {}

    for ds in datasets_present:
        f = combined[(combined["variant"] == "C60.ai-Full") & (combined["dataset"] == ds)]["accuracy"].astype(float)
        h = combined[(combined["variant"] == "C60.ai-HPO")  & (combined["dataset"] == ds)]["accuracy"].astype(float)
        if len(f) < 2 or len(h) < 2:
            continue
        means_full[ds] = f.mean() * 100
        means_hpo[ds]  = h.mean() * 100
        sems_full[ds]  = f.sem() * 100
        sems_hpo[ds]   = h.sem() * 100
        if len(f) == len(h):
            _, p = scipy_stats.wilcoxon(f.values, h.values, alternative="two-sided", zero_method="wilcox")
        else:
            _, p = scipy_stats.mannwhitneyu(f.values, h.values, alternative="two-sided")
        pvals[ds] = p

    ds_plot = [d for d in datasets_present if d in means_full and d in means_hpo]
    if not ds_plot:
        print("Not enough paired data for fig10")
        return

    x = np.arange(len(ds_plot))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Left: grouped bar comparison ────────────────────────────────────────
    ax = axes[0]
    bars_full = ax.bar(x - width/2,
                       [means_full[d] for d in ds_plot],
                       width, yerr=[sems_full[d] for d in ds_plot],
                       label="C60.ai-Full (structural + HPO)",
                       color=C60_COLOR, alpha=0.88,
                       error_kw=dict(elinewidth=1.0, capsize=3),
                       edgecolor="white")
    bars_hpo  = ax.bar(x + width/2,
                       [means_hpo[d] for d in ds_plot],
                       width, yerr=[sems_hpo[d] for d in ds_plot],
                       label="C60.ai-HPO (HPO only, fixed topology)",
                       color="#78909C", alpha=0.80,
                       error_kw=dict(elinewidth=1.0, capsize=3),
                       edgecolor="white")

    # Annotate p-values above paired bars
    y_max = ax.get_ylim()[1]
    for i, ds in enumerate(ds_plot):
        p = pvals.get(ds, 1.0)
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        top = max(means_full[ds] + sems_full[ds], means_hpo[ds] + sems_hpo[ds]) + 0.5
        ax.text(i, top, star, ha="center", va="bottom", fontsize=9,
                color="#333" if star != "n.s." else "#999")

    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABELS_ABL.get(d, d) for d in ds_plot], fontsize=9)
    ax.set_ylabel("Mean CV Accuracy (%)")
    ax.set_title("C60.ai-Full vs C60.ai-HPO\n(* p<0.05, ** p<0.01, *** p<0.001 Wilcoxon)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    lo = min(min(means_full[d] for d in ds_plot),
             min(means_hpo[d]  for d in ds_plot)) - 2.0
    ax.set_ylim(lo, None)

    # ── Right: accuracy gain (Full − HPO) ───────────────────────────────────
    ax = axes[1]
    gains = [means_full[d] - means_hpo[d] for d in ds_plot]
    bar_colors = [GREEN_COLOR if g > 0 else RED_COLOR for g in gains]
    bars = ax.bar(x, gains, 0.55,
                  color=bar_colors, alpha=0.85, edgecolor="white")
    for bar, g in zip(bars, gains):
        sign = "+" if g >= 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2,
                g + (0.06 if g >= 0 else -0.06),
                f"{sign}{g:.2f}pp",
                ha="center", va="bottom" if g >= 0 else "top",
                fontsize=8, fontweight="bold")

    ax.axhline(0, color="#666", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABELS_ABL.get(d, d) for d in ds_plot], fontsize=9)
    ax.set_ylabel("Accuracy Gain (Full − HPO, pp)")
    ax.set_title("Gain from Graph-Structural Search\n(positive = Full wins)",
                 fontweight="bold", fontsize=10)
    avg_gain = np.mean(gains)
    ax.axhline(avg_gain, color=C60_COLOR, lw=1.2, ls="--", alpha=0.8)
    ax.text(len(ds_plot) - 0.5, avg_gain + 0.05,
            f"mean +{avg_gain:.2f}pp", ha="right", fontsize=8,
            color=C60_COLOR, fontweight="bold")

    fig.suptitle(
        "Ablation Study: Contribution of Graph-Structural Genetic Operators",
        fontweight="bold", fontsize=12
    )
    fig.tight_layout()
    path = OUT / "fig10_ablation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

    # Print a summary table
    print("\n--- Ablation Summary ---")
    print(f"{'Dataset':<16} {'Full (%)':>10} {'HPO (%)':>10} {'Gain (pp)':>10} {'p-value':>10}")
    print("-" * 58)
    for ds in ds_plot:
        p = pvals.get(ds, float("nan"))
        print(f"{ds:<16} {means_full[ds]:>10.2f} {means_hpo[ds]:>10.2f} "
              f"{means_full[ds]-means_hpo[ds]:>+10.2f} {p:>10.4f}")
    overall_gain = np.mean([means_full[d] - means_hpo[d] for d in ds_plot])
    print(f"\nMean accuracy gain from structural search: {overall_gain:+.2f} pp")

# ════════════════════════════════════════════════════════════════════════════
# FIG 7  —  Summary panel (for paper intro / overview)
# ════════════════════════════════════════════════════════════════════════════
def fig7_summary_panel():
    systems_order = ["C60.ai","SVM-RBF","VotingEnsemble","RandomForest",
                     "GradientBoosting","RandomSearch-GBT","KNN-10",
                     "SelectKBest+GBT","LR","PCA+LR"]

    ds_order = DATASETS_7
    mean_by_sys = {}
    for sys in systems_order:
        accs = [skl_means.loc[sys, ds] for ds in ds_order if ds in skl_means.columns]
        mean_by_sys[sys] = np.mean(accs) * 100 if accs else np.nan

    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_sys = sorted(systems_order, key=lambda s: mean_by_sys[s], reverse=True)
    vals = [mean_by_sys[s] for s in sorted_sys]
    colors = [C60_COLOR if s == "C60.ai" else GRAY_COLOR for s in sorted_sys]

    bars = ax.bar(range(len(sorted_sys)), vals, color=colors, alpha=0.88,
                  edgecolor="white", linewidth=0.5, width=0.6)

    for bar, v, sys in zip(bars, vals, sorted_sys):
        label = "C60.ai\n(ours)" if sys == "C60.ai" else sys.replace("Ensemble","Ens.").replace("RandomSearch-GBT","RandSrch")
        fontw = "bold" if sys == "C60.ai" else "normal"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=7, fontweight=fontw)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.5,
                label, ha="center", va="top", fontsize=6.5,
                rotation=20, color="white" if sys == "C60.ai" else "#444")

    ax.set_ylabel("Mean Accuracy (%) across 7 Datasets")
    ax.set_title("C60.ai Achieves Highest Mean Accuracy Among All Baselines",
                 fontweight="bold")
    ax.set_ylim(86, 97)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)

    # annotation arrow
    c60_idx = sorted_sys.index("C60.ai")
    ax.annotate("Best overall\n(94.95%)",
                xy=(c60_idx, mean_by_sys["C60.ai"]),
                xytext=(c60_idx + 1.5, mean_by_sys["C60.ai"] - 1.0),
                arrowprops=dict(arrowstyle="->", color=C60_COLOR, lw=1.5),
                fontsize=8, color=C60_COLOR, fontweight="bold")

    fig.tight_layout()
    path = OUT / "fig7_summary.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

# ════════════════════════════════════════════════════════════════════════════
# FIG 8  —  Win/loss/tie matrix vs all baselines
# ════════════════════════════════════════════════════════════════════════════
def fig8_winloss():
    systems_order = ["SVM-RBF","VotingEnsemble","RandomForest",
                     "GradientBoosting","RandomSearch-GBT","KNN-10",
                     "SelectKBest+GBT","LR","PCA+LR"]

    wins, losses, ties = [], [], []
    for sys in systems_order:
        w = l = t = 0
        for ds in DATASETS_7:
            if ds not in skl_means.columns: continue
            c60_acc = skl_means.loc["C60.ai", ds]
            sys_acc = skl_means.loc[sys, ds]
            diff = c60_acc - sys_acc
            if abs(diff) < 0.002:
                t += 1
            elif diff > 0:
                w += 1
            else:
                l += 1
        wins.append(w); losses.append(l); ties.append(t)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(systems_order))
    width = 0.25

    sys_labels = [s.replace("RandomSearch-GBT","RandSearch\n-GBT")
                   .replace("VotingEnsemble","Voting\nEns.") for s in systems_order]

    ax.bar(x - width, wins,   width, label="C60.ai Wins",  color=GREEN_COLOR, alpha=0.85)
    ax.bar(x,         ties,   width, label="Tie",          color="#FDD835",   alpha=0.85)
    ax.bar(x + width, losses, width, label="C60.ai Loses", color=RED_COLOR,   alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(sys_labels, fontsize=8)
    ax.set_ylabel("Number of Datasets (out of 7)")
    ax.set_yticks(range(8))
    ax.set_title("C60.ai vs. Each Baseline: Win / Tie / Loss across 7 Datasets",
                 fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.axhline(3.5, color="#ccc", lw=0.7, ls="--")

    fig.tight_layout()
    path = OUT / "fig8_winloss.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="Only regenerate these figures (e.g. fig6 fig10)")
    args = parser.parse_args()

    all_figs = {
        "fig1":  fig1_grouped_bar,
        "fig2":  fig2_heatmap,
        "fig3":  fig3_rank_scatter,
        "fig4":  fig4_pipeline_dag,
        "fig5":  fig5_automl,
        "fig6":  fig6_fitness_evolution,
        "fig7":  fig7_summary_panel,
        "fig8":  fig8_winloss,
        "fig10": fig10_ablation,
    }

    to_run = args.only if args.only else list(all_figs.keys())
    print(f"Generating figures: {to_run}")
    for name in to_run:
        if name in all_figs:
            all_figs[name]()
        else:
            print(f"Unknown figure: {name}")
    print(f"\nAll figures saved to {OUT}/")
