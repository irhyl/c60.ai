"""
visualizer.py — Pipeline DAG and evolution-progress rendering.

Two public entry points:

  render_pipeline(pipeline)  → matplotlib Figure of the typed DAG
  render_evolution(log)      → matplotlib Figure of score-over-generation

Both return a matplotlib.figure.Figure and do NOT call plt.show() so the
caller controls display / saving:

    fig = render_pipeline(best_pipeline)
    fig.savefig("pipeline.png", dpi=150)
    # or: plt.show()

If matplotlib is unavailable, both functions raise ImportError with a
helpful install hint.  networkx is already a dependency (used in Pipeline
itself), so it is always available.
"""

from __future__ import annotations

from typing import Optional

from c60.core.pipeline import Pipeline
from c60.evolution.engine import EvolutionLog


# ---------------------------------------------------------------------------
# Pipeline DAG visualisation
# ---------------------------------------------------------------------------

def render_pipeline(
    pipeline: Pipeline,
    title: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """
    Render the pipeline as a directed-acyclic-graph figure.

    Nodes are coloured by step_type and labelled with the operation name.
    Edges show the direction of data flow.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to render (need not be fitted).
    title : str, optional
        Figure title.  Defaults to the pipeline's name.
    figsize : tuple
        (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "render_pipeline() requires matplotlib.  "
            "Install it with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title or pipeline.name or "Pipeline DAG", fontsize=13, pad=12)
    ax.axis("off")

    G = pipeline.graph
    if len(G) == 0:
        ax.text(
            0.5, 0.5, "(empty pipeline)",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )
        return fig

    # ---- Node labels -------------------------------------------------
    labels: dict = {}
    for step in pipeline.get_nodes():
        labels[step.id] = f"{step.name}\n[{step.step_type}]"

    # ---- Layout ------------------------------------------------------
    # Try graphviz hierarchical layout; fall back to spring if unavailable
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        try:
            # Sugiyama-style layered layout via multipartite
            topo = list(nx.topological_sort(G))
            layers: dict = {}
            for node in topo:
                preds = list(G.predecessors(node))
                layers[node] = (max(layers[p] for p in preds) + 1) if preds else 0
            # Spread nodes within the same layer vertically
            layer_counts: dict = {}
            pos = {}
            for node in topo:
                layer = layers[node]
                idx = layer_counts.get(layer, 0)
                layer_counts[layer] = idx + 1
                pos[node] = (layer * 2.0, -idx * 1.5)
        except Exception:
            pos = nx.spring_layout(G, seed=42)

    # ---- Colours by step_type ----------------------------------------
    _TYPE_COLORS = {
        "scaler":                   "#4fc3f7",   # light blue
        "dim_reducer":              "#ce93d8",   # purple
        "feature_selector":         "#80cbc4",   # teal
        "feature_selector_regression": "#80cbc4",
        "feature_engineer":         "#fff176",   # yellow
        "classifier":               "#ef9a9a",   # red
        "regressor":                "#ffcc80",   # orange
    }

    node_ids = list(G.nodes())
    node_colors = []
    for nid in node_ids:
        step = pipeline.get_step(nid)
        color = _TYPE_COLORS.get(
            step.step_type if step else "", "#b0bec5"  # grey fallback
        )
        node_colors.append(color)

    # ---- Draw --------------------------------------------------------
    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        labels=labels,
        node_color=node_colors,
        node_size=2200,
        font_size=8,
        font_weight="bold",
        arrows=True,
        arrowsize=18,
        edge_color="#555555",
        width=1.5,
    )

    # ---- Legend (only types present in this graph) -------------------
    present_types = {
        pipeline.get_step(nid).step_type
        for nid in node_ids
        if pipeline.get_step(nid) is not None
    }
    patches = [
        mpatches.Patch(color=_TYPE_COLORS.get(t, "#b0bec5"), label=t)
        for t in sorted(present_types)
    ]
    if patches:
        ax.legend(handles=patches, loc="upper left", fontsize=7, framealpha=0.85)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Evolution progress visualisation
# ---------------------------------------------------------------------------

def render_evolution(
    log: EvolutionLog,
    title: Optional[str] = None,
    metric_name: str = "score",
    figsize: tuple = (8, 4),
):
    """
    Render best and mean score over generations.

    Parameters
    ----------
    log : EvolutionLog
        History returned by engine.history().
    title : str, optional
        Figure title.
    metric_name : str
        Y-axis label (e.g. "accuracy", "r2").
    figsize : tuple
        (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "render_evolution() requires matplotlib.  "
            "Install it with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title or "Evolution Progress", fontsize=13, pad=10)
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel(metric_name.capitalize(), fontsize=11)

    if not log.records:
        ax.text(
            0.5, 0.5, "(no data)",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )
        return fig

    generations = [r.generation for r in log.records]
    best_scores = log.best_scores()
    mean_scores = log.mean_scores()

    ax.plot(
        generations, best_scores,
        "o-", color="#e53935", label="Best", linewidth=2, markersize=5,
    )
    ax.plot(
        generations, mean_scores,
        "s--", color="#1e88e5", label="Mean", linewidth=1.5, alpha=0.75, markersize=4,
    )
    ax.fill_between(
        generations, mean_scores, best_scores,
        alpha=0.08, color="#e53935",
    )

    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate final best
    if best_scores:
        ax.annotate(
            f"{best_scores[-1]:.4f}",
            xy=(generations[-1], best_scores[-1]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, color="#e53935",
        )

    fig.tight_layout()
    return fig
