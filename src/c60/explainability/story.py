"""
story.py — PipelineStory: human-readable narrative of an evolution run.

PipelineStory wraps an EvolutionLog (from engine.history()) and optionally
the best Pipeline, and produces three kinds of output:

  narrate()            — full multi-paragraph narrative of the entire run
  pipeline_summary()   — one paragraph describing a specific pipeline
  generation_table()   — compact fixed-width table of per-generation stats

All output is plain text (no HTML/Markdown) so it renders well in a terminal
or can be embedded in reports without extra dependencies.
"""

from __future__ import annotations

from typing import List, Optional

from c60.core.pipeline import Pipeline
from c60.evolution.engine import EvolutionLog
from c60.explainability.introspector import PipelineIntrospector


class PipelineStory:
    """
    Generate human-readable narratives about an evolution run.

    Parameters
    ----------
    log : EvolutionLog
        The history returned by engine.history().
    best_pipeline : Pipeline, optional
        The best pipeline found.  When provided, narrate() appends a
        pipeline summary.
    feature_names : list of str, optional
        Input feature names used for top-feature reporting.
    task : str
        "classification" or "regression" — affects metric labelling.
    metric : str
        Display name of the optimisation metric (e.g. "accuracy", "r2").
    """

    def __init__(
        self,
        log: EvolutionLog,
        best_pipeline: Optional[Pipeline] = None,
        feature_names: Optional[List[str]] = None,
        task: str = "classification",
        metric: str = "accuracy",
    ) -> None:
        self._log = log
        self._best = best_pipeline
        self._feature_names = feature_names
        self._task = task
        self._metric = metric

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def narrate(self) -> str:
        """Return a full narrative of the evolution run."""
        parts: List[str] = []

        n_gen = self._log.generations_run()
        parts.append("=== C60.ai Evolution Run Summary ===\n")

        if n_gen == 0:
            parts.append("No generations were completed.")
            return "\n".join(parts)

        first = self._log.records[0]
        last = self._log.records[-1]

        parts.append(
            f"Task: {self._task}  |  Metric: {self._metric}  |  "
            f"Generations completed: {n_gen}"
        )
        delta = last.best_score - first.best_score
        parts.append(
            f"Initial best {self._metric}: {first.best_score:.4f}  →  "
            f"Final best {self._metric}: {last.best_score:.4f}  "
            f"(improvement: {delta:+.4f})"
        )
        avg_time = last.wall_time / n_gen if n_gen > 0 else 0.0
        parts.append(
            f"Wall time: {last.wall_time:.1f}s  "
            f"(avg {avg_time:.2f}s/generation)"
        )
        parts.append("")

        # --- Score progression ---
        parts.append("-- Score Progression --")
        best_scores = self._log.best_scores()
        mean_scores = self._log.mean_scores()
        improvement_gens: List[int] = []
        for i, (b, m) in enumerate(zip(best_scores, mean_scores)):
            marker = ""
            if i > 0 and b > best_scores[i - 1] + 1e-6:
                marker = " ← new best"
                improvement_gens.append(i)
            parts.append(
                f"  Gen {i:3d}: best={b:.4f}, mean={m:.4f}{marker}"
            )
        parts.append("")

        if improvement_gens:
            parts.append(
                f"Best score improved in {len(improvement_gens)} "
                f"generation(s): {improvement_gens}"
            )
        else:
            parts.append(
                "Best score did not improve after generation 0 "
                "(population may have converged immediately)."
            )
        parts.append("")

        # --- Convergence / plateau ---
        plateau_len = _plateau_length(best_scores)
        if plateau_len >= 2:
            parts.append(
                f"Convergence detected: best score was stable for the "
                f"last {plateau_len} generation(s)."
            )
            parts.append("")

        # --- Best pipeline ---
        if self._best is not None:
            parts.append("-- Best Pipeline --")
            parts.append(self.pipeline_summary(self._best))

        return "\n".join(parts)

    def pipeline_summary(
        self,
        pipeline: Pipeline,
        feature_names: Optional[List[str]] = None,
    ) -> str:
        """
        Return a concise description of a pipeline's structure.

        If the pipeline has been fitted and a predictive step contains
        importances, the top-5 features are appended.
        """
        names = feature_names or self._feature_names
        steps = pipeline.topological_order()

        # Build the step chain string
        step_descs: List[str] = []
        for step in steps:
            desc = step.name
            if step.params:
                # Show at most two params to keep lines short
                param_str = ", ".join(
                    f"{k}={v}" for k, v in list(step.params.items())[:2]
                )
                desc += f"({param_str})"
            step_descs.append(f"[{step.step_type}] {desc}")

        chain = " → ".join(step_descs)
        lines = [
            f"Pipeline '{pipeline.name}' — {len(steps)} step(s):",
            f"  {chain}",
        ]

        # Optionally append top-feature importance
        if names is not None:
            report = PipelineIntrospector().inspect(pipeline)
            top = report.top_features(feature_names=names, k=5)
            if top:
                top_str = ", ".join(f"{n} ({w:.3f})" for n, w in top)
                lines.append(f"  Top features: {top_str}")

        return "\n".join(lines)

    def generation_table(self) -> str:
        """Return a compact fixed-width table of generation statistics."""
        header = f"{'Gen':>4}  {'Best':>8}  {'Mean':>8}  {'Time(s)':>8}"
        sep = "-" * len(header)
        rows = [header, sep]
        for rec in self._log.records:
            rows.append(
                f"{rec.generation:>4}  {rec.best_score:>8.4f}  "
                f"{rec.mean_score:>8.4f}  {rec.wall_time:>8.2f}"
            )
        return "\n".join(rows)

    def __repr__(self) -> str:
        return (
            f"PipelineStory(generations={self._log.generations_run()}, "
            f"task={self._task!r}, metric={self._metric!r})"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _plateau_length(scores: List[float]) -> int:
    """Return the number of trailing generations with no improvement."""
    if len(scores) < 2:
        return 0
    count = 0
    for i in range(len(scores) - 1, 0, -1):
        if scores[i] > scores[i - 1] + 1e-6:
            break
        count += 1
    return count
