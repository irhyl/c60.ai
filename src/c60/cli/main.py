"""
main.py — Click CLI entry point for C60.ai.

Commands
--------
  c60 version              Print version and quit.
  c60 info                 Show the default registry contents.
  c60 run   --input CSV    Run evolution on a CSV dataset.
  c60 explain --pipeline PKL  Explain a saved pipeline.

All commands write to stdout (plain text).  The `run` command also
accepts --output to save the best pipeline as a pickle file and
--story to print the full PipelineStory narrative.

Examples
--------
    c60 version
    c60 info
    c60 run --input iris.csv --target species --task classification \\
            --generations 10 --population 20 --output best.pkl
    c60 explain --pipeline best.pkl --data iris.csv --target species
"""

from __future__ import annotations

import pickle
import sys
from typing import Optional

import click


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """C60.ai — Molecular Evolution AutoML Framework."""


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

@cli.command()
def version():
    """Print the installed version of c60."""
    from c60 import __version__
    click.echo(f"c60 version {__version__}")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--type", "step_type", default=None,
    help="Filter by step type (e.g. classifier, scaler, dim_reducer).",
)
def info(step_type: Optional[str]):
    """Print the contents of the default operation registry."""
    from c60.core.registry import default_registry

    if step_type:
        specs = default_registry.get_by_type(step_type)
        if not specs:
            click.echo(f"No operations found for step type '{step_type}'.")
            return
        click.echo(f"Operations with step_type='{step_type}':")
        for spec in specs:
            click.echo(
                f"  {spec.op_class.__name__:<35} "
                f"in={spec.input_type.__name__} → out={spec.output_type.__name__}"
            )
    else:
        click.echo(default_registry.summary())


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--input",  "input_path",  required=True,
              type=click.Path(exists=True), help="Path to input CSV file.")
@click.option("--target", "target_col",  required=True,
              help="Name of the target column in the CSV.")
@click.option("--task",   default="classification",
              type=click.Choice(["classification", "regression"]),
              help="Supervised learning task type.")
@click.option("--metric", default=None,
              help="Fitness metric (accuracy, r2, …). Uses task default if omitted.")
@click.option("--generations", default=10, show_default=True,
              help="Maximum number of GA generations.")
@click.option("--population",  default=20, show_default=True,
              help="Population size per generation.")
@click.option("--cv",          default=3,  show_default=True,
              help="Number of cross-validation folds.")
@click.option("--timeout",     default=60.0, show_default=True,
              help="Per-pipeline evaluation timeout in seconds.")
@click.option("--seed",        default=None, type=int,
              help="Random seed for reproducibility.")
@click.option("--output",      default=None,
              type=click.Path(), help="Save best pipeline to this pickle file.")
@click.option("--story/--no-story", default=True, show_default=True,
              help="Print the full PipelineStory narrative.")
def run(
    input_path, target_col, task, metric, generations, population,
    cv, timeout, seed, output, story,
):
    """Run genetic algorithm evolution on a CSV dataset."""
    try:
        import pandas as pd
    except ImportError:
        click.echo("ERROR: pandas is required for the run command.  "
                   "Install it with: pip install pandas", err=True)
        sys.exit(1)

    # --- Load data ---
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        click.echo(f"ERROR reading '{input_path}': {exc}", err=True)
        sys.exit(1)

    if target_col not in df.columns:
        click.echo(
            f"ERROR: target column '{target_col}' not found in CSV.  "
            f"Available columns: {list(df.columns)}", err=True
        )
        sys.exit(1)

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    click.echo(
        f"Loaded '{input_path}': {X.shape[0]} samples × {X.shape[1]} features, "
        f"task={task}"
    )

    # --- Run evolution ---
    from c60.evolution.engine import EvolutionEngine
    engine = EvolutionEngine(
        population_size=population,
        max_generations=generations,
        cv=cv,
        task=task,
        metric=metric,
        eval_timeout=timeout,
        random_seed=seed,
    )

    click.echo(f"Running evolution: pop={population}, max_gen={generations} ...")
    best_pipeline = engine.fit(X, y)

    # --- Report ---
    final_score = engine.history().final_best()
    metric_label = metric or ("accuracy" if task == "classification" else "r2")
    click.echo(f"\nBest {metric_label}: {final_score:.4f}")

    if story:
        from c60.explainability.story import PipelineStory
        feature_names = list(df.drop(columns=[target_col]).columns)
        narrator = PipelineStory(
            engine.history(),
            best_pipeline=best_pipeline,
            feature_names=feature_names,
            task=task,
            metric=metric_label,
        )
        click.echo("\n" + narrator.narrate())

    # --- Save ---
    if output:
        try:
            with open(output, "wb") as f:
                pickle.dump(best_pipeline, f)
            click.echo(f"\nBest pipeline saved to '{output}'.")
        except Exception as exc:
            click.echo(f"WARNING: could not save pipeline: {exc}", err=True)


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--pipeline", "pipeline_path", required=True,
              type=click.Path(exists=True),
              help="Path to a pickled Pipeline file.")
@click.option("--data",     "data_path",     default=None,
              type=click.Path(exists=True),
              help="CSV file to fit for feature importances (optional).")
@click.option("--target",   "target_col",    default=None,
              help="Target column in the CSV (required if --data is given).")
@click.option("--top-k",    default=10, show_default=True,
              help="Number of top features to display.")
def explain(pipeline_path, data_path, target_col, top_k):
    """Explain a previously saved pipeline."""
    # --- Load pipeline ---
    try:
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)
    except Exception as exc:
        click.echo(f"ERROR loading pipeline: {exc}", err=True)
        sys.exit(1)

    feature_names = None

    # --- Optionally fit on data for importances ---
    if data_path:
        try:
            import pandas as pd
            df = pd.read_csv(data_path)
        except ImportError:
            click.echo("ERROR: pandas is required when --data is given.", err=True)
            sys.exit(1)
        except Exception as exc:
            click.echo(f"ERROR reading '{data_path}': {exc}", err=True)
            sys.exit(1)

        if target_col and target_col in df.columns:
            X = df.drop(columns=[target_col]).values
            y = df[target_col].values
            feature_names = list(df.drop(columns=[target_col]).columns)
            try:
                pipeline.fit(X, y)
            except Exception as exc:
                click.echo(f"WARNING: could not fit pipeline: {exc}", err=True)

    # --- Introspect ---
    from c60.explainability.introspector import PipelineIntrospector
    report = PipelineIntrospector().inspect(pipeline)
    click.echo(report.summary())

    top = report.top_features(feature_names=feature_names, k=top_k)
    if top:
        click.echo("\nFeature importances:")
        for name, weight in top:
            bar = "█" * int(weight * 40)
            click.echo(f"  {name:<30} {weight:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cli()


if __name__ == "__main__":
    main()
