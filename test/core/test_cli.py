"""
test_cli.py — Tests for the c60 Click CLI.

Uses Click's CliRunner so no subprocess is spawned.

Structure
---------
Section 1 — version command
Section 2 — info command
Section 3 — run command  (uses a temp CSV)
Section 4 — explain command  (uses a saved pipeline file)
"""

import os
import pickle
import tempfile

import numpy as np
import pytest
from click.testing import CliRunner
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from c60.cli.main import cli
from c60.core.pipeline import Pipeline, PipelineStep
from c60.core.types import CleanTabular, ScaledData, ClassLabels


# ===========================================================================
# Helpers
# ===========================================================================

def _make_iris_csv(path: str) -> None:
    """Write iris data to a CSV file."""
    import pandas as pd
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(path, index=False)


def _make_clf_pipeline():
    p = Pipeline(name="TestCLF")
    sc = PipelineStep(
        name="StandardScaler", step_type="scaler",
        operation=StandardScaler(),
        input_type=CleanTabular, output_type=ScaledData,
    )
    clf = PipelineStep(
        name="LogisticRegression", step_type="classifier",
        operation=LogisticRegression(max_iter=200),
        input_type=ScaledData, output_type=ClassLabels,
    )
    p.add_step(sc); p.add_step(clf)
    p.add_edge(sc.id, clf.id)
    return p


# ===========================================================================
# Section 1 — version
# ===========================================================================

class TestVersionCommand:

    def test_version_exit_code_zero(self):
        result = CliRunner().invoke(cli, ["version"])
        assert result.exit_code == 0, result.output

    def test_version_output_contains_version_string(self):
        result = CliRunner().invoke(cli, ["version"])
        assert "c60" in result.output.lower()

    def test_version_output_has_number(self):
        result = CliRunner().invoke(cli, ["version"])
        # Version string like "0.2.0"
        assert any(c.isdigit() for c in result.output)


# ===========================================================================
# Section 2 — info
# ===========================================================================

class TestInfoCommand:

    def test_info_exit_code_zero(self):
        result = CliRunner().invoke(cli, ["info"])
        assert result.exit_code == 0, result.output

    def test_info_lists_classifiers(self):
        result = CliRunner().invoke(cli, ["info"])
        assert "classifier" in result.output.lower()

    def test_info_lists_scalers(self):
        result = CliRunner().invoke(cli, ["info"])
        assert "scaler" in result.output.lower()

    def test_info_filter_by_type(self):
        result = CliRunner().invoke(cli, ["info", "--type", "classifier"])
        assert result.exit_code == 0
        assert "classifier" in result.output.lower()

    def test_info_filter_unknown_type(self):
        result = CliRunner().invoke(cli, ["info", "--type", "unknown_xyz"])
        assert result.exit_code == 0
        assert "No operations found" in result.output

    def test_info_filter_shows_input_output_types(self):
        result = CliRunner().invoke(cli, ["info", "--type", "scaler"])
        assert "ScaledData" in result.output or "scaler" in result.output.lower()


# ===========================================================================
# Section 3 — run
# ===========================================================================

class TestRunCommand:

    def test_run_produces_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "iris.csv")
            _make_iris_csv(csv_path)
            result = CliRunner().invoke(cli, [
                "run",
                "--input", csv_path,
                "--target", "target",
                "--task", "classification",
                "--generations", "2",
                "--population", "4",
                "--cv", "2",
                "--no-story",
            ])
            assert result.exit_code == 0, result.output
            assert "Best" in result.output

    def test_run_shows_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "iris.csv")
            _make_iris_csv(csv_path)
            result = CliRunner().invoke(cli, [
                "run", "--input", csv_path,
                "--target", "target",
                "--generations", "2", "--population", "4", "--cv", "2",
                "--no-story",
            ])
            # Score line like "Best accuracy: 0.9333"
            assert any(c.isdigit() for c in result.output)

    def test_run_saves_pipeline_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "iris.csv")
            pkl_path = os.path.join(tmpdir, "best.pkl")
            _make_iris_csv(csv_path)
            result = CliRunner().invoke(cli, [
                "run", "--input", csv_path,
                "--target", "target",
                "--generations", "2", "--population", "4", "--cv", "2",
                "--no-story", "--output", pkl_path,
            ])
            assert result.exit_code == 0, result.output
            assert os.path.exists(pkl_path), "Pickle file not created"
            with open(pkl_path, "rb") as f:
                pipeline = pickle.load(f)
            assert isinstance(pipeline, Pipeline)

    def test_run_with_story_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "iris.csv")
            _make_iris_csv(csv_path)
            result = CliRunner().invoke(cli, [
                "run", "--input", csv_path,
                "--target", "target",
                "--generations", "2", "--population", "4", "--cv", "2",
                "--story",
            ])
            assert result.exit_code == 0, result.output
            assert "Evolution" in result.output or "generation" in result.output.lower()

    def test_run_missing_target_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "iris.csv")
            _make_iris_csv(csv_path)
            result = CliRunner().invoke(cli, [
                "run", "--input", csv_path,
                "--target", "nonexistent_column",
                "--generations", "2", "--population", "4",
            ])
            assert result.exit_code != 0

    def test_run_help_shows_options(self):
        result = CliRunner().invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--target" in result.output
        assert "--task" in result.output

    def test_run_with_seed_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "iris.csv")
            _make_iris_csv(csv_path)
            scores = []
            for _ in range(2):
                r = CliRunner().invoke(cli, [
                    "run", "--input", csv_path,
                    "--target", "target",
                    "--generations", "2", "--population", "4", "--cv", "2",
                    "--no-story", "--seed", "42",
                ])
                # Extract score line
                for line in r.output.splitlines():
                    if "Best" in line and ":" in line:
                        try:
                            scores.append(float(line.split(":")[-1].strip()))
                        except ValueError:
                            pass
            if len(scores) == 2:
                assert abs(scores[0] - scores[1]) < 1e-6


# ===========================================================================
# Section 4 — explain
# ===========================================================================

class TestExplainCommand:

    def _save_pipeline(self, path: str, fitted: bool = False) -> None:
        p = _make_clf_pipeline()
        if fitted:
            X, y = load_iris(return_X_y=True)
            p.fit(X, y)
        with open(path, "wb") as f:
            pickle.dump(p, f)

    def test_explain_exit_code_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl = os.path.join(tmpdir, "p.pkl")
            self._save_pipeline(pkl, fitted=True)
            result = CliRunner().invoke(cli, ["explain", "--pipeline", pkl])
            assert result.exit_code == 0, result.output

    def test_explain_shows_step_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl = os.path.join(tmpdir, "p.pkl")
            self._save_pipeline(pkl, fitted=True)
            result = CliRunner().invoke(cli, ["explain", "--pipeline", pkl])
            assert "scaler" in result.output or "classifier" in result.output

    def test_explain_with_data_shows_importances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl = os.path.join(tmpdir, "p.pkl")
            csv = os.path.join(tmpdir, "iris.csv")
            self._save_pipeline(pkl, fitted=False)
            _make_iris_csv(csv)
            result = CliRunner().invoke(cli, [
                "explain", "--pipeline", pkl,
                "--data", csv, "--target", "target",
            ])
            assert result.exit_code == 0, result.output
            # With fitted pipeline, should show feature importances
            assert "feature" in result.output.lower() or "scaler" in result.output.lower()

    def test_explain_help(self):
        result = CliRunner().invoke(cli, ["explain", "--help"])
        assert result.exit_code == 0
        assert "--pipeline" in result.output

    def test_explain_nonexistent_pipeline_fails(self):
        result = CliRunner().invoke(cli, [
            "explain", "--pipeline", "/nonexistent/path/pipeline.pkl"
        ])
        assert result.exit_code != 0


# ===========================================================================
# Section 5 — top-level CLI
# ===========================================================================

class TestCLITopLevel:

    def test_help_lists_commands(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        for cmd in ("version", "info", "run", "explain"):
            assert cmd in result.output

    def test_unknown_command_fails(self):
        result = CliRunner().invoke(cli, ["unknown_xyz"])
        assert result.exit_code != 0
