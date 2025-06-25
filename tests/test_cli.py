import subprocess
import sys
import os
import tempfile
import pandas as pd

def test_cli_load_dataset():
    # Create a temporary CSV file
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    try:
        result = subprocess.run([
            sys.executable, os.path.join("interface", "cli.py"), "dataset", "load", tmp_path, "--validate"
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "Loaded dataset with shape" in result.stdout
        assert "No missing values detected" in result.stdout
    finally:
        os.remove(tmp_path)

def test_cli_generate_pipeline():
    # Create a temporary CSV file with a target column
    df = pd.DataFrame({"feature1": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    try:
        result = subprocess.run([
            sys.executable, os.path.join("interface", "cli.py"), "pipeline", "generate", "--dataset", tmp_path, "--target", "target"
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "Pipeline generated and trained" in result.stdout
        assert "Saved best pipeline" in result.stdout
        # Clean up generated model file
        if os.path.exists("best_automl_pipeline.joblib"):
            os.remove("best_automl_pipeline.joblib")
    finally:
        os.remove(tmp_path)
