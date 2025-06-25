import os
import tempfile
import pandas as pd
from c60 import AutoML

def test_automl_save_load():
    # Create dummy data
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    X = df[["x"]]
    y = df["y"]
    automl = AutoML(task="classification")
    automl.fit(X, y)
    # Save model
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        model_path = tmp.name
    automl.save(model_path)
    assert os.path.exists(model_path)
    # Load model
    loaded = AutoML.load(model_path)
    assert isinstance(loaded, AutoML)
    assert loaded.best_score_ == automl.best_score_
    os.remove(model_path)
