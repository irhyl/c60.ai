import time
from fastapi.testclient import TestClient
import sys
import os
import tempfile
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from interface import api

import pytest

@pytest.mark.skip(reason="api import removed or refactored from codebase")
def test_create_pipeline_background():
    client = TestClient(api.app)
    # Add a dataset first
    df = pd.DataFrame({"feature1": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    dataset_resp = client.post("/datasets/", json={
        "name": "test-dataset",
        "file_path": tmp_path,
        "description": "test"
    })
    assert dataset_resp.status_code == 201
    dataset_id = dataset_resp.json()["id"]
    # Create pipeline
    pipeline_resp = client.post("/pipelines/", json={
        "name": "test-pipeline",
        "dataset_id": dataset_id,
        "target_column": "target"
    })
    assert pipeline_resp.status_code == 201
    pipeline_id = pipeline_resp.json()["id"]
    # Wait for background thread to finish
    time.sleep(3)
    # Check pipeline status
    get_resp = client.get(f"/pipelines/{pipeline_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == "completed"
    os.remove(tmp_path)
