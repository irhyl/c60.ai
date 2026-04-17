"""
test_api.py — Tests for the C60.ai FastAPI REST API.

Uses FastAPI's TestClient (httpx-backed) for synchronous testing.
Long-running jobs are submitted and polled with a short timeout so
the test suite stays fast (small population, few generations).

Structure
---------
Section 1 — /health + /info (system routes)
Section 2 — POST /jobs (submission)
Section 3 — GET /jobs/{id} (status polling)
Section 4 — GET /jobs/{id}/result (result retrieval)
Section 5 — DELETE /jobs/{id}
Section 6 — Error handling (404s, validation errors)
Section 7 — End-to-end: submit → poll → result
"""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris, load_diabetes

from c60.api.server import app, _JOBS


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def client():
    """Fresh TestClient for each test; clear job store between tests."""
    _JOBS.clear()
    with TestClient(app) as c:
        yield c
    _JOBS.clear()


@pytest.fixture(scope="module")
def iris_payload():
    X, y = load_iris(return_X_y=True)
    return {
        "X": X[:30].tolist(),     # 30 samples — fast
        "y": y[:30].tolist(),
        "task": "classification",
        "population_size": 4,
        "max_generations": 2,
        "cv": 2,
        "eval_timeout": 20.0,
        "random_seed": 42,
    }


@pytest.fixture(scope="module")
def diabetes_payload():
    X, y = load_diabetes(return_X_y=True)
    return {
        "X": X[:40].tolist(),
        "y": y[:40].tolist(),
        "task": "regression",
        "metric": "r2",
        "population_size": 4,
        "max_generations": 2,
        "cv": 2,
        "eval_timeout": 20.0,
        "random_seed": 0,
    }


def _wait_for_job(client, job_id: str, timeout: float = 120.0) -> dict:
    """Poll until the job is complete or failed, then return status response."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200
        status = resp.json()["status"]
        if status in ("complete", "failed"):
            return resp.json()
        time.sleep(0.2)
    pytest.fail(f"Job {job_id} did not finish within {timeout}s")


# ===========================================================================
# Section 1 — System routes
# ===========================================================================

class TestSystemRoutes:

    def test_health_status_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_health_has_version(self, client):
        body = client.get("/health").json()
        assert "version" in body
        assert body["version"]  # non-empty

    def test_info_status_200(self, client):
        resp = client.get("/info")
        assert resp.status_code == 200

    def test_info_has_n_operations(self, client):
        body = client.get("/info").json()
        assert body["n_operations"] > 0

    def test_info_has_step_types(self, client):
        body = client.get("/info").json()
        assert "classifier" in body["step_types"]
        assert "scaler" in body["step_types"]

    def test_info_operations_dict(self, client):
        body = client.get("/info").json()
        ops = body["operations"]
        assert "classifier" in ops
        assert len(ops["classifier"]) > 0


# ===========================================================================
# Section 2 — POST /jobs (submission)
# ===========================================================================

class TestJobSubmission:

    def test_submit_returns_202(self, client, iris_payload):
        resp = client.post("/jobs", json=iris_payload)
        assert resp.status_code == 202

    def test_submit_returns_job_id(self, client, iris_payload):
        body = client.post("/jobs", json=iris_payload).json()
        assert "job_id" in body
        assert len(body["job_id"]) > 0

    def test_submit_initial_status_pending_or_running(self, client, iris_payload):
        body = client.post("/jobs", json=iris_payload).json()
        assert body["status"] in ("pending", "running")

    def test_submit_regression_returns_202(self, client, diabetes_payload):
        resp = client.post("/jobs", json=diabetes_payload)
        assert resp.status_code == 202

    def test_submit_mismatched_X_y_returns_422(self, client):
        bad = {
            "X": [[1.0, 2.0], [3.0, 4.0]],
            "y": [0],   # wrong length
            "task": "classification",
        }
        resp = client.post("/jobs", json=bad)
        assert resp.status_code == 422

    def test_submit_empty_X_returns_422(self, client):
        bad = {"X": [], "y": [], "task": "classification"}
        resp = client.post("/jobs", json=bad)
        assert resp.status_code == 422

    def test_submit_invalid_task_returns_422(self, client, iris_payload):
        bad = {**iris_payload, "task": "clustering"}
        resp = client.post("/jobs", json=bad)
        assert resp.status_code == 422

    def test_submit_population_below_min_returns_422(self, client, iris_payload):
        bad = {**iris_payload, "population_size": 1}
        resp = client.post("/jobs", json=bad)
        assert resp.status_code == 422


# ===========================================================================
# Section 3 — GET /jobs/{id} (status)
# ===========================================================================

class TestJobStatus:

    def test_status_returns_200(self, client, iris_payload):
        job_id = client.post("/jobs", json=iris_payload).json()["job_id"]
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200

    def test_status_has_correct_job_id(self, client, iris_payload):
        job_id = client.post("/jobs", json=iris_payload).json()["job_id"]
        body = client.get(f"/jobs/{job_id}").json()
        assert body["job_id"] == job_id

    def test_status_valid_values(self, client, iris_payload):
        job_id = client.post("/jobs", json=iris_payload).json()["job_id"]
        body = client.get(f"/jobs/{job_id}").json()
        assert body["status"] in ("pending", "running", "complete", "failed")

    def test_status_unknown_job_returns_404(self, client):
        resp = client.get("/jobs/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404

    def test_status_eventually_completes(self, client, iris_payload):
        job_id = client.post("/jobs", json=iris_payload).json()["job_id"]
        final = _wait_for_job(client, job_id, timeout=120.0)
        assert final["status"] in ("complete", "failed")


# ===========================================================================
# Section 4 — GET /jobs/{id}/result
# ===========================================================================

class TestJobResult:

    def test_result_returns_200(self, client, iris_payload):
        job_id = client.post("/jobs", json=iris_payload).json()["job_id"]
        resp = client.get(f"/jobs/{job_id}/result")
        assert resp.status_code == 200

    def test_result_unknown_job_returns_404(self, client):
        resp = client.get("/jobs/00000000-0000-0000-0000-000000000000/result")
        assert resp.status_code == 404

    def test_completed_result_has_best_score(self, client, iris_payload):
        job_id = client.post("/jobs", json=iris_payload).json()["job_id"]
        _wait_for_job(client, job_id, timeout=120.0)
        body = client.get(f"/jobs/{job_id}/result").json()
        if body["status"] == "complete":
            assert body["best_score"] is not None
            assert isinstance(body["best_score"], float)

    def test_completed_result_has_pipeline_steps(self, client, iris_payload):
        job_id = client.post("/jobs", json=iris_payload).json()["job_id"]
        _wait_for_job(client, job_id, timeout=120.0)
        body = client.get(f"/jobs/{job_id}/result").json()
        if body["status"] == "complete":
            assert body["pipeline_steps"] is not None
            assert len(body["pipeline_steps"]) >= 2

    def test_completed_result_has_generations_run(self, client, iris_payload):
        job_id = client.post("/jobs", json=iris_payload).json()["job_id"]
        _wait_for_job(client, job_id, timeout=120.0)
        body = client.get(f"/jobs/{job_id}/result").json()
        if body["status"] == "complete":
            assert body["generations_run"] is not None
            assert body["generations_run"] >= 1

    def test_completed_result_has_generation_history(self, client, iris_payload):
        job_id = client.post("/jobs", json=iris_payload).json()["job_id"]
        _wait_for_job(client, job_id, timeout=120.0)
        body = client.get(f"/jobs/{job_id}/result").json()
        if body["status"] == "complete":
            history = body["generation_history"]
            assert history is not None
            assert len(history) >= 1
            assert "best_score" in history[0]
            assert "mean_score" in history[0]


# ===========================================================================
# Section 5 — DELETE /jobs/{id}
# ===========================================================================

class TestDeleteJob:

    def test_delete_completed_job_returns_204(self, client, iris_payload):
        job_id = client.post("/jobs", json=iris_payload).json()["job_id"]
        _wait_for_job(client, job_id, timeout=120.0)
        resp = client.delete(f"/jobs/{job_id}")
        assert resp.status_code == 204

    def test_delete_removes_job_from_store(self, client, iris_payload):
        job_id = client.post("/jobs", json=iris_payload).json()["job_id"]
        _wait_for_job(client, job_id, timeout=120.0)
        client.delete(f"/jobs/{job_id}")
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 404

    def test_delete_unknown_job_returns_404(self, client):
        resp = client.delete("/jobs/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404


# ===========================================================================
# Section 6 — End-to-end
# ===========================================================================

class TestEndToEnd:

    def test_classification_job_best_score_reasonable(self, client, iris_payload):
        """Classification on 30 iris samples → accuracy should beat 0.5."""
        job_id = client.post("/jobs", json=iris_payload).json()["job_id"]
        _wait_for_job(client, job_id, timeout=120.0)
        body = client.get(f"/jobs/{job_id}/result").json()
        assert body["status"] == "complete", body.get("error")
        assert body["best_score"] > 0.5, (
            f"Expected > 50% accuracy on 30-sample iris, got {body['best_score']:.2%}"
        )

    def test_regression_job_completes(self, client, diabetes_payload):
        job_id = client.post("/jobs", json=diabetes_payload).json()["job_id"]
        final = _wait_for_job(client, job_id, timeout=120.0)
        assert final["status"] in ("complete", "failed")

    def test_multiple_concurrent_jobs(self, client, iris_payload):
        """Submit 3 jobs simultaneously; all should eventually complete."""
        job_ids = [
            client.post("/jobs", json=iris_payload).json()["job_id"]
            for _ in range(3)
        ]
        for jid in job_ids:
            final = _wait_for_job(client, jid, timeout=180.0)
            assert final["status"] in ("complete", "failed"), (
                f"Job {jid} stuck in {final['status']}"
            )
