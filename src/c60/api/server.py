"""
server.py — FastAPI application for C60.ai async evolution jobs.

Architecture
------------
Job lifecycle:
  POST /jobs          → creates a Job, starts a background thread, returns 202
  GET  /jobs/{id}     → returns current status (pending / running / complete / failed)
  GET  /jobs/{id}/result → returns full JobResult once complete

Jobs are stored in an in-memory dict (_JOBS) keyed by UUID.  This is
intentionally simple — Phase 9 would replace this with a Redis-backed
queue and persistent storage.

The background thread runs EvolutionEngine.fit() and writes its result
back to the Job object.  Thread safety is guaranteed by Python's GIL for
dict reads/writes and by a per-job threading.Event for completion signalling.

Running
-------
    uvicorn c60.api.server:app --reload --port 8000

    # or from Python:
    import uvicorn
    from c60.api.server import app
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException

from c60.api.models import (
    GenerationSummary,
    HealthResponse,
    JobResult,
    JobStatus,
    RegistryInfo,
    RunRequest,
)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="C60.ai Evolution API",
    description="Async REST API for running AutoML pipeline evolution jobs.",
    version="0.2.0",
)


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------

@dataclass
class _Job:
    job_id: str
    status: str = "pending"
    result: Optional[JobResult] = None
    done: threading.Event = field(default_factory=threading.Event)


_JOBS: Dict[str, _Job] = {}
_JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_evolution(job_id: str, req: RunRequest) -> None:
    """Execute the evolution job in a background thread."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        return

    job.status = "running"

    try:
        from c60.evolution.engine import EvolutionEngine

        X = np.array(req.X, dtype=np.float64)
        y = np.array(req.y)

        engine = EvolutionEngine(
            population_size=req.population_size,
            max_generations=req.max_generations,
            cv=req.cv,
            task=req.task,
            metric=req.metric,
            eval_timeout=req.eval_timeout,
            random_seed=req.random_seed,
        )

        best_pipeline = engine.fit(X, y)
        log = engine.history()

        # Build step list in topological order
        pipeline_steps = [
            step.name for step in best_pipeline.topological_order()
        ]

        gen_history = [
            GenerationSummary(
                generation=rec.generation,
                best_score=rec.best_score,
                mean_score=rec.mean_score,
                wall_time=rec.wall_time,
            )
            for rec in log.records
        ]

        job.result = JobResult(
            job_id=job_id,
            status="complete",
            best_score=log.final_best(),
            generations_run=log.generations_run(),
            pipeline_steps=pipeline_steps,
            generation_history=gen_history,
        )
        job.status = "complete"

    except Exception as exc:
        job.result = JobResult(
            job_id=job_id,
            status="failed",
            error=str(exc),
        )
        job.status = "failed"

    finally:
        job.done.set()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Health check — always returns 200 if the server is running."""
    from c60 import __version__
    return HealthResponse(status="ok", version=__version__)


@app.get("/info", response_model=RegistryInfo, tags=["system"])
def registry_info():
    """Return a summary of the default operation registry."""
    from c60.core.registry import default_registry
    specs = list(default_registry._specs)

    step_types = sorted({s.step_type for s in specs})
    operations: Dict[str, List[str]] = {st: [] for st in step_types}
    for spec in specs:
        operations[spec.step_type].append(spec.op_class.__name__)

    return RegistryInfo(
        n_operations=len(specs),
        step_types=step_types,
        operations=operations,
    )


@app.post("/jobs", response_model=JobStatus, status_code=202, tags=["jobs"])
def submit_job(req: RunRequest):
    """
    Submit a new evolution job.

    Returns 202 Accepted with a job_id.  Poll GET /jobs/{job_id} for status.
    """
    if len(req.X) != len(req.y):
        raise HTTPException(
            status_code=422,
            detail=f"X has {len(req.X)} rows but y has {len(req.y)} elements.",
        )

    job_id = str(uuid.uuid4())
    job = _Job(job_id=job_id)

    with _JOBS_LOCK:
        _JOBS[job_id] = job

    thread = threading.Thread(
        target=_run_evolution,
        args=(job_id, req),
        daemon=True,
        name=f"c60-job-{job_id[:8]}",
    )
    thread.start()

    return JobStatus(
        job_id=job_id,
        status="pending",
        message="Job submitted. Poll GET /jobs/{job_id} for status.",
    )


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["jobs"])
def get_job_status(job_id: str):
    """Return the current status of a job."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatus(job_id=job_id, status=job.status)


@app.get("/jobs/{job_id}/result", response_model=JobResult, tags=["jobs"])
def get_job_result(job_id: str):
    """
    Return the full result of a completed job.

    Returns the current status (running / pending) if the job is not yet done.
    """
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job.result is not None:
        return job.result

    # Job still in progress — return a stub
    return JobResult(job_id=job_id, status=job.status)


@app.delete("/jobs/{job_id}", status_code=204, tags=["jobs"])
def delete_job(job_id: str):
    """Remove a completed job from the in-memory store."""
    with _JOBS_LOCK:
        if job_id not in _JOBS:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        job = _JOBS.pop(job_id)

    if not job.done.is_set():
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a running job.",
        )
