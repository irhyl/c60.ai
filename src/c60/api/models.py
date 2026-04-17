"""
models.py — Pydantic request / response models for the C60.ai REST API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class RunRequest(BaseModel):
    """
    Request body for POST /jobs — submit a new evolution job.

    X and y are passed as plain JSON arrays (list-of-lists for X,
    flat list for y).  This keeps the API self-contained without
    requiring file uploads.  For larger datasets, use the CLI or
    submit pre-split arrays.
    """
    X: List[List[float]] = Field(..., description="Feature matrix (n_samples × n_features).")
    y: List[Any]         = Field(..., description="Target vector (n_samples,).")
    task: Literal["classification", "regression"] = "classification"
    metric: Optional[str] = None
    population_size: int  = Field(default=10, ge=2, le=200)
    max_generations: int  = Field(default=5,  ge=1, le=500)
    cv: int               = Field(default=3,  ge=2, le=10)
    eval_timeout: float   = Field(default=30.0, ge=1.0, le=300.0)
    random_seed: Optional[int] = None

    @field_validator("X")
    @classmethod
    def x_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("X must not be empty.")
        return v

    @field_validator("y")
    @classmethod
    def y_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("y must not be empty.")
        return v


class JobStatus(BaseModel):
    """Minimal status response returned immediately after job submission."""
    job_id: str
    status: Literal["pending", "running", "complete", "failed"]
    message: str = ""


class GenerationSummary(BaseModel):
    """Per-generation statistics for the result payload."""
    generation: int
    best_score: float
    mean_score: float
    wall_time: float


class JobResult(BaseModel):
    """
    Full result returned by GET /jobs/{job_id}/result once the job completes.
    """
    job_id: str
    status: Literal["pending", "running", "complete", "failed"]
    best_score: Optional[float] = None
    generations_run: Optional[int] = None
    pipeline_steps: Optional[List[str]] = None   # step names in topological order
    generation_history: Optional[List[GenerationSummary]] = None
    error: Optional[str] = None


class RegistryInfo(BaseModel):
    """Summary of the operation registry returned by GET /info."""
    n_operations: int
    step_types: List[str]
    operations: Dict[str, List[str]]   # step_type → [op class names]


class HealthResponse(BaseModel):
    status: str
    version: str
