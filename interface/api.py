"""
REST API for C60 AutoML system.

This module provides a FastAPI-based REST API for interacting with the
C60 AutoML system programmatically.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="C60 AutoML API",
    description="REST API for C60 AutoML Pipeline Generation",
    version="0.1.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demonstration
# In production, use a proper database
_datasets = {}
_pipelines = {}


class DatasetCreate(BaseModel):
    """Schema for dataset creation."""
    name: str
    file_path: str
    description: Optional[str] = None


class PipelineCreate(BaseModel):
    """Schema for pipeline creation."""
    name: str
    dataset_id: str
    target_column: str
    parameters: Optional[Dict[str, Any]] = None


@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": "C60 AutoML API",
        "version": "0.1.0",
        "documentation": "/docs"
    }


@app.post("/datasets/", status_code=status.HTTP_201_CREATED)
async def create_dataset(dataset: DatasetCreate):
    """Create a new dataset."""
    dataset_id = f"dataset_{len(_datasets) + 1}"
    _datasets[dataset_id] = {
        "id": dataset_id,
        **dataset.dict(),
        "status": "created"
    }
    logger.info(f"Created dataset {dataset_id}")
    return {"id": dataset_id, **dataset.dict()}


@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset by ID."""
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return _datasets[dataset_id]


@app.post("/pipelines/", status_code=status.HTTP_201_CREATED)
async def create_pipeline(pipeline: PipelineCreate):
    """Create and run a new pipeline."""
    if pipeline.dataset_id not in _datasets:
        raise HTTPException(status_code=400, detail="Dataset not found")
    
    pipeline_id = f"pipeline_{len(_pipelines) + 1}"
    _pipelines[pipeline_id] = {
        "id": pipeline_id,
        **pipeline.dict(),
        "status": "created"
    }
    
    # TODO: Start pipeline execution in background
    logger.info(f"Created pipeline {pipeline_id}")
    return {"id": pipeline_id, **pipeline.dict()}


@app.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get pipeline by ID."""
    if pipeline_id not in _pipelines:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return _pipelines[pipeline_id]


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_api()
