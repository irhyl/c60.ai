"""
FastAPI REST API for C60.ai AutoML system.
"""
from fastapi import FastAPI, UploadFile, File
from c60.core.automl import AutoML
import tempfile

app = FastAPI(title="C60.ai AutoML API")

automl = AutoML()

@app.post("/run")
async def run_pipeline(pipeline: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as tmp:
        tmp.write(await pipeline.read())
        tmp.flush()
        result = automl.run_from_yaml(tmp.name)
    return {"status": "completed", "result": result}

@app.get("/health")
def health():
    return {"status": "ok"}
