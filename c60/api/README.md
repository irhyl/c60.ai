# C60 API

This package provides a REST API for running C60.ai pipelines via HTTP.

## Usage

Start the API server:

```bash
uvicorn c60.api.app:app --reload
```

### Endpoints
- `POST /run`: Upload a pipeline YAML to execute
- `GET /health`: Health check
