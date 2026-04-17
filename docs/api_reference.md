# C60.ai — API Reference

*Complete reference for the Python API and REST API.*

---

## Python API

### `EvolutionEngine`

`src/c60/evolution/engine.py`

The central orchestrator. Holds the population, operators, evaluator, and GA loop.

```python
class EvolutionEngine:
    def __init__(
        self,
        population_size: int = 20,
        max_generations: int = 10,
        task: str = "classification",       # "classification" | "regression"
        metric: str | None = None,          # default: "accuracy" / "r2"
        n_folds: int = 3,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.3,
        k_elite: int = 2,
        k_tournament: int = 3,
        complexity_penalty: float = 0.002,
        eval_timeout: float = 30.0,
        cache_size: int = 512,
        plateau_patience: int = 5,
        plateau_tolerance: float = 1e-4,
        random_seed: int | None = None,
        registry: OperationRegistry | None = None,
        evaluator: FitnessEvaluator | None = None,
        parallel_evaluator: ParallelEvaluator | None = None,
    )
```

**Methods**

| Method | Returns | Description |
| --- | --- | --- |
| `fit(X, y)` | `Pipeline` | Run the GA, return the best pipeline refit on all data |
| `history()` | `EvolutionLog` | Log of per-generation statistics |
| `best_score_` | `float` | Best CV score found (property, available after `fit`) |

---

### `Pipeline`

`src/c60/core/pipeline.py`

A typed Directed Acyclic Graph of sklearn-compatible steps.

```python
class Pipeline:
    def add_step(self, step: PipelineStep) -> None
    def add_edge(self, from_id: str, to_id: str) -> None
    def fit(self, X, y) -> "Pipeline"
    def predict(self, X) -> np.ndarray
    def score(self, X, y) -> float
    def clone(self) -> "Pipeline"
    def structure_hash(self) -> str
    def topological_order(self) -> list[PipelineStep]
    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "Pipeline"
```

`add_edge` raises `TypeError` if the edge is type-incompatible (i.e.
`output_type(from_step)` is not a subtype of `input_type(to_step)`).

---

### `PipelineStep`

```python
@dataclass
class PipelineStep:
    id: str                  # UUID
    name: str
    step_type: str           # "scaler" | "classifier" | "dim_reducer" | ...
    operation: BaseEstimator
    input_type: type         # DataType subclass
    output_type: type        # DataType subclass
```

---

### `OperationRegistry`

`src/c60/core/registry.py`

Catalogue of available ML operations.

```python
class OperationRegistry:
    def register(self, spec: OperationSpec) -> None
    def sample_step(self, step_type: str) -> PipelineStep
    def get_by_type(self, step_type: str) -> list[OperationSpec]
    def summary(self) -> str
```

`default_registry` is the pre-populated singleton with 20+ operations.

**`OperationSpec`**

```python
@dataclass
class OperationSpec:
    name: str
    step_type: str           # "scaler" | "classifier" | "dim_reducer" | ...
    op_class: type           # sklearn estimator class
    input_type: type
    output_type: type
    param_space: dict        # {param_name: [value1, value2, ...]}
    default_params: dict
```

---

### `FitnessEvaluator`

`src/c60/evaluation/fitness.py`

```python
class FitnessEvaluator:
    def __init__(
        self,
        n_folds: int = 3,
        metric: str = "accuracy",
        eval_timeout: float = 30.0,
        complexity_penalty: float = 0.002,
        cache: EvaluationCache | None = None,
    )

    def evaluate(self, pipeline: Pipeline, X, y) -> FitnessResult
```

**`FitnessResult`**

```python
@dataclass
class FitnessResult:
    adjusted_score: float    # CV score minus complexity penalty
    raw_score: float         # mean CV score without penalty
    n_folds_completed: int
    timed_out: bool
```

---

### `EvaluationCache`

`src/c60/evaluation/cache.py`

```python
class EvaluationCache:
    def __init__(self, max_size: int = 512)
    def get(self, hash: str) -> FitnessResult | None
    def put(self, hash: str, result: FitnessResult) -> None
    def stats(self) -> dict   # {"hits": int, "misses": int, "hit_rate": float, "size": int}
    def clear(self) -> None
```

---

### `ParallelEvaluator`

`src/c60/execution/parallel.py`

```python
class ParallelEvaluator:
    def __init__(self, n_jobs: int = 4)

    def evaluate_population(
        self,
        population: list[Individual],
        evaluator: FitnessEvaluator,
        X, y,
        reevaluate: bool = False,
    ) -> None
```

Evaluates all `None`-scored individuals in `population` in parallel using
`ThreadPoolExecutor(max_workers=n_jobs)`. Results are written back to each
`Individual.score` in place.

---

### `PipelineStory`

`src/c60/explainability/story.py`

```python
class PipelineStory:
    def __init__(
        self,
        log: EvolutionLog,
        pipeline: Pipeline,
        feature_names: list[str] | None = None,
    )

    def narrate(self) -> str
    def generation_table(self) -> str
```

---

### `PipelineIntrospector`

`src/c60/explainability/introspector.py`

```python
class PipelineIntrospector:
    def inspect(self, pipeline: Pipeline, fitness: FitnessResult) -> PipelineReport

class PipelineReport:
    steps: list[StepReport]
    def top_features(self, feature_names: list[str], k: int = 10) -> list[tuple[str, float]]
```

---

### Visualiser

`src/c60/explainability/visualizer.py`

```python
def render_pipeline(pipeline: Pipeline) -> matplotlib.figure.Figure
def render_evolution(log: EvolutionLog) -> matplotlib.figure.Figure
```

Both return `Figure` objects without calling `plt.show()`.

---

### Hybrid Nodes

`src/c60/hybrid/`

```python
class NeuralAutoencoder(BaseEstimator, TransformerMixin):
    """sklearn-compatible bottleneck autoencoder (PyTorch backend)."""
    def __init__(
        self,
        bottleneck_dim: int = 16,
        hidden_dim: int = 64,
        epochs: int = 50,
        lr: float = 1e-3,
        random_seed: int = 0,
    )
    def fit(self, X, y=None) -> "NeuralAutoencoder"
    def transform(self, X) -> np.ndarray

class NeuralClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-compatible 2-hidden-layer MLP (PyTorch backend)."""
    def __init__(
        self,
        hidden_dim: int = 64,
        epochs: int = 100,
        lr: float = 1e-3,
        dropout: float = 0.2,
        random_seed: int = 0,
    )
    def fit(self, X, y) -> "NeuralClassifier"
    def predict(self, X) -> np.ndarray
    def predict_proba(self, X) -> np.ndarray

def register_hybrid_nodes(registry: OperationRegistry) -> None:
    """Add NeuralAutoencoder and NeuralClassifier to the registry.
    No-op (with warning) if PyTorch is not installed."""
```

---

### Data Types

`src/c60/core/types.py`

```python
class DataType: pass
class UnscaledData(DataType): pass
class ScaledData(UnscaledData): pass
class EmbeddedData(ScaledData): pass   # output of NeuralAutoencoder
class ClassLabels(DataType): pass
class RegressionTargets(DataType): pass
```

Edge compatibility: `issubclass(output_type, input_type)` must be `True`.

---

## REST API

Base URL: `http://localhost:8000`

Interactive documentation: `http://localhost:8000/docs` (Swagger UI)

---

### `POST /jobs`

Submit a new evolution job.

**Request body** (`RunRequest`)

```json
{
  "X": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]],
  "y": [0, 0],
  "task": "classification",
  "population_size": 15,
  "max_generations": 8,
  "cv": 3,
  "eval_timeout": 30.0,
  "random_seed": 42
}
```

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `X` | `[[float]]` | required | Feature matrix |
| `y` | `[any]` | required | Target vector |
| `task` | `string` | `"classification"` | `"classification"` or `"regression"` |
| `metric` | `string \| null` | `null` | Override default metric |
| `population_size` | `int` | `10` | 2–200 |
| `max_generations` | `int` | `5` | 1–500 |
| `cv` | `int` | `3` | 2–10 |
| `eval_timeout` | `float` | `30.0` | 1–300 seconds |
| `random_seed` | `int \| null` | `null` | Fixed seed for reproducibility |

**Response** `202 Accepted`

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "pending",
  "message": "Job submitted. Poll GET /jobs/{job_id} for status."
}
```

---

### `GET /jobs/{job_id}`

Poll job status.

**Response** `200 OK`

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "running"
}
```

`status` values: `"pending"` | `"running"` | `"complete"` | `"failed"`

---

### `GET /jobs/{job_id}/result`

Retrieve full result for a completed job.

**Response** `200 OK`

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "complete",
  "best_score": 0.9600,
  "pipeline_steps": [
    {"name": "StandardScaler", "step_type": "scaler", "params": {}},
    {"name": "PCA",            "step_type": "dim_reducer", "params": {"n_components": 3}},
    {"name": "SVC",            "step_type": "classifier", "params": {"C": 8.2, "kernel": "rbf"}}
  ],
  "generation_history": [
    {"generation": 0, "best_score": 0.6133, "mean_score": 0.4821, "wall_time": 1.2},
    {"generation": 1, "best_score": 0.7733, "mean_score": 0.6104, "wall_time": 1.8}
  ],
  "n_generations": 8,
  "elapsed_seconds": 14.3,
  "error": null
}
```

If the job failed: `"status": "failed"`, `"error": "<exception message>"`.
If the job is still running: returns current status stub without `pipeline_steps`.

---

### `DELETE /jobs/{job_id}`

Remove a completed job from the in-memory store.

**Response** `204 No Content`

Returns `409 Conflict` if the job is still running.

---

### `GET /health`

Health check.

**Response** `200 OK`

```json
{"status": "ok", "version": "0.2.0"}
```

---

### `GET /registry`

List available operations.

**Response** `200 OK`

```json
{
  "operations": [
    {"name": "StandardScaler", "step_type": "scaler",    "input_type": "UnscaledData", "output_type": "ScaledData"},
    {"name": "PCA",            "step_type": "dim_reducer","input_type": "ScaledData",  "output_type": "ScaledData"},
    {"name": "SVC",            "step_type": "classifier", "input_type": "ScaledData",  "output_type": "ClassLabels"}
  ]
}
```

---

## Error Codes

| Code | Condition |
| --- | --- |
| `422 Unprocessable Entity` | Invalid request body (Pydantic validation failure) |
| `404 Not Found` | Job ID does not exist |
| `409 Conflict` | Attempt to delete a running job |
