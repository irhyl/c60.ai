# C60.ai — Getting Started

*Step-by-step tutorial: install, fit, explain, and extend.*

---

## 1. Installation

### Prerequisites

- Python 3.9+
- pip

### From source (recommended)

```bash
git clone https://github.com/aditirkrishna/c60.ai.git
cd c60.ai
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

### Optional extras

```bash
pip install torch          # hybrid neuro-symbolic nodes (NeuralAutoencoder, NeuralClassifier)
pip install matplotlib     # pipeline and evolution visualisation
pip install fastapi uvicorn   # REST API server
```

### Verify installation

```bash
c60 version
# c60 version 0.2.0

python -c "import c60; print('OK')"
```

---

## 2. Your First Run (Python)

```python
from sklearn.datasets import load_iris
from c60.evolution.engine import EvolutionEngine

X, y = load_iris(return_X_y=True)

engine = EvolutionEngine(
    population_size=20,
    max_generations=10,
    task="classification",
    random_seed=42,
)

best_pipeline = engine.fit(X, y)
print(f"Best CV accuracy: {engine.best_score_:.4f}")
```

`engine.fit()` returns a pipeline already **refit on the full training data**,
ready to call `.predict()` on.

---

## 3. Reading the Evolution Story

```python
from c60.explainability.story import PipelineStory

story = PipelineStory(
    engine.history(),
    best_pipeline,
    feature_names=["sepal_len", "sepal_wid", "petal_len", "petal_wid"],
)

print(story.narrate())
```

Example output:

```text
Evolution ran 8 generations (+2 early-stopped), improving from 0.6133 to 0.9600 (+0.3467).
Significant jumps: gen 1 (+0.1600), gen 2 (+0.0800).

Best pipeline:
  StandardScaler → PCA(n_components=3) → SVC(C=8.2, kernel='rbf', gamma='scale')

Top-5 features:
  petal_len  0.486
  petal_wid  0.374
  sepal_len  0.087
  sepal_wid  0.053
```

```python
# ASCII generation table
print(story.generation_table())
```

```text
Gen │  Best   │  Mean   │  Time
────┼─────────┼─────────┼──────
  0 │  0.6133 │  0.4821 │  1.2s
  1 │  0.7733 │  0.6104 │  1.8s
  2 │  0.8533 │  0.7213 │  2.1s
  3 │  0.9467 │  0.8321 │  2.3s
  ...
  8 │  0.9600 │  0.9113 │  1.9s
```

---

## 4. Saving and Loading Pipelines

```python
import pickle

# Save
with open("best_pipeline.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

# Load and predict
with open("best_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

predictions = pipeline.predict(X_new)
```

---

## 5. CLI Walkthrough

### Run on a CSV file

```bash
c60 run data.csv --target label --task classification \
    --generations 10 --population 20 \
    --output best_pipeline.pkl \
    --story
```

Flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--target` | required | Name of the target column in the CSV |
| `--task` | `classification` | `classification` or `regression` |
| `--generations` | `10` | Number of GA generations |
| `--population` | `20` | Population size |
| `--cv` | `3` | Inner CV folds |
| `--timeout` | `30` | Per-pipeline eval timeout (seconds) |
| `--seed` | `42` | Random seed |
| `--output` | None | Path to save the best pipeline (pickle) |
| `--story` | False | Print the PipelineStory narrative |

### Explain a saved pipeline

```bash
c60 explain best_pipeline.pkl --data data.csv --target label
```

Prints feature importances extracted from each step of the pipeline.

### Browse the registry

```bash
c60 info                           # all operations
c60 info --type classifier         # classifiers only
c60 info --type dim_reducer        # dimensionality reducers only
```

---

## 6. REST API Walkthrough

Start the server:

```bash
uvicorn c60.api.server:app --reload --port 8000
```

Browse the interactive docs at `http://localhost:8000/docs`.

Submit a job:

```python
import requests, time
import numpy as np
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

resp = requests.post("http://localhost:8000/jobs", json={
    "X": X.tolist(),
    "y": y.tolist(),
    "task": "classification",
    "population_size": 15,
    "max_generations": 8,
    "random_seed": 42,
})
job_id = resp.json()["job_id"]

# Poll until complete
while True:
    status = requests.get(f"http://localhost:8000/jobs/{job_id}").json()["status"]
    if status in ("complete", "failed"):
        break
    time.sleep(2)

result = requests.get(f"http://localhost:8000/jobs/{job_id}/result").json()
print(f"Best score: {result['best_score']:.4f}")
print(f"Pipeline: {result['pipeline_steps']}")
```

---

## 7. Using Hybrid Neural Nodes

Requires PyTorch:

```bash
pip install torch
```

```python
from c60.core.registry import default_registry
from c60.hybrid.registry_ext import register_hybrid_nodes
from c60.evolution.engine import EvolutionEngine

# Extend the registry with neural nodes
register_hybrid_nodes(default_registry)

engine = EvolutionEngine(
    population_size=15,
    max_generations=8,
    task="classification",
    random_seed=0,
)
best_pipeline = engine.fit(X, y)
```

With hybrid nodes registered, the GA can discover pipelines like:

```text
StandardScaler → NeuralAutoencoder(bottleneck=16) → RandomForestClassifier
```

---

## 8. Extending the Registry

Add a custom sklearn-compatible operation:

```python
from c60.core.registry import OperationSpec, default_registry
from c60.core.types import ScaledData
from sklearn.preprocessing import QuantileTransformer

spec = OperationSpec(
    name="QuantileTransformer",
    step_type="scaler",
    op_class=QuantileTransformer,
    input_type=ScaledData,
    output_type=ScaledData,
    param_space={"n_quantiles": [50, 100, 200, 500], "output_distribution": ["uniform", "normal"]},
    default_params={"n_quantiles": 100, "output_distribution": "uniform"},
)

default_registry.register(spec)
```

The GA will now sample `QuantileTransformer` during population initialisation and mutation.

---

## 9. Parallel Evaluation

By default, C60.ai evaluates pipelines sequentially. Enable parallel evaluation with:

```python
from c60.execution.parallel import ParallelEvaluator
from c60.evaluation.fitness import FitnessEvaluator

evaluator = FitnessEvaluator(n_folds=3, eval_timeout=30)
parallel_evaluator = ParallelEvaluator(n_jobs=4)

engine = EvolutionEngine(
    population_size=20,
    max_generations=10,
    task="classification",
    evaluator=evaluator,
    parallel_evaluator=parallel_evaluator,
)
```

---

## 10. Visualisation

Requires matplotlib:

```bash
pip install matplotlib
```

```python
from c60.explainability.visualizer import render_pipeline, render_evolution

# Draw the best pipeline as a DAG
fig = render_pipeline(best_pipeline)
fig.savefig("pipeline.png", dpi=150, bbox_inches="tight")

# Draw score progression over generations
fig = render_evolution(engine.history())
fig.savefig("evolution.png", dpi=150, bbox_inches="tight")
```

---

## 11. Running Tests

```bash
# Full suite
pytest

# Specific module
pytest test/core/test_evolution.py -v

# Coverage
pytest --cov=src/c60 --cov-report=html
open htmlcov/index.html
```

---

## 12. Common Pitfalls

**Timeout too short for large datasets**: The default `eval_timeout=30` may not be enough
for slow estimators on datasets with many samples. Increase it:

```python
EvolutionEngine(..., eval_timeout=120)
```

**Small population / few generations**: With `population_size=5` and `max_generations=3`,
the search space is barely scratched. Use at least 15 and 8 respectively for meaningful
results.

**Fixed random seed gives deterministic results**: If you need to compare against a
baseline, use the same `random_seed` across runs. If you need diversity, vary the seed.

**Memory on Windows**: Spawning multiple threads for parallel evaluation increases memory
usage. If you hit "paging file too small" errors, reduce `n_jobs` or `population_size`.
