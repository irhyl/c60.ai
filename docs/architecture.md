# C60.ai — System Architecture

*How the code is organised, what each module does, and the design decisions behind each
choice.*

---

## Overview

C60.ai is structured as a layered system. Lower layers are stable, well-tested, and have
no knowledge of the layers above them. Higher layers build on lower ones and add
orchestration, user interface, and evaluation.

```text
┌───────────────────────────────────────────────────────────┐
│  User Interface Layer                                       │
│  cli/main.py  ·  api/server.py  ·  api/models.py           │
├───────────────────────────────────────────────────────────┤
│  Evolution Layer                                            │
│  evolution/engine.py  ·  evolution/population.py           │
│  evolution/operators.py  ·  evolution/selection.py         │
├───────────────────────────────────────────────────────────┤
│  Evaluation Layer                                           │
│  evaluation/fitness.py  ·  evaluation/cache.py             │
│  execution/parallel.py                                      │
├───────────────────────────────────────────────────────────┤
│  Explainability Layer                                       │
│  explainability/introspector.py  ·  story.py  ·  visualizer│
├───────────────────────────────────────────────────────────┤
│  Hybrid Layer  (optional)                                   │
│  hybrid/node.py  ·  hybrid/registry_ext.py                 │
├───────────────────────────────────────────────────────────┤
│  Core Layer                                                 │
│  core/types.py  ·  core/pipeline.py  ·  core/registry.py  │
└───────────────────────────────────────────────────────────┘
```

Dependencies only flow **upward** — lower layers never import from higher ones.

---

## Core Layer (`src/c60/core/`)

### `types.py` — Data-type lattice

Defines the type hierarchy for pipeline data:

```python
class DataType: pass
class UnscaledData(DataType): pass
class ScaledData(UnscaledData): pass
class EmbeddedData(ScaledData): pass  # neural encoder output
class ClassLabels(DataType): pass
class RegressionTargets(DataType): pass
```

`compatible_input_types` maps each type to all types that can feed into it. The check
`issubclass(output_type, input_type)` is used to validate edges.

**Design decision**: Using Python's class hierarchy for type compatibility means adding a
new type only requires subclassing — no registry changes, no configuration files.

---

### `pipeline.py` — Typed DAG

`PipelineStep` wraps a single sklearn estimator with type annotations and a UUID:

```python
@dataclass
class PipelineStep:
    id: str                  # UUID, used as graph node key
    name: str
    step_type: str           # "scaler", "classifier", etc.
    operation: BaseEstimator # the sklearn object
    input_type: type
    output_type: type
```

`Pipeline` manages the DAG:

- `add_step(step)` — register a node
- `add_edge(from_id, to_id)` — add a directed edge with type-compatibility check
- `topological_order()` — Kahn's algorithm
- `fit(X, y)` / `predict(X)` — execute through the topological order
- `clone()` — deep copy via `sklearn.base.clone` + `copy.deepcopy`
- `structure_hash()` — deterministic hash of the graph topology and hyperparameters
- `to_dict()` / `from_dict()` — JSON serialisation

**Design decision**: `structure_hash()` uses the canonical adjacency string with sorted
hyperparameters, making it topology-sensitive but order-insensitive. Two pipelines with
identical structure but different construction order produce the same hash.

---

### `registry.py` — Operation registry

`OperationRegistry` maintains the catalogue of available ML operations:

```python
@dataclass
class OperationSpec:
    name: str
    step_type: str
    cls: type                      # sklearn estimator class
    input_type: type
    output_type: type
    param_space: dict              # hyperparameter search space
    default_params: dict
```

`sample_step(step_type)` creates a `PipelineStep` with parameters sampled uniformly from
`param_space`. This is used during random population initialisation and the
`replace_node` mutation.

`default_registry` contains 20+ operations out of the box:

| Type | Operations |
| --- | --- |
| scaler | StandardScaler, MinMaxScaler, RobustScaler |
| dim_reducer | PCA, TruncatedSVD, NMF |
| feature_selector | SelectKBest(f_classif), SelectKBest(mutual_info), VarianceThreshold |
| classifier | LogisticRegression, SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier |
| regressor | LinearRegression, Ridge, SVR, RandomForestRegressor, GradientBoostingRegressor |

**Design decision**: `param_space` uses lists of discrete values rather than continuous
distributions. This makes hyperparameter mutation simple (uniform sample from list) and
reproducible (no distribution fitting required).

---

## Evaluation Layer (`src/c60/evaluation/`)

### `fitness.py` — FitnessEvaluator

```python
class FitnessEvaluator:
    def evaluate(self, pipeline, X, y) -> FitnessResult
```

For each evaluation:
1. Clone the pipeline (so the original is not mutated by fitting).
2. Run `StratifiedKFold` (classification) or `KFold` (regression) cross-validation.
3. Each fold runs in a thread with `eval_timeout` seconds wall-clock limit.
4. Compute mean CV score, apply complexity penalty.
5. Return `FitnessResult(adjusted_score, raw_score, n_folds_completed, timed_out)`.

**Design decision**: Thread-based timeout (not process-based) because spawning processes
on Windows requires `multiprocessing`, which has known issues with the Microsoft Store
Python build. Threads share the GIL but are sufficient for I/O-bound sklearn estimators.

---

### `cache.py` — EvaluationCache

```python
class EvaluationCache:
    def get(self, hash: str) -> Optional[FitnessResult]
    def put(self, hash: str, result: FitnessResult)
    def stats(self) -> dict   # hits, misses, hit_rate, size
```

Implemented as an `OrderedDict` for O(1) lookup and O(1) FIFO eviction. When `max_size`
is exceeded, the oldest entry is removed.

**Design decision**: FIFO rather than LRU, because pipelines do not have temporal locality.
A pipeline seen in generation 1 is as likely to reappear in generation 8 as in generation
2. FIFO is simpler and equally effective here.

---

## Evolution Layer (`src/c60/evolution/`)

### `population.py` — Population and Individual

`Individual` wraps a `Pipeline` with its fitness score. `Population` wraps a list of
`Individual`s with methods for evaluation, scoring, and iteration.

`pop.evaluate_all(evaluator, X, y)` skips already-evaluated individuals (score is not
`None`). This is the "lazy evaluation" that makes elitism efficient — elites from the
previous generation carry their cached scores.

---

### `operators.py` — Genetic operators

`SubgraphCrossover.apply(p1, p2)` implements the crossover described in `theory.md`.
Falls back to cloning both parents if no valid crossover point is found (this happens
when the two pipelines have incompatible types at all potential cut points).

`Mutator.mutate(pipeline)` tries each of the five mutation operators in random order,
returning the first valid mutant. If all operators fail (e.g., the pipeline is too
small to remove a node), returns a clone of the original.

**Design decision**: Returning the original on mutation failure (rather than raising an
exception) ensures the GA loop never crashes due to degenerate pipeline configurations.

---

### `selection.py` — Selection strategies

`TournamentSelector` implements k-tournament selection. `ElitismStrategy` extracts the
top-k individuals by score for direct carry-over.

**Design decision**: Tournament selection over roulette-wheel selection because tournament
is less sensitive to fitness scaling (roulette fails when all fitnesses are close together,
giving effectively random selection).

---

### `engine.py` — EvolutionEngine

The central orchestrator. Holds references to the evaluator, population, selector,
crossover, mutator, and elitism strategy. Runs the GA loop and maintains the
`EvolutionLog`.

`engine.fit(X, y)` returns the best `Pipeline` found. After the loop, the best pipeline
is **refit on the full training data** before being returned, so it is ready to call
`predict()` directly.

`engine.history()` returns the `EvolutionLog` for use by `PipelineStory`.

---

## Explainability Layer (`src/c60/explainability/`)

### `introspector.py`

`PipelineIntrospector.inspect(pipeline, fitness)` walks the fitted pipeline in
topological order and extracts feature importances from each step. Returns a
`PipelineReport` (collection of `StepReport`s).

`PipelineReport.top_features(feature_names, k)` returns the k most important features
across all steps, using the importance scores from the predictive step (classifier or
regressor).

---

### `story.py`

`PipelineStory.narrate()` produces a human-readable summary combining:
- Score progression across generations
- Which generations showed significant improvement
- Plateau detection
- Best pipeline topology description
- Top-5 feature importances (if available)

`PipelineStory.generation_table()` produces a fixed-width ASCII table suitable for
logging or reporting.

---

### `visualizer.py`

`render_pipeline(pipeline)` draws the DAG using `networkx` with nodes coloured by
step type. Attempts graphviz layout (`dot`); falls back to a manual layered layout if
graphviz is not installed.

`render_evolution(log)` draws best/mean score over generations with a shaded band
showing the gap between them.

Both functions return `matplotlib.Figure` objects without calling `plt.show()`, so they
work in headless environments (CI, servers).

---

## Hybrid Layer (`src/c60/hybrid/`)

`NeuralAutoencoder` and `NeuralClassifier` implement the sklearn estimator interface
(`fit`, `transform`/`predict`, `get_params`, `set_params`) on top of PyTorch
`nn.Sequential` modules.

**Design decision**: Custom `__deepcopy__` using `torch.save`/`torch.load` on a
`BytesIO` buffer, rather than relying on `copy.deepcopy` for nn.Module objects.
Standard deepcopy of PyTorch modules sometimes fails on Windows due to pickling
restrictions on GPU tensors. The save/load approach works universally.

`register_hybrid_nodes(registry)` extends the default registry with these nodes.
It is a no-op (with a warning) if PyTorch is not installed, so the rest of the system
works without torch.

---

## Execution Layer (`src/c60/execution/`)

`ParallelEvaluator` wraps `FitnessEvaluator` with `ThreadPoolExecutor`:

```python
class ParallelEvaluator:
    def evaluate_population(self, population, evaluator, X, y, reevaluate=False)
```

When `n_jobs=1` or the population has only one unevaluated individual, falls back to
sequential evaluation (avoids thread-pool overhead for small cases).

**Design decision**: Thread-level parallelism rather than process-level. sklearn estimators
release the GIL during C-extension calls (numpy, scipy), so multiple CV evaluations can
genuinely run concurrently in threads without Python's GIL being a bottleneck.

---

## CLI (`src/c60/cli/`)

Built with [Click](https://click.palletsprojects.com/). Four subcommands:

| Command | What it does |
| --- | --- |
| `c60 version` | Print version |
| `c60 info [--type TYPE]` | List available operations in the registry |
| `c60 run CSV --target COL --task TASK` | Load CSV, run evolution, print story, optionally save pipeline |
| `c60 explain PKL [--data CSV]` | Load pickled pipeline, print feature importances |

---

## REST API (`src/c60/api/`)

Built with [FastAPI](https://fastapi.tiangolo.com/).

### Job lifecycle

```text
POST /jobs          → 202 Accepted  {job_id, status: "pending"}
GET  /jobs/{id}     → 200 OK        {job_id, status: "running"|"complete"|"failed"}
GET  /jobs/{id}/result → 200 OK     {best_score, pipeline_steps, generation_history, ...}
DELETE /jobs/{id}   → 204 No Content
```

Each job runs in a **daemon thread** so the FastAPI event loop is never blocked. A
`threading.Event` signals completion. The in-memory job store (`_JOBS: Dict[str, _Job]`)
is protected by a `threading.Lock`.

**Design decision**: In-memory store rather than a database. This is intentional for a
single-node research tool — adding persistence (Redis, SQLite) is a documented extension
point but not needed for the current scope.

---

## Benchmark (`benchmark/`)

Three files form the benchmark harness:

- `baselines.py` — defines all 9 baseline estimators and the `C60Estimator` sklearn
  wrapper. C60Estimator follows the sklearn estimator contract so it can be used inside
  `cross_val_score`, GridSearchCV, or any pipeline.
- `runner.py` — `BenchmarkRunner` handles the nested CV loop, exception isolation
  (failed evaluations record NaN rather than crashing), and progress printing.
- `report.py` — `ResultsReporter` computes summary statistics, average ranks, Wilcoxon
  signed-rank tests, and generates matplotlib figures.

**Design decision**: `C60Estimator.fit()` calls `engine.fit()` (which uses CV internally)
and then **refits the best pipeline on the full training data**. This is the standard
AutoML contract: find the best structure using CV, then train the final model on all
available data for maximum predictive power.
