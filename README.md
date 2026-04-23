# C60.ai — Molecular Evolution AutoML

> *Named after Buckminsterfullerene (C60): a 60-carbon molecule whose highly stable,
> non-obvious lattice emerges entirely from self-organisation — never from top-down design.
> The same principle drives this framework.*

C60.ai is a research-grade **Automated Machine Learning (AutoML)** framework that treats
every machine learning pipeline as a **graph molecule** and evolves it with a genetic
algorithm. Unlike every mainstream AutoML tool, C60.ai does not assume the pipeline has a
fixed shape — it searches over **arbitrary directed acyclic graphs**, discovering topologies
no human would design by hand.

---

## Why C60.ai Exists

Every major AutoML framework (auto-sklearn, TPOT, H2O, Google AutoML) shares one hidden
limitation:

> **They assume the pipeline shape is fixed.**

The search space is always `Preprocessor → FeatureSelector → Model`. Systems search over
*which components* fill the slots, not *what the slots should be*. This creates hard ceilings:

| Problem with today's AutoML | C60.ai's answer |
| --- | --- |
| Fixed sequential topology | Pipelines are arbitrary DAGs — parallel branches, skip connections |
| Hyperparameter tuning only | Structural mutation: insert, delete, replace nodes; redirect edges |
| No memory across evaluations | EvaluationCache keyed by structure hash, FIFO-evicted |
| Black-box output | PipelineStory — human-readable narrative of the entire evolution |
| Manual feature engineering | Genetic operators discover useful subgraph patterns automatically |

---

## Benchmark Results

### vs. 9 sklearn baselines (core: 3-fold × 3 seeds; extended: 2-fold × 2 seeds)

| Dataset | Samples | Features | Classes | C60.ai | Best Baseline | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| digits | 1 797 | 64 | 10 | **98.94%** | SVM-RBF 98.05% | +0.89 pp |
| pendigits | 8 000 | 16 | 10 | 99.54% | SVM-RBF 99.56% | −0.02 pp |
| breast_cancer | 569 | 30 | 2 | 97.01% | PCA+LR 97.77% | −0.76 pp |
| wine | 178 | 13 | 3 | 97.57% | LR 98.13% | −0.56 pp |
| iris | 150 | 4 | 3 | **96.22%** | KNN-10 96.00% | +0.22 pp |

**C60.ai ranks #1 overall** (avg rank 2.75), **significantly outperforms 7 of 9 baselines**
(Wilcoxon signed-rank, p < 0.05).

### vs. 10 AutoML frameworks (2-fold × 2 seeds)

| Rank | System | Mean Accuracy | digits | iris |
| --- | --- | --- | --- | --- |
| 1 | OptunaEnsemble | 97.01% | 98.58% | 94.00% |
| 2 | GreedyEnsemble | 96.84% | **98.66%** | 93.67% |
| 3 | AutoStack | 96.72% | 97.77% | 93.67% |
| 4 | FeatEngAutoML | 96.73% | 98.50% | 93.33% |
| **5** | **C60.ai** | **96.40%** | **98.83%** | **96.00%** |
| 6 | BayesSearchCV | 96.54% | 98.19% | 94.33% |
| 7 | OptunaSearch | 96.62% | 97.61% | **96.00%** |

**C60.ai ranks #5 of 11** overall and is the **best single system on digits** (98.83%).
No AutoML system is statistically significantly better than C60.ai (Wilcoxon p > 0.05).
C60.ai's structural advantage is clearest on high-dimensional multi-class problems.

Full results, plots, and statistical analysis: [`benchmark/results/`](benchmark/results/)

---

## Installation

```bash
git clone https://github.com/aditirkrishna/c60.ai.git
cd c60.ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install torch        # hybrid neuro-symbolic nodes
pip install matplotlib   # evolution plots and pipeline visualisation
```

---

## Quick Start

### Fit on any dataset

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
best_pipeline.fit(X, y)
print(f"Accuracy: {best_pipeline.score(X, y):.4f}")
```

### Read the evolution story

```python
from c60.explainability.story import PipelineStory

story = PipelineStory(
    engine.history(), best_pipeline,
    feature_names=["sepal_len", "sepal_wid", "petal_len", "petal_wid"],
)
print(story.narrate())
```

Output:

```text
Evolution ran 10 generations, improving from 0.6133 to 0.9600 (+0.3467).
Best pipeline: StandardScaler -> PCA(n=3) -> SVC(C=8.2, kernel=rbf)
Top features: petal_len 0.486 | petal_wid 0.374 | sepal_len 0.087
```

### CLI

```bash
c60 run data.csv --target label --task classification
c60 explain best_pipeline.pkl --data data.csv
c60 info --type classifier
```

### REST API

```bash
uvicorn c60.api.server:app --reload
```

```python
import requests

resp = requests.post("http://localhost:8000/jobs", json={
    "X": X.tolist(), "y": y.tolist(),
    "task": "classification",
    "population_size": 15,
    "max_generations": 8,
})
job_id = resp.json()["job_id"]
# poll GET /jobs/{job_id} until status == "complete"
result = requests.get(f"http://localhost:8000/jobs/{job_id}/result").json()
print(result["best_score"], result["pipeline_steps"])
```

---

## How It Works

```text
Dataset (X, y)
     |
     v
Population of random Pipeline DAGs
  A: Scaler -> PCA -> SVM
  B: Scaler -> SelectKBest -> GBT
  C: Scaler -> RandomForest

For each generation:
  1. EVALUATE  — cross-val accuracy per pipeline (cached by structure hash)
  2. SELECT    — tournament selection (higher score = more likely to reproduce)
  3. CROSSOVER — swap subgraphs between two parent pipelines
  4. MUTATE    — insert/delete/replace nodes; redirect edges; tweak hyperparams
  5. ELITISM   — best K individuals carry forward unchanged

     |
     v
Best pipeline found -> refit on full training data -> ready to predict
```

Steps 3 and 4 operate on **graph structure** — this is what distinguishes C60.ai from all
template-based AutoML.

---

## Architecture

```text
src/c60/
  core/           Typed DAG pipeline, operation registry, data-type lattice
  evaluation/     Fitness evaluator (stratified k-fold + timeout), eval cache
  evolution/      Population, genetic operators, tournament selection, GA engine
  explainability/ Feature introspection, PipelineStory narrative, visualisation
  hybrid/         PyTorch autoencoder + MLP classifier as first-class pipeline nodes
  execution/      Parallel population evaluation (ThreadPoolExecutor)
  cli/            Click CLI: run / explain / info / version
  api/            FastAPI async job server with Pydantic models

benchmark/
  baselines.py    9 sklearn baselines + C60Estimator sklearn-compatible wrapper
  runner.py       BenchmarkRunner — nested CV, standard + OpenML datasets
  report.py       ResultsReporter — tables, Wilcoxon tests, plots
  _run.py         Executable benchmark script
  results/        results_full.csv, report.txt, summary.md, PNG charts

test/             250+ pytest tests, full suite completes in < 60 s
docs/             Full documentation (concept / theory / architecture / results)
research/         Original research document and open problems
```

---

## Comparison with Other AutoML Frameworks

| Feature | auto-sklearn | TPOT | H2O AutoML | C60.ai |
| --- | --- | --- | --- | --- |
| Pipeline topology | Fixed | Fixed | Fixed | **Arbitrary DAG** |
| Search method | Bayesian | Genetic (DEAP) | Grid/random | **Graph-level GA** |
| Structural mutation | No | Limited | No | **Yes (5 operators)** |
| Explainability | Limited | No | No | **PipelineStory** |
| Neural hybrid nodes | No | No | No | **Yes (PyTorch)** |
| REST API | No | No | Yes | **Yes (FastAPI)** |
| Structure-hash cache | No | No | No | **Yes** |

---

## Documentation

| File | Contents |
| --- | --- |
| [docs/concept.md](docs/concept.md) | What is AutoML? The molecular evolution metaphor for anyone |
| [docs/theory.md](docs/theory.md) | Mathematical formulation — DAGs, fitness, genetic operators |
| [docs/architecture.md](docs/architecture.md) | Code organisation and design decisions |
| [docs/algorithms.md](docs/algorithms.md) | Selection, crossover, mutation, plateau detection in depth |
| [docs/results.md](docs/results.md) | Full benchmark results with statistical analysis |
| [docs/getting_started.md](docs/getting_started.md) | Step-by-step tutorial: install → fit → explain → extend |
| [docs/api_reference.md](docs/api_reference.md) | Python API and REST API reference |
| [research/molecular_concept.md](research/molecular_concept.md) | Original research document |

---

## Running Tests

```bash
pytest                                      # full suite (~60 s)
pytest test/core/test_evolution.py -v      # GA engine
pytest test/core/test_benchmark.py -v      # benchmark infrastructure
pytest --cov=src/c60 --cov-report=html     # coverage report
```

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{c60ai2026,
  title  = {C60.ai: Molecular Evolution for Automated Machine Learning},
  author = {Ramakrishnan, Aditi},
  year   = {2026},
  url    = {https://github.com/aditirkrishna/c60.ai}
}
```
