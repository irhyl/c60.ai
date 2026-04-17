# C60.ai — Algorithms in Depth

*A detailed walkthrough of every algorithm in the genetic engine — selection, crossover,
mutation, elitism, and plateau detection. Pseudocode is given for each.*

---

## 1. Random Population Initialisation

Before evolution can start, we need a diverse initial population. Each individual is a
randomly sampled pipeline DAG.

```text
function random_initialize(N, registry, task):
    population = []
    for i in 1..N:
        pipeline = random_pipeline(registry, task)
        population.append(Individual(pipeline, score=None))
    return population

function random_pipeline(registry, task):
    # Always start with a scaler
    scaler = registry.sample_step("scaler")
    pipeline = Pipeline()
    pipeline.add_step(scaler)

    # Optionally add a dim reducer or feature selector
    if rand() < 0.4:
        middle = registry.sample_step("dim_reducer" or "feature_selector")
        pipeline.add_step(middle)
        pipeline.add_edge(scaler.id, middle.id)
        last = middle
    else:
        last = scaler

    # Always end with a classifier or regressor
    predictor = registry.sample_step(task_predictor_type(task))
    pipeline.add_step(predictor)
    pipeline.add_edge(last.id, predictor.id)

    return pipeline
```

The initial population intentionally uses simple 2–3 step pipelines. More complex
structures emerge through evolution.

---

## 2. Fitness Evaluation

### 2.1 Cross-validation with timeout

```text
function evaluate(pipeline, X, y, k_folds, metric, timeout, cache):
    h = pipeline.structure_hash()
    if h in cache:
        return cache[h]                    # cache hit

    result = run_with_timeout(
        fn      = lambda: cross_val(pipeline, X, y, k_folds, metric),
        seconds = timeout
    )

    if result.timed_out:
        fitness = FitnessResult(score=-inf, timed_out=True)
    else:
        penalty = complexity_penalty * len(pipeline.steps)
        fitness = FitnessResult(score=result.mean_cv - penalty)

    cache.put(h, fitness)
    return fitness

function cross_val(pipeline, X, y, k_folds, metric):
    splitter = StratifiedKFold(n_splits=k_folds)
    scores = []
    for train_idx, test_idx in splitter.split(X, y):
        p = clone(pipeline)
        p.fit(X[train_idx], y[train_idx])
        scores.append(metric(y[test_idx], p.predict(X[test_idx])))
    return mean(scores)
```

### 2.2 Timeout implementation

The timeout is implemented using `threading.Thread` with `thread.join(timeout)`. If the
thread is still alive after the timeout, the evaluation is marked as failed with score
`-inf`. The thread continues running in the background but its result is discarded.

This is safe because each evaluation works on a cloned pipeline, so there is no shared
mutable state between concurrent evaluations.

---

## 3. Tournament Selection

```text
function tournament_select(population, k):
    contestants = sample(population, k, with_replacement=True)
    return max(contestants, key=lambda ind: ind.score)
```

### Why tournament over roulette wheel?

Roulette wheel selection assigns selection probability proportional to fitness:

```text
P(select individual i) = score_i / sum(all scores)
```

This fails in two common cases:
1. **All similar fitness**: when scores are close (e.g. 0.920, 0.922, 0.919), all
   probabilities are nearly equal and selection is essentially random.
2. **Negative or -inf fitness**: roulette cannot handle negative values without
   offsetting, which introduces another hyperparameter.

Tournament selection has neither problem: it only requires a relative ordering, handles
any real-valued scores including `-inf`, and its selection pressure is controlled by the
single parameter k.

---

## 4. Subgraph-Exchange Crossover

This is the most important and novel operator in C60.ai.

```text
function crossover(P1, P2):
    # Find compatible cut points
    candidates = []
    for node_1 in P1.internal_nodes():
        for node_2 in P2.internal_nodes():
            if types_compatible(node_1, node_2, P1, P2):
                candidates.append((node_1, node_2))

    if not candidates:
        return clone(P1), clone(P2)    # no valid crossover point

    (v1, v2) = random_choice(candidates)

    # Extract subgraphs
    sub1 = subgraph_below(P1, v1)
    sub2 = subgraph_below(P2, v2)

    # Swap subgraphs
    C1 = replace_subgraph(P1, v1, sub2)
    C2 = replace_subgraph(P2, v2, sub1)

    # Validate
    if not valid(C1): C1 = clone(P1)
    if not valid(C2): C2 = clone(P2)

    return C1, C2

function types_compatible(v1, v2, P1, P2):
    # The subgraph rooted at v2 must produce an output type compatible
    # with whatever follows v1 in P1, and vice versa
    return (output_type(sub_root_v2) <: expected_input_after(v1, P1) and
            output_type(sub_root_v1) <: expected_input_after(v2, P2))
```

### Example

```text
Parent 1:  Scaler → PCA(n=5) → SVM
Parent 2:  Scaler → SelectKBest(k=10) → RandomForest

Cut at PCA / SelectKBest (both produce ScaledData, compatible with classifiers):

Child 1:   Scaler → SelectKBest(k=10) → SVM
Child 2:   Scaler → PCA(n=5) → RandomForest
```

---

## 5. Mutation Operators

All five operators are applied stochastically. For each individual in the offspring, each
operator fires with probability `p_mutate` (default 0.3).

### 5.1 `add_node`

```text
function add_node(pipeline):
    # Choose a random edge (u, v) in the pipeline
    edge = random_choice(pipeline.edges)
    u, v = edge

    # Sample a new step compatible with the edge's type
    mid_type = output_type(u)    # must accept this as input
    candidates = registry.steps_accepting(mid_type)
    new_step = random_choice(candidates)

    # Insert: remove edge (u,v), add step, add edges (u, new), (new, v)
    pipeline.remove_edge(u, v)
    pipeline.add_step(new_step)
    pipeline.add_edge(u, new_step.id)
    pipeline.add_edge(new_step.id, v)
    return pipeline
```

### 5.2 `remove_node`

```text
function remove_node(pipeline):
    # Must keep source and sink; only remove internal nodes
    internal = pipeline.internal_nodes()
    if len(internal) == 0:
        return pipeline    # nothing to remove

    victim = random_choice(internal)

    # Re-wire: connect all parents of victim to all children of victim
    parents  = pipeline.parents_of(victim)
    children = pipeline.children_of(victim)

    for parent in parents:
        for child in children:
            if types_compatible(parent, child):
                pipeline.add_edge(parent, child)

    pipeline.remove_step(victim)
    return pipeline
```

### 5.3 `replace_node`

```text
function replace_node(pipeline):
    # Pick a random node that is not the source or sink
    target = random_choice(pipeline.internal_nodes() + [pipeline.sink])
    step_type = target.step_type

    # Sample a replacement of the same type
    replacement = registry.sample_step(step_type)

    # Preserve edges; just swap the operation
    pipeline.replace_step(target, replacement)
    return pipeline
```

### 5.4 `mutate_hyperparams`

```text
function mutate_hyperparams(pipeline):
    # Pick a random node
    node = random_choice(pipeline.steps)
    spec = registry.spec_for(node.operation.__class__)

    if not spec.param_space:
        return pipeline    # no hyperparameters to mutate

    # Pick a random hyperparameter and resample it
    param_name = random_choice(list(spec.param_space.keys()))
    new_value   = random_choice(spec.param_space[param_name])
    node.operation.set_params(**{param_name: new_value})
    return pipeline
```

### 5.5 `add_skip_edge`

```text
function add_skip_edge(pipeline):
    # Find a pair (u, v) where u is an ancestor of v (not direct parent)
    # and output_type(u) <: input_type(v)
    topo = pipeline.topological_order()
    for i, u in enumerate(topo):
        for v in topo[i+2:]:    # skip direct successor
            if (u, v) not in pipeline.edges:
                if types_compatible(u, v):
                    pipeline.add_edge(u, v)
                    return pipeline
    return pipeline    # no valid skip edge found
```

Skip edges create bypass connections that give later steps access to earlier
(less-transformed) representations — a structural analogue of ResNet's residual connections.

---

## 6. Elitism

```text
function get_elites(population, k):
    sorted_pop = sorted(population, key=lambda i: i.score, reverse=True)
    return [clone(ind) for ind in sorted_pop[:k]]
```

Elites are cloned before insertion into the next generation to prevent the same object
from appearing multiple times in the population (which would cause the evaluator to
skip re-evaluation incorrectly).

Elites retain their fitness scores from the previous generation, so they are never
re-evaluated — a significant cache-free speedup when k is large.

---

## 7. Plateau Detection

```text
function check_plateau(log, patience, tolerance):
    if len(log) < 2:
        return False

    recent_scores = [record.best_score for record in log[-patience:]]
    improvements  = [b - a for a, b in zip(recent_scores, recent_scores[1:])]

    # Plateau if no single improvement exceeds tolerance
    return all(imp <= tolerance for imp in improvements)
```

When a plateau is detected, evolution terminates early. This prevents wasting computation
when the population has converged to a local optimum.

**Tuning guidance**:
- Higher `patience` (e.g. 10): allows longer exploration before giving up; better on
  complex datasets where improvement can stall temporarily.
- Lower `tolerance` (e.g. 1e-6): stricter definition of "no improvement"; terminates
  earlier, less exploration.
- Setting `patience = max_generations` disables plateau detection entirely.

---

## 8. The Complete Generation Loop

```text
function run_generation(pop, evaluator, X, y, selector, crossover, mutator,
                        elitism, population_size, crossover_rate, mutation_rate):

    # 1. Evaluate any unevaluated individuals
    for ind in pop:
        if ind.score is None:
            ind.score = evaluator.evaluate(ind.pipeline, X, y).adjusted_score

    # 2. Extract elites (carry forward unchanged)
    elites = elitism.get_elites(pop)

    # 3. Generate offspring
    offspring = []
    while len(offspring) + len(elites) < population_size:
        p1 = selector.select(pop)
        p2 = selector.select(pop)

        if rand() < crossover_rate:
            c1, c2 = crossover.apply(p1.pipeline, p2.pipeline)
        else:
            c1, c2 = clone(p1.pipeline), clone(p2.pipeline)

        if rand() < mutation_rate:
            c1 = mutator.mutate(c1)
        if rand() < mutation_rate:
            c2 = mutator.mutate(c2)

        offspring.append(Individual(c1))
        if len(offspring) + len(elites) < population_size:
            offspring.append(Individual(c2))

    # 4. Form new population
    new_pop = elites + offspring
    assert len(new_pop) == population_size

    return new_pop
```

---

## 9. Default Hyperparameter Settings

| Parameter | Default | Effect |
| --- | --- | --- |
| `population_size` | 20 | Larger = more diversity, slower per generation |
| `max_generations` | 10 | More generations = longer search |
| `n_folds` (inner CV) | 3 | More folds = more reliable fitness, slower |
| `crossover_rate` | 0.7 | Probability that two parents recombine (vs. clone) |
| `mutation_rate` | 0.3 | Probability each operator fires per individual |
| `k_elite` | 2 | Elites carried forward unchanged |
| `k_tournament` | 3 | Tournament size for parent selection |
| `complexity_penalty` | 0.002 | Fitness penalty per step |
| `plateau_patience` | 5 | Generations of no improvement before stopping |
| `plateau_tolerance` | 1e-4 | Minimum improvement to reset plateau counter |
| `eval_timeout` | 30.0 s | Max wall time per pipeline evaluation |
| `cache_size` | 512 | Max cached fitness results |

---

## 10. Worked Example: One Complete Run

Dataset: iris (150 samples, 4 features, 3 classes)

```text
Gen 0: init 20 random pipelines
  best=0.6133 (Scaler → KNN)  mean=0.4821

Evaluate: 20 pipelines, 15 cache misses (5 duplicates)
  Pipeline A: Scaler → PCA(n=2) → LogReg  → CV=0.8667
  Pipeline B: Scaler → SVM(C=1)           → CV=0.9333
  ...

Gen 1 parents: tournament selects B (score 0.9333) and C (0.8933)
  Crossover: swap PCA subgraph into B → Child: Scaler → PCA → SVM(C=1)
  Mutate child: replace PCA with SelectKBest → Scaler → SelectKBest → SVM(C=1)

  Evaluate: 20 pipelines, 12 cache misses
  best=0.9467  mean=0.7823  (+0.0133 improvement)

...

Gen 8: plateau detected (no improvement for 5 generations)
  best=0.9600  (Scaler → PCA(n=3) → SVM(C=8.2, gamma=0.1))

Refit best pipeline on full X, y.
Return fitted pipeline.
```

The evolution found that PCA with 3 components + a tuned SVM outperforms all simpler
pipelines on this dataset — a result consistent with the known structure of the iris
problem (first 3 PCA components explain > 99.8% of variance).
