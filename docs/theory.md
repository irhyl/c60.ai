# C60.ai — Theoretical Foundations

*Mathematical formulation of the C60.ai framework. Assumes familiarity with basic linear
algebra, probability, and graph theory.*

---

## 1. The Pipeline as a Typed DAG

### 1.1 Formal definition

A **C60 pipeline** is a tuple:

```text
P = (V, E, τ, θ)
```

where:

- **V** is a finite set of nodes (pipeline steps)
- **E ⊆ V × V** is a set of directed edges (data-flow connections)
- **τ : V → (T_in × T_out)** assigns input and output data types to each node
- **θ : V → Params** assigns hyperparameter configurations to each node

The graph (V, E) must be a **Directed Acyclic Graph (DAG)** — no cycles, so data always
flows forward.

### 1.2 Type lattice

Data types form a partial order (lattice) T:

```text
UnscaledData
     |
 ScaledData
     |
EmbeddedData    ClassLabels    RegressionTargets
```

`EmbeddedData <: ScaledData <: UnscaledData` (subtype relation).

An edge (u, v) is **type-compatible** iff:

```text
output_type(u) <: input_type(v)
```

This is enforced at graph construction time. Attempting to add a type-incompatible edge
raises `TypeError`.

### 1.3 Topological execution

Given P and input data X, the pipeline executes as follows:

1. Compute a topological ordering `v_1, v_2, ..., v_n` of V.
2. For each `v_i` in order: `X_i = step_i.transform(X_{parent(i)})` (or `.fit_transform`
   during training).
3. The output of the final node (the sink, `v_n`) is the pipeline output.

For a source node with no parents, the input is the raw data X.

For nodes with multiple parents, outputs are **concatenated** along the feature axis.

---

## 2. Fitness Function

### 2.1 Base fitness

Given pipeline P, training data (X, y), and evaluation protocol CV:

```text
F(P, X, y) = (1/k) Σ_{i=1}^{k} metric(y_test_i, P.predict(X_test_i))
```

where k is the number of CV folds and `metric` is accuracy (classification) or R²
(regression).

### 2.2 Complexity penalty

To discourage unnecessarily large pipelines (genetic bloat):

```text
F_adj(P, X, y) = F(P, X, y) - λ · |V(P)|
```

where λ is the complexity penalty coefficient (default 0.002) and |V(P)| is the number of
steps.

### 2.3 Timeout handling

Each fitness evaluation runs in a thread with a wall-clock timeout T_max (default 30s).
If evaluation does not complete within T_max, the pipeline receives fitness −∞ and is
excluded from selection.

### 2.4 Evaluation cache

Pipelines are identified by a **structure hash**:

```text
hash(P) = SHA256(canonical_adjacency_string + sorted_hyperparams)
```

If hash(P) exists in the cache, the cached fitness is returned immediately without
re-running cross-validation. The cache uses FIFO eviction with a configurable maximum size
(default 512 entries).

---

## 3. Population Model

### 3.1 Individual

An individual `i` wraps a pipeline P with its fitness score s:

```text
i = (P_i, s_i)    where s_i ∈ ℝ ∪ {-∞}
```

### 3.2 Population

A population `pop` at generation `g` is an ordered list of N individuals:

```text
pop_g = [(P_1, s_1), ..., (P_N, s_N)]
```

N is the population size (default 20).

### 3.3 Mean and best

```text
mean(pop_g) = (1/N) Σ s_i    (excluding -∞)
best(pop_g) = argmax_i s_i
```

---

## 4. Genetic Operators

### 4.1 Tournament selection

Given tournament size k (default 3):

1. Sample k individuals uniformly at random from the population (with replacement).
2. Return the individual with the highest fitness score.

The probability that the best individual in a population of size N wins a tournament of
size k is:

```text
P(best wins) = 1 - ((N-1)/N)^k
```

For N=20, k=3: P(best wins) ≈ 0.143 per tournament.

Selection pressure can be increased by raising k; k=N degenerates to always selecting
the best (elitism-only).

### 4.2 Subgraph-exchange crossover

Given two parent pipelines P₁ and P₂:

1. Find a valid "cut point" in P₁: a node v such that the subgraph rooted at v is
   type-compatible with the corresponding position in P₂.
2. Extract the subgraph S₁ below v from P₁ and S₂ from the matching position in P₂.
3. Produce child C₁ by replacing S₁ with S₂ in P₁, and C₂ by replacing S₂ with S₁ in P₂.
4. Validate both children (acyclicity, type compatibility). Fall back to cloning parents if
   validation fails.

This preserves valid structures while mixing topologies from both parents.

### 4.3 Mutation operators

Five mutation operators are applied independently, each with probability p_mut (default 0.3
per operator):

| Operator | Description |
| --- | --- |
| `add_node` | Insert a randomly sampled compatible step between two existing nodes |
| `remove_node` | Delete a non-essential intermediate step and reconnect its neighbors |
| `replace_node` | Swap one step for another of the same type from the registry |
| `mutate_hyperparams` | Resample one hyperparameter from the step's declared search space |
| `add_skip_edge` | Add a direct edge from an earlier node to a later compatible node |

After each mutation, the pipeline is validated. Invalid mutations are discarded silently.

---

## 5. The GA Loop

```text
Inputs: X, y, N, G, k_cv, λ, T_max, p_cross, p_mut, k_elite

pop_0 ← random_initialize(N)

for g = 0 to G-1:
    evaluate_unevaluated(pop_g, X, y)       # fitness with cache + timeout
    log(g, best(pop_g), mean(pop_g))

    elites ← top_k(pop_g, k_elite)          # elitism

    offspring ← []
    while |offspring| < N - k_elite:
        p1 ← tournament_select(pop_g)
        p2 ← tournament_select(pop_g)

        if rand() < p_cross:
            c1, c2 ← crossover(p1, p2)
        else:
            c1, c2 ← clone(p1), clone(p2)

        c1 ← maybe_mutate(c1, p_mut)
        c2 ← maybe_mutate(c2, p_mut)
        offspring.append(c1, c2)

    pop_{g+1} ← elites + offspring[:N - k_elite]

    if plateau_detected(log, patience):
        break

# Final pass: evaluate any unevaluated individuals
evaluate_all(pop_G, X, y)
return best_across_all_generations
```

### 5.1 Plateau detection

A plateau occurs when the best fitness improves by less than `tolerance` (default 1e-4)
for `patience` consecutive generations (default 5). When a plateau is detected, evolution
terminates early.

```text
improvement_g = best(pop_g).score - best(pop_{g-1}).score
plateau_counter += 1 if improvement_g < tolerance else reset to 0
stop if plateau_counter >= patience
```

---

## 6. Type System Properties

### 6.1 Compatibility check

```python
def compatible(output_type, input_type):
    return issubclass(output_type, input_type)
```

This uses Python's built-in class hierarchy, making the type lattice extensible: any user
can add a new data type by subclassing an existing one.

### 6.2 Correctness guarantee

**Lemma**: If all edges in P satisfy type compatibility, then topological execution of P
on data of type `input_type(source)` will never encounter a type mismatch at any step.

**Proof sketch**: By induction on topological order. The source receives data of its
declared input type. For each subsequent node v, all parents produce output of type T_out.
By compatibility, T_out <: input_type(v), so v receives compatible data.

---

## 7. Complexity Analysis

| Operation | Complexity |
| --- | --- |
| Fitness evaluation (one pipeline) | O(k · n · fit_cost) where n = dataset size |
| Cache lookup | O(1) average (hash map) |
| Tournament selection | O(k_tournament) |
| Crossover | O(\|V\|) |
| Mutation (single operator) | O(\|V\| + \|E\|) |
| One generation | O(N · k · n · fit_cost) without cache |
| Full evolution (G generations) | O(G · N · k · n · fit_cost) worst case |

The cache is the dominant optimisation: in practice, many pipelines recur across
generations. Cache hit rates of 30–60% are typical, roughly halving wall-clock time.

---

## 8. Fitness Landscape Analysis

### 8.1 Neutrality

Many mutations produce pipelines with nearly identical fitness (neutral drift). This is
desirable: neutral mutations allow the population to explore topology space without
immediate selection pressure.

### 8.2 Epistasis

Pipeline steps interact: the benefit of adding PCA depends on which classifier follows.
High epistasis makes the fitness landscape rugged and the global optimum hard to find
by local search. Crossover helps by recombining subgraphs that co-evolved as compatible
units.

### 8.3 No free lunch

By the No Free Lunch theorem (Wolpert & Macready, 1997), no search algorithm is
universally better than random. C60.ai is not exempt. It is designed to perform well
on the class of problems where pipeline topology matters — structured tabular data with
diverse feature types and moderate sample sizes.

---

## 9. The Explainability Framework

### 9.1 Feature importances

For each step v in the pipeline, the introspector extracts importances by checking, in
order:

1. `step.feature_importances_` (tree models)
2. `|step.coef_|` averaged over classes (linear models)
3. `step.explained_variance_ratio_` (PCA)
4. `step.scores_` (SelectKBest)

Importances are L1-normalised to sum to 1 across all features.

### 9.2 PipelineStory

The story is generated from the `EvolutionLog`, which records for each generation g:

```text
record_g = (generation, best_score_g, mean_score_g, wall_time_g, best_pipeline_hash_g)
```

The narrative identifies:

- Total improvement: `best_score_G - best_score_0`
- Generations with significant improvement (delta > 1% of total range)
- Plateau length: number of trailing non-improving generations
- Best pipeline topology at termination

---

## 10. Hybrid Neural Nodes

### 10.1 NeuralAutoencoder

Implements a bottleneck autoencoder as a sklearn-compatible transformer:

```text
Encoder: n_features → hidden_dim → bottleneck_dim   (ReLU activations)
Decoder: bottleneck_dim → hidden_dim → n_features    (for reconstruction loss)
```

Training minimises MSE reconstruction loss using Adam. At transform time, only the encoder
is applied, producing `bottleneck_dim`-dimensional embeddings.

Output type: `EmbeddedData <: ScaledData`, so the output is compatible with any step that
accepts `ScaledData`.

### 10.2 NeuralClassifier

Implements a two-hidden-layer MLP:

```text
n_features → hidden_dim → Dropout(0.2) → hidden_dim → n_classes
```

Trained with CrossEntropyLoss and Adam. Produces class predictions and probabilities.

### 10.3 Reproducibility

Both nodes accept a `random_seed` parameter. PyTorch RNG state is seeded at fit time.
Deep-copying (required by the GA for population management) uses `torch.save/load` on a
`BytesIO` buffer to safely serialise weight tensors.

---

## References

- Wolpert, D. H., & Macready, W. G. (1997). No free lunch theorems for optimization.
  *IEEE Transactions on Evolutionary Computation*, 1(1), 67–82.
- Koza, J. R. (1992). *Genetic Programming*. MIT Press.
- Miller, J. F., & Smith, S. L. (2006). Redundancy and computational efficiency in
  Cartesian genetic programming. *IEEE TEVC*, 10(2), 167–174.
- Eiben, A. E., & Smith, J. E. (2015). *Introduction to Evolutionary Computing* (2nd ed.).
  Springer.
- Olson, R. S., et al. (2016). TPOT: A tree-based pipeline optimization tool for
  automating machine learning. *AutoML Workshop, ICML*.
- Feurer, M., et al. (2015). Efficient and robust automated machine learning. *NeurIPS*.
