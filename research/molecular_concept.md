# C60.ai: Molecular Evolution for AutoML
## Complete Research, Architecture & Implementation Guide

> **Version:** 0.2.0-research  
> **Status:** Living document — updated as implementation progresses  
> **Scope:** Theory → Mathematics → Architecture → Phase-by-phase build plan

---

## Table of Contents

1. [The Problem Space](#1-the-problem-space)
2. [Core Philosophy: Why Molecular Evolution](#2-core-philosophy-why-molecular-evolution)
3. [Theoretical Foundations](#3-theoretical-foundations)
4. [Mathematical Formalization](#4-mathematical-formalization)
5. [Type System & Compatibility](#5-type-system--compatibility)
6. [System Architecture](#6-system-architecture)
7. [Implementation: Phase by Phase](#7-implementation-phase-by-phase)
8. [Advanced Research Directions](#8-advanced-research-directions)
9. [Benchmarking Strategy](#9-benchmarking-strategy)
10. [Open Problems](#10-open-problems)

---

## 1. The Problem Space

### 1.1 What Existing AutoML Does Well

Automated Machine Learning frameworks like auto-sklearn, TPOT, H2O AutoML, and Google AutoML have genuinely moved the needle. They automate the tedious loop of: pick a preprocessing step → pick a model → tune hyperparameters → evaluate. For practitioners without deep ML expertise, they produce competitive baselines quickly.

### 1.2 Where They All Break Down

Every major AutoML framework shares a hidden assumption: **the pipeline has a fixed shape**. The search space is not over arbitrary graphs — it is over parameterized templates. The system knows in advance that a pipeline is always `[Preprocessor → FeatureSelector → Classifier]`, and it searches over which components fill each slot.

This creates hard ceilings:

| Limitation | Consequence |
|---|---|
| Fixed topology | Cannot discover that two parallel feature extractors with a late ensemble outperforms a sequential chain |
| No structural mutation | Hyperparameter tuning can only exploit — it cannot explore new shapes |
| No memory of evolution | Each candidate is evaluated in isolation; there is no lineage, no "this worked before in this context" |
| Black-box outputs | User receives a pipeline object with no explanation of why this structure was chosen |
| No cross-task transfer | Learning from one dataset does not inform search on the next |

### 1.3 The C60.ai Hypothesis

The central claim of C60.ai:

> **An ML pipeline is not a sequence. It is a molecule — a graph of functional units whose topology is itself the primary optimization target, above and beyond the choice of components or their hyperparameters.**

If we represent pipelines as directed acyclic graphs and apply graph-level genetic operators (structural mutations, subgraph crossover), we can search over a vastly larger space than any template-based system — and discover pipeline shapes that no human would design by hand.

The name C60 references the Buckminsterfullerene molecule (60 carbon atoms in a highly stable, non-obvious spherical lattice) — a structure that would never have been predicted from first principles but emerges naturally from the laws of molecular self-organization. The same principle applies here.

### 1.4 What Success Looks Like

A successful C60.ai system will:

1. Accept a raw dataset and a task type (classification, regression)
2. Initialize a population of diverse pipeline DAGs
3. Evolve them over generations using graph mutation and crossover
4. Return the best pipeline with a full evolutionary trace — what structural changes led to improvement
5. Be competitive with TPOT and auto-sklearn on standard benchmarks (OpenML-CC18 suite)
6. Produce human-readable explanations of why the winning pipeline is structured the way it is

---

## 2. Core Philosophy: Why Molecular Evolution

### 2.1 The Biological Analogy — Taken Seriously

In molecular biology, a molecule's *function* is inseparable from its *structure*. Hemoglobin works because of its quaternary structure — four subunits arranged in a specific spatial configuration. Changing one bond changes the function. This is not metaphor: it is the mechanism.

We apply the same principle. In a pipeline DAG:
- **Nodes** = functional units (atoms / amino acids)
- **Edges** = data flow (covalent bonds)
- **Topology** = the structural configuration that determines behavior
- **Fitness** = performance on a held-out validation set (the molecule's "function")
- **Evolution** = genetic algorithm iterating toward higher fitness

The analogy is tight enough to be load-bearing, not just decorative.

### 2.2 Why Genetic Algorithms Over Bayesian Optimization

Bayesian Optimization (BO) is excellent at optimizing *continuous* hyperparameter spaces. It models the objective function as a Gaussian Process and selects the next evaluation point to maximize expected improvement.

It is poorly suited to *structural* search because:
- DAG topologies are discrete and combinatorial, not continuous
- The "distance" between two DAGs is not naturally expressible as a kernel
- BO does not naturally represent population diversity — it converges to a single best point

Genetic Algorithms (GAs) operate on populations of discrete structures, use explicit diversity mechanisms (crossover, mutation), and maintain a Pareto front of solutions. They are the right tool for structural search.

### 2.3 Why DAGs and Not Trees

Earlier graph GP systems (like TPOT's underlying DEAP framework) use trees. Trees are a subset of DAGs. DAGs are strictly more expressive:

- DAGs allow **fan-out** (one step feeds multiple downstream steps)
- DAGs allow **fan-in** (multiple steps merge into one — e.g., ensemble)
- DAGs allow **parallel branches** (independent preprocessing paths for different feature groups)

Restricting to trees forces every "ensemble" into an awkward nested structure. DAGs represent ensembles, stacking, and feature-level branching natively.

---

## 3. Theoretical Foundations

### 3.1 Directed Acyclic Graphs

A **Directed Acyclic Graph (DAG)** $G = (V, E)$ consists of:
- A finite set of vertices $V$
- A set of directed edges $E \subseteq V \times V$
- The acyclicity constraint: there is no path $v \to \cdots \to v$ for any $v \in V$

Key properties we rely on:
- **Topological ordering always exists** for any DAG. This gives us a valid execution order.
- **Sources** (in-degree 0) are the pipeline's data entry points.
- **Sinks** (out-degree 0) are the pipeline's output nodes (final predictions).
- **Reachability** determines which nodes a given node depends on — used for subgraph extraction during crossover.

### 3.2 Genetic Algorithms — A Rigorous Treatment

A Genetic Algorithm maintains a population $\mathcal{P} = \{P_1, P_2, \ldots, P_N\}$ of candidate solutions. Each iteration (generation) applies:

**1. Evaluation:** Compute fitness $f(P_i)$ for each individual.

**2. Selection:** Choose parents probabilistically, biased toward higher fitness. Common strategies:
- *Tournament selection:* Pick $k$ random individuals; return the fittest. Parameter $k$ controls selection pressure.
- *Fitness-proportionate (roulette wheel):* Probability of selection proportional to $f(P_i) / \sum_j f(P_j)$.
- *Rank-based:* Probability proportional to fitness rank, not raw value. More robust to fitness scaling.

**3. Genetic Operators:**
- *Crossover:* Combine two parents to produce offspring.
- *Mutation:* Randomly alter a single individual.

**4. Replacement:** Form the next generation from offspring, optionally preserving the best individuals (*elitism*).

**Convergence** is typically declared when:
- A maximum number of generations is reached
- Fitness improvement over the last $k$ generations is below a threshold $\epsilon$
- Wall-clock time budget is exhausted

### 3.3 Schema Theorem and Building Blocks

Holland's Schema Theorem provides the theoretical justification for why GAs work. A **schema** is a template that matches a subset of the population (e.g., "all pipelines that include a StandardScaler followed by a RandomForest"). The theorem states that short, low-order schemata with above-average fitness receive exponentially increasing representation over generations.

In graph terms: if a particular subgraph (e.g., `PCA → SVM`) consistently appears in high-fitness pipelines, the GA will propagate and recombine that subgraph. This is the graph-level building block hypothesis — the theoretical basis for crossover being useful.

### 3.4 Bloat and Complexity Control

A known failure mode of symbolic regression / graph GP is **bloat** — population members grow larger each generation without a corresponding fitness increase. This happens because:
- Large structures are harder to "break" by mutation (they have more nodes to absorb damage)
- Crossover preferentially exchanges large subtrees (larger targets)

We control bloat with **parsimony pressure**: penalizing pipeline complexity in the fitness function. The adjusted fitness is:

$$f_{\text{adj}}(P) = f_{\text{raw}}(P) - \lambda \cdot \text{complexity}(P)$$

where $\text{complexity}(P)$ can be node count, total parameter count, or estimated inference latency. $\lambda$ is a tunable penalty coefficient.

---

## 4. Mathematical Formalization

### 4.1 The Pipeline as a Typed DAG

A pipeline $P$ is a typed DAG:

$$P = (V, E, \tau, \theta)$$

Where:
- $V = \{v_1, \ldots, v_n\}$ — set of pipeline steps (nodes)
- $E \subseteq V \times V$ — directed data-flow edges, subject to acyclicity
- $\tau: V \rightarrow \mathcal{T}$ — type assignment mapping each node to a step type from the type lattice $\mathcal{T}$
- $\theta: V \rightarrow \Theta$ — hyperparameter assignment, mapping each node to its configuration

Each node $v_i$ is defined by:

$$v_i = (\text{id}_i,\ \text{op}_i,\ \tau_i,\ \theta_i)$$

- $\text{id}_i$ — globally unique identifier (UUID)
- $\text{op}_i$ — the callable operation (e.g., a scikit-learn transformer instance)
- $\tau_i \in \mathcal{T}$ — step type
- $\theta_i \in \Theta_i$ — hyperparameter configuration, drawn from the step's search space $\Theta_i$

An edge $(v_i, v_j) \in E$ is **type-valid** if and only if the output type of $v_i$ is compatible with the input type of $v_j$ (defined formally in Section 5).

### 4.2 The Fitness Function

Given a dataset $\mathcal{D} = \{(x_k, y_k)\}_{k=1}^{m}$, the raw fitness of pipeline $P$ is estimated via $K$-fold cross-validation:

$$f_{\text{raw}}(P) = \frac{1}{K} \sum_{k=1}^{K} \mathcal{M}(y_{\text{val}}^{(k)},\ P.\text{predict}(X_{\text{val}}^{(k)}))$$

where $\mathcal{M}$ is a task-appropriate metric (accuracy for classification, $R^2$ or RMSE for regression).

The adjusted fitness with parsimony pressure:

$$f_{\text{adj}}(P) = f_{\text{raw}}(P) - \lambda \cdot \frac{|V(P)|}{|V_{\max}|}$$

where $|V(P)|$ is the node count and $|V_{\max}|$ is a configurable maximum pipeline size. $\lambda \in [0, 1]$ controls the trade-off.

For multi-objective optimization (e.g., accuracy vs. inference latency), we use **Pareto dominance**: pipeline $P_1$ dominates $P_2$ if $P_1$ is at least as good on all objectives and strictly better on at least one. The GA maintains the Pareto front rather than a single best solution.

### 4.3 Mutation Operators

Each mutation operator $\mu_i: P \rightarrow P'$ transforms a pipeline into a modified version while preserving DAG validity.

#### Node Insertion
Select an edge $(v_a, v_b) \in E$. Sample a new step $v_{\text{new}}$ compatible with both endpoints. Replace the edge with $(v_a, v_{\text{new}})$ and $(v_{\text{new}}, v_b)$.

$$\mu_{\text{insert}}(V, E) = (V \cup \{v_{\text{new}}\},\ (E \setminus \{(v_a, v_b)\}) \cup \{(v_a, v_{\text{new}}), (v_{\text{new}}, v_b)\})$$

**Constraint:** $v_{\text{new}}$ must be type-compatible: $\tau_{\text{out}}(v_a)$ must be accepted by $v_{\text{new}}$, and $v_{\text{new}}$'s output type must be accepted by $v_b$.

#### Node Deletion
Select a non-source, non-sink node $v_i$ with exactly one predecessor $v_a$ and one or more successors $\{v_{b_1}, \ldots, v_{b_k}\}$. Remove $v_i$ and reconnect:

$$\mu_{\text{delete}}(V, E) = (V \setminus \{v_i\},\ (E \setminus \delta(v_i)) \cup \{(v_a, v_{b_j})\ \forall j\})$$

where $\delta(v_i)$ is the set of all edges incident to $v_i$.

**Constraint:** The resulting edges must be type-valid. If reconnection is type-invalid, the mutation is rejected.

#### Node Replacement
Select a node $v_i$. Sample a new operation $\text{op}'$ of the same step type $\tau_i$ from the registry. Replace the operation and resample hyperparameters:

$$\mu_{\text{replace}}(v_i) = v_i' = (\text{id}_i,\ \text{op}',\ \tau_i,\ \theta'_i)$$

Because $\tau_i$ is preserved, type compatibility of all incident edges is automatically maintained.

#### Hyperparameter Mutation
Select a node $v_i$ and a hyperparameter $h \in \theta_i$. Perturb it according to its type:
- *Continuous:* $h' = h + \mathcal{N}(0, \sigma^2)$, clipped to the valid range
- *Integer:* $h' = h + \text{Uniform}(\{-\delta, \ldots, +\delta\})$
- *Categorical:* $h' \sim \text{Uniform}(\text{domain}(h))$

#### Edge Modification
Select a node $v_i$ and one of its outgoing edges $(v_i, v_j)$. Redirect it to a different valid target $v_k$ (where $(v_i, v_k)$ does not already exist and would not create a cycle). This operator can create branching structures and is the highest-risk mutation — it requires full DAG revalidation.

### 4.4 Crossover Operator

The primary crossover operator is **compatible-point subgraph exchange**.

Given parents $P_1 = (V_1, E_1)$ and $P_2 = (V_2, E_2)$:

1. Find a **cut node** pair $(v_i \in V_1, v_j \in V_2)$ where $\tau(v_i) = \tau(v_j)$ (same step type — compatible cut point)
2. Let $\text{Sub}(P, v)$ denote the subgraph reachable from $v$ (i.e., $v$ and all its descendants)
3. Offspring $O_1$ is formed by replacing $\text{Sub}(P_1, v_i)$ with $\text{Sub}(P_2, v_j)$
4. Validate $O_1$ is a valid DAG and type-consistent
5. If validation fails, fall back to returning $P_1$ unchanged

$$O_1 = (V_1 \setminus \text{Sub}(P_1, v_i)) \cup \text{Sub}(P_2, v_j)$$

**Finding valid cut points** is the hardest part of graph crossover. The algorithm:
1. Build type-indexed maps: for each type $\tau$, list all nodes of that type in each parent
2. For each type that appears in both parents, identify candidate cut pairs
3. From candidate pairs, filter those where the subgraph swap produces a valid DAG
4. If multiple valid pairs exist, select randomly or by subgraph fitness contribution

### 4.5 Selection Mechanisms

**Tournament Selection** (recommended default):

```
select_parent(population, k):
    contestants = random.sample(population, k)
    return max(contestants, key=fitness)
```

Tournament size $k$ controls selection pressure:
- Small $k$ (e.g., 2): low pressure, high diversity
- Large $k$ (e.g., 7): high pressure, fast convergence but premature convergence risk

**Elitism:** Always carry the top $e$ individuals unchanged into the next generation. Prevents losing the best-found solution. Typical: $e = 2$.

### 4.6 Population Initialization

The initial population must be **diverse**. Strategies:

- *Seeded templates:* Include a handful of known-good linear pipelines (Scaler → PCA → Classifier). These serve as "genetic anchors."
- *Random DAGs:* Generate random type-valid DAGs of varying depth (1–5 nodes). Ensure at least one source and one sink.
- *Type-guided random:* Start from a source node (data ingestion), repeatedly sample a type-compatible next node, with a probability of branching at each step.

The initial population should cover a range of complexities: simple (2-node) to moderately complex (6-8 nodes). Do not start with very large pipelines.

---

## 5. Type System & Compatibility

This is the most important engineering component and the one most often skipped in research implementations. Without a type system, mutation and crossover operators will constantly produce invalid pipelines (e.g., connecting a Classifier output to a Scaler input), wasting evaluation budget.

### 5.1 The Type Lattice

Define a type lattice $\mathcal{T}$ with the following base types:

```
DataType
├── TabularData          # 2D array of floats (n_samples × n_features)
│   ├── RawTabular       # May contain NaN, categorical, mixed
│   └── CleanTabular     # Imputed, encoded, numeric-only
│       ├── ScaledData   # Normalized / standardized
│       └── UnscaledData
├── Features             # Transformed feature matrix (output of feature engineering)
├── Predictions          # Final output (class labels or continuous values)
│   ├── ClassLabels      # Discrete predictions
│   └── RegressionValues # Continuous predictions
└── Probabilities        # Probability estimates (output of predict_proba)
```

Each `PipelineStep` declares:
- `input_type: DataType` — what it accepts
- `output_type: DataType` — what it produces

An edge $(v_i, v_j)$ is valid iff `output_type(v_i)` is a subtype of `input_type(v_j)` in the lattice.

### 5.2 Step Type Taxonomy

```
StepType
├── Imputer              # RawTabular → CleanTabular
├── Encoder              # RawTabular → CleanTabular (handles categoricals)
├── Scaler               # CleanTabular → ScaledData
├── FeatureSelector      # CleanTabular | ScaledData → Features
├── FeatureEngineer      # CleanTabular | ScaledData → Features
├── DimensionReducer     # Features | ScaledData → Features
├── Classifier           # Features | ScaledData → ClassLabels | Probabilities
├── Regressor            # Features | ScaledData → RegressionValues
└── Ensembler            # (ClassLabels | Probabilities)+ → ClassLabels | Probabilities
```

The `+` on `Ensembler` input means it accepts multiple inputs (fan-in node).

### 5.3 Compatibility Check Algorithm

```python
def is_compatible(output_type: DataType, input_type: DataType) -> bool:
    return is_subtype(output_type, input_type)

def is_subtype(child: DataType, parent: DataType) -> bool:
    # Walk up the type lattice from child; return True if parent is encountered
    ...
```

This function is called:
1. During `Pipeline.add_edge()` — reject type-invalid edges immediately
2. During mutation — filter candidate operations before applying
3. During crossover — validate the resulting DAG before accepting

---

## 6. System Architecture

### 6.1 Component Map

```
c60.ai
│
├── core/                        # Foundational data structures
│   ├── pipeline.py              # Pipeline, PipelineStep (typed DAG)
│   ├── types.py                 # DataType lattice, compatibility checks
│   └── registry.py              # OperationRegistry — catalog of all ops
│
├── execution/                   # Running a pipeline on data
│   ├── executor.py              # PipelineExecutor — topological execution
│   └── validator.py             # PipelineValidator — structural + type checks
│
├── evaluation/                  # Measuring pipeline fitness
│   ├── fitness.py               # FitnessEvaluator — cross-validation wrapper
│   ├── metrics.py               # Metric functions (accuracy, RMSE, F1, AUC)
│   └── cache.py                 # EvaluationCache — hash → score lookup
│
├── evolution/                   # Genetic algorithm engine
│   ├── operators/
│   │   ├── mutation.py          # All mutation operators
│   │   └── crossover.py        # Subgraph crossover
│   ├── selection.py             # TournamentSelector, ElitismSelector
│   ├── population.py            # Population — manages N pipeline individuals
│   └── engine.py                # EvolutionEngine — main GA loop
│
├── explainability/              # Lineage tracking and human-readable output
│   ├── log.py                   # EvolutionLog — structured event log
│   ├── introspector.py          # PipelineIntrospector — query the log
│   └── story.py                 # PipelineStory — render narrative + viz
│
├── hybrid/                      # Neuro-symbolic extensions (Phase 7)
│   ├── hybrid_node.py           # HybridNode — symbolic + neural in one step
│   └── nas.py                   # Neural Architecture Search integration
│
├── api/                         # External interfaces (Phase 8)
│   ├── rest.py                  # FastAPI REST endpoints
│   └── cli.py                   # Click-based CLI
│
└── automl.py                    # Top-level AutoML class (user-facing entry point)
```

### 6.2 Data Flow Through the System

```
User Input (X, y, task, budget)
        │
        ▼
  ┌─────────────────┐
  │  AutoML.fit()   │   ← top-level entry point
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Population     │   ← generates initial diverse pipeline DAGs
  │  Initializer    │     using OperationRegistry + type-guided sampling
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────────────────┐
  │              GA Loop                    │
  │                                         │
  │  ┌──────────────┐   ┌────────────────┐  │
  │  │ FitnessEval  │   │EvolutionLog    │  │
  │  │ (cross-val)  │   │(logs all events│  │
  │  └──────┬───────┘   └────────────────┘  │
  │         │                               │
  │  ┌──────▼───────┐                       │
  │  │  Selection   │ ← TournamentSelector  │
  │  └──────┬───────┘                       │
  │         │                               │
  │  ┌──────▼───────┐                       │
  │  │  Crossover   │ ← SubgraphExchange    │
  │  └──────┬───────┘                       │
  │         │                               │
  │  ┌──────▼───────┐                       │
  │  │  Mutation    │ ← 5 operator types    │
  │  └──────┬───────┘                       │
  │         │                               │
  │  ┌──────▼───────┐                       │
  │  │  Validation  │ ← DAG + type checks   │
  │  └──────┬───────┘                       │
  │         │                               │
  │  ┌──────▼───────┐                       │
  │  │  Replacement │ ← next generation     │
  │  └─────────────-┘                       │
  └──────────────────────────────┬──────────┘
                                 │ (stopping criteria met)
                                 ▼
                    ┌────────────────────────┐
                    │  Best Pipeline +       │
                    │  PipelineStory report  │
                    └────────────────────────┘
```

### 6.3 Key Interface Contracts

**PipelineStep** must expose:
- `fit(X, y=None) → self`
- `transform(X) → X_transformed` (for transformers)
- `predict(X) → y_pred` (for estimators)
- `clone() → PipelineStep` (deep copy for safe mutation)

**Pipeline** must expose:
- `fit(X, y) → self`
- `predict(X) → y_pred`
- `clone() → Pipeline`
- `topological_order() → List[PipelineStep]`
- `to_dict() / from_dict()` (serialization)

**FitnessEvaluator** must expose:
- `evaluate(pipeline: Pipeline, X, y) → float`

**EvolutionEngine** must expose:
- `fit(X, y) → Pipeline` (returns best pipeline)
- `history() → EvolutionLog`

---

## 7. Implementation: Phase by Phase

Each phase is self-contained and produces testable deliverables. Never start Phase N+1 until Phase N passes all tests.

---

### Phase 1 — Core Data Structures

**Goal:** Typed pipeline graph with full serialization and cloning support.

**Deliverables:**
- `src/c60/core/types.py` — DataType lattice + compatibility check
- `src/c60/core/pipeline.py` — PipelineStep, Pipeline (rewritten with types)
- `src/c60/core/registry.py` — OperationRegistry

**File: `types.py`**
```python
class DataType:
    """Base class for all data types in the type lattice."""
    pass

class TabularData(DataType): pass
class RawTabular(TabularData): pass
class CleanTabular(TabularData): pass
class ScaledData(CleanTabular): pass

class Features(DataType): pass
class Predictions(DataType): pass
class ClassLabels(Predictions): pass
class RegressionValues(Predictions): pass
class Probabilities(DataType): pass

def is_compatible(output_type: type, input_type: type) -> bool:
    """Returns True if output_type is a subclass of input_type."""
    return issubclass(output_type, input_type)
```

**File: `pipeline.py`** — Key additions over current version:
```python
class PipelineStep:
    input_type: type      # DataType subclass this step accepts
    output_type: type     # DataType subclass this step produces

    def fit(self, X, y=None) -> 'PipelineStep': ...
    def transform(self, X): ...
    def predict(self, X): ...
    def clone(self) -> 'PipelineStep': ...

class Pipeline:
    def fit(self, X, y) -> 'Pipeline': ...
    def predict(self, X): ...
    def clone(self) -> 'Pipeline': ...
    def topological_order(self) -> List[PipelineStep]: ...
    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict) -> 'Pipeline': ...
```

**File: `registry.py`**
```python
class OperationRegistry:
    """
    Central catalog of all available pipeline operations.
    Indexed by StepType. Each entry includes the operation class,
    its input/output types, and its hyperparameter search space.
    """
    def get_by_type(self, step_type: str) -> List[OperationSpec]: ...
    def sample_operation(self, step_type: str) -> PipelineStep: ...
    def sample_hyperparams(self, op_class, space: dict) -> dict: ...
```

**Acceptance Criteria:**
- [ ] `is_compatible(ScaledData, CleanTabular)` returns `True`
- [ ] `is_compatible(ClassLabels, CleanTabular)` returns `False`
- [ ] `Pipeline.clone()` produces a deep copy — mutating clone does not affect original
- [ ] `Pipeline.to_dict()` / `from_dict()` round-trips correctly
- [ ] `Pipeline.topological_order()` returns steps in valid execution order
- [ ] `registry.sample_operation("scaler")` returns a valid `PipelineStep`

---

### Phase 2 — Execution Engine

**Goal:** Execute a pipeline on real data, fitting each step in topological order.

**Deliverables:**
- `src/c60/execution/executor.py` — PipelineExecutor
- `src/c60/execution/validator.py` — PipelineValidator

**File: `executor.py`**
```python
class PipelineExecutor:
    def fit(self, pipeline: Pipeline, X, y) -> Pipeline:
        """
        Fit pipeline to data by executing steps in topological order.
        For fan-in nodes (Ensemblers), collect outputs of all predecessors.
        For fan-out nodes, pass the same data to all successors.
        """
        ...

    def predict(self, pipeline: Pipeline, X) -> np.ndarray:
        """
        Run prediction through fitted pipeline.
        """
        ...
```

**Fan-out handling:** When a node has multiple successors, its output is passed independently to each successor. The executor maintains a `node_output: Dict[step_id, ndarray]` map.

**Fan-in handling:** When a node has multiple predecessors (e.g., an Ensembler), the executor collects all predecessor outputs and passes them as a list. The Ensembler operation must handle list input.

**Execution pseudocode:**
```
node_outputs = {}
for step in pipeline.topological_order():
    predecessors = pipeline.predecessors(step)
    if len(predecessors) == 0:
        input_data = X          # source node
    elif len(predecessors) == 1:
        input_data = node_outputs[predecessors[0].id]
    else:
        input_data = [node_outputs[p.id] for p in predecessors]  # fan-in

    node_outputs[step.id] = step.transform(input_data)  # or predict for sinks
```

**File: `validator.py`**
```python
class PipelineValidator:
    def validate_structure(self, pipeline: Pipeline) -> ValidationResult:
        """Checks: is DAG, has at least one source, has at least one sink."""

    def validate_types(self, pipeline: Pipeline) -> ValidationResult:
        """Checks: all edges are type-compatible."""

    def validate(self, pipeline: Pipeline) -> ValidationResult:
        """Runs all checks. Returns list of errors."""
```

**Acceptance Criteria:**
- [ ] Linear pipeline `Imputer → Scaler → Classifier` fits and predicts on Iris dataset
- [ ] Branching pipeline `Scaler → [Model1, Model2] → Ensembler` executes correctly
- [ ] Invalid type edge raises `TypeCompatibilityError` in validator
- [ ] Fan-in node receives outputs from all predecessors

---

### Phase 3 — Operation Registry (Populated)

**Goal:** A fully populated registry of sklearn operations with hyperparameter search spaces.

**Deliverables:**
- `src/c60/core/registry.py` — populated with real operations
- `src/c60/core/operations/` — operation specification files

**Operations to register (minimum viable set):**

| Step Type | Operations |
|---|---|
| Imputer | `SimpleImputer(strategy='mean')`, `SimpleImputer(strategy='median')` |
| Scaler | `StandardScaler`, `MinMaxScaler`, `RobustScaler` |
| Encoder | `OneHotEncoder`, `OrdinalEncoder` |
| FeatureSelector | `SelectKBest`, `SelectFromModel` |
| DimensionReducer | `PCA`, `TruncatedSVD` |
| FeatureEngineer | `PolynomialFeatures` |
| Classifier | `LogisticRegression`, `RandomForestClassifier`, `SVC`, `GradientBoostingClassifier`, `KNeighborsClassifier` |
| Regressor | `LinearRegression`, `RandomForestRegressor`, `SVR`, `GradientBoostingRegressor` |
| Ensembler | `VotingClassifier`, `StackingClassifier` |

**Hyperparameter search spaces** are defined per operation:
```python
REGISTRY = {
    "RandomForestClassifier": OperationSpec(
        op_class=RandomForestClassifier,
        input_type=ScaledData,
        output_type=ClassLabels,
        search_space={
            "n_estimators": IntRange(10, 500),
            "max_depth": IntRange(2, 20, allow_none=True),
            "min_samples_split": IntRange(2, 20),
            "criterion": Categorical(["gini", "entropy"]),
        }
    ),
    ...
}
```

**Acceptance Criteria:**
- [ ] `registry.get_by_type("scaler")` returns at least 3 operations
- [ ] `registry.sample_operation("classifier")` returns a fitted-ready `PipelineStep`
- [ ] All registered operations have input/output types assigned
- [ ] All registered operations have at least one hyperparameter in their search space

---

### Phase 4 — Fitness Evaluation

**Goal:** Robustly evaluate pipeline fitness with caching to avoid re-evaluation.

**Deliverables:**
- `src/c60/evaluation/fitness.py`
- `src/c60/evaluation/metrics.py`
- `src/c60/evaluation/cache.py`

**File: `fitness.py`**
```python
class FitnessEvaluator:
    def __init__(
        self,
        metric: str = "accuracy",   # or "f1", "roc_auc", "rmse", "r2"
        cv: int = 5,
        timeout_seconds: float = 60.0,
        complexity_penalty: float = 0.01,
    ): ...

    def evaluate(self, pipeline: Pipeline, X, y) -> FitnessResult:
        """
        Returns FitnessResult(raw_score, adjusted_score, cv_scores, wall_time).
        Applies complexity penalty.
        Catches exceptions from broken pipelines and returns -inf fitness.
        Enforces timeout.
        """
        ...
```

**File: `cache.py`**
```python
class EvaluationCache:
    """
    Caches fitness evaluations keyed by pipeline structure hash.
    Pipeline hash = hash of (sorted node operations + hyperparams + sorted edges).
    Saves significant compute when the GA produces duplicate pipelines.
    """
    def get(self, pipeline: Pipeline) -> Optional[FitnessResult]: ...
    def put(self, pipeline: Pipeline, result: FitnessResult) -> None: ...
    def pipeline_hash(self, pipeline: Pipeline) -> str: ...
```

**Important:** The evaluator must catch and handle:
- Pipelines that throw during `fit` (e.g., incompatible shapes)
- Pipelines that exceed the timeout
- NaN fitness scores

Return `-inf` for broken pipelines rather than crashing the evolution loop.

**Acceptance Criteria:**
- [ ] `FitnessEvaluator.evaluate()` returns a valid float for a working pipeline on Iris
- [ ] Broken pipeline (e.g., no classifier at sink) returns `float('-inf')` not an exception
- [ ] Evaluating the same pipeline twice: second call hits cache, not cross-val
- [ ] Complexity penalty reduces score for a 10-node pipeline vs. a 3-node pipeline with equal raw score

---

### Phase 5 — Genetic Operators

**Goal:** Implement all mutation types and subgraph crossover, each preserving DAG validity and type correctness.

**Deliverables:**
- `src/c60/evolution/operators/mutation.py`
- `src/c60/evolution/operators/crossover.py`

**File: `mutation.py`**
```python
class MutationOperator(ABC):
    @abstractmethod
    def apply(self, pipeline: Pipeline, registry: OperationRegistry) -> Pipeline:
        """Apply mutation. Return new pipeline (do not modify in place)."""

class NodeInsertionMutation(MutationOperator): ...
class NodeDeletionMutation(MutationOperator): ...
class NodeReplacementMutation(MutationOperator): ...
class HyperparameterMutation(MutationOperator): ...
class EdgeRedirectionMutation(MutationOperator): ...

class MutationEngine:
    """
    Selects and applies a mutation operator randomly.
    Operator probabilities are configurable.
    Retries up to max_attempts if the mutation produces an invalid pipeline.
    """
    def __init__(self, operators: List[MutationOperator], probabilities: List[float]): ...
    def mutate(self, pipeline: Pipeline) -> Pipeline: ...
```

**File: `crossover.py`**
```python
class CrossoverOperator(ABC):
    @abstractmethod
    def apply(self, parent1: Pipeline, parent2: Pipeline) -> Tuple[Pipeline, Pipeline]:
        """Return two offspring pipelines."""

class SubgraphExchangeCrossover(CrossoverOperator):
    def find_cut_points(self, p1: Pipeline, p2: Pipeline) -> List[Tuple[str, str]]:
        """
        Find pairs of nodes (one from each parent) with the same step_type.
        These are valid cut points for subgraph exchange.
        """
        ...

    def apply(self, parent1, parent2) -> Tuple[Pipeline, Pipeline]:
        """
        Attempt crossover at a random compatible cut point.
        Validate result. If invalid, return copies of parents unchanged.
        """
        ...
```

**Acceptance Criteria:**
- [ ] `NodeInsertionMutation` always produces a valid DAG
- [ ] `NodeDeletionMutation` never disconnects a source from a sink (unless pipeline becomes a no-op)
- [ ] `NodeReplacementMutation` preserves edge type compatibility
- [ ] `HyperparameterMutation` stays within declared search space bounds
- [ ] `SubgraphExchangeCrossover` produces two valid DAGs, or falls back to parent copies
- [ ] All operators tested with both simple (2-node) and complex (8-node branching) pipelines

---

### Phase 6 — Evolution Engine

**Goal:** Tie everything together into a working GA loop.

**Deliverables:**
- `src/c60/evolution/population.py`
- `src/c60/evolution/selection.py`
- `src/c60/evolution/engine.py`

**File: `population.py`**
```python
class Individual:
    pipeline: Pipeline
    fitness: Optional[float] = None

class Population:
    def __init__(self, size: int): ...
    def initialize(self, registry, X, y, task: str) -> None: ...
    def evaluate_all(self, evaluator: FitnessEvaluator, X, y) -> None: ...
    def best(self) -> Individual: ...
    def sorted(self) -> List[Individual]: ...
```

**File: `selection.py`**
```python
class TournamentSelector:
    def __init__(self, tournament_size: int = 5): ...
    def select(self, population: Population) -> Individual: ...

class ElitismPreserver:
    def __init__(self, elite_count: int = 2): ...
    def get_elites(self, population: Population) -> List[Individual]: ...
```

**File: `engine.py`**
```python
class EvolutionEngine:
    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        elite_count: int = 2,
        fitness_plateau_patience: int = 10,
        max_time_seconds: float = 3600.0,
        metric: str = "accuracy",
        task: str = "classification",
    ): ...

    def fit(self, X, y) -> Pipeline:
        """
        Run the full GA loop. Returns the best pipeline found.
        Logs all events to self._log.
        """
        ...

    def history(self) -> EvolutionLog:
        """Return the full evolution log for introspection."""
        ...
```

**GA Loop Pseudocode:**
```
Initialize population (diverse, type-valid DAGs)
Evaluate all (cross-val fitness)
Log generation 0

For generation in 1..max_generations:
    If time budget exceeded: break
    If fitness plateau for patience generations: break

    next_gen = elites (top-e unchanged)

    While len(next_gen) < population_size:
        parent1 = tournament_select(population)
        parent2 = tournament_select(population)

        if random() < crossover_rate:
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = parent1.clone(), parent2.clone()

        if random() < mutation_rate:
            child1 = mutate(child1)
        if random() < mutation_rate:
            child2 = mutate(child2)

        evaluate(child1), evaluate(child2)   # uses cache if already seen
        next_gen.append(child1, child2)

    population = next_gen
    Log generation stats

Return best individual's pipeline
```

**Acceptance Criteria:**
- [ ] `engine.fit(X_iris, y_iris)` completes without error and returns a `Pipeline`
- [ ] Returned pipeline achieves > 80% accuracy on Iris (basic sanity check)
- [ ] Fitness improves monotonically (or stays flat due to elitism) — never decreases across generations
- [ ] `engine.history()` contains one entry per generation with best fitness, mean fitness, population size

---

### Phase 7 — Explainability Layer

**Goal:** Make the evolutionary process auditable and human-readable.

**Deliverables:**
- `src/c60/explainability/log.py`
- `src/c60/explainability/introspector.py`
- `src/c60/explainability/story.py`

**File: `log.py`**
```python
@dataclass
class GenerationRecord:
    generation: int
    timestamp: float
    best_fitness: float
    mean_fitness: float
    population_snapshot: List[dict]  # serialized pipelines + fitness

@dataclass
class MutationEvent:
    generation: int
    operator: str           # e.g., "NodeInsertionMutation"
    parent_hash: str
    offspring_hash: str
    fitness_delta: float    # offspring fitness - parent fitness

class EvolutionLog:
    def log_generation(self, record: GenerationRecord) -> None: ...
    def log_mutation(self, event: MutationEvent) -> None: ...
    def log_crossover(self, event: CrossoverEvent) -> None: ...
    def to_json(self) -> str: ...
```

**File: `introspector.py`**
```python
class PipelineIntrospector:
    def __init__(self, log: EvolutionLog): ...

    def lineage(self, pipeline: Pipeline) -> List[Pipeline]:
        """Trace the ancestral chain of a pipeline back to generation 0."""

    def most_impactful_mutation(self) -> MutationEvent:
        """Return the single mutation event with the largest fitness delta."""

    def operator_effectiveness(self) -> Dict[str, float]:
        """Return mean fitness delta per mutation operator type."""

    def fitness_trajectory(self) -> List[Tuple[int, float]]:
        """Return (generation, best_fitness) pairs."""
```

**File: `story.py`**
```python
class PipelineStory:
    def __init__(self, pipeline: Pipeline, log: EvolutionLog): ...

    def render_text(self) -> str:
        """
        Generate a human-readable narrative:
        - What is the final pipeline structure?
        - What were the 3 most impactful mutations in its lineage?
        - How did fitness evolve over generations?
        """

    def render_dag(self, output_path: str) -> None:
        """
        Render the pipeline DAG as an image using matplotlib + networkx.
        Nodes colored by step_type, edges labeled with data type.
        """

    def render_evolution_curve(self, output_path: str) -> None:
        """Plot best/mean fitness per generation."""
```

**Acceptance Criteria:**
- [ ] After `engine.fit()`, `engine.history()` contains at least one mutation event per generation
- [ ] `introspector.fitness_trajectory()` shows monotonically non-decreasing best fitness
- [ ] `story.render_text()` produces readable narrative without crashing
- [ ] `story.render_dag()` saves a valid PNG of the pipeline graph

---

### Phase 8 — HybridNode (Neuro-Symbolic Extension)

**Goal:** Allow individual pipeline nodes to contain PyTorch neural networks, enabling gradient-based learning within the evolutionary framework.

**This phase requires its own sub-research document** before implementation. Key open questions:

1. **Architecture search space:** What does a `HybridNode`'s neural component look like? Fully connected? CNN? How is it parameterized for structural mutation?
2. **Training protocol:** When does the neural component get trained? Within each cross-val fold? Separately?
3. **Gradient vs. evolution:** Do we use backprop *inside* the GA, or is the GA a black-box optimizer over the neural architecture?
4. **Memory and compute:** Neural operations are orders of magnitude more expensive than sklearn transformers. How do we budget evaluation time?

**Recommended approach for v1 of HybridNode:**
- A `HybridNode` wraps a fixed-architecture MLP (configurable hidden sizes)
- The MLP is trained per-fold during fitness evaluation (same as an sklearn estimator)
- Structural mutation for `HybridNode` only applies hyperparameter mutation (hidden sizes, learning rate, dropout) — not architecture search
- Full NAS integration is deferred to a later research phase

---

### Phase 9 — Infrastructure

**Goal:** Production-ready interfaces and distributed evaluation.

**Deliverables:**
- `src/c60/api/cli.py` — Click CLI
- `src/c60/api/rest.py` — FastAPI server
- Distributed evaluation via Ray

**CLI interface:**
```bash
c60 fit --data train.csv --target label --task classification \
        --generations 50 --population 30 --metric accuracy \
        --output best_pipeline.pkl

c60 predict --pipeline best_pipeline.pkl --data test.csv --output predictions.csv

c60 explain --pipeline best_pipeline.pkl --format text
c60 explain --pipeline best_pipeline.pkl --format png --output dag.png
```

**REST API endpoints:**
```
POST   /experiments          → create and start a new evolution run
GET    /experiments/{id}     → get status + current best fitness
GET    /experiments/{id}/best → download best pipeline
GET    /experiments/{id}/story → get text narrative
DELETE /experiments/{id}     → cancel run
```

**Distributed evaluation with Ray:**
```python
@ray.remote
def evaluate_pipeline(pipeline_dict, X, y, metric, cv):
    pipeline = Pipeline.from_dict(pipeline_dict)
    return FitnessEvaluator(metric=metric, cv=cv).evaluate(pipeline, X, y)

# In EvolutionEngine, evaluate population in parallel:
futures = [evaluate_pipeline.remote(p.to_dict(), X, y, ...) for p in population]
results = ray.get(futures)
```

---

## 8. Advanced Research Directions

### 8.1 Meta-Learning: Warm-Starting from Prior Tasks

Rather than always starting from random pipelines, we can warm-start the population using pipeline structures that performed well on "similar" datasets (measured by meta-features: n_samples, n_features, class imbalance ratio, feature correlations, etc.). This is the **meta-learning** approach and is the basis for systems like Auto-sklearn 2.0.

For C60.ai, this means:
1. After each successful run, store the winning pipeline structure in a **meta-knowledge base**
2. At the start of a new run, retrieve the top-k most similar past tasks (by meta-feature distance)
3. Seed the initial population with pipelines from those tasks
4. The GA then refines from a warm start rather than exploring from scratch

### 8.2 Multi-Fidelity Evaluation

Full cross-validation on every candidate pipeline is expensive. **Multi-fidelity** approaches evaluate cheap proxies first and only spend full budget on promising candidates:

- *Successive Halving:* Start with low-budget eval (1-fold, 10% data). Eliminate bottom half. Repeat with double budget. This is Hyperband applied to pipeline search.
- *Learning Curve Extrapolation:* Train on increasing fractions of data; extrapolate to predict full-data performance.
- *Surrogate Models:* Train a cheap model to predict pipeline fitness from pipeline structure features.

### 8.3 LLM-Driven Pipeline Generation

An emerging direction: use a Large Language Model to propose initial pipeline structures or suggest mutations, guided by task description and dataset characteristics.

The LLM acts as a **domain-knowledge oracle** — it can suggest "this dataset has high-cardinality categoricals, so include an encoder before scaling" — while the GA provides the optimization signal.

This requires:
- A structured prompt format describing the dataset and current population state
- A parser that converts LLM output into valid `Pipeline` objects
- A fitness oracle to validate LLM suggestions against real data

### 8.4 Reinforcement Learning as a Search Policy

Instead of random tournament selection, a trained RL agent learns a selection policy: given the current population state, which individuals to select as parents, and which mutation operators to apply. The reward signal is fitness improvement.

This converts the GA from a stateless stochastic search into a learned, adaptive search policy.

---

## 9. Benchmarking Strategy

### 9.1 Benchmark Suite

Use the **OpenML-CC18** benchmark suite: 72 diverse classification datasets covering tabular data of varying sizes, class counts, and feature types. This is the same suite used by auto-sklearn, TPOT, and Auto-WEKA for head-to-head comparisons.

For regression: use the **OpenML regression benchmark** (27 datasets).

### 9.2 Baselines to Beat

| Baseline | What it represents |
|---|---|
| Default RandomForest | Lower bound — no AutoML |
| Grid search (Scaler + Classifier) | Template-based baseline |
| TPOT (100 generations) | Best open-source GP-based AutoML |
| auto-sklearn | Best open-source BO-based AutoML |
| Random search over same pipeline space | Ablation: does structure matter, or is random good enough? |

### 9.3 Evaluation Protocol

- 10-fold outer cross-validation (test set never seen during AutoML search)
- Inner 5-fold cross-validation for fitness evaluation
- Fixed wall-clock budget: 1 hour per dataset per method
- Report: mean ± std accuracy across 10 outer folds

### 9.4 Ablation Studies

To justify specific design choices:

| Ablation | Tests |
|---|---|
| No crossover (mutation only) | Value of structural recombination |
| No type system (random connections) | Value of type-guided mutation |
| No parsimony penalty | Effect of bloat control |
| Random selection vs. tournament | Selection mechanism impact |
| Fixed topology (template) vs. free DAG | Core hypothesis of the project |

---

## 10. Open Problems

These are genuine research questions — not implementation gaps, but areas where the right answer is unknown:

1. **The Cut-Point Problem:** How do you efficiently find valid crossover cut points in two structurally dissimilar DAGs? Current approach (match by step type) is coarse. A finer approach might use learned graph embeddings to find "semantically similar" subgraphs across parents.

2. **Fitness Landscape Characterization:** How smooth or rugged is the fitness landscape over pipeline DAGs? Are there many local optima? Does the graph representation affect the landscape structure compared to tree representations? This determines whether gradient-free search (GA) or gradient-based search (BO with a graph kernel) is more appropriate.

3. **Diversity Preservation at Scale:** As population size increases, selection pressure tends to collapse diversity (genetic drift). What explicit diversity mechanisms (niching, fitness sharing, novelty search) work best in DAG space?

4. **Transfer Across Task Types:** Can a pipeline evolved for classification on dataset A transfer useful substructures to regression on dataset B? What is the right representation for cross-task transfer in graph space?

5. **Execution Semantics of Fan-In Nodes:** When multiple pipeline branches merge at an Ensembler, how should the ensembler operate? Voting (classification)? Averaging (regression)? Learned stacking? What is the right default, and how does the choice interact with evolution?

---

*This document is the authoritative design reference for C60.ai. All implementation decisions should be traceable to a section here. When implementation reveals that a design decision was wrong, update this document first, then the code.*
