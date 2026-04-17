# C60.ai — The Concept

*This document explains the ideas behind C60.ai for someone who has never built a machine
learning system before. No maths required.*

---

## 1. What is Machine Learning?

Machine learning is a way of teaching computers to make decisions by showing them examples,
rather than by writing explicit rules.

For example:

- You show a computer 10 000 photos labelled "cat" or "not cat".
- The computer learns patterns in the pixel values that distinguish the two.
- Now it can classify new photos it has never seen.

The key ingredient is a **model** — a mathematical function that maps inputs (photo pixels)
to outputs (cat / not cat). Training means adjusting the model's internal numbers until it
makes good predictions on the examples you gave it.

---

## 2. What is a Machine Learning Pipeline?

Real-world data is messy. Before feeding it to a model you almost always need to:

1. **Clean it** — handle missing values, fix data types
2. **Scale it** — bring all numbers into a similar range so no single feature dominates
3. **Select or transform features** — reduce noise, find the most useful signals
4. **Train a model** — logistic regression, decision tree, neural network, etc.

This chain of steps is called a **pipeline**.

```text
Raw Data
  |
  v  Step 1: Scale (StandardScaler)
  |
  v  Step 2: Select best 10 features (SelectKBest)
  |
  v  Step 3: Classify (GradientBoostingClassifier)
  |
  v  Prediction
```

Building a good pipeline for a new dataset is hard. Which scaler? Which feature selector?
Which model? Which hyperparameters for each? Getting these choices right — or even close to
right — can take a data scientist days or weeks.

---

## 3. What is AutoML?

AutoML (Automated Machine Learning) is software that builds the pipeline for you.

You give it a dataset and a task ("predict which customers will churn"). It tries many
combinations of steps and hyperparameters automatically, evaluates each one on a held-out
test set, and returns the best pipeline it found.

Popular AutoML tools: **auto-sklearn**, **TPOT**, **H2O AutoML**, **Google AutoML**,
**Azure AutoML**.

These tools are genuinely useful. But they all share one limitation that nobody talks about.

---

## 4. The Hidden Assumption All AutoML Tools Make

Every AutoML tool assumes the pipeline has a **fixed shape**:

```text
[one scaler] → [one feature selector] → [one model]
```

The software searches over *which* components fill the three slots. It never questions
*whether three slots is the right number*, or *whether a sequential chain is the right
shape*.

This is like trying to build the best possible sandwich but assuming it always has exactly
three layers — and never considering that some of the best sandwiches have five layers, or
use two different fillings side by side, or skip a layer entirely.

The fixed-shape assumption creates real limits:

- It cannot discover that combining outputs from *two different feature extractors* gives
  better signal than any single extractor.
- It cannot find that skipping scaling entirely is better on certain tree-based models.
- It cannot stumble onto an ensemble of three different classifiers trained on different
  subsets of features.

These are the kinds of structures that expert data scientists discover over years of
experience — and that AutoML, as currently built, will never find.

---

## 5. C60.ai's Idea: Pipelines Are Molecules

C60.ai removes the fixed-shape assumption.

In C60.ai, a pipeline is not a sequence. It is a **graph** — specifically a **Directed
Acyclic Graph (DAG)**.

```text
          StandardScaler
         /              \
        /                \
  PCA(n=8)         SelectKBest(k=15)
        \                /
         \              /
          [concatenate]
               |
               v
     GradientBoostingClassifier
               |
               v
           Prediction
```

This pipeline has a branching structure that no sequential AutoML tool can represent.
C60.ai can discover and evaluate it.

The name C60 comes from **Buckminsterfullerene** — a molecule made of 60 carbon atoms
arranged in an elegant, non-obvious sphere. Nobody designed it from the top down. It
emerges from the laws of chemistry. We believe good ML pipelines can emerge the same way —
from evolution.

---

## 6. How Evolution Works Here

C60.ai uses a **genetic algorithm** — a search technique inspired by biological evolution.

### Step 1 — Create a population

Start with, say, 20 random pipeline graphs. Each one is different — different steps,
different connections, different hyperparameters.

### Step 2 — Evaluate fitness

Run each pipeline on your data using cross-validation (train on part of the data, test on
the rest, to get an honest accuracy estimate). Each pipeline gets a fitness score
(e.g. accuracy = 0.87).

### Step 3 — Select parents

Pipelines with higher fitness scores are more likely to be chosen as parents for the next
generation. This is like natural selection: better-adapted individuals reproduce more.

### Step 4 — Crossover (recombination)

Take two parent pipelines. Swap a subgraph from one into the other. This produces two
"child" pipelines that share parts of both parents — just like a child inherits genes from
both parents.

### Step 5 — Mutation

Randomly change one thing in a pipeline:

- Add a new step (e.g. insert a `PCA` between two existing steps)
- Remove a step
- Replace a step with a different one (e.g. swap `SVM` for `RandomForest`)
- Change a hyperparameter (e.g. increase the number of trees)
- Add or remove an edge in the graph

### Step 6 — Elitism

Always keep the best few pipelines from the previous generation unchanged, so the
population never loses its best-found solution.

### Repeat

Do steps 2–6 for a fixed number of generations (e.g. 10 generations). Return the best
pipeline found across all generations.

---

## 7. Why "Molecular Evolution"?

The molecular metaphor is more than just a name.

In chemistry, a molecule's properties depend on **which atoms are present** *and* **how
they are connected** (the topology). Two molecules with the same atoms but different
connections are different substances with different properties.

The same is true for ML pipelines. Two pipelines using the exact same components but
connected differently behave very differently. C60.ai's genetic operators mutate and
recombine **topology** — the connections — not just the component choices.

The C60 molecule (Buckminsterfullerene) was discovered in 1985 and surprised everyone.
Nobody had predicted that 60 carbon atoms would arrange themselves into a stable sphere.
The structure emerged from the chemistry. We believe the best ML pipeline topologies also
surprise us — and should be discovered, not designed.

---

## 8. What Makes This Hard

Building this system involves several genuinely difficult problems:

**Type safety**: Not every step can connect to every other step. You can't feed the output
of a classifier back into a scaler. C60.ai defines a type lattice
(`UnscaledData → ScaledData → ClassLabels`) and checks compatibility on every edge
before inserting it.

**Exponential search space**: The number of possible pipeline graphs grows exponentially
with the number of available operations. You cannot try them all. The genetic algorithm
explores intelligently but is not guaranteed to find the global optimum.

**Evaluation cost**: Every fitness evaluation runs cross-validation, which fits the
pipeline multiple times. With 20 pipelines per generation and 10 generations, that is
200 cross-validations per run. C60.ai addresses this with an **evaluation cache** (never
re-evaluate a pipeline you've seen before) and a **per-pipeline timeout** (abandon
evaluations that take too long).

**Bloat**: Genetic algorithms tend to accumulate unnecessary complexity over time
(a phenomenon called "genetic bloat"). C60.ai adds a **complexity penalty** to the fitness
function, subtracting a small amount per step so that simpler pipelines are preferred when
accuracy is equal.

---

## 9. What C60.ai Can Do That Others Cannot

| Capability | Traditional AutoML | C60.ai |
| --- | --- | --- |
| Find a pipeline with parallel feature branches | No | Yes |
| Discover that a custom step ordering outperforms the default | No | Yes |
| Explain *why* the chosen structure was selected | No | Yes (PipelineStory) |
| Use a neural network layer inside the pipeline alongside sklearn steps | No | Yes (HybridNode) |
| Show a narrative of how the pipeline improved over generations | No | Yes |
| Expose a REST API for remote job submission | No | Yes |

---

## 10. What C60.ai Cannot Do (Yet)

- **Multi-task learning**: optimising the same pipeline across multiple datasets
  simultaneously.
- **Online learning**: adapting to a data stream without re-running the full evolution.
- **Transfer learning**: using knowledge from a previous run to warm-start the next one.
- **GPU-native execution**: the current implementation runs on CPU; very large datasets
  will be slow.
- **Guaranteed global optimum**: like all genetic algorithms, C60.ai may return a good but
  not perfect solution.

These are documented open problems in [`research/molecular_concept.md`](../research/molecular_concept.md).

---

## Summary

| Concept | One-line definition |
| --- | --- |
| Pipeline | A chain (or graph) of data transformation and modelling steps |
| AutoML | Software that builds and selects the best pipeline automatically |
| DAG pipeline | A pipeline represented as an arbitrary graph, not just a sequence |
| Genetic algorithm | A search method inspired by biological evolution |
| Fitness function | The score used to evaluate how good a pipeline is (e.g. accuracy) |
| Structural mutation | Changing the graph topology (add/remove/replace nodes and edges) |
| EvaluationCache | A lookup table so already-tested pipelines are never re-evaluated |
| PipelineStory | A human-readable explanation of how the best pipeline was found |
| Molecular metaphor | Treating pipeline topology the same way chemistry treats bond structure |
