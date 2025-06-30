# C60.ai: Molecular Evolutionary AutoML Framework

## Overview
C60.ai is an advanced AutoML framework inspired by molecular evolution. It treats machine learning pipelines as evolving molecules—flexible, graph-based structures that can mutate, recombine, and adapt to complex tasks. This approach enables open-ended, explainable, and hybrid symbolic/neural pipeline search, going far beyond template-based AutoML systems.

---

## Molecular Concept in C60.ai

- **Pipelines as Molecules:**
  - Each ML pipeline is represented as a directed acyclic graph (DAG), where nodes are processing steps (e.g., transformers, estimators, rules, neural nets).
  - Pipelines can be arbitrarily complex, not restricted to fixed templates.
  - Mutation and crossover operators act on the graph, just as chemical reactions modify molecules.

- **Graph-based Evolution:**
  - Populations of pipeline graphs are evolved using genetic algorithms, guided by a GNN (graph neural network) predictor.
  - Mutation: Random or learned changes to node/edge features (e.g., swapping steps, changing hyperparameters, inserting/removing nodes).
  - Crossover: Combining subgraphs from two parent pipelines to create new offspring.
  - Hybrid nodes allow mixing symbolic rules and neural models in the same pipeline.

- **RL and NAS Integration:**
  - Reinforcement Learning (RL) agents can guide pipeline mutation/selection using Q-learning or policy gradients.
  - Neural Architecture Search (NAS) agents search over neural node architectures, integrated as part of the graph.
  - Both RL and NAS agents are pluggable and extensible for research.

- **Explainability and Introspection:**
  - Every pipeline evolution, mutation, and evaluation step is logged by the `PipelineIntrospector`.
  - Users can query the full history and explanations for any pipeline, making the search process transparent and auditable.

- **Self-Documenting Pipelines:**
  - The `PipelineStory` module generates human-readable Markdown/HTML stories of each pipeline's evolution and structure.
  - GUI DAG visualization (using Plotly/NetworkX) lets users see and explore pipeline graphs interactively.

- **Hybrid Symbolic + Neural DAGs:**
  - Pipelines can contain both symbolic (rule-based) and neural (learned) nodes.
  - Mutation/crossover logic is hybrid-aware, supporting research into neuro-symbolic AutoML.

- **Agent-Based Code Generation:**
  - LLM agents can be integrated as pipeline nodes or used to generate/modify code for pipeline steps.

---

## Example: Molecular Evolution in Action

```python
from c60.core.pipeline import Pipeline, PipelineStep
from c60.introspect import HybridNode

# Define symbolic and neural components
symbolic_rule = lambda x: x + 1
neural_model = lambda x: x * 2

# Create a hybrid node
hnode = HybridNode("h1", "hybrid", symbolic_rule, neural_model)
step1 = PipelineStep("hybrid_step", hnode)
step2 = PipelineStep("neural_step", neural_model)

# Create two pipelines
pipe1 = Pipeline([step1, step2])
pipe2 = Pipeline([PipelineStep("symbolic_step", symbolic_rule)])

# Mutate and crossover
pipe1.mutate()
crossed = pipe1.crossover(pipe2)
print("After mutation:", pipe1)
print("After crossover:", crossed)
```

---

## Key Modules and APIs

- `PipelineIntrospector`: Logs and explains every pipeline evolution step.
- `RLSearchAgent`: Q-learning agent for RL-based pipeline search.
- `NASearchAgent`: Random or guided search over neural architectures.
- `HybridNode`: Enables hybrid symbolic/neural pipeline steps.
- `PipelineStory`: Generates Markdown/HTML stories and visualizes DAGs.
- `EvolutionarySearch`: Core genetic algorithm with GNN guidance, RL/NAS agent support, and introspection hooks.

---

## Philosophy
C60.ai is designed as a research sandbox for AGI/AutoML, enabling:
- Open-ended, composable, and extensible pipeline search
- Deep explainability and auditability
- Hybrid neuro-symbolic architectures
- Integration of RL, NAS, and LLM agents

If you want to extend, visualize, or audit any part of the evolutionary process, C60.ai provides the hooks, logs, and APIs to make it possible.

---

## Getting Started
1. Define your pipeline steps (symbolic, neural, or hybrid).
2. Initialize `EvolutionarySearch` with GNN, RL/NAS agents if desired, and introspector.
3. Run the search and visualize or query the results using `PipelineStory` and `PipelineIntrospector`.

For more, see the in-code docstrings and examples.
