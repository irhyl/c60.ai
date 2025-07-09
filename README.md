# C60.ai: AutoML Framework

## Overview

C60.ai is a high-performance Automated Machine Learning (AutoML) framework that redefines pipeline optimization. Inspired by molecular evolution, it treats machine learning pipelines as flexible, graph-based structures that can mutate, recombine, and adapt to complex tasks. This innovative approach enables open-ended, explainable, and hybrid symbolic/neural pipeline search, moving beyond traditional template-based AutoML systems.

Our mission is to empower researchers and practitioners with an AutoML solution that is both powerful and transparent, capable of discovering novel and highly optimized ML workflows for diverse problem sets.

## Key Features

- **Evolutionary Pipeline Optimization:**
  - Molecular graph-based pipeline representation (Directed Acyclic Graphs - DAGs).
  - Genetic algorithm-driven pipeline evolution.
  - Support for multi-objective optimization (e.g., accuracy vs. latency).

- **Advanced ML Capabilities:**
  - Automated feature engineering and selection.
  - Hyperparameter optimization.
  - Foundational support for model interpretability and explainability.
  - Designed for classification and regression tasks.

- **Performance & Scalability (Conceptual for Portfolio):**
  - Architecture designed for distributed computing.
  - Efficient memory management principles applied.
  - GPU acceleration hooks for future integration.
  - Parallel pipeline evaluation (local simulation for portfolio).

- **Developer Experience:**
  - Clean, modular API design for easy extension.
  - Comprehensive test suite (will be built out).
  - Type hints throughout the codebase for robust development.
  - Pre-commit hooks for consistent code quality.

## Installation

This project uses `pyproject.toml` as the canonical source for build and dependency configuration, ensuring a modern and reproducible setup.

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- git (for cloning the repository)

### Quick Install

To get C60.ai up and running on your local machine:

```bash
# 1. Clone the repository
git clone [https://github.com/aditirkrishna/c60.ai.git](https://github.com/aditirkrishna/c60.ai.git) 
cd c60.ai

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 3. Install C60.ai with development dependencies
pip install -e ".[dev]"
```

## Quick Start

Once installed, you can begin interacting with C60.ai. More detailed examples will be provided in the `examples/` directory as the project progresses.

### Basic Usage (Conceptual)

```python
# from c60 import AutoML # This will be our main entry point
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split

# # Load sample dataset
# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize AutoML with molecular evolution
# automl = AutoML(
# #   task='classification',
# #   evolution_budget=10, # Number of generations for evolution
# #   metric='accuracy'
# )

# # Fit the molecular pipeline
# # automl.fit(X_train, y_train)

# # Make predictions
# # predictions = automl.predict(X_test)

# # Evaluate performance
# # from sklearn.metrics import accuracy_score
# # print(f"Model accuracy: {accuracy_score(y_test, predictions):.4f}")
```

## Documentation

Comprehensive documentation, including API reference, advanced usage guides, and detailed explanations of the molecular evolution paradigm, will be available soon.

[Visit our documentation site (coming soon)](https://c60-ai.readthedocs.io)

---

## Development Setup

For contributors looking to extend or modify C60.ai:

```bash
# After following "Quick Install" steps:
# Run tests
pytest

# Run linting and formatting checks
black .
flake8
mypy .
