# C60.ai - nothin', just a cool AutoML Framework

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

C60.ai is a high-performance AutoML framework that combines evolutionary algorithms with modern machine learning techniques to automate and optimize the entire ML pipeline development process. Designed with scalability and flexibility in mind, it's perfect for both research and production use cases.

## Key Features

- **Evolutionary Pipeline Optimization**
  - Molecular graph-based pipeline representation
  - Genetic algorithm-driven pipeline evolution
  - Multi-objective optimization support

- **Advanced ML Capabilities**
  - Automated feature engineering and selection
  - Hyperparameter optimization
  - Model interpretability and explainability
  - Support for classification and regression tasks

- **Performance & Scalability**
  - Distributed computing support
  - Efficient memory management
  - GPU acceleration
  - Parallel pipeline evaluation

- **Developer Experience**
  - Clean, modular API
  - Extensive documentation
  - Comprehensive test suite
  - Type hints throughout
  - Pre-commit hooks

## Installation

**Note:** This project uses `pyproject.toml` as the canonical source for build and dependency configuration. The legacy `setup.cfg` is deprecated and only kept for rare backward compatibility. Always use `pyproject.toml` for installation and packaging tasks.

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/c60.ai.git
cd c60.ai

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install with pip
pip install -e .
```

## 🏁 Quick Start

---

## Command-Line Interface (CLI)

You can interact with C60.ai via the command line:

```bash
# Load a dataset and validate
python interface/cli.py dataset load path/to/data.csv --validate

# Generate a pipeline for a dataset
python interface/cli.py pipeline generate --dataset path/to/data.csv --target target_column
```

---

## REST API

C60.ai provides a FastAPI-based REST API. To run the API server:

```bash
uvicorn interface.api:app --reload
```

### Example API usage (Python):
```python
import requests
# Create a dataset
resp = requests.post('http://localhost:8000/datasets/', json={
    'name': 'my-dataset',
    'file_path': 'path/to/data.csv'
})
# Create a pipeline
resp = requests.post('http://localhost:8000/pipelines/', json={
    'name': 'my-pipeline',
    'dataset_id': 'dataset_1',
    'target_column': 'target'
})
```

---

## Model Serialization

You can save and load trained AutoML models:

```python
from c60 import AutoML
# ... train your model ...
automl.save('best_model.joblib')
# Later, reload it:
loaded = AutoML.load('best_model.joblib')
```

---

## Cloud Execution

C60.ai supports simulated cloud/local pipeline execution:

```python
from interface.cloud_runner import CloudRunner, CloudProvider
runner = CloudRunner(provider=CloudProvider.LOCAL)
status = runner.run_pipeline(pipeline_config={"pipeline_id": "test"}, dataset_path="data.csv")
print(status)
```

---

### Basic Usage

```python
from c60 import AutoML
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load sample dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize AutoML
automl = AutoML(
    task='classification',
    time_budget=300,  # 5 minutes
    metric='accuracy',
    n_jobs=-1  # Use all available cores
)

# Fit the model
print("Training model...")
automl.fit(X_train, y_train)

# Make predictions
predictions = automl.predict(X_test)

# Evaluate performance
from sklearn.metrics import accuracy_score
print(f"Model accuracy: {accuracy_score(y_test, predictions):.4f}")

# Save the model
automl.save('best_model.joblib')
```

### Advanced Usage

Check out our comprehensive examples in the `examples/` directory:

1. [Classification Example](examples/classification.ipynb)
2. [Regression Example](examples/regression.ipynb)
3. [Implementation Walkthrough](examples/implementation_walkthrough.ipynb)

## Documentation

For comprehensive documentation, including API reference and advanced usage, please visit our [documentation site](coming soon).


### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/c60.ai.git
cd c60.ai

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black .
flake8
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or feedback, please open an issue or reach out to aditirkrishna@gmail.com

## Cite

If you use C60.ai in your research, please cite my paper (coming soon).

---

_built with ❤️ by aditi ramakrishnan | 2025_
