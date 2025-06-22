# C60.ai - Generative Evolutionary AutoML Framework

C60.ai is an advanced AutoML framework that leverages generative and evolutionary algorithms to automate the machine learning pipeline development process.

## Features

- **Automated Pipeline Generation**: Generate end-to-end ML pipelines using evolutionary algorithms
- **Hyperparameter Optimization**: Advanced optimization techniques for model tuning
- **Modular Architecture**: Easily extensible with custom components
- **Distributed Computing**: Support for distributed training and optimization
- **Comprehensive Tracking**: Built-in experiment tracking and model versioning

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/c60.ai.git
   cd c60.ai
   ```

2. Create and activate a virtual environment:
   ```bash
   # Using venv
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

```python
from c60 import AutoML

# Initialize AutoML
automl = AutoML(
    task='classification',
    time_budget=3600,  # 1 hour
    metric='accuracy',
    n_jobs=-1
)

# Load your dataset
import pandas as pd
X, y = pd.read_csv('data.csv'), pd.read_csv('target.csv')

# Run AutoML
pipeline = automl.fit(X, y)

# Make predictions
y_pred = pipeline.predict(X_test)
```

## Project Structure

```
c60.ai/
├── c60/                    # Source code
│   ├── core/               # Core framework components
│   ├── engine/             # Core engine implementation
│   ├── interface/          # User interfaces (CLI, API)
│   ├── optimizers/         # Optimization algorithms
│   └── utils/              # Utility functions
├── tests/                  # Test suite
├── examples/               # Example notebooks
├── docs/                   # Documentation
├── config/                 # Configuration files
└── scripts/                # Utility scripts
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Cite

If you use C60.ai in your research, please cite our paper (coming soon).

## Contact

For questions or feedback, please open an issue or contact us at [email protected]
