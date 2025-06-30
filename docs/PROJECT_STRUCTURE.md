# C60.ai Project Structure

## Core Directories
- `c60/` - Main package
  - `core/` - Core AutoML logic and base classes
  - `engine/` - Search and optimization engines
  - `gnn/` - Graph Neural Network components
  - `introspect/` - Explainability and visualization
  - `utils/` - Utility functions and helpers
  - `api/` - Web API components (scaffold)
  - `cli/` - Command-line interface (scaffold)
  - `cloud/` - Cloud integration (scaffold)

## Supporting Directories
- `docs/` - Documentation and Sphinx config
- `examples/` - Example usage and tutorials
- `notebooks/` - Jupyter notebooks
- `tests/` - Unit and integration tests
- `.github/` - GitHub workflows and templates

## Key Files
- `pyproject.toml` - Build system configuration
- `setup.py` - Package installation script
- `requirements.txt` - Production dependencies
- `requirements-dev.txt` - Development dependencies
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Service orchestration

## Documentation
- `README.md` - Main project documentation
- `README_MOLECULAR_AUTOML.md` - Technical deep dive
- `docs/PROJECT_STRUCTURE.md` - This file

## Development
- `.pre-commit-config.yaml` - Pre-commit hooks
- `.gitignore` - Git ignore rules
- `Makefile` - Common development commands
