# Contributing to C60.ai

Thank you for your interest in contributing to C60.ai! We welcome contributions from the community to help improve this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/your-username/c60.ai.git
   cd c60.ai
   ```
3. **Set up** the development environment (see below).

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- [Poetry](https://python-poetry.org/) (recommended) or pip
- Git

### Using Poetry (Recommended)

1. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies:
   ```bash
   poetry install --with dev,docs
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

### Using pip

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix or MacOS:
   source .venv/bin/activate
   ```

2. Install the package in development mode with all dependencies:
   ```bash
   pip install -e ".[dev,docs]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Making Changes

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-short-description
   ```

2. Make your changes following the [code style](#code-style) guidelines.

3. Add tests for your changes.

4. Run the tests and ensure they pass:
   ```bash
   make test
   ```

5. Commit your changes with a descriptive commit message:
   ```bash
   git commit -m "Add feature: brief description of changes"
   ```

6. Push your branch to your fork:
   ```bash
   git push origin your-branch-name
   ```

## Testing

Run the test suite:

```bash
# Run all tests with coverage
make test

# Run tests quickly without coverage
make test-fast

# Run a specific test file
make test-file tests/test_module.py

# Run tests matching a pattern
pytest -k "test_pattern"
```

## Code Style

We use several tools to maintain code quality and style:

- **Black** for code formatting
- **isort** for import sorting
- **Flake8** for linting
- **Mypy** for static type checking

To automatically format and check your code:

```bash
# Format code with Black and isort
make format

# Check code style
make lint
```

Pre-commit hooks will automatically run these checks when you commit.

## Documentation

We use Sphinx for documentation. To build the documentation locally:

```bash
# Build the docs
make docs

# Serve the docs locally
make serve-docs
```

Documentation should be updated when adding new features or changing existing behavior.

## Submitting a Pull Request

1. Ensure your fork is up to date with the main repository:
   ```bash
   git remote add upstream https://github.com/aditirkrishna/c60.ai.git
   git fetch upstream
   git checkout main
   git merge upstream/main
   git push origin main
   ```

2. Push your changes to your fork:
   ```bash
   git push origin your-branch-name
   ```

3. Open a Pull Request from your fork to the main repository's `main` branch.

4. Fill out the PR template with details about your changes.

5. Ensure all CI checks pass and address any review comments.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with the following information:

- A clear title and description
- Steps to reproduce the issue (if applicable)
- Expected vs. actual behavior
- Environment information (Python version, OS, etc.)
- Any relevant logs or error messages

## License

By contributing to C60.ai, you agree that your contributions will be licensed under the [MIT License](LICENSE).
