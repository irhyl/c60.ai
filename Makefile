# Makefile for C60.ai development

# Variables
PYTHON = python
PIP = pip
PYTEST = pytest
COVERAGE = coverage
BLACK = black
ISORT = isort
FLAKE8 = flake8
MYPY = mypy

# Directories
SRC_DIR = c60
TESTS_DIR = tests
DOCS_DIR = docs
EXAMPLES_DIR = examples

# Default target
.DEFAULT_GOAL := help

# Help target to show all available commands
help:
	@echo "C60.ai Development Commands:"
	@echo "  setup           - Set up the development environment"
	@echo "  install         - Install the package in development mode"
	@echo "  test            - Run tests with coverage"
	@echo "  test-fast       - Run tests quickly without coverage"
	@echo "  lint            - Check code style with black, isort, flake8, and mypy"
	@echo "  format          - Format code with black and isort"
	@echo "  clean           - Remove build artifacts and cache"
	@echo "  docs            - Build documentation"
	@echo "  check           - Run all checks (lint, test, docs)"

# Set up the development environment
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e ".[dev]"
	pre-commit install

# Install the package in development mode
install:
	$(PIP) install -e .


# Run tests with coverage
test:
	$(PYTEST) --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=xml -v $(TESTS_DIR)

# Run tests quickly without coverage
test-fast:
	$(PYTEST) -v $(TESTS_DIR)

# Run a specific test file
test-%:
	$(PYTEST) -v $(TESTS_DIR)/$*.py -k $(TEST)

# Check code style with black, isort, flake8, and mypy
lint: lint-black lint-isort lint-flake8 lint-mypy

# Format code with black and isort
format: format-black format-isort

# Clean build artifacts and cache
clean:
	rm -rf build/ dist/ .eggs/ .pytest_cache/ .mypy_cache/ .coverage htmlcov/ *.egg-info .ipynb_checkpoints
	find . -type d -name '__pycache__' -exec rm -rf {} \;
	find . -type f -name '*.py[co]' -delete

# Documentation
docs:
	$(MAKE) -C $(DOCS_DIR) clean
	$(MAKE) -C $(DOCS_DIR) html

# Serve documentation
serve-docs: docs
	python -m http.server 8000 --directory $(DOCS_DIR)/_build/html

# Check everything
check: lint test docs

# Black
lint-black:
	$(BLACK) --check $(SRC_DIR) $(TESTS_DIR) $(EXAMPLES_DIR)

format-black:
	$(BLACK) $(SRC_DIR) $(TESTS_DIR) $(EXAMPLES_DIR)

# isort
lint-isort:
	$(ISORT) --check-only $(SRC_DIR) $(TESTS_DIR) $(EXAMPLES_DIR)

format-isort:
	$(ISORT) $(SRC_DIR) $(TESTS_DIR) $(EXAMPLES_DIR)

# flake8
lint-flake8:
	$(FLAKE8) $(SRC_DIR) $(TESTS_DIR) $(EXAMPLES_DIR)

# mypy
lint-mypy:
	$(MYPY) $(SRC_DIR) $(TESTS_DIR) $(EXAMPLES_DIR)

# Build distribution
build:
	$(PYTHON) -m build

# Upload to PyPI (requires twine)
publish: build
	twine upload dist/*

# Run Jupyter lab
jupyter:
	jupyter lab

.PHONY: help setup install test test-fast lint format clean docs serve-docs check lint-black format-black lint-isort format-isort lint-flake8 lint-mypy build publish jupyter
