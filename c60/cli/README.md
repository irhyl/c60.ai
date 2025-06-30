# C60 CLI

This package provides a command-line interface for running C60.ai pipelines.

## Usage

```bash
python -m c60.cli.cli --input examples/01_fraud_detection_bank/pipeline_dag.yaml
```

Arguments:
- `--input`: Path to the pipeline YAML file (required)
- `--data`: Path to input data (optional)
- `--output`: Path to save results (optional)
