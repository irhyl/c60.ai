"""
Command-line interface for C60 AutoML system.

This module provides a command-line interface for users to interact with
the C60 AutoML system, including dataset loading, pipeline generation,
and model training.
"""

"""
Command-line interface for C60 AutoML system.

This module provides a command-line interface for users to interact with
C60 AutoML, including dataset loading, pipeline generation, and model training.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def main(args: Optional[list[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        args: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        int: Exit code (0 for success, non-zero for error).
    """
    parser = argparse.ArgumentParser(
        description="C60 AutoML - Automated Machine Learning Pipeline Generation"
    )
    
    # Global arguments
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level"
    )

    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Dataset command
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")
    
    # Load dataset command
    load_parser = dataset_subparsers.add_parser("load", help="Load a dataset")
    load_parser.add_argument("filepath", help="Path to dataset file")
    load_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate dataset after loading"
    )
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Pipeline operations")
    pipeline_subparsers = pipeline_parser.add_subparsers(dest="pipeline_command")
    
    # Generate pipeline command
    generate_parser = pipeline_subparsers.add_parser(
        "generate",
        help="Generate a new pipeline"
    )
    generate_parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset file or name of loaded dataset"
    )
    generate_parser.add_argument(
        "--target",
        required=True,
        help="Name of the target column"
    )
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Handle commands
    if not parsed_args.command:
        parser.print_help()
        return 0
    
    if parsed_args.command == "dataset" and parsed_args.dataset_command == "load":
        return handle_load_dataset(parsed_args)
    elif parsed_args.command == "pipeline" and parsed_args.pipeline_command == "generate":
        return handle_generate_pipeline(parsed_args)
    
    print(f"Unknown command: {parsed_args.command}", file=sys.stderr)
    return 1


def handle_load_dataset(args) -> int:
    """
    Handle the 'dataset load' CLI command.
    Loads a dataset from a CSV file and optionally validates for missing values.

    Args:
        args: Parsed CLI arguments with 'filepath' and 'validate' attributes.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    try:
        import pandas as pd
        print(f"Loading dataset from {args.filepath}")
        df = pd.read_csv(args.filepath)
        print(f"Loaded dataset with shape: {df.shape}")
        if args.validate:
            print("Validating dataset for missing values...")
            missing = df.isnull().sum().sum()
            if missing > 0:
                print(f"Warning: Dataset contains {missing} missing values.")
            else:
                print("No missing values detected.")
        return 0
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return 1


def handle_generate_pipeline(args) -> int:
    """
    Handle the 'pipeline generate' CLI command.
    Loads a dataset, trains an AutoML pipeline, and saves the trained model.

    Args:
        args: Parsed CLI arguments with 'dataset' and 'target' attributes.

    Returns:
        int: Exit code (0 for success, 1 for error).
    """
    try:
        import pandas as pd
        from c60 import AutoML
        print(f"Generating pipeline for dataset: {args.dataset}")
        print(f"Target column: {args.target}")
        df = pd.read_csv(args.dataset)
        X = df.drop(columns=[args.target])
        y = df[args.target]
        automl = AutoML(task='classification')  # TODO: infer task from data or args
        automl.fit(X, y)
        print(f"Pipeline generated and trained. Best score: {automl.best_score_}")
        # Save the best pipeline
        automl.save('best_automl_pipeline.joblib')
        print("Saved best pipeline to 'best_automl_pipeline.joblib'")
        return 0
    except Exception as e:
        print(f"Error generating pipeline: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
