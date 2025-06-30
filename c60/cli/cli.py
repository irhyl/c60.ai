"""
Command-line interface for C60.ai AutoML system.
"""
import argparse
from c60.core.automl import AutoML

def main():
    parser = argparse.ArgumentParser(description="C60.ai AutoML CLI")
    parser.add_argument('--input', type=str, required=True, help='Path to pipeline_dag.yaml')
    parser.add_argument('--data', type=str, help='Path to input data (optional)')
    parser.add_argument('--output', type=str, help='Path to save results (optional)')
    args = parser.parse_args()

    print(f"[C60] Loading pipeline: {args.input}")
    automl = AutoML()
    automl.run_from_yaml(args.input, data_path=args.data, output_path=args.output)
    print("[C60] Pipeline run complete.")

if __name__ == "__main__":
    main()
