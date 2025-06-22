import pandas as pd
import numpy as np
from typing import Tuple

def load_csv(filepath: str) -> Tuple[pd.DataFrame, str]:
    """
    Load a CSV file and infer the machine learning task type.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        Tuple[pd.DataFrame, str]: The loaded DataFrame and inferred task type ('classification' or 'regression').

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or invalid.
    """
    # Load the CSV file
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError("Dataset file is empty")

    # Validate the dataset
    validate_dataset(df)

    # Infer the task type based on the target column (assumed to be the last column)
    target_col = df.columns[-1]
    if df[target_col].dtype in [np.object_, np.str_, np.bool_] or df[target_col].nunique() < 20:
        task = 'classification'
    else:
        task = 'regression'

    return df, task

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the integrity of the dataset.

    Args:
        df (pd.DataFrame): The dataset to validate.

    Raises:
        ValueError: If the dataset contains missing values, has no columns, or has no rows.
    """
    if df.empty:
        raise ValueError("Dataset is empty")
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")
    if len(df.columns) < 2:
        raise ValueError("Dataset must have at least one feature and one target column")