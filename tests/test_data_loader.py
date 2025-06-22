import pytest
import pandas as pd
import numpy as np
from engine.data_loader import load_csv, validate_dataset
import os

# Path to the test dataset
TEST_DATA_PATH = "datasets/test_iris.csv"

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.0, 6.0],
        "target": ["A", "B", "A"]
    })

@pytest.fixture
def regression_df():
    """Create a sample DataFrame for regression testing."""
    return pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.0, 6.0],
        "target": [10.0, 20.0, 30.0]
    })

def test_load_csv_valid_file():
    """Test loading a valid CSV file."""
    df, task = load_csv(TEST_DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert task == "classification"
    assert df.shape == (150, 5)  # Iris dataset: 150 rows, 5 columns
    assert "species" in df.columns

def test_load_csv_nonexistent_file():
    """Test loading a nonexistent file."""
    with pytest.raises(FileNotFoundError):
        load_csv("nonexistent.csv")

def test_load_csv_empty_file(tmp_path):
    """Test loading an empty CSV file."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="Dataset file is empty"):
        load_csv(str(empty_file))

def test_validate_dataset_valid(sample_df):
    """Test validating a valid dataset."""
    validate_dataset(sample_df)  # Should not raise any exception

def test_validate_dataset_empty():
    """Test validating an empty dataset."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_dataset(empty_df)

def test_validate_dataset_missing_values(sample_df):
    """Test validating a dataset with missing values."""
    sample_df.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Dataset contains missing values"):
        validate_dataset(sample_df)

def test_validate_dataset_insufficient_columns():
    """Test validating a dataset with too few columns."""
    single_col_df = pd.DataFrame({"feature": [1, 2, 3]})
    with pytest.raises(ValueError, match="Dataset must have at least one feature and one target column"):
        validate_dataset(single_col_df)

def test_load_csv_regression(regression_df, tmp_path):
    """Test inferring regression task type."""
    temp_file = tmp_path / "regression.csv"
    regression_df.to_csv(temp_file, index=False)
    df, task = load_csv(str(temp_file))
    assert task == "regression"