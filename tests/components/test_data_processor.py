import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from loanApprovalPrediction.components.data_processor import DataProcessor
from loanApprovalPrediction.entity import DataProcessingConfig
from loanApprovalPrediction.utils.common import read_json


# Fixture for creating a sample raw data file for testing
@pytest.fixture
def sample_raw_data(tmp_path: Path) -> str:
    """Creates a dummy raw data CSV and returns its path."""
    data = read_json("tests/data/data.json")

    df = pd.DataFrame(data)
    raw_data_path = tmp_path / "raw_data.csv"
    df.to_csv(raw_data_path, index=False)
    return str(raw_data_path)


# Fixture for providing a DataProcessingConfig instance for testing
@pytest.fixture
def data_processing_config(tmp_path: Path) -> DataProcessingConfig:
    """Provides a temporary DataProcessingConfig for testing."""
    return DataProcessingConfig(
        root_dir=str(tmp_path),
        preprocessor_object_file_name="preprocessor.joblib",
    )


# Fixture for providing a DataProcessor instance for testing
@pytest.fixture
def data_processor(
    sample_raw_data: str, data_processing_config: DataProcessingConfig
) -> DataProcessor:
    """Provides a DataProcessor instance configured for testing."""
    return DataProcessor(raw_data_path=sample_raw_data, config=data_processing_config)


# Test for the `load_data` method
def test_load_data(data_processor: DataProcessor):
    """Tests if data is loaded correctly."""
    df = data_processor.load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Loan_ID" in df.columns


# Test for `_identify_features` method
def test_identify_features(data_processor: DataProcessor):
    """Tests if numerical and categorical features are identified correctly."""
    df = data_processor.load_data()
    X = df.drop("Loan_Status", axis=1)
    numerical_features, categorical_features = data_processor._identify_features(X)

    # Based on the sample data
    expected_numerical = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]
    expected_categorical = [
        "Loan_ID",
        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "Property_Area",
        "Dependents",
    ]

    assert sorted(numerical_features) == sorted(expected_numerical)
    assert sorted(categorical_features) == sorted(expected_categorical)


# # Test for `get_preprocessing_pipeline` method
def test_get_preprocessing_pipeline(data_processor: DataProcessor):
    """Tests if the preprocessing pipeline is created correctly."""
    df = data_processor.load_data()
    X = df.drop("Loan_Status", axis=1)
    numerical, categorical = data_processor._identify_features(X)
    pipeline = data_processor.get_preprocessing_pipeline(numerical, categorical)

    assert isinstance(pipeline, ColumnTransformer)

    # Check transformer names from the initialized 'transformers' list
    transformer_names = [name for name, _, _ in pipeline.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


# # Test for `initiate_data_transformation` method
def test_initiate_data_transformation(data_processor: DataProcessor):
    """Tests the entire data transformation process."""
    (
        X_processed,
        y,
        preprocessor,
        feature_names,
        original_X_df,
    ) = data_processor.initiate_data_transformation()

    assert isinstance(X_processed, np.ndarray)
    assert isinstance(y, pd.Series)
    assert isinstance(preprocessor, ColumnTransformer)
    assert isinstance(feature_names, list)
    assert isinstance(original_X_df, pd.DataFrame)
    assert X_processed.shape[0] == y.shape[0]  # Same number of samples
    assert not np.isnan(X_processed).any()  # Imputation should handle NaNs


# # Test for `split_data` method
def test_split_data(data_processor: DataProcessor):
    """Tests if the data is split correctly into train, validation, and test sets."""
    (
        X_processed,
        y,
        _,
        _,
        _,
    ) = data_processor.initiate_data_transformation()
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
        X_processed, y
    )

    assert X_train.shape[0] > 0
    assert X_val.shape[0] >= 0  # Can be zero if the dataset is tiny
    assert X_test.shape[0] > 0
    assert y_train.shape[0] == X_train.shape[0]
    assert y_val.shape[0] == X_val.shape[0]
    assert y_test.shape[0] == X_test.shape[0]


# # Test for `save_artifacts` method
def test_save_artifacts(data_processor: DataProcessor):
    """Tests if the preprocessor artifact is saved correctly."""
    _, _, preprocessor, _, _ = data_processor.initiate_data_transformation()
    data_processor.save_artifacts(preprocessor)

    expected_path = data_processor.preprocessor_save_path
    assert os.path.exists(expected_path)

    loaded_preprocessor = joblib.load(expected_path)
    assert isinstance(loaded_preprocessor, ColumnTransformer)
    os.remove(expected_path)  # Clean up the created artifact
