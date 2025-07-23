import os
import tempfile
from pathlib import Path
import pytest
import pandas as pd
from unittest import mock
from sklearn.compose import ColumnTransformer

from loanApprovalPrediction.components.data_processor import DataProcessor
from loanApprovalPrediction.components.model_trainer import ModelTrainer
from loanApprovalPrediction.entity import DataProcessingConfig, ModelTrainerConfig
from loanApprovalPrediction.utils.common import read_json

# Sample best_params as in params.json
BEST_PARAMS = {
    "n_estimators": 10,  # Use small values for test speed
    "max_depth": 3,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": True
}

@pytest.fixture
def sample_raw_data(tmp_path: Path) -> str:
    data = read_json("tests/data/data.json")
    df = pd.DataFrame(data)
    raw_data_path = tmp_path / "raw_data.csv"
    df.to_csv(raw_data_path, index=False)
    return str(raw_data_path)

@pytest.fixture
def data_processing_config(tmp_path: Path) -> DataProcessingConfig:
    return DataProcessingConfig(
        root_dir=str(tmp_path),
        preprocessor_object_file_name="preprocessor.joblib"
    )

@pytest.fixture
def data_processor(sample_raw_data: str, data_processing_config: DataProcessingConfig) -> DataProcessor:
    return DataProcessor(raw_data_path=sample_raw_data, config=data_processing_config)

@pytest.fixture
def model_trainer_config(tmp_path: Path) -> ModelTrainerConfig:
    return ModelTrainerConfig(
        root_dir=tmp_path,
        experiment_name="test_experiment",
        registered_model_name="test_model",
        artifact_path="test_artifact",
        best_params=BEST_PARAMS,
        cv_folds=2,
        random_state=42,
        test_size=0.2,
        val_size=0.25,
        mlflow_tracking_uri="file:///tmp/mlruns"
    )

@mock.patch("mlflow.start_run")
@mock.patch("mlflow.set_experiment")
@mock.patch("mlflow.set_tracking_uri")
@mock.patch("mlflow.log_params")
@mock.patch("mlflow.log_metric")
@mock.patch("mlflow.log_dict")
@mock.patch("mlflow.sklearn.log_model")
def test_model_trainer_train_model(
    mock_log_model,
    mock_log_dict,
    mock_log_metric,
    mock_log_params,
    mock_set_tracking_uri,
    mock_set_experiment,
    mock_start_run,
    data_processor,
    model_trainer_config
):
    # Mock mlflow.start_run to be a context manager
    class DummyRun:
        info = type("info", (), {"run_id": "dummy_run_id"})()
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    mock_start_run.return_value = DummyRun()

    trainer = ModelTrainer(config=model_trainer_config, preprocessor=data_processor)
    # Should not raise
    trainer.train_model()
    # Check that mlflow logging was called
    assert mock_log_params.called
    assert mock_log_metric.called
    assert mock_log_dict.called
    assert mock_log_model.called 