from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    mongo_db_name: str
    mongo_collection_name: str
    file_name: str


@dataclass(frozen=True)
class DataProcessingConfig:
    root_dir: Path
    preprocessor_object_file_name: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    experiment_name: str
    registered_model_name: str
    artifact_path: str
    best_params: dict
    cv_folds: int
    random_state: int
    test_size: float
    val_size: float
    mlflow_tracking_uri: str
