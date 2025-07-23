import os
from pathlib import Path

from dotenv import load_dotenv

from loanApprovalPrediction.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    ROOT_DIR,
)
from loanApprovalPrediction.entity import (
    DataIngestionConfig,
    DataProcessingConfig,
    ModelTrainerConfig,
)
from loanApprovalPrediction.utils.common import create_directories, read_json

load_dotenv()


class ConfigurationManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_json(Path.joinpath(ROOT_DIR, config_filepath))
        self.params = read_json(Path.joinpath(ROOT_DIR, params_filepath))
        create_directories([Path.joinpath(ROOT_DIR, self.config["artifacts_root"])])

    def get_data_ingestion_config(self):
        config = self.config["data_ingestion"]

        source_url = config["source_url"].replace(
            "<password>", os.getenv("MONGO_DB_PASSWORD")
        )

        create_directories([Path.joinpath(ROOT_DIR, config["root_dir"])])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config["root_dir"],
            source_url=source_url,
            mongo_db_name=config["mongo_db_name"],
            mongo_collection_name=config["mongo_collection_name"],
            file_name=config["file_name"],
        )

        return data_ingestion_config

    def get_data_processing_config(self):
        config = self.config["data_processing"]
        create_directories([Path.joinpath(ROOT_DIR, config["root_dir"])])

        data_processing_config = DataProcessingConfig(
            root_dir=config["root_dir"],
            preprocessor_object_file_name=config["preprocessor_object_file_name"],
        )

        return data_processing_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config["model_training"]
        return ModelTrainerConfig(
            root_dir=config["root_dir"],
            experiment_name=config["experiment_name"],
            registered_model_name=config["registered_model_name"],
            artifact_path=config["artifact_path"],
            best_params=self.params,
            cv_folds=config["cv_folds"],
            random_state=config["random_state"],
            test_size=config["test_size"],
            val_size=config["val_size"],
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        )


if __name__ == "__main__":
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
