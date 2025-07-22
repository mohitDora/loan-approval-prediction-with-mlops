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
        print(self.config)
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

        return ModelTrainerConfig(
            root_dir=Path("artifacts/models"),
            experiment_name="Loan_Approval_RandomForest_Direct_Train",  # Updated experiment name
            registered_model_name="LoanApprovalRandomForestModel",
            artifact_path="loan_approval_model",
            best_params=self.params,  # Pass the best parameters directly
            cv_folds=3,  # Not directly used for single model train, but kept in config
            random_state=42,
            test_size=0.2,
            val_size=0.25,
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        )


if __name__ == "__main__":
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
