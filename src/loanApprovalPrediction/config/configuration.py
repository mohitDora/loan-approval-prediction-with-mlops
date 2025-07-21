import os
from pathlib import Path

from dotenv import load_dotenv

from loanApprovalPrediction.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    ROOT_DIR,
)
from loanApprovalPrediction.entity import DataIngestionConfig
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
        create_directories([Path.joinpath(ROOT_DIR, config["root_dir"], "raw_data")])
        create_directories(
            [Path.joinpath(ROOT_DIR, config["root_dir"], "processed_data")]
        )

        data_ingestion_config = DataIngestionConfig(
            root_dir=config["root_dir"],
            source_url=source_url,
            mongo_db_name=config["mongo_db_name"],
            mongo_collection_name=config["mongo_collection_name"],
        )

        return data_ingestion_config


if __name__ == "__main__":
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    print(data_ingestion_config)
