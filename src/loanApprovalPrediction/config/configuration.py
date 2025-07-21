import os
from loanApprovalPrediction.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from loanApprovalPrediction.utils.common import create_directories, read_json
from loanApprovalPrediction.entity import DataIngestionConfig
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

project_root = Path(__file__).resolve().parent.parent.parent.parent


class ConfigurationManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):

        self.config = read_json(Path.joinpath(project_root, config_filepath))
        self.params = read_json(Path.joinpath(project_root, params_filepath))
        print(self.config)
        create_directories([Path.joinpath(project_root, self.config["artifacts_root"])])

    def get_data_ingestion_config(self):
        config = self.config["data_ingestion"]

        source_url = config["source_url"].replace(
            "<password>", os.getenv("MONGO_DB_PASSWORD")
        )

        create_directories(
            [Path.joinpath(project_root, config["root_dir"])] + [config["root_dir"]]
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
