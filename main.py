from loanApprovalPrediction.components.data_ingestion import DataIngestion
from loanApprovalPrediction.components.data_processor import DataProcessor

# from loanApprovalPrediction.components.model_trainer import DataIngestion
from loanApprovalPrediction.config import configuration
from loanApprovalPrediction.constants import ROOT_DIR

if __name__ == "__main__":
    config = configuration.ConfigurationManager()
    ingestionConfig = config.get_data_ingestion_config()
    preprocessorConfig = config.get_data_processing_config()

    raw_data_path = f"{ROOT_DIR}/{ingestionConfig.root_dir}/{ingestionConfig.file_name}"

    data_ingestion = DataIngestion(ingestionConfig)
    data_ingestion.ingest_data()

    data_processor = DataProcessor(raw_data_path, preprocessorConfig)
    _, _, preprocessor, _, _ = data_processor.initiate_data_transformation()
    data_processor.save_artifacts(preprocessor=preprocessor)
