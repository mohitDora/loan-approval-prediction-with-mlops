from loanApprovalPrediction.components.data_ingestion import DataIngestion
from loanApprovalPrediction.components.data_processor import DataProcessor
from loanApprovalPrediction.components.model_trainer import ModelTrainer
from loanApprovalPrediction.config import configuration
from loanApprovalPrediction.constants import ROOT_DIR
from loanApprovalPrediction.logger import logger


def run_pipeline():
    config = configuration.ConfigurationManager()
    ingestionConfig = config.get_data_ingestion_config()
    preprocessorConfig = config.get_data_processing_config()
    model_trainer_config = config.get_model_trainer_config()

    raw_data_path = f"{ROOT_DIR}/{ingestionConfig.root_dir}/{ingestionConfig.file_name}"

    logger.info("=" * 10 + "Data Ingestion Started" + "=" * 10)
    print("\n" * 2)
    data_ingestion = DataIngestion(ingestionConfig)
    data_ingestion.ingest_data()
    print("\n" * 2)
    logger.info("=" * 10 + "Data Ingestion Ended" + "=" * 10)
    print("\n" * 2)
    logger.info("=" * 10 + "Data Preprocessing Started" + "=" * 10)
    print("\n" * 2)
    data_processor = DataProcessor(raw_data_path, preprocessorConfig)
    _, _, preprocessor, _, _ = data_processor.initiate_data_transformation()
    data_processor.save_artifacts(preprocessor=preprocessor)
    print("\n" * 2)
    logger.info("=" * 10 + "Data Processing Ended" + "=" * 10)
    print("\n" * 2)
    logger.info("=" * 10 + "Model Training Started" + "=" * 10)
    print("\n" * 2)
    model_trainer = ModelTrainer(
        config=model_trainer_config, preprocessor=data_processor
    )
    model_trainer.train_model()
    print("\n" * 2)
    logger.info("=" * 10 + "Model Training Ended" + "=" * 10)
