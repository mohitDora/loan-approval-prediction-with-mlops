from loanApprovalPrediction.components.data_ingestion import DataIngestion
from loanApprovalPrediction.config import configuration 


if __name__ == "__main__":
    config = configuration.ConfigurationManager()

    data_ingestion = DataIngestion(config.get_data_ingestion_config())
    data_ingestion.ingest_data()