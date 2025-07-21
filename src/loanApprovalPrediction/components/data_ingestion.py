import sys

import pandas as pd
from pymongo import MongoClient

from loanApprovalPrediction.config import configuration
from loanApprovalPrediction.constants import ROOT_DIR
from loanApprovalPrediction.entity import DataIngestionConfig
from loanApprovalPrediction.logger import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.client = None
        self.db = None
        self.collection = None

    def _connect_to_mongodb(self):
        """Establishes a connection to MongoDB."""
        try:
            self.client = MongoClient(self.config.source_url)
            self.db = self.client[self.config.mongo_db_name]
            self.collection = self.db[self.config.mongo_collection_name]
            logger.info("Successfully connected to MongoDB.")
        except Exception as e:
            logger.error(
                f"Failed to connect to MongoDB at {self.config.source_url}: {e}",
                exc_info=True,
            )

    def _close_mongodb_connection(self):
        """Closes the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")

    def ingest_data(self):
        """Ingests data from MongoDB and saves it to a raw CSV file."""
        logger.info(
            f"Attempting to ingest data from MongoDB collection: {self.config.mongo_collection_name}"
        )
        try:
            self._connect_to_mongodb()

            # Fetch all documents from the collection
            cursor = self.collection.find({})
            data_list = list(cursor)

            if not data_list:
                logger.warning("No data found in MongoDB collection.")
                raise Exception("No data found in MongoDB collection.", sys)

            # Convert list of dictionaries to pandas DataFrame
            df = pd.DataFrame(data_list)

            # Remove the '_id' column added by MongoDB if it exists
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])

            # Save the raw data to CSV
            path = (
                f"{ROOT_DIR}/{self.config.root_dir}/raw_data/loanApprovalPrediction.csv"
            )
            print(path)
            df.to_csv(
                path,
                index=False,
            )
            logger.info(
                f"Successfully ingested {df.shape[0]} records and saved to {path}"
            )

            return df

        except Exception as e:
            logger.error(f"Error during data ingestion: {e}", exc_info=True)
        finally:
            self._close_mongodb_connection()


if __name__ == "__main__":
    try:
        config = configuration.ConfigurationManager()
        data_ingestor = DataIngestion(config.get_data_ingestion_config())
        raw_df = data_ingestor.ingest_data()
        logger.info(f"Raw data loaded from MongoDB, shape: {raw_df.shape}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during data ingestion: {e}", exc_info=True
        )
