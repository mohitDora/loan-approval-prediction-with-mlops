import subprocess
from pathlib import Path

import joblib
import mlflow

from loanApprovalPrediction.config import configuration
from loanApprovalPrediction.constants import ROOT_DIR
from loanApprovalPrediction.logger import logger


def load_model_artifacts():
    """
    Loads the preprocessor and the trained model from joblib files.
    Ensures DVC-tracked artifacts are pulled if not present.
    """
    logger.info("Attempting to load model artifacts...")
    try:
        # config = configuration.ConfigurationManager()
        # training_config = config.get_model_trainer_config()

        # model_name = training_config.registered_model_name
        # model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Staging")

        logged_model = "runs:/14c21c85b30841a89ceb420a91e8c216/loan_approval_model"

        model = mlflow.pyfunc.load_model(logged_model)

        logger.info("Model artifacts loaded successfully.")
        return model

    except FileNotFoundError as e:
        logger.error(
            f"Required model artifact not found even after DVC pull attempt: {e}"
        )
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during model loading: {e}")
        raise e


if __name__ == "__main__":
    load_model_artifacts()
