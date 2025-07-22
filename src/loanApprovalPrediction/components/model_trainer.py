import os
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from loanApprovalPrediction.entity import ModelTrainerConfig
from loanApprovalPrediction.logger import logger

load_dotenv()


class ModelTrainer:
    """
    A class to handle model training, evaluation, and MLflow logging
    using predefined best parameters for the Loan Approval Prediction project.
    """

    def __init__(
        self,
        config: ModelTrainerConfig,
        preprocessor: ColumnTransformer = None,
    ):
        self.config = config
        self.data_processor = preprocessor
        logger.info("Initialized ModelTrainer.")

    def _set_mlflow_environment(self):
        """Sets the MLflow tracking URI and experiment name."""
        try:
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            mlflow.set_experiment(self.config.experiment_name)
            logger.info(
                f"MLflow tracking URI set to: {os.getenv('MLFLOW_TRACKING_URI')}"
            )
            logger.info(f"MLflow experiment set to: {self.config.experiment_name}")
        except Exception as e:
            error_msg = f"Error setting MLflow environment: {e}"
            logger.error(error_msg)
            raise e

    def train_model(self):
        """
        Orchestrates the entire model training and logging process.
        This includes data transformation, splitting, direct model training
        with best parameters, evaluation, and MLflow logging.
        """
        self._set_mlflow_environment()

        try:
            # 1. Data Transformation
            logger.info("Initiating data transformation...")
            X_processed_np, y_series, preprocessor, feature_names, original_X_df = (
                self.data_processor.initiate_data_transformation()
            )
            logger.info("Data transformation completed.")

            # 2. Data Splitting
            logger.info("Splitting data into train, validation, and test sets...")
            (
                X_train_processed,
                X_val_processed,
                X_test_processed,
                y_train,
                y_val,
                y_test,
            ) = self.data_processor.split_data(
                X_processed_np,
                y_series,
                test_size=self.config.test_size,
                val_size=self.config.val_size,
                random_state=self.config.random_state,
            )
            logger.info("Data splitting completed.")

            # 3. Define Classifier using best_params
            classifier = RandomForestClassifier(
                **self.config.best_params, random_state=self.config.random_state
            )
            logger.info(
                f"Initialized RandomForestClassifier with best parameters: {self.config.best_params}"
            )

            # 4. Perform Training and Logging within an MLflow run
            self._perform_training_and_logging(
                model_name="Random_Forest_Best_Model_Training",  # Updated run name
                classifier=classifier,
                y_train=y_train,
                y_test=y_test,
                preprocessor=preprocessor,
                original_X_df=original_X_df,  # Pass original X for fetching train/test splits
            )
            logger.info("Model training and logging process completed successfully.")

        except Exception as e:
            logger.error(
                f"An unexpected error occurred in ModelTrainer.train_model: {e}",
                exc_info=True,
            )
            raise e

    def _perform_training_and_logging(
        self,
        model_name: str,
        classifier: RandomForestClassifier,
        y_train: pd.Series,
        y_test: pd.Series,
        preprocessor: ColumnTransformer,
        original_X_df: pd.DataFrame,  # This is the full original X (raw)
    ):
        """
        Performs the actual training, evaluation, and MLflow logging
        for a single model using predefined best parameters.
        """
        with mlflow.start_run(run_name=model_name) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}")

            # Log the best parameters that are being used for this run
            mlflow.log_params(self.config.best_params)
            logger.info(
                f"Logged best parameters used for training: {self.config.best_params}"
            )

            full_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),  # The fitted preprocessor object
                    ("classifier", classifier),
                ]
            )
            logger.info(
                "Created full scikit-learn pipeline (preprocessor + classifier)."
            )

            # Get the original (raw) training data corresponding to y_train's index
            X_train_original_format = original_X_df.loc[y_train.index]
            logger.info(
                f"Fitting pipeline on training data (shape: {X_train_original_format.shape})..."
            )
            full_pipeline.fit(X_train_original_format, y_train)
            logger.info("Pipeline fitting completed.")

            # --- Evaluate on Test Set ---
            X_test_original_format = original_X_df.loc[y_test.index]

            y_test_pred = full_pipeline.predict(X_test_original_format)
            y_test_proba = full_pipeline.predict_proba(X_test_original_format)[:, 1]

            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_roc_auc = roc_auc_score(y_test, y_test_proba)

            logger.info(f"--- Test Metrics for {model_name} ---")
            logger.info(f"Accuracy: {test_accuracy:.4f}")
            logger.info(f"Precision: {test_precision:.4f}")
            logger.info(f"Recall: {test_recall:.4f}")
            logger.info(f"F1-Score: {test_f1:.4f}")
            logger.info(f"ROC AUC: {test_roc_auc:.4f}")

            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_f1_score", test_f1)
            mlflow.log_metric("test_roc_auc", test_roc_auc)

            # Log classification report
            report = classification_report(y_test, y_test_pred, output_dict=True)
            mlflow.log_dict(report, "classification_report.json")
            logger.info("Logged test metrics and classification report to MLflow.")

            try:
                tracking_uri_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                if tracking_uri_type_store != "file":
                    mlflow.sklearn.log_model(
                        sk_model=full_pipeline,
                        artifact_path=self.config.artifact_path,
                        registered_model_name=self.config.registered_model_name,
                    )
                else:
                    mlflow.sklearn.log_model(
                        sk_model=full_pipeline,
                        artifact_path=self.config.artifact_path,
                    )

                logger.info(
                    f"Model logged to MLflow and registered as '{self.config.registered_model_name}' (Run ID: {run_id})."
                )

                return run_id
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred in ModelTrainer._perform_training_and_logging: {e}",
                )
                raise e
