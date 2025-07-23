from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from loanApprovalPrediction.constants import ROOT_DIR
from loanApprovalPrediction.entity import DataProcessingConfig
from loanApprovalPrediction.logger import logger


class DataProcessor:
    """
    A class to handle data loading, preprocessing, and splitting for the
    Loan Approval Prediction project.
    """

    def __init__(self, raw_data_path: str, config: DataProcessingConfig):
        """
        Initializes the DataProcessor with paths for raw data and preprocessor output.
        """
        self.config = config
        self.raw_data_path = raw_data_path
        self.preprocessor_save_path = Path.joinpath(
            ROOT_DIR, self.config.root_dir, self.config.preprocessor_object_file_name
        )

    def load_data(self) -> pd.DataFrame:
        """
        Loads the raw loan approval data from the specified filepath.

        :return: A pandas DataFrame containing the raw data.
        :raises MyException: If the data file is not found or other loading errors occur.
        """
        try:
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Data loaded successfully from {self.raw_data_path}.")
            return df
        except FileNotFoundError:
            error_msg = (
                f"Data file not found at {self.raw_data_path}. Please ensure it exists."
            )
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Error loading data from {self.raw_data_path}: {e}"
            logger.error(error_msg)
            raise e

    def _identify_features(self, X: pd.DataFrame) -> tuple[list, list]:
        """
        Identifies numerical and categorical features from the DataFrame.

        :param X: Features DataFrame.
        :return: A tuple containing lists of numerical and categorical feature names.
        """
        numerical_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        categorical_features = X.select_dtypes(include="object").columns.tolist()

        if "Dependents" in numerical_features:
            numerical_features.remove("Dependents")
        if "Dependents" not in categorical_features:
            categorical_features.append("Dependents")

        logger.info(f"Identified numerical features: {numerical_features}")
        logger.info(f"Identified categorical features: {categorical_features}")
        return numerical_features, categorical_features

    def get_preprocessing_pipeline(
        self, numerical_features: list, categorical_features: list
    ) -> ColumnTransformer:
        """
        Creates and returns the ColumnTransformer for preprocessing.

        :param numerical_features: List of numerical feature names.
        :param categorical_features: List of categorical feature names.
        :return: A fitted ColumnTransformer.
        """
        try:
            numerical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_transformer, numerical_features),
                    ("cat", categorical_transformer, categorical_features),
                ],
                remainder="passthrough",
            )
            logger.info("Preprocessing pipeline (ColumnTransformer) created.")
            return preprocessor
        except Exception as e:
            error_msg = f"Error creating preprocessing pipeline: {e}"
            logger.error(error_msg)
            raise e

    def initiate_data_transformation(
        self,
    ) -> tuple[np.ndarray, pd.Series, ColumnTransformer, list, pd.DataFrame]:
        """
        Orchestrates the entire data transformation process:
        loads, preprocesses, and prepares data for model training.

        :return: A tuple containing:
                 - X_processed (np.ndarray): Processed features.
                 - y (pd.Series): Target variable.
                 - preprocessor (ColumnTransformer): Fitted preprocessor.
                 - feature_names (list): List of feature names after preprocessing.
                 - original_X_df (pd.DataFrame): Original features DataFrame (before preprocessing).
        :raises MyException: If any error occurs during data transformation.
        """
        try:
            df = self.load_data()
            if df is None:
                logger.error("Data loading failed.")

            # Drop Loan_ID
            df = df.drop("Loan_ID", axis=1)
            logger.info("Dropped 'Loan_ID' column.")

            # Handle 'Dependents' as string/object before feature identification
            df["Dependents"] = df["Dependents"].astype(str)
            logger.info("Converted 'Dependents' column to string type.")

            # Convert 'Loan_Status' to numerical (Y=1, N=0)
            if "Loan_Status" not in df.columns:
                logger.error("'Loan_Status' column not found in data.")

            le = LabelEncoder()
            df["Loan_Status"] = le.fit_transform(df["Loan_Status"])
            y = df["Loan_Status"]
            original_X_df = df.drop(
                "Loan_Status", axis=1
            )  # Keep original X for input_example
            logger.info("Encoded 'Loan_Status' and separated X and y.")

            numerical_features, categorical_features = self._identify_features(
                original_X_df
            )
            preprocessor = self.get_preprocessing_pipeline(
                numerical_features, categorical_features
            )

            # Fit the preprocessor on the entire original_X_df
            preprocessor.fit(original_X_df)
            X_processed = preprocessor.transform(original_X_df)
            logger.info("Data preprocessing completed using ColumnTransformer.")

            # Get feature names after transformation
            feature_names = numerical_features + list(
                preprocessor.named_transformers_["cat"]
                .named_steps["onehot"]
                .get_feature_names_out(categorical_features)
            )
            logger.info(f"Generated {len(feature_names)} features after preprocessing.")

            return X_processed, y, preprocessor, feature_names, original_X_df

        except Exception as e:
            error_msg = f"Error during data transformation: {e}"
            logger.error(error_msg)
            raise e

    def split_data(
        self,
        X: np.ndarray,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.25,
        random_state: int = 42,
    ) -> tuple:
        """
        Splits processed data into training, validation, and test sets.

        :param X: Processed features (NumPy array).
        :param y: Target variable (Pandas Series).
        :param test_size: Proportion of the dataset to include in the test split.
        :param val_size: Proportion of the remaining training set to include in the validation split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing X_train, X_val, X_test, y_train, y_val, y_test.
        :raises MyException: If any error occurs during data splitting.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=val_size,
                random_state=random_state,
                stratify=y_train,
            )
            logger.info(
                f"Data split into Train ({X_train.shape[0]} samples), Validation ({X_val.shape[0]} samples), Test ({X_test.shape[0]} samples) sets."
            )
            return X_train, X_val, X_test, y_train, y_val, y_test
        except Exception as e:
            error_msg = f"Error during data splitting: {e}"
            logger.error(error_msg)
            raise e
