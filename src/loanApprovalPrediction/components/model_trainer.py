# src/model_trainer.py
import os

import mlflow
import mlflow.sklearn
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import (
    Pipeline,
)  # Important: We are building a pipeline with preprocessor and classifier

from loanApprovalPrediction.components.data_processor import (
    initiate_data_transformation,
    load_data,
    split_data,
)

load_dotenv()

# Import functions from data_processor


def train_and_log_model(
    model_name,
    classifier,
    param_grid,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    preprocessor,
    original_X_for_example,
):
    """
    Trains a model with given classifier and hyperparameters,
    logs results to MLflow, and registers the model.
    """
    mlflow.set_experiment("Loan_Approval_RandomForest_Experiments")
    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        # Log the full parameter grid for the GridSearchCV
        mlflow.log_params(
            {
                f"grid_param_{k.replace('classifier__', '')}": v
                for k, v in param_grid.items()
            }
        )

        # Create a full pipeline that includes preprocessing and the classifier
        # This pipeline will be saved as the MLflow model
        full_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),  # The fitted preprocessor object
                ("classifier", classifier),
            ]
        )

        print(f"Starting GridSearchCV for {model_name}...")
        grid_search = GridSearchCV(
            full_pipeline,
            param_grid,
            cv=KFold(n_splits=3, shuffle=True, random_state=42),
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
        )

        # Fit GridSearchCV on the training data
        grid_search.fit(
            original_X_for_example.loc[y_train.index], y_train
        )  # Fit on the original X_train part
        # GridSearchCV will handle internal preprocessing

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_

        print(f"\nBest hyperparameters found for {model_name}: {best_params}")
        print(f"Best cross-validation accuracy for {model_name}: {best_cv_score:.4f}")

        # Log best hyperparameters and best CV score
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_accuracy", best_cv_score)

        # --- Evaluate on Validation Set (using the best model from GridSearchCV) ---
        # Note: The best_model is already a Pipeline that includes preprocessing
        y_val_pred = best_model.predict(
            original_X_for_example.loc[y_val.index]
        )  # Predict on original X_val part
        y_val_proba = best_model.predict_proba(original_X_for_example.loc[y_val.index])[
            :, 1
        ]

        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_roc_auc = roc_auc_score(y_val, y_val_proba)

        print(f"\n--- Validation Metrics for {model_name} ---")
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"Precision: {val_precision:.4f}")
        print(f"Recall: {val_recall:.4f}")
        print(f"F1-Score: {val_f1:.4f}")
        print(f"ROC AUC: {val_roc_auc:.4f}")

        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_precision", val_precision)
        mlflow.log_metric("val_recall", val_recall)
        mlflow.log_metric("val_f1_score", val_f1)
        mlflow.log_metric("val_roc_auc", val_roc_auc)

        # --- Evaluate on Test Set ---
        y_test_pred = best_model.predict(
            original_X_for_example.loc[y_test.index]
        )  # Predict on original X_test part
        y_test_proba = best_model.predict_proba(
            original_X_for_example.loc[y_test.index]
        )[:, 1]

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)

        print(f"\n--- Test Metrics for {model_name} ---")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"F1-Score: {test_f1:.4f}")
        print(f"ROC AUC: {test_roc_auc:.4f}")

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_roc_auc", test_roc_auc)

        # Log classification report
        report = classification_report(y_test, y_test_pred, output_dict=True)
        mlflow.log_dict(report, "classification_report.json")

        # --- Log the best model to MLflow and register it ---
        # The input_example should be a Pandas DataFrame that matches the input format
        # of your model_pipeline (i.e., the raw features before any preprocessing).
        # input_example = original_X_for_example.iloc[
        #     [0]
        # ]  # Use the first row of original X as input example

        # Infer the model signature for robust deployment
        # Pass the original (unprocessed) input example to infer_signature
        # signature = infer_signature(input_example, best_model.predict(input_example))

        # mlflow.sklearn.log_model(
        #     sk_model=best_model,
        #     artifact_path="loan_approval_model",
        #     signature=signature,
        #     input_example=input_example,
        #     tags={"model_type": "RandomForestClassifier", "dataset": "LoanApproval"},
        # )

        print(
            f"\nModel logged to MLflow and registered as 'LoanApprovalRandomForestModel' (Run ID: {run_id})."
        )

        return run_id


if __name__ == "__main__":
    # Ensure MLflow tracking URI is set (e.g., to a local directory or a server)
    mlflow.set_experiment("Loan_Approval_RandomForest_Experiments")
    np.random.seed(42)  # For reproducibility

    # Load data
    df_raw = load_data()
    if df_raw is None:
        print("Exiting as data could not be loaded.")
        exit()

    # Preprocess data and get the fitted preprocessor and original X for input example
    X_processed_np, y_series, preprocessor, feature_names, original_X_df = (
        initiate_data_transformation(df_raw.copy())
    )

    # Split the processed data (for training, validation, test sets)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_processed_np, y_series
    )

    # Define the classifier and its parameter grid
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    }

    # Train and log the Random Forest model
    train_and_log_model(
        model_name="Random_Forest_Hyperparameter_Tuning",
        classifier=rf_classifier,
        param_grid=rf_param_grid,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        preprocessor=preprocessor,
        original_X_for_example=original_X_df,  # Pass the original X DataFrame for input_example and GridSearchCV
    )

    print(
        "\nTraining and logging complete. Run 'mlflow ui' in your terminal to view runs."
    )
