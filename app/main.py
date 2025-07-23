import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import ValidationError

from app.schema import LoanApplication, PredictionResponse
from app.utils import load_model_artifacts
from loanApprovalPrediction.logger import logger
import uvicorn

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts house prices based on various features.",
    version="0.1.0",
)

preprocessor = None
model = None
model_version = "1.0.0"


@app.on_event("startup")
async def startup_event():
    """
    Load the model and preprocessor when the FastAPI application starts.
    """
    global model
    logger.info("Application startup: Loading model and preprocessor...")
    try:
        model = load_model_artifacts()
        logger.info("Model and preprocessor loaded successfully.")
    except Exception as e:
        logger.critical(
            f"An unexpected error occurred during model loading: {e}", exc_info=True
        )
        model = None


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to ensure the API is running.
    """
    return {"status": "ok", "message": "API is healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: LoanApplication):
    """
    Predicts the price of a house based on the provided features.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs for errors.",
        )

    logger.info(f"Received prediction request: {features.model_dump()}")

    try:
        # Convert LoanApplication to DataFrame
        input_df = pd.DataFrame([features.model_dump()])

        prediction = model.predict(input_df)
        logger.info(f"Predicted price: {prediction}")
        return PredictionResponse(prediction="Y" if prediction[0] == 1 else "N")

    except ValidationError as e:
        logger.error(f"Input validation error: {e.errors()}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Validation Error", "errors": e.errors()},
        ) from e
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during prediction: {e}",
        ) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
