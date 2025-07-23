# Loan Approval Prediction with MLOps

A robust, end-to-end machine learning project for predicting loan approval using MLOps best practices. This project features a modular pipeline for data ingestion, preprocessing, model training, and a FastAPI-based prediction service, all orchestrated with DVC and CI/CD workflows.

## Features
- **Data Ingestion**: Loads data from MongoDB into CSV artifacts.
- **Data Processing**: Cleans, encodes, and transforms data for modeling.
- **Model Training**: Trains a Random Forest classifier with hyperparameter tuning and logs experiments to MLflow.
- **API Service**: FastAPI app for real-time loan approval predictions.
- **MLOps**: DVC for pipeline/data/model versioning, GitHub Actions for CI/CD, Docker for containerization.

## Project Structure
```
loan-approval-prediction-with-mlops/
├── app/                  # FastAPI app (main.py, schema.py, utils.py)
├── src/loanApprovalPrediction/
│   ├── components/       # Data ingestion, processing, model training modules
│   ├── config/           # Configuration management
│   ├── constants/        # Project-wide constants
│   ├── entity/           # Data classes for configs
│   ├── logger/           # Logging setup
│   ├── pipeline/         # Pipeline orchestration
│   └── utils/            # Utility functions
├── artifacts/            # DVC-tracked data, preprocessor, models
├── config/config.json    # Main configuration file
├── params.json           # Model hyperparameters
├── dvc.yaml              # DVC pipeline definition
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker containerization
├── tests/                # Unit tests
├── notebooks/            # EDA, research, and data loading notebooks
└── ...
```

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/mohitDora/loan-approval-prediction-with-mlops.git
cd loan-approval-prediction-with-mlops
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory and set the following (see `config/config.json` for details):
```
MONGO_DB_PASSWORD=your_mongodb_password
MLFLOW_TRACKING_URI=your_mlflow_tracking_uri
MLFLOW_TRACKING_USERNAME=your_mlflow_username
MLFLOW_TRACKING_PASSWORD=your_mlflow_password
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

### 4. Run the DVC Pipeline
Ensure DVC is installed and configured. This will run data ingestion, preprocessing, and model training:
```bash
dvc repro
```

### 5. Start the FastAPI Service
```bash
uvicorn app.main:app --reload
```
The API will be available at [http://localhost:8000](http://localhost:8000)

### 6. Run Tests
```bash
pytest
```

## API Usage

### Health Check
```
GET /health
```
Response:
```json
{"status": "ok", "message": "API is healthy"}
```

### Predict Loan Approval
```
POST /predict
Content-Type: application/json
```
Request body (example):
```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": 1.0,
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 0,
  "LoanAmount": 120,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```
Response:
```json
{
  "prediction": "Y",
  "probability": null
}
```

## Pipeline Stages
Defined in `dvc.yaml`:
- **data_ingestion**: Loads data from MongoDB to CSV
- **data_preprocessing**: Cleans and transforms data
- **model_trainer**: Trains and logs the model

## Configuration
- **config/config.json**: Main pipeline and data config
- **params.json**: Model hyperparameters (e.g., n_estimators, max_depth)

## Docker
Build and run the API in a container:
```bash
docker build -t loan-approval-api .
docker run -p 8000:8000 --env-file .env loan-approval-api
```

## Testing
Unit tests are in the `tests/` directory and use `pytest`:
```bash
pytest
```

## Notebooks
- `notebooks/research.ipynb`: EDA, feature engineering, model selection
- `notebooks/load_data_to_mongodb.ipynb`: Data upload to MongoDB

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
