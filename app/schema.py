from typing import Literal, Optional

from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    Gender: Literal["Male", "Female"]
    Married: Literal["Yes", "No"]
    Dependents: float
    Education: Literal["Graduate", "Not Graduate"]
    Self_Employed: Literal["Yes", "No"]
    ApplicantIncome: int
    CoapplicantIncome: int
    LoanAmount: int
    Loan_Amount_Term: int
    Credit_History: int  # 1 or 0
    Property_Area: Literal["Urban", "Rural", "Semiurban"]


class PredictionResponse(BaseModel):
    prediction: Literal["Y", "N"]
    probability: Optional[float] = Field(
        None, description="Probability of loan approval (if available)"
    )
