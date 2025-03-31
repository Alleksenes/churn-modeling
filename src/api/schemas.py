# ============================================
# File: src/api/schemas.py
# ============================================
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field  # , validator


# --- Enums for Validation ---
class GeographyEnum(str, Enum):
    france = "France"
    spain = "Spain"
    germany = "Germany"


class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"


# --- Input Schemas ---
class ChurnPredictionInput(BaseModel):
    """Input schema for single churn prediction."""

    CreditScore: int = Field(..., ge=0, description="Customer's credit score")
    Geography: GeographyEnum = Field(..., description="Customer's country")
    Gender: GenderEnum = Field(..., description="Customer's gender")
    Age: int = Field(..., ge=18, le=100, description="Customer's age")
    Tenure: int = Field(..., ge=0, description="Number of years as customer")
    Balance: float = Field(..., ge=0, description="Customer's account balance")
    NumOfProducts: int = Field(
        ..., ge=1, le=4, description="Number of products customer uses (usually 1-4)"
    )
    HasCrCard: int = Field(
        ...,
        ge=0,
        le=1,
        description="Does the customer have a credit card? (1=Yes, 0=No)",
    )
    IsActiveMember: int = Field(
        ..., ge=0, le=1, description="Is the customer an active member? (1=Yes, 0=No)"
    )
    EstimatedSalary: float = Field(..., ge=0, description="Customer's estimated salary")

    class Config:
        """Pydantic configuration."""

        schema_extra = {
            "example": {
                "CreditScore": 608,
                "Geography": "Spain",
                "Gender": "Female",
                "Age": 41,
                "Tenure": 1,
                "Balance": 83807.86,
                "NumOfProducts": 1,
                "HasCrCard": 0,
                "IsActiveMember": 1,
                "EstimatedSalary": 112542.58,
            }
        }
        use_enum_values = True


class BulkChurnPredictionInput(BaseModel):
    """Input schema for bulk churn predictions."""

    inputs: List[ChurnPredictionInput] = Field(..., min_items=1)


# --- Output Schemas ---
class ChurnPredictionOutput(BaseModel):
    """Output schema for single churn prediction."""

    prediction: Optional[int] = Field(
        None, description="Churn prediction (1=Churn, 0=No Churn)"
    )
    probability_churn: Optional[float] = Field(
        None, ge=0, le=1, description="Predicted probability of churning (Class 1)"
    )
    error: Optional[str] = Field(
        None, description="Error message if prediction failed for this input"
    )


class BulkChurnPredictionOutput(BaseModel):
    """Output schema for bulk churn predictions."""

    results: List[ChurnPredictionOutput]
