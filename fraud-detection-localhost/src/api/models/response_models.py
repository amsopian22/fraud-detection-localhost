from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: int
    fraud_probability: float
    model_version: Optional[str] = "1.0.0"
