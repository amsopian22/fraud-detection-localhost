import joblib
import pandas as pd
import os
import logging
from fastapi import APIRouter, HTTPException
from src.api.models.request_models import Transaction
from src.api.models.response_models import PredictionResponse
from src.feature_engineering import MasterFeatureEngineer

# --- Start of inlined model_service.py ---
class ModelService:
    def __init__(self, model_path: str = "models/trained_models"):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.load_model()

    def load_model(self):
        model_file = os.path.join(self.model_path, "xgboost_fraud_detector.joblib")
        feature_names_file = os.path.join(self.model_path, "feature_names.joblib")
        try:
            self.model = joblib.load(model_file)
            self.feature_names = joblib.load(feature_names_file)
            logging.info(f"Model loaded from {self.model_path}")
        except FileNotFoundError as e:
            logging.error(f"Error loading model files: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> tuple[int, float]:
        if self.model is None or self.feature_names is None:
            raise RuntimeError("Model is not loaded.")
        try:
            # Ensure all required feature names are present
            missing_cols = set(self.feature_names) - set(data.columns)
            if missing_cols:
                raise RuntimeError(f"Missing columns in prediction data: {missing_cols}")
            data = data[self.feature_names]
        except KeyError as e:
            raise RuntimeError(f"Column mismatch error: {e}") from e

        prediction = self.model.predict(data)[0]
        probability = self.model.predict_proba(data)[0][1]
        return int(prediction), float(probability)

model_service = ModelService()
feature_engineer = MasterFeatureEngineer()
# --- End of inlined model_service.py ---


router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    try:
        logger.info(f"Received prediction request for transaction: {transaction.trans_num}")
        transaction_dict = transaction.model_dump()
        data_df = pd.DataFrame([transaction_dict])

        # Apply feature engineering
        features_df = feature_engineer.create_all_features(data_df)

        is_fraud, prob = model_service.predict(features_df)

        logger.info(f"Prediction for {transaction.trans_num}: is_fraud={is_fraud}, probability={prob:.4f}")

        return PredictionResponse(
            transaction_id=transaction.trans_num,
            is_fraud=is_fraud,
            fraud_probability=prob,
            model_version="2.0.0"
        )
    except RuntimeError as e:
        logger.error(f"Model service runtime error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except KeyError as e:
        logger.error(f"Missing feature in input data: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid transaction data. Missing feature: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during prediction.")
