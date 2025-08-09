import joblib
import pandas as pd
import os
import logging

class ModelService:
    def __init__(self, model_path: str = "/app/models/trained_models"):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.load_model()

    def load_model(self):
        """Loads the model and feature names from disk."""
        model_file = os.path.join(self.model_path, "xgboost_fraud_detector.joblib")
        feature_names_file = os.path.join(self.model_path, "feature_names.joblib")

        try:
            self.model = joblib.load(model_file)
            self.feature_names = joblib.load(feature_names_file)
            logging.info("Model and feature names loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"Error loading model files: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> tuple[int, float]:
        """Makes a fraud prediction."""
        if self.model is None or self.feature_names is None:
            raise RuntimeError("Model is not loaded.")

        # Ensure columns are in the same order as during training
        data = data[self.feature_names]

        prediction = self.model.predict(data)[0]
        probability = self.model.predict_proba(data)[0][1] # Probability of fraud class

        return int(prediction), float(probability)

# Create a singleton instance
model_service = ModelService()
