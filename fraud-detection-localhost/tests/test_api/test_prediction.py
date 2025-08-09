# tests/test_api/test_prediction.py
import sys
import os
import pandas as pd

# Add project root to the path to resolve import issues
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

import numpy as np
from unittest.mock import patch

from src.api.main import app
from fastapi.testclient import TestClient

class TestPredictionAPI:
    """Test cases for prediction API endpoints"""
    
    def setup_method(self):
        """Setup test client and sample data"""
        self.client = TestClient(app)
        self.sample_transaction = {
            "cc_num": 1234567890123456,
            "merchant": "test_merchant",
            "category": "grocery_pos",
            "amt": 50.0,
            "first": "John",
            "last": "Doe",
            "gender": "M",
            "street": "123 Main St",
            "city": "Test City",
            "state": "CA",
            "zip": 12345,
            "lat": 40.7128,
            "long": -74.0060,
            "city_pop": 50000,
            "job": "Engineer",
            "dob": "1980-01-01",
            "trans_num": "c2a0c8a2-7948-4bb6-817d-2b7e63351d5c",
            "unix_time": 1672531200,
            "merch_lat": 40.7580,
            "merch_long": -73.9855,
            "merch_zipcode": 10001
        }
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/monitoring/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
    
    def test_predict_valid_transaction(self):
        """Test prediction with valid transaction data"""
        with patch('src.api.endpoints.prediction.feature_engineer.create_all_features') as mock_feature_engineer, \
             patch('src.api.endpoints.prediction.model_service.predict') as mock_predict:

            mock_feature_engineer.return_value = pd.DataFrame([self.sample_transaction])
            mock_predict.return_value = (0, 0.2)
            
            response = self.client.post("/predict/", json=self.sample_transaction)
            assert response.status_code == 200
            
            data = response.json()
            assert "transaction_id" in data
            assert "fraud_probability" in data
            assert "is_fraud" in data
            assert isinstance(data["fraud_probability"], float)
            assert isinstance(data["is_fraud"], int)
    
    def test_predict_invalid_amount(self):
        """Test prediction with invalid amount"""
        invalid_transaction = self.sample_transaction.copy()
        invalid_transaction["amt"] = -10.0
        
        response = self.client.post("/predict/", json=invalid_transaction)
        assert response.status_code == 500

    def test_predict_missing_fields(self):
        """Test prediction with missing required fields"""
        incomplete_transaction = {
            "cc_num": 1234567890123456,
            "amt": 50.0
        }
        
        response = self.client.post("/predict/", json=incomplete_transaction)
        assert response.status_code == 422
    
    def test_predict_high_fraud_probability(self):
        """Test prediction returning high fraud probability"""
        with patch('src.api.endpoints.prediction.feature_engineer.create_all_features') as mock_feature_engineer, \
             patch('src.api.endpoints.prediction.model_service.predict') as mock_predict:
            
            mock_feature_engineer.return_value = pd.DataFrame([self.sample_transaction])
            mock_predict.return_value = (1, 0.8)
            
            response = self.client.post("/predict/", json=self.sample_transaction)
            assert response.status_code == 200
            
            data = response.json()
            assert data["fraud_probability"] > 0.5
            assert data["is_fraud"] == 1

    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = self.client.get("/monitoring/metrics")
        assert response.status_code == 200
        assert "uptime_seconds" in response.text