# tests/test_api/test_prediction.py
import numpy as np
from unittest.mock import patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from api.main import app
from fastapi.testclient import TestClient

class TestPredictionAPI:
    """Test cases for prediction API endpoints"""
    
    def setup_method(self):
        """Setup test client and sample data"""
        self.client = TestClient(app)
        self.sample_transaction = {
            "cc_num": "1234567890123456",
            "merchant": "test_merchant",
            "category": "grocery_pos",
            "amt": 50.0,
            "first": "John",
            "last": "Doe",
            "gender": "M",
            "street": "123 Main St",
            "city": "Test City",
            "state": "CA",
            "zip": "12345",
            "lat": 40.7128,
            "long": -74.0060,
            "city_pop": 50000,
            "job": "Engineer",
            "dob": "1980-01-01",
            "merch_lat": 40.7580,
            "merch_long": -73.9855,
            "merch_zipcode": "10001"
        }
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
    
    def test_predict_valid_transaction(self):
        """Test prediction with valid transaction data"""
        with patch('api.main.model') as mock_model:
            # Mock model prediction
            mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
            mock_model.predict.return_value = np.array([0])
            
            response = self.client.post("/predict", json=self.sample_transaction)
            assert response.status_code == 200
            
            data = response.json()
            assert "transaction_id" in data
            assert "fraud_probability" in data
            assert "is_fraud" in data
            assert "risk_level" in data
            assert "processing_time_ms" in data
            
            # Validate response structure
            assert isinstance(data["fraud_probability"], float)
            assert isinstance(data["is_fraud"], bool)
            assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    
    def test_predict_invalid_amount(self):
        """Test prediction with invalid amount"""
        invalid_transaction = self.sample_transaction.copy()
        invalid_transaction["amt"] = -10.0  # Negative amount
        
        response = self.client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_fields(self):
        """Test prediction with missing required fields"""
        incomplete_transaction = {
            "cc_num": "1234567890123456",
            "amt": 50.0
            # Missing other required fields
        }
        
        response = self.client.post("/predict", json=incomplete_transaction)
        assert response.status_code == 422
    
    def test_predict_high_fraud_probability(self):
        """Test prediction returning high fraud probability"""
        with patch('api.main.model') as mock_model:
            # Mock high fraud probability
            mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
            mock_model.predict.return_value = np.array([1])
            
            response = self.client.post("/predict", json=self.sample_transaction)
            assert response.status_code == 200
            
            data = response.json()
            assert data["fraud_probability"] > 0.5
            assert data["is_fraud"] == True
            assert data["risk_level"] == "HIGH"
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        batch_request = {
            "transactions": [self.sample_transaction, self.sample_transaction]
        }
        
        with patch('api.main.model') as mock_model:
            mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.7, 0.3]])
            mock_model.predict.return_value = np.array([0, 0])
            
            response = self.client.post("/predict/batch", json=batch_request)
            assert response.status_code == 200
            
            data = response.json()
            assert "predictions" in data
            assert "total" in data
            assert data["total"] == 2
            assert len(data["predictions"]) == 2
    
    def test_model_info_endpoint(self):
        """Test model information endpoint"""
        with patch('api.main.model') as mock_model:
            mock_model.__class__.__name__ = "XGBClassifier"
            
            response = self.client.get("/model/info")
            
            if response.status_code == 200:
                data = response.json()
                assert "name" in data or "version" in data
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "uptime_seconds" in data
        assert "model_loaded" in data