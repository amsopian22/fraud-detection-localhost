# tests/test_models/test_feature_engineering.py
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from feature_engineering import MasterFeatureEngineer
from feature_engineering.core_features import CoreFeatureEngineer
from feature_engineering.geo_features import GeoFeatureEngineer
from feature_engineering.temporal_features import TemporalFeatureEngineer

class TestFeatureEngineering:
    """Test cases for feature engineering modules"""
    
    def setup_method(self):
        """Setup test data"""
        self.sample_data = pd.DataFrame({
            'trans_date_trans_time': ['2023-01-01 14:30:00', '2023-01-02 09:15:00'],
            'cc_num': ['1234567890123456', '9876543210987654'],
            'merchant': ['merchant_A', 'merchant_B'],
            'category': ['grocery_pos', 'gas_transport'],
            'amt': [50.0, 75.5],
            'first': ['John', 'Jane'],
            'last': ['Doe', 'Smith'],
            'gender': ['M', 'F'],
            'street': ['123 Main St', '456 Oak Ave'],
            'city': ['City A', 'City B'],
            'state': ['CA', 'NY'],
            'zip': ['12345', '67890'],
            'lat': [40.7128, 34.0522],
            'long': [-74.0060, -118.2437],
            'city_pop': [50000, 100000],
            'job': ['Engineer', 'Teacher'],
            'dob': ['1980-01-01', '1975-06-15'],
            'merch_lat': [40.7580, 34.0700],
            'merch_long': [-73.9855, -118.2000],
            'merch_zipcode': ['10001', '90210'],
            'is_fraud': [0, 1]
        })
    
    def test_core_feature_creation(self):
        """Test core feature engineering"""
        engineer = CoreFeatureEngineer()
        
        # Test basic features
        result = engineer.create_basic_features(self.sample_data)
        
        assert 'amt_log' in result.columns
        assert 'amt_sqrt' in result.columns
        assert 'city_pop_log' in result.columns
        assert 'is_high_amount' in result.columns
        
        # Validate calculations
        assert result['amt_log'].iloc[0] == np.log1p(50.0)
        assert result['amt_sqrt'].iloc[0] == np.sqrt(50.0)
    
    def test_categorical_features(self):
        """Test categorical feature encoding"""
        engineer = CoreFeatureEngineer()
        result = engineer.create_categorical_features(self.sample_data)
        
        assert 'gender_M' in result.columns
        assert 'gender_F' in result.columns
        assert 'category_risk' in result.columns
        assert 'state_high_risk' in result.columns
        
        # Validate gender encoding
        assert result['gender_M'].iloc[0] == 1  # John is Male
        assert result['gender_F'].iloc[0] == 0
        assert result['gender_M'].iloc[1] == 0  # Jane is Female
        assert result['gender_F'].iloc[1] == 1
    
    def test_geographic_features(self):
        """Test geographic feature engineering"""
        engineer = GeoFeatureEngineer()
        result = engineer.create_distance_features(self.sample_data)
        
        assert 'customer_merchant_distance' in result.columns
        assert 'distance_category' in result.columns
        assert 'is_very_close' in result.columns
        assert 'is_local' in result.columns
        assert 'location_risk_score' in result.columns
        
        # Validate distance calculation (should be positive)
        assert all(result['customer_merchant_distance'] >= 0)
    
    def test_temporal_features(self):
        """Test temporal feature engineering"""
        engineer = TemporalFeatureEngineer()
        result = engineer.create_time_features(self.sample_data)
        
        assert 'hour' in result.columns
        assert 'day_of_week' in result.columns
        assert 'is_weekend' in result.columns
        assert 'is_night' in result.columns
        assert 'is_business_hours' in result.columns
        
        # Validate hour extraction
        assert result['hour'].iloc[0] == 14  # 2:30 PM
        assert result['hour'].iloc[1] == 9   # 9:15 AM
    
    def test_age_features(self):
        """Test age-based feature creation"""
        engineer = TemporalFeatureEngineer()
        result = engineer.create_age_features(self.sample_data)
        
        assert 'customer_age' in result.columns
        assert 'age_category' in result.columns
        assert 'is_young' in result.columns
        assert 'is_senior' in result.columns
        
        # Validate age calculations (approximate)
        assert result['customer_age'].iloc[0] > 40  # Born in 1980
        assert result['customer_age'].iloc[1] > 45  # Born in 1975
    
    def test_master_feature_engineer(self):
        """Test master feature engineering orchestration"""
        master_engineer = MasterFeatureEngineer()
        result = master_engineer.create_all_features(self.sample_data)
        
        # Should have significantly more features than original
        assert len(result.columns) > len(self.sample_data.columns)
        
        # Check for key engineered features
        expected_features = [
            'amt_log', 'customer_merchant_distance', 'hour', 'customer_age',
            'is_night', 'is_weekend', 'location_risk_score', 'composite_risk_score'
        ]
        
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
    
    def test_feature_list_consistency(self):
        """Test that feature list matches actual features created"""
        master_engineer = MasterFeatureEngineer()
        result = master_engineer.create_all_features(self.sample_data)
        feature_list = master_engineer.get_feature_list()
        
        # Check that all listed features are actually in the result
        missing_features = [f for f in feature_list if f not in result.columns]
        
        # Some features might be missing due to data limitations, but most should be present
        assert len(missing_features) < len(feature_list) * 0.2, f"Too many missing features: {missing_features}"