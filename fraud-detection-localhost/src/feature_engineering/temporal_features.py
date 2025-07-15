# src/feature_engineering/temporal_features.py
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TemporalFeatureEngineer:
    """Temporal feature engineering for fraud detection"""
    
    def create_time_features(self, df: pd.DataFrame, timestamp_col: str = 'trans_date_trans_time') -> pd.DataFrame:
        """Create comprehensive time-based features"""
        features = df.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(features[timestamp_col]):
            features['datetime'] = pd.to_datetime(features[timestamp_col])
        else:
            features['datetime'] = features[timestamp_col]
        
        # Basic time components
        features['hour'] = features['datetime'].dt.hour
        features['day_of_week'] = features['datetime'].dt.dayofweek
        features['day_of_month'] = features['datetime'].dt.day
        features['month'] = features['datetime'].dt.month
        features['quarter'] = features['datetime'].dt.quarter
        
        # Time-based binary features
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_night'] = ((features['hour'] < 6) | (features['hour'] > 22)).astype(int)
        features['is_business_hours'] = ((features['hour'] >= 9) & (features['hour'] <= 17)).astype(int)
        features['is_late_night'] = ((features['hour'] >= 23) | (features['hour'] <= 5)).astype(int)
        features['is_early_morning'] = ((features['hour'] >= 6) & (features['hour'] <= 9)).astype(int)
        features['is_evening'] = ((features['hour'] >= 18) & (features['hour'] <= 22)).astype(int)
        
        # Holiday and special day features (simplified)
        features['is_month_end'] = (features['day_of_month'] >= 28).astype(int)
        features['is_month_start'] = (features['day_of_month'] <= 3).astype(int)
        features['is_mid_month'] = ((features['day_of_month'] >= 14) & (features['day_of_month'] <= 16)).astype(int)
        
        # Cyclical encoding for circular features
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        logger.info("Created temporal features")
        return features
    
    def create_age_features(self, df: pd.DataFrame, dob_col: str = 'dob') -> pd.DataFrame:
        """Create age-based features"""
        features = df.copy()
        
        # Convert DOB to datetime
        features['dob_datetime'] = pd.to_datetime(features[dob_col], errors='coerce')
        
        # Calculate age
        current_date = datetime.now()
        features['customer_age'] = (
            (current_date - features['dob_datetime']).dt.days / 365.25
        )
        
        # Handle missing ages
        median_age = features['customer_age'].median()
        features['customer_age'] = features['customer_age'].fillna(median_age)
        
        # Age categories
        features['age_category'] = pd.cut(
            features['customer_age'],
            bins=[0, 25, 35, 50, 65, float('inf')],
            labels=['young', 'young_adult', 'middle_aged', 'senior', 'elderly']
        ).astype(str)
        
        # Age-based binary features
        features['is_young'] = (features['customer_age'] < 25).astype(int)
        features['is_senior'] = (features['customer_age'] >= 65).astype(int)
        features['is_prime_age'] = ((features['customer_age'] >= 25) & (features['customer_age'] < 65)).astype(int)
        
        # Age risk scoring
        features['age_risk_score'] = np.where(
            features['customer_age'] < 25, 0.3,  # Young people higher risk
            np.where(features['customer_age'] > 70, 0.2, 0.1)  # Elderly people medium risk
        )
        
        logger.info("Created age-based features")
        return features
