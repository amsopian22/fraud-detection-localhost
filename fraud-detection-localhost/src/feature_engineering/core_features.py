# src/feature_engineering/core_features.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CoreFeatureEngineer:
    """Core feature engineering for fraud detection"""
    
    def __init__(self):
        self.feature_stats = {}
        self.encoders = {}
        
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic mathematical and statistical features"""
        features = df.copy()
        
        # Amount-based features
        features['amt_log'] = np.log1p(features['amt'])
        features['amt_sqrt'] = np.sqrt(features['amt'])
        features['amt_zscore'] = (features['amt'] - features['amt'].mean()) / features['amt'].std()
        
        # Population features
        features['city_pop_log'] = np.log1p(features['city_pop'])
        features['city_pop_zscore'] = (features['city_pop'] - features['city_pop'].mean()) / features['city_pop'].std()
        
        # Binary features
        features['is_high_amount'] = (features['amt'] > features['amt'].quantile(0.9)).astype(int)
        features['is_round_amount'] = (features['amt'] % 1 == 0).astype(int)
        features['is_large_city'] = (features['city_pop'] > features['city_pop'].quantile(0.8)).astype(int)
        
        logger.info("Created basic mathematical features")
        return features
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create and encode categorical features"""
        features = df.copy()
        
        # One-hot encoding for categories with few unique values
        if 'gender' in features.columns:
            features['gender_M'] = (features['gender'] == 'M').astype(int)
            features['gender_F'] = (features['gender'] == 'F').astype(int)
        
        # Category mappings (simplified for common categories)
        category_risk_map = {
            'grocery_pos': 1,
            'gas_transport': 2, 
            'misc_net': 3,
            'grocery_net': 1,
            'entertainment': 4,
            'misc_pos': 3
        }
        
        features['category_risk'] = features['category'].map(category_risk_map).fillna(0)
        
        # State risk encoding (based on fraud rates - simplified)
        high_risk_states = ['CA', 'FL', 'TX', 'NY']
        features['state_high_risk'] = features['state'].isin(high_risk_states).astype(int)
        
        logger.info("Created categorical features")
        return features
    
    def create_amount_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount percentile features"""
        features = df.copy()
        
        # Overall percentiles
        features['amt_percentile'] = features['amt'].rank(pct=True)
        
        # Category-specific percentiles
        for category in features['category'].unique():
            mask = features['category'] == category
            if mask.sum() > 1:
                features.loc[mask, 'amt_percentile_in_category'] = (
                    features.loc[mask, 'amt'].rank(pct=True)
                )
        
        features['amt_percentile_in_category'] = features['amt_percentile_in_category'].fillna(0.5)
        
        logger.info("Created amount percentile features")
        return features