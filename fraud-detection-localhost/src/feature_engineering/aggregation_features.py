# src/feature_engineering/aggregation_features.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AggregationFeatureEngineer:
    """Aggregation and derived feature engineering"""
    
    def __init__(self):
        self.customer_stats = {}
        self.merchant_stats = {}
        
    def create_customer_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-level aggregation features"""
        features = df.copy()
        
        # Customer transaction statistics
        customer_stats = df.groupby('cc_num').agg({
            'amt': ['count', 'mean', 'std', 'min', 'max', 'sum'],
            'customer_merchant_distance': ['mean', 'std'],
            'is_fraud': ['sum', 'mean']
        }).round(4)
        
        # Flatten column names
        customer_stats.columns = ['_'.join(col).strip() for col in customer_stats.columns]
        
        # Merge back to original dataframe
        features = features.merge(
            customer_stats, 
            left_on='cc_num', 
            right_index=True, 
            how='left', 
            suffixes=('', '_customer_hist')
        )
        
        # Fill missing values for new customers
        numeric_cols = customer_stats.columns
        for col in numeric_cols:
            if col in features.columns:
                features[col] = features[col].fillna(features[col].median())
        
        logger.info("Created customer aggregation features")
        return features
    
    def create_merchant_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create merchant-level aggregation features"""
        features = df.copy()
        
        # Merchant transaction statistics
        merchant_stats = df.groupby('merchant').agg({
            'amt': ['count', 'mean', 'std'],
            'is_fraud': ['sum', 'mean'],
            'customer_merchant_distance': ['mean']
        }).round(4)
        
        # Flatten column names
        merchant_stats.columns = ['_'.join(col).strip() for col in merchant_stats.columns]
        
        # Merge back to original dataframe
        features = features.merge(
            merchant_stats,
            left_on='merchant',
            right_index=True,
            how='left',
            suffixes=('', '_merchant_hist')
        )
        
        # Fill missing values for new merchants
        numeric_cols = merchant_stats.columns
        for col in numeric_cols:
            if col in features.columns:
                features[col] = features[col].fillna(features[col].median())
        
        logger.info("Created merchant aggregation features")
        return features
    
    def create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transaction velocity features"""
        features = df.copy()
        
        # Sort by customer and time
        features = features.sort_values(['cc_num', 'trans_date_trans_time'])
        
        # Time between transactions
        features['prev_transaction_time'] = features.groupby('cc_num')['trans_date_trans_time'].shift(1)
        features['time_since_last_transaction'] = (
            pd.to_datetime(features['trans_date_trans_time']) - 
            pd.to_datetime(features['prev_transaction_time'])
        ).dt.total_seconds() / 3600  # Convert to hours
        
        # Fill missing values for first transactions
        features['time_since_last_transaction'] = features['time_since_last_transaction'].fillna(24)
        
        # Velocity features
        features['is_rapid_transaction'] = (features['time_since_last_transaction'] < 1).astype(int)
        features['is_very_rapid'] = (features['time_since_last_transaction'] < 0.1).astype(int)
        
        # Transaction frequency
        features['transactions_last_hour'] = 0  # Would be calculated from real-time data
        features['transactions_last_day'] = 0   # Would be calculated from real-time data
        
        logger.info("Created velocity features")
        return features
    
    def create_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk scores"""
        features = df.copy()
        
        # Amount risk score
        features['amount_risk'] = np.where(
            features['amt'] > features['amt'].quantile(0.95), 0.8,
            np.where(features['amt'] > features['amt'].quantile(0.9), 0.5, 0.1)
        )
        
        # Time risk score
        features['time_risk'] = (
            features['is_night'] * 0.3 +
            features['is_weekend'] * 0.2 +
            features['is_late_night'] * 0.5
        )
        
        # Location risk score
        features['location_risk'] = np.where(
            features['customer_merchant_distance'] > 200, 0.6,
            np.where(features['customer_merchant_distance'] > 100, 0.4, 0.1)
        )
        
        # Composite risk score
        features['composite_risk_score'] = (
            features['amount_risk'] * 0.4 +
            features['time_risk'] * 0.3 +
            features['location_risk'] * 0.3
        )
        
        logger.info("Created composite risk scores")
        return features