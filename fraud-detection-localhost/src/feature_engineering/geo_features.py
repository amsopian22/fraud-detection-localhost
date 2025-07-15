# src/feature_engineering/geo_features.py
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import logging

logger = logging.getLogger(__name__)

class GeoFeatureEngineer:
    """Geographic feature engineering for fraud detection"""
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on earth (in km)"""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        
        return c * r
    
    def create_distance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create distance-based features"""
        features = df.copy()
        
        # Primary distance feature
        features['customer_merchant_distance'] = features.apply(
            lambda row: self.haversine_distance(
                row['lat'], row['long'], row['merch_lat'], row['merch_long']
            ), axis=1
        )
        
        # Distance categories
        features['distance_category'] = pd.cut(
            features['customer_merchant_distance'],
            bins=[0, 10, 50, 200, 1000, float('inf')],
            labels=['very_close', 'close', 'medium', 'far', 'very_far']
        ).astype(str)
        
        # Binary distance features
        features['is_very_close'] = (features['customer_merchant_distance'] < 10).astype(int)
        features['is_local'] = (features['customer_merchant_distance'] < 50).astype(int)
        features['is_distant'] = (features['customer_merchant_distance'] > 200).astype(int)
        
        # Same state/zip indicators
        features['same_state'] = (features['state'] == features['merch_zipcode'].str[:2]).astype(int)
        features['same_zip_prefix'] = (
            features['zip'].str[:3] == features['merch_zipcode'].str[:3]
        ).astype(int)
        
        # Location risk scores
        features['location_risk_score'] = (
            (features['customer_merchant_distance'] > 100).astype(int) * 0.4 +
            (features['same_state'] == 0).astype(int) * 0.3 +
            (features['city_pop'] < 10000).astype(int) * 0.3
        )
        
        logger.info("Created geographic distance features")
        return features
    
    def create_location_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based cluster features"""
        features = df.copy()
        
        # Simple geographic clustering by rounding coordinates
        features['lat_rounded'] = np.round(features['lat'], 1)
        features['long_rounded'] = np.round(features['long'], 1)
        features['location_cluster'] = (
            features['lat_rounded'].astype(str) + '_' + 
            features['long_rounded'].astype(str)
        )
        
        # Merchant location clustering
        features['merch_lat_rounded'] = np.round(features['merch_lat'], 1)
        features['merch_long_rounded'] = np.round(features['merch_long'], 1)
        features['merch_location_cluster'] = (
            features['merch_lat_rounded'].astype(str) + '_' + 
            features['merch_long_rounded'].astype(str)
        )
        
        logger.info("Created location cluster features")
        return features