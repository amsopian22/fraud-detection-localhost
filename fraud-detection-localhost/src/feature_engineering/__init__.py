# src/feature_engineering/__init__.py
from .core_features import CoreFeatureEngineer
from .geo_features import GeoFeatureEngineer
from .temporal_features import TemporalFeatureEngineer
from .aggregation_features import AggregationFeatureEngineer

class MasterFeatureEngineer:
    """Master class that orchestrates all feature engineering"""
    
    def __init__(self):
        self.core_engineer = CoreFeatureEngineer()
        self.geo_engineer = GeoFeatureEngineer()
        self.temporal_engineer = TemporalFeatureEngineer()
        self.aggregation_engineer = AggregationFeatureEngineer()
        
    def create_all_features(self, df: pd.DataFrame, include_aggregations: bool = False) -> pd.DataFrame:
        """Create all features in the correct order"""
        logger.info("Starting comprehensive feature engineering...")
        
        # Start with core features
        features = self.core_engineer.create_basic_features(df)
        features = self.core_engineer.create_categorical_features(features)
        features = self.core_engineer.create_amount_percentiles(features)
        
        # Add geographic features
        features = self.geo_engineer.create_distance_features(features)
        features = self.geo_engineer.create_location_clusters(features)
        
        # Add temporal features
        features = self.temporal_engineer.create_time_features(features)
        features = self.temporal_engineer.create_age_features(features)
        
        # Add aggregation features (optional, for training)
        if include_aggregations:
            features = self.aggregation_engineer.create_customer_aggregations(features)
            features = self.aggregation_engineer.create_merchant_aggregations(features)
            features = self.aggregation_engineer.create_velocity_features(features)
        
        # Add risk scores
        features = self.aggregation_engineer.create_risk_scores(features)
        
        logger.info(f"Feature engineering complete. Created {len(features.columns)} total features")
        return features
    
    def get_feature_list(self) -> List[str]:
        """Get list of all engineered features"""
        return [
            # Core features
            'amt_log', 'amt_sqrt', 'amt_zscore', 'city_pop_log', 'city_pop_zscore',
            'is_high_amount', 'is_round_amount', 'is_large_city',
            'gender_M', 'gender_F', 'category_risk', 'state_high_risk',
            'amt_percentile', 'amt_percentile_in_category',
            
            # Geographic features
            'customer_merchant_distance', 'is_very_close', 'is_local', 'is_distant',
            'same_state', 'same_zip_prefix', 'location_risk_score',
            
            # Temporal features
            'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_business_hours',
            'is_late_night', 'is_early_morning', 'is_evening',
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'customer_age', 'is_young', 'is_senior', 'is_prime_age', 'age_risk_score',
            
            # Risk scores
            'amount_risk', 'time_risk', 'location_risk', 'composite_risk_score'
        ]