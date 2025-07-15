# src/models/model_server.py
import time
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from fastapi import HTTPException
import logging
from .model_registry import ModelRegistry
import json

logger = logging.getLogger(__name__)

class ModelServer:
    """High-performance model serving with caching and batch processing"""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.current_model = None
        self.current_metadata = None
        self.prediction_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.batch_queue = []
        self.batch_size = 100
        self.batch_timeout = 1.0  # seconds
        
        # Load production model
        self._load_production_model()
    
    def _load_production_model(self):
        """Load the current production model"""
        try:
            result = self.registry.get_production_model()
            if result:
                self.current_model, self.current_metadata = result
                logger.info(f"Loaded production model: {self.current_metadata.model_id}")
            else:
                logger.warning("No production model available")
        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
    
    def reload_model(self):
        """Reload the production model (for hot-swapping)"""
        logger.info("Reloading production model...")
        self._load_production_model()
    
    def _create_cache_key(self, features: Dict[str, Any]) -> str:
        """Create cache key from features"""
        import hashlib
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - timestamp < self.cache_ttl
    
    async def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single transaction"""
        if not self.current_model:
            raise HTTPException(status_code=503, detail="No model available")
        
        start_time = time.time()
        
        # Check cache
        cache_key = self._create_cache_key(features)
        if cache_key in self.prediction_cache:
            cached_result, timestamp = self.prediction_cache[cache_key]
            if self._is_cache_valid(timestamp):
                cached_result['processing_time_ms'] = (time.time() - start_time) * 1000
                cached_result['cached'] = True
                return cached_result
        
        try:
            # Prepare features
            feature_vector = self._prepare_features(features)
            
            # Make prediction
            fraud_prob = float(self.current_model.predict_proba([feature_vector])[0][1])
            is_fraud = fraud_prob > 0.5
            
            # Determine risk level
            if fraud_prob < 0.3:
                risk_level = "LOW"
            elif fraud_prob < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            result = {
                'fraud_probability': fraud_prob,
                'is_fraud': is_fraud,
                'risk_level': risk_level,
                'model_version': self.current_metadata.version,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'cached': False
            }
            
            # Cache result
            self.prediction_cache[cache_key] = (result.copy(), time.time())
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions for a batch of transactions"""
        if not self.current_model:
            raise HTTPException(status_code=503, detail="No model available")
        
        start_time = time.time()
        results = []
        
        try:
            # Prepare all features
            feature_vectors = [self._prepare_features(features) for features in features_list]
            feature_matrix = np.array(feature_vectors)
            
            # Batch prediction
            fraud_probs = self.current_model.predict_proba(feature_matrix)[:, 1]
            is_fraud_batch = fraud_probs > 0.5
            
            # Process results
            for i, (features, fraud_prob, is_fraud) in enumerate(zip(features_list, fraud_probs, is_fraud_batch)):
                # Determine risk level
                if fraud_prob < 0.3:
                    risk_level = "LOW"
                elif fraud_prob < 0.7:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "HIGH"
                
                result = {
                    'fraud_probability': float(fraud_prob),
                    'is_fraud': bool(is_fraud),
                    'risk_level': risk_level,
                    'model_version': self.current_metadata.version,
                    'processing_time_ms': (time.time() - start_time) * 1000 / len(features_list),
                    'cached': False
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    def _prepare_features(self, features: Dict[str, Any]) -> List[float]:
        """Prepare features for model input"""
        # This should match the feature engineering pipeline
        from feature_engineering import MasterFeatureEngineer
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Apply feature engineering
        engineer = MasterFeatureEngineer()
        engineered_df = engineer.create_all_features(df, include_aggregations=False)
        
        # Select model features
        model_features = self.current_metadata.features
        feature_vector = []
        
        for feature in model_features:
            if feature in engineered_df.columns:
                value = engineered_df[feature].iloc[0]
                if pd.isna(value):
                    value = 0.0  # Handle missing values
                feature_vector.append(float(value))
            else:
                feature_vector.append(0.0)  # Default for missing features
        
        return feature_vector
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        if not self.current_metadata:
            return {"status": "no_model_loaded"}
        
        return {
            "model_id": self.current_metadata.model_id,
            "version": self.current_metadata.version,
            "name": self.current_metadata.name,
            "algorithm": self.current_metadata.algorithm,
            "status": self.current_metadata.status.value,
            "created_at": self.current_metadata.created_at.isoformat(),
            "metrics": self.current_metadata.metrics,
            "features": self.current_metadata.features,
            "model_size_mb": self.current_metadata.model_size_mb
        }
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.prediction_cache.clear()
        logger.info("Prediction cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        valid_entries = sum(1 for _, (_, timestamp) in self.prediction_cache.items() 
                           if self._is_cache_valid(timestamp))
        
        return {
            "total_entries": len(self.prediction_cache),
            "valid_entries": valid_entries,
            "cache_hit_rate": 0.0,  # Would be tracked in real implementation
            "cache_ttl_seconds": self.cache_ttl
        }