# src/models/model_validator.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ModelValidator:
    """Model validation and testing utilities"""
    
    def __init__(self):
        self.validation_thresholds = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77,
            'roc_auc': 0.85
        }
    
    def validate_model_performance(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Validate model performance against thresholds"""
        
        logger.info("Starting model validation...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        # Check thresholds
        validation_results = {}
        overall_passed = True
        
        for metric, value in metrics.items():
            threshold = self.validation_thresholds.get(metric, 0.0)
            passed = value >= threshold
            overall_passed = overall_passed and passed
            
            validation_results[metric] = {
                'value': value,
                'threshold': threshold,
                'passed': passed
            }
        
        validation_results['overall_passed'] = overall_passed
        validation_results['sample_size'] = len(y_test)
        validation_results['fraud_rate'] = y_test.mean()
        
        logger.info(f"Validation completed. Overall passed: {overall_passed}")
        return validation_results
    
    def validate_model_stability(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                                n_iterations: int = 10) -> Dict[str, Any]:
        """Test model stability across multiple random samples"""
        
        logger.info(f"Testing model stability with {n_iterations} iterations...")
        
        stability_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc': []
        }
        
        for i in range(n_iterations):
            # Random sample
            indices = np.random.choice(len(X_test), size=min(1000, len(X_test)), replace=False)
            X_sample = X_test[indices]
            y_sample = y_test[indices]
            
            # Make predictions
            y_pred = model.predict(X_sample)
            y_prob = model.predict_proba(X_sample)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Calculate metrics
            if len(np.unique(y_sample)) > 1:  # Ensure both classes present
                stability_metrics['accuracy'].append(accuracy_score(y_sample, y_pred))
                stability_metrics['precision'].append(precision_score(y_sample, y_pred, zero_division=0))
                stability_metrics['recall'].append(recall_score(y_sample, y_pred, zero_division=0))
                stability_metrics['f1_score'].append(f1_score(y_sample, y_pred, zero_division=0))
                stability_metrics['roc_auc'].append(roc_auc_score(y_sample, y_prob))
        
        # Calculate stability statistics
        stability_results = {}
        for metric, values in stability_metrics.items():
            if values:
                stability_results[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                }
        
        # Overall stability score (lower CV is better)
        cv_scores = [result['coefficient_of_variation'] for result in stability_results.values()]
        stability_results['overall_stability_score'] = 1.0 - np.mean(cv_scores)  # Higher is more stable
        
        logger.info(f"Stability testing completed. Overall stability: {stability_results['overall_stability_score']:.3f}")
        return stability_results
    
    def validate_feature_importance(self, model, feature_names: List[str]) -> Dict[str, Any]:
        """Validate feature importance and detect potential issues"""
        
        if not hasattr(model, 'feature_importances_'):
            return {"status": "no_feature_importance", "message": "Model does not support feature importance"}
        
        importance = model.feature_importances_
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Analysis
        total_importance = importance.sum()
        top_10_importance = importance_df.head(10)['importance'].sum()
        
        results = {
            'total_features': len(feature_names),
            'non_zero_features': (importance > 0).sum(),
            'top_10_importance_ratio': top_10_importance / total_importance if total_importance > 0 else 0,
            'max_importance': importance.max(),
            'min_importance': importance.min(),
            'importance_std': importance.std(),
            'top_features': importance_df.head(10).to_dict('records')
        }
        
        # Warnings
        warnings = []
        if results['top_10_importance_ratio'] > 0.9:
            warnings.append("High concentration of importance in top 10 features")
        
        if results['non_zero_features'] < len(feature_names) * 0.1:
            warnings.append("Very few features have non-zero importance")
        
        results['warnings'] = warnings
        
        return results