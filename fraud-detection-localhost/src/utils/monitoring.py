# src/utils/monitoring.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import redis
import json

logger = logging.getLogger(__name__)

class FraudDetectionMonitor:
    """Real-time monitoring for fraud detection system"""
    
    def __init__(self, db_connection_string: str, redis_connection_string: str):
        self.db_engine = create_engine(db_connection_string)
        self.redis_client = redis.from_url(redis_connection_string, decode_responses=True)
        self.alerts = []
        
    def log_prediction(self, transaction_id: str, prediction_data: Dict[str, Any]):
        """Log prediction to monitoring system"""
        try:
            # Store in Redis for real-time access
            prediction_key = f"prediction:{transaction_id}"
            prediction_data['timestamp'] = datetime.now().isoformat()
            self.redis_client.setex(prediction_key, timedelta(hours=24), json.dumps(prediction_data))
            
            # Update metrics
            self.update_prediction_metrics(prediction_data)
            
            logger.info(f"Logged prediction for transaction {transaction_id}")
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
    
    def update_prediction_metrics(self, prediction_data: Dict[str, Any]):
        """Update real-time prediction metrics"""
        try:
            # Increment prediction counter
            self.redis_client.incr("metrics:prediction_count")
            
            # Update fraud detection counters
            if prediction_data.get('is_fraud', False):
                self.redis_client.incr("metrics:fraud_detected_count")
            
            # Update risk level counters
            risk_level = prediction_data.get('risk_level', 'UNKNOWN')
            self.redis_client.incr(f"metrics:risk_level:{risk_level}")
            
            # Update processing time metrics
            processing_time = prediction_data.get('processing_time_ms', 0)
            self.redis_client.lpush("metrics:processing_times", processing_time)
            self.redis_client.ltrim("metrics:processing_times", 0, 999)  # Keep last 1000 values
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def check_model_drift(self, recent_predictions: List[Dict[str, Any]], 
                         threshold: float = 0.1) -> Dict[str, Any]:
        """Check for model drift in recent predictions"""
        if len(recent_predictions) < 100:
            return {"drift_detected": False, "message": "Insufficient data for drift detection"}
        
        # Convert to DataFrame
        df = pd.DataFrame(recent_predictions)
        
        # Check distribution drift
        current_fraud_rate = df['is_fraud'].mean()
        expected_fraud_rate = 0.05  # Expected baseline fraud rate
        
        drift_detected = abs(current_fraud_rate - expected_fraud_rate) > threshold
        
        drift_info = {
            "drift_detected": drift_detected,
            "current_fraud_rate": current_fraud_rate,
            "expected_fraud_rate": expected_fraud_rate,
            "drift_magnitude": abs(current_fraud_rate - expected_fraud_rate),
            "threshold": threshold,
            "sample_size": len(recent_predictions)
        }
        
        if drift_detected:
            self.create_alert("model_drift", drift_info)
            logger.warning(f"Model drift detected: {drift_info}")
        
        return drift_info
    
    def check_performance_degradation(self, recent_feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for performance degradation based on feedback"""
        if len(recent_feedback) < 50:
            return {"degradation_detected": False, "message": "Insufficient feedback data"}
        
        # Calculate recent performance metrics
        df = pd.DataFrame(recent_feedback)
        
        # Assuming feedback contains actual fraud labels
        if 'actual_fraud' in df.columns:
            recent_accuracy = (df['predicted_fraud'] == df['actual_fraud']).mean()
            expected_accuracy = 0.90  # Expected baseline accuracy
            
            degradation_detected = recent_accuracy < (expected_accuracy - 0.05)
            
            performance_info = {
                "degradation_detected": degradation_detected,
                "recent_accuracy": recent_accuracy,
                "expected_accuracy": expected_accuracy,
                "sample_size": len(recent_feedback)
            }
            
            if degradation_detected:
                self.create_alert("performance_degradation", performance_info)
                logger.warning(f"Performance degradation detected: {performance_info}")
            
            return performance_info
        
        return {"degradation_detected": False, "message": "No actual labels available"}
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        health_status = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check database connection
            with self.db_engine.connect() as conn:
                conn.execute("SELECT 1")
            health_status["checks"]["database"] = "healthy"
        except Exception as e:
            health_status["checks"]["database"] = f"unhealthy: {e}"
            health_status["status"] = "degraded"
        
        try:
            # Check Redis connection
            self.redis_client.ping()
            health_status["checks"]["redis"] = "healthy"
        except Exception as e:
            health_status["checks"]["redis"] = f"unhealthy: {e}"
            health_status["status"] = "degraded"
        
        # Check prediction volume
        try:
            prediction_count = self.redis_client.get("metrics:prediction_count")
            if prediction_count:
                health_status["checks"]["prediction_volume"] = int(prediction_count)
            else:
                health_status["checks"]["prediction_volume"] = 0
        except Exception as e:
            health_status["checks"]["prediction_volume"] = f"error: {e}"
        
        # Check processing times
        try:
            processing_times = self.redis_client.lrange("metrics:processing_times", 0, -1)
            if processing_times:
                avg_processing_time = np.mean([float(t) for t in processing_times])
                health_status["checks"]["avg_processing_time_ms"] = round(avg_processing_time, 2)
                
                if avg_processing_time > 1000:  # Alert if processing time > 1 second
                    health_status["status"] = "degraded"
                    self.create_alert("high_latency", {"avg_processing_time": avg_processing_time})
        except Exception as e:
            health_status["checks"]["processing_times"] = f"error: {e}"
        
        return health_status
    
    def create_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Create monitoring alert"""
        alert = {
            "type": alert_type,
            "timestamp": datetime.now().isoformat(),
            "data": alert_data,
            "severity": self.get_alert_severity(alert_type)
        }
        
        self.alerts.append(alert)
        
        # Store in Redis
        alert_key = f"alert:{alert_type}:{datetime.now().timestamp()}"
        self.redis_client.setex(alert_key, timedelta(days=7), json.dumps(alert))
        
        logger.warning(f"Alert created: {alert}")
    
    def get_alert_severity(self, alert_type: str) -> str:
        """Get severity level for alert type"""
        severity_map = {
            "model_drift": "high",
            "performance_degradation": "high",
            "high_latency": "medium",
            "system_error": "high",
            "data_quality": "medium"
        }
        return severity_map.get(alert_type, "low")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alerts 
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
        
        return recent_alerts
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        try:
            metrics = {}
            
            # Prediction metrics
            metrics['total_predictions'] = int(self.redis_client.get("metrics:prediction_count") or 0)
            metrics['fraud_detected'] = int(self.redis_client.get("metrics:fraud_detected_count") or 0)
            
            # Risk level distribution
            for risk_level in ['LOW', 'MEDIUM', 'HIGH']:
                count = self.redis_client.get(f"metrics:risk_level:{risk_level}")
                metrics[f'risk_level_{risk_level.lower()}'] = int(count or 0)
            
            # Processing time metrics
            processing_times = self.redis_client.lrange("metrics:processing_times", 0, -1)
            if processing_times:
                times = [float(t) for t in processing_times]
                metrics['avg_processing_time_ms'] = round(np.mean(times), 2)
                metrics['p95_processing_time_ms'] = round(np.percentile(times, 95), 2)
                metrics['p99_processing_time_ms'] = round(np.percentile(times, 99), 2)
            
            # Fraud detection rate
            if metrics['total_predictions'] > 0:
                metrics['fraud_detection_rate'] = metrics['fraud_detected'] / metrics['total_predictions']
            else:
                metrics['fraud_detection_rate'] = 0
            
            # Recent alerts
            metrics['recent_alerts'] = len(self.get_recent_alerts(24))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}