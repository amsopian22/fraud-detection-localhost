import requests
import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Any, Optional
import time
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Prediction result structure"""
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    processing_time_ms: float
    model_version: str
    timestamp: str

class FraudDetectionClient:
    """Client for fraud detection API"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 timeout: int = 30,
                 max_retries: int = 3,
                 api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key
        self.session = requests.Session()
        
        # Set headers
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
        self.session.headers.update({'Content-Type': 'application/json'})
        
        self.logger = logging.getLogger(__name__)
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def predict_single(self, transaction: Dict[str, Any]) -> Optional[PredictionResult]:
        """Make fraud prediction for single transaction"""
        url = f"{self.base_url}/predict"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(url, json=transaction, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                return PredictionResult(
                    transaction_id=data.get('transaction_id', 'N/A'), # Added .get for robustness
                    fraud_probability=data['fraud_probability'],
                    is_fraud=data['is_fraud'],
                    risk_level=data['risk_level'],
                    processing_time_ms=data['processing_time_ms'],
                    model_version=data['model_version'],
                    timestamp=datetime.now().isoformat()
                )
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    self.logger.error("All attempts failed for single prediction")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def predict_batch(self, transactions: List[Dict[str, Any]], 
                     batch_size: int = 100) -> List[Optional[PredictionResult]]:
        """Make fraud predictions for batch of transactions"""
        results = []
        
        # Process in batches
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            batch_request = {"transactions": batch}
            
            url = f"{self.base_url}/predict/batch"
            
            for attempt in range(self.max_retries):
                try:
                    response = self.session.post(url, json=batch_request, timeout=self.timeout * 2)
                    response.raise_for_status()
                    
                    data = response.json()
                    batch_results = []
                    
                    for pred_data in data.get('predictions', []):
                        batch_results.append(PredictionResult(
                            transaction_id=pred_data.get('transaction_id', 'N/A'),
                            fraud_probability=pred_data['fraud_probability'],
                            is_fraud=pred_data['is_fraud'],
                            risk_level=pred_data['risk_level'],
                            processing_time_ms=pred_data['processing_time_ms'],
                            model_version=pred_data['model_version'],
                            timestamp=datetime.now().isoformat()
                        ))
                    
                    results.extend(batch_results)
                    break
                    
                except requests.exceptions.RequestException as e:
                    self.logger.warning(f"Batch attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        self.logger.error(f"All attempts failed for batch {i//batch_size + 1}")
                        results.extend([None] * len(batch))
                    time.sleep(2 ** attempt)
        
        return results
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get current model information"""
        try:
            response = self.session.get(f"{self.base_url}/model/info", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return None
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get API metrics"""
        try:
            response = self.session.get(f"{self.base_url}/metrics", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            return None
    
    def predict_from_csv(self, csv_file: str, output_file: str = None) -> pd.DataFrame:
        """Make predictions from CSV file"""
        self.logger.info(f"Processing CSV file: {csv_file}")
        
        # Load data
        df = pd.read_csv(csv_file)
        transactions = df.to_dict('records')
        
        # Make predictions
        results = self.predict_batch(transactions)
        
        # Create results DataFrame
        results_data = []
        for i, result in enumerate(results):
            row = transactions[i].copy()
            if result:
                row.update({
                    'predicted_fraud_probability': result.fraud_probability,
                    'predicted_is_fraud': result.is_fraud,
                    'predicted_risk_level': result.risk_level,
                    'prediction_processing_time_ms': result.processing_time_ms,
                    'model_version': result.model_version,
                    'prediction_timestamp': result.timestamp
                })
            else:
                row.update({
                    'predicted_fraud_probability': None,
                    'predicted_is_fraud': None,
                    'predicted_risk_level': None,
                    'prediction_processing_time_ms': None,
                    'model_version': None,
                    'prediction_timestamp': datetime.now().isoformat(),
                    'prediction_error': True
                })
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Save if output file specified
        if output_file:
            results_df.to_csv(output_file, index=False)
            self.logger.info(f"Results saved to: {output_file}")
        
        return results_df

class AsyncFraudDetectionClient:
    """Async client for high-performance operations"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 timeout: int = 30,
                 max_concurrent: int = 10,
                 api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_concurrent = max_concurrent
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
    
    async def predict_batch_async(self, transactions: List[Dict[str, Any]]) -> List[Optional[PredictionResult]]:
        """Make async batch predictions"""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def predict_single_async(transaction):
            async with semaphore:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    try:
                        async with session.post(
                            f"{self.base_url}/predict",
                            json=transaction,
                            headers=headers
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                return PredictionResult(
                                    transaction_id=data.get('transaction_id', 'N/A'),
                                    fraud_probability=data['fraud_probability'],
                                    is_fraud=data['is_fraud'],
                                    risk_level=data['risk_level'],
                                    processing_time_ms=data['processing_time_ms'],
                                    model_version=data['model_version'],
                                    timestamp=datetime.now().isoformat()
                                )
                            else:
                                self.logger.error(f"API error: {response.status}")
                                return None
                    except Exception as e:
                        self.logger.error(f"Async prediction failed: {e}")
                        return None
        
        # Execute all predictions concurrently
        tasks = [predict_single_async(transaction) for transaction in transactions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Task failed: {result}")
                final_results.append(None)
            else:
                final_results.append(result)
        
        return final_results