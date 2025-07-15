# src/data_processing/stream_processor.py
import json
import logging
from typing import Dict, Any
from datetime import datetime
import redis
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class StreamProcessor(ABC):
    """Abstract base class for stream processing"""
    
    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single message"""
        pass

class FraudDetectionStreamProcessor(StreamProcessor):
    """Real-time fraud detection stream processor"""
    
    def __init__(self, model_server, feature_engineer):
        self.model_server = model_server
        self.feature_engineer = feature_engineer
        self.logger = logging.getLogger(__name__)
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a transaction message for fraud detection"""
        try:
            # Extract transaction data
            transaction_data = message.get('transaction', {})
            
            # Make prediction
            prediction = await self.model_server.predict_single(transaction_data)
            
            # Add metadata
            result = {
                'transaction_id': message.get('transaction_id'),
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'original_message': message
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process message: {e}")
            return {
                'error': str(e),
                'transaction_id': message.get('transaction_id'),
                'timestamp': datetime.now().isoformat()
            }

class MessageQueue:
    """Redis-based message queue for stream processing"""
    
    def __init__(self, redis_url: str = "redis://redis:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.logger = logging.getLogger(__name__)
    
    async def publish(self, channel: str, message: Dict[str, Any]):
        """Publish message to channel"""
        try:
            message_json = json.dumps(message)
            self.redis_client.publish(channel, message_json)
            self.logger.debug(f"Published message to {channel}")
        except Exception as e:
            self.logger.error(f"Failed to publish message: {e}")
    
    async def subscribe(self, channel: str, processor: StreamProcessor):
        """Subscribe to channel and process messages"""
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(channel)
        
        self.logger.info(f"Subscribed to channel: {channel}")
        
        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        result = await processor.process_message(data)
                        
                        # Optionally publish result to output channel
                        output_channel = f"{channel}_results"
                        await self.publish(output_channel, result)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process message from {channel}: {e}")
        except KeyboardInterrupt:
            self.logger.info(f"Unsubscribing from {channel}")
            pubsub.unsubscribe(channel)
        finally:
            pubsub.close()