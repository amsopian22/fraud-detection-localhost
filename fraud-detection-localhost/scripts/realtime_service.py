#!/usr/bin/env python3
"""
Real-time Fraud Detection Simulation Service
Continuously generates transactions and makes predictions
"""

import asyncio
import time
import random
import json
import logging
from datetime import datetime, timedelta
from typing import Dict
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
import redis
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealtimeFraudSimulator:
    def __init__(self):
        self.model = None
        self.feature_encoder = None
        self.feature_names = None
        self.label_encoders = None
        
        # Database and Redis
        self.db_engine = None
        self.redis_client = None
        
        # Simulation parameters
        self.transactions_per_minute = 10  # Adjustable rate
        self.fraud_rate = 0.02  # 2% fraud rate
        self.batch_size = 5
        
        # Transaction templates
        self.merchants = [
            "Amazon Store", "Walmart", "Target", "CVS Pharmacy", "Starbucks",
            "McDonald's", "Shell Gas", "Home Depot", "Best Buy", "Whole Foods",
            "Apple Store", "Netflix", "Uber", "DoorDash", "PayPal"
        ]
        
        self.categories = [
            "grocery_pos", "gas_transport", "misc_net", "entertainment", 
            "misc_pos", "grocery_net", "shopping_net", "food_dining"
        ]
        
        self.states = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
        self.jobs = ["Engineer", "Teacher", "Doctor", "Sales", "Manager", "Nurse", "Lawyer"]
        
        # Initialize connections and load model
        self._init_connections()
        self._load_model()
    
    def _init_connections(self):
        """Initialize database and Redis connections"""
        try:
            # Database
            db_url = os.getenv('DATABASE_URL', 'postgresql://frauduser:fraudpass123@postgres:5432/frauddb')
            self.db_engine = create_engine(db_url, pool_pre_ping=True)
            logger.info("✅ Database connection established")
            
            # Redis
            self.redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
    
    def _load_model(self):
        """Load trained model and artifacts"""
        try:
            model_path = '/app/models/trained_models'
            
            # Load model
            self.model = joblib.load(f'{model_path}/xgboost_fraud_detector.joblib')
            self.feature_encoder = joblib.load(f'{model_path}/feature_scaler.joblib')  # FIXED: correct filename
            self.feature_names = joblib.load(f'{model_path}/feature_names.joblib')
            
            # Load label encoders if they exist
            try:
                self.label_encoders = joblib.load(f'{model_path}/label_encoders.joblib')
                logger.info(f"✅ Label encoders loaded: {list(self.label_encoders.keys())}")
            except Exception as e:
                logger.warning(f"⚠️ Label encoders not found: {e}")
                self.label_encoders = {}
            
            logger.info(f"✅ Model loaded with features: {self.feature_names}")
            logger.info("✅ Model and artifacts loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    def generate_transaction(self, force_fraud: bool = False) -> Dict:
        """Generate a realistic transaction - FIXED VERSION"""
        # Basic transaction info
        transaction_id = f"sim_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        # Determine if this should be fraudulent
        is_simulated_fraud = force_fraud or random.random() < self.fraud_rate
        
        # Generate amounts (fraudulent ones tend to be higher) - FIXED
        if is_simulated_fraud:
            # Fraudulent transactions: higher amounts, round numbers
            amt = random.choice([
                round(random.uniform(500, 5000), 2),    # High amounts
                round(random.uniform(1000, 10000), 0),  # Round numbers
                9999.99, 4999.99, 2999.99             # Suspicious amounts
            ])
        else:
            # Normal transactions: realistic distribution - FIXED
            # Use numpy lognormal instead of random.lognormal
            amt = round(np.random.lognormal(3.5, 1.2), 2)  # Log-normal distribution
            amt = max(1.0, min(amt, 500.0))  # Cap at reasonable amount
        
        # Geographic data
        if is_simulated_fraud:
            # Fraudulent: more distance between customer and merchant
            customer_lat = round(random.uniform(25, 45), 6)
            customer_long = round(random.uniform(-125, -65), 6)
            # Merchant far away
            merchant_lat = round(customer_lat + random.uniform(-10, 10), 6)
            merchant_long = round(customer_long + random.uniform(-15, 15), 6)
        else:
            # Normal: customer and merchant close
            customer_lat = round(random.uniform(32, 42), 6)
            customer_long = round(random.uniform(-120, -75), 6)
            # Merchant nearby
            merchant_lat = round(customer_lat + random.uniform(-0.5, 0.5), 6)
            merchant_long = round(customer_long + random.uniform(-0.5, 0.5), 6)
        
        # Time-based features
        now = datetime.now()
        if is_simulated_fraud:
            # Fraudulent: more likely at night or weekends
            if random.random() < 0.6:  # 60% chance to be at suspicious time
                hour = random.choice(list(range(0, 6)) + list(range(23, 24)))
            else:
                hour = random.randint(0, 23)
        else:
            # Normal: business hours more likely
            hour = random.choices(
                range(24), 
                weights=[1,1,1,1,1,1,2,4,6,8,10,10,10,8,8,8,8,6,4,3,2,2,1,1]
            )[0]
        
        # Customer info
        first_names = ["John", "Jane", "Michael", "Sarah", "David", "Emily", "Chris", "Jessica"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller"]
        
        transaction = {
            "cc_num": f"4000{random.randint(100000000000, 999999999999)}",
            "merchant": random.choice(self.merchants),
            "category": random.choice(self.categories),
            "amt": amt,
            "first": random.choice(first_names),
            "last": random.choice(last_names),
            "gender": random.choice(["M", "F"]),
            "street": f"{random.randint(1, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Elm'])} St",
            "city": f"City_{random.randint(1, 100)}",
            "state": random.choice(self.states),
            "zip": f"{random.randint(10000, 99999)}",
            "lat": customer_lat,
            "long": customer_long,
            "city_pop": random.randint(1000, 1000000),
            "job": random.choice(self.jobs),
            "dob": f"19{random.randint(50, 95)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            "merch_lat": merchant_lat,
            "merch_long": merchant_long,
            "merch_zipcode": f"{random.randint(10000, 99999)}",
            "trans_date_trans_time": now.strftime('%Y-%m-%d %H:%M:%S'),
            "_simulated_fraud": is_simulated_fraud  # For verification
        }
        
        return transaction

    def engineer_features(self, transaction: Dict) -> Dict:
        """Engineer features for prediction (compatible with both approaches)"""
        features = {}
        
        # Determine approach based on feature names
        is_enhanced = any('transaction_' in name for name in self.feature_names) if self.feature_names else False
        
        if is_enhanced:
            # Enhanced approach features (notebook method)
            features['amt'] = float(transaction.get('amt', 0))
            features['lat'] = float(transaction.get('lat', 0))
            features['long'] = float(transaction.get('long', 0))
            features['merch_lat'] = float(transaction.get('merch_lat', 0))
            features['merch_long'] = float(transaction.get('merch_long', 0))
            
            # Time features
            try:
                dt = pd.to_datetime(transaction.get('trans_date_trans_time', datetime.now()))
                features['hour'] = dt.hour
                features['day'] = dt.day
                features['month'] = dt.month
            except:
                now = datetime.now()
                features['hour'] = now.hour
                features['day'] = now.day
                features['month'] = now.month
            
            # Age calculation (2025 reference)
            try:
                dob = pd.to_datetime(transaction.get('dob', '1980-01-01'))
                features['age'] = 2025 - dob.year
            except:
                features['age'] = 40  # Default age
            
            # Per-card statistics (simulated for real-time)
            features['transaction_time'] = random.uniform(0.1, 24.0)  # Hours since last transaction
            features['transaction_std'] = random.uniform(10, 500)     # Amount std for this card
            features['transaction_avg'] = random.uniform(50, 300)     # Amount avg for this card
            
            # Categorical encoding for enhanced approach
            for col in ['merchant', 'category', 'gender']:
                if col in transaction and col in self.label_encoders:
                    try:
                        val = str(transaction[col])
                        if val in self.label_encoders[col].classes_:
                            features[col] = self.label_encoders[col].transform([val])[0]
                        else:
                            features[col] = -1  # Unknown category
                    except:
                        features[col] = -1
                else:
                    features[col] = -1
                    
        else:
            # Original approach features
            features['amt'] = float(transaction.get('amt', 0))
            features['lat'] = float(transaction.get('lat', 0))
            features['long'] = float(transaction.get('long', 0))
            features['merch_lat'] = float(transaction.get('merch_lat', 0))
            features['merch_long'] = float(transaction.get('merch_long', 0))
            features['city_pop'] = float(transaction.get('city_pop', 10000))
            
            # Time features
            try:
                dt = pd.to_datetime(transaction.get('trans_date_trans_time', datetime.now()))
                features['hour'] = dt.hour
                features['day_of_week'] = dt.dayofweek
            except:
                now = datetime.now()
                features['hour'] = now.hour
                features['day_of_week'] = now.weekday()
            
            # Engineered features
            features['amt_log'] = np.log1p(features['amt'])
            features['city_pop_log'] = np.log1p(features['city_pop'])
            
            # Haversine distance calculation
            def haversine_distance(lat1, lon1, lat2, lon2):
                lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                return c * 6371  # Earth radius in km
            
            features['customer_merchant_distance'] = haversine_distance(
                features['lat'], features['long'], 
                features['merch_lat'], features['merch_long']
            )
            
            # Time-based features
            features['is_weekend'] = 1 if features['day_of_week'] >= 5 else 0
            features['is_night'] = 1 if (features['hour'] < 6 or features['hour'] > 22) else 0
            
            # Age calculation
            try:
                dob = pd.to_datetime(transaction.get('dob', '1980-01-01'))
                features['customer_age'] = (datetime.now() - dob).days / 365.25
            except:
                features['customer_age'] = 40  # Default age
            
            # Categorical encoding for original approach
            for col in ['gender', 'category', 'state']:
                if col in transaction and col in self.label_encoders:
                    try:
                        val = str(transaction[col])
                        if val in self.label_encoders[col].classes_:
                            features[f'{col}_encoded'] = self.label_encoders[col].transform([val])[0]
                        else:
                            features[f'{col}_encoded'] = 0  # Unknown category
                    except:
                        features[f'{col}_encoded'] = 0
                else:
                    features[f'{col}_encoded'] = 0
        
        return features
    
    def predict_fraud(self, transaction: Dict) -> Dict:
        """Make fraud prediction"""
        start_time = time.time()
        
        try:
            # Engineer features
            features = self.engineer_features(transaction)
            
            # Use the exact feature names from the trained model
            if self.feature_names:
                feature_vector = []
                for name in self.feature_names:
                    feature_vector.append(features.get(name, 0))
                logger.debug(f"Using model features: {self.feature_names}")
                logger.debug(f"Feature vector: {feature_vector}")
            else:
                # Fallback to expected features if feature_names not available
                logger.warning("Feature names not available, using fallback")
                expected_features = ['amt', 'lat', 'long', 'merch_lat', 'merch_long', 'city_pop', 'unix_time',
                                   'amt_log', 'city_pop_log', 'lat_diff', 'long_diff', 'hour', 'day_of_week',
                                   'category_encoded', 'gender_encoded', 'state_encoded']
                
                feature_vector = []
                for name in expected_features:
                    feature_vector.append(features.get(name, 0))
            
            # Make prediction
            X = np.array([feature_vector])
            X_scaled = self.feature_encoder.transform(X)
            
            fraud_prob = float(self.model.predict_proba(X_scaled)[0][1])
            
            # Use threshold based on approach (0.4 for enhanced, 0.5 for original)
            is_enhanced = any('transaction_' in name for name in self.feature_names) if self.feature_names else False
            threshold = 0.4 if is_enhanced else 0.5
            is_fraud = fraud_prob > threshold
            
            # Risk level
            if fraud_prob < 0.3:
                risk_level = "LOW"
            elif fraud_prob < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                'transaction_id': f"pred_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
                'fraud_probability': fraud_prob,
                'is_fraud': is_fraud,
                'risk_level': risk_level,
                'confidence_score': max(fraud_prob, 1 - fraud_prob),
                'processing_time_ms': processing_time,
                'model_version': 'v4.0.0',
                'threshold_used': threshold,
                'approach': 'enhanced' if is_enhanced else 'original'
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                'transaction_id': f"error_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
                'fraud_probability': 0.0,
                'is_fraud': False,
                'risk_level': "LOW",
                'confidence_score': 0.0,
                'processing_time_ms': 0,
                'model_version': 'error'
            }
    
    def store_prediction(self, transaction: Dict, prediction: Dict):
        """Store prediction in database and Redis"""
        try:
            if not self.db_engine:
                return
            
            # Prepare data for database
            insert_data = {
                'transaction_id': prediction['transaction_id'],
                'amount': transaction['amt'],
                'merchant_name': transaction['merchant'],
                'category': transaction['category'],
                'customer_location': f"{transaction['city']}, {transaction['state']}",
                'merchant_location': f"{transaction['merch_lat']}, {transaction['merch_long']}",
                'fraud_probability': prediction['fraud_probability'],
                'is_fraud': prediction['is_fraud'],
                'risk_level': prediction['risk_level'],
                'confidence_score': prediction['confidence_score'],
                'model_version': prediction['model_version'],
                'processing_time_ms': prediction['processing_time_ms'],
                'transaction_data': json.dumps(transaction)
            }
            
            # Insert into database
            insert_sql = """
            INSERT INTO realtime_predictions (
                transaction_id, amount, merchant_name, category, 
                customer_location, merchant_location, fraud_probability, 
                is_fraud, risk_level, confidence_score, model_version, 
                processing_time_ms, transaction_data
            ) VALUES (
                :transaction_id, :amount, :merchant_name, :category,
                :customer_location, :merchant_location, :fraud_probability,
                :is_fraud, :risk_level, :confidence_score, :model_version,
                :processing_time_ms, :transaction_data
            )
            """
            
            with self.db_engine.connect() as conn:
                conn.execute(text(insert_sql), insert_data)
                conn.commit()
            
            # Store in Redis for real-time access
            if self.redis_client:
                redis_data = {**transaction, **prediction, 'timestamp': datetime.now().isoformat()}
                self.redis_client.setex(
                    f"realtime:{prediction['transaction_id']}", 
                    timedelta(hours=1), 
                    json.dumps(redis_data)
                )
                
                # Update counters
                self.redis_client.incr("realtime:total_predictions")
                if prediction['is_fraud']:
                    self.redis_client.incr("realtime:fraud_detected")
                    
        except Exception as e:
            if "duplicate key value violates unique constraint" in str(e):
                logger.warning(f"Duplicate transaction ID skipped: {prediction['transaction_id']}")
            else:
                logger.error(f"Storage error: {e}")
    
    def run_simulation_batch(self):
        """Run a batch of transaction simulations"""
        transactions = []
        predictions = []
        
        for i in range(self.batch_size):
            # Generate transaction
            transaction = self.generate_transaction()
            
            # Make prediction
            prediction = self.predict_fraud(transaction)
            
            # Store result
            self.store_prediction(transaction, prediction)
            
            transactions.append(transaction)
            predictions.append(prediction)
            
            # Log interesting cases
            if prediction['is_fraud']:
                logger.warning(
                    f"🚨 FRAUD DETECTED - Amount: ${transaction['amt']:.2f}, "
                    f"Probability: {prediction['fraud_probability']:.4f}, "
                    f"Merchant: {transaction['merchant']}"
                )
        
        # Batch summary
        fraud_count = sum(1 for p in predictions if p['is_fraud'])
        avg_prob = np.mean([p['fraud_probability'] for p in predictions])
        avg_amount = np.mean([t['amt'] for t in transactions])
        
        logger.info(
            f"📊 Batch processed: {len(predictions)} transactions, "
            f"{fraud_count} frauds detected, "
            f"avg probability: {avg_prob:.4f}, "
            f"avg amount: ${avg_amount:.2f}"
        )
        
        return transactions, predictions
    
    async def run_continuous_simulation(self):
        """Run continuous simulation"""
        logger.info("🚀 Starting continuous fraud detection simulation...")
        
        # Setup database schema
        try:
            schema_sql = open('/app/sql/realtime_schema.sql', 'r').read()
            with self.db_engine.connect() as conn:
                conn.execute(text(schema_sql))
                conn.commit()
            logger.info("✅ Database schema initialized")
        except Exception as e:
            logger.warning(f"⚠️  Schema setup warning: {e}")
        
        batch_interval = 60 / self.transactions_per_minute * self.batch_size  # seconds
        
        while True:
            try:
                start_time = time.time()
                
                # Run simulation batch
                transactions, predictions = self.run_simulation_batch()
                
                # Calculate sleep time to maintain rate
                elapsed = time.time() - start_time
                sleep_time = max(0, batch_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except KeyboardInterrupt:
                logger.info("🛑 Simulation stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Simulation error: {e}")
                await asyncio.sleep(5)  # Wait before retry

async def main():
    """Main function"""
    simulator = RealtimeFraudSimulator()
    
    if not simulator.model:
        logger.error("❌ Model not loaded. Please train model first.")
        return
    
    print("🎯 Real-time Fraud Detection Simulator")
    print(f"📊 Rate: {simulator.transactions_per_minute} transactions/minute")
    print(f"🎲 Fraud rate: {simulator.fraud_rate:.1%}")
    print("🔄 Starting continuous simulation...")
    print("Press Ctrl+C to stop")
    
    await simulator.run_continuous_simulation()

if __name__ == "__main__":
    asyncio.run(main())