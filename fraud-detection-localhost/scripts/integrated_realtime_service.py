#!/usr/bin/env python3
"""
Integrated Real-time Fraud Detection Service
Generates transactions, makes predictions, and stores results for dashboard consumption
"""

import asyncio
import aiohttp
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List
import logging
import uuid
from sqlalchemy import create_engine, text
import redis
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegratedRealtimeService:
    def __init__(self):
        self.api_url = "http://ml-api:8000"  # Use internal service name
        self.session = None
        
        # Database and Redis connections
        self.db_engine = None
        self.redis_client = None
        
        # Service configuration
        self.transactions_per_minute = 6  # 1 transaction every 10 seconds
        self.fraud_rate = 0.15  # 15% should be fraud
        
        # Running statistics
        self.stats = {
            'total_transactions': 0,
            'fraud_detected': 0,
            'start_time': time.time(),
            'last_batch_time': time.time()
        }
        
        # Sample data for transaction generation
        self.setup_sample_data()
        
    def setup_sample_data(self):
        """Setup sample data for realistic transaction generation"""
        self.merchants = [
            "Amazon Store", "Walmart", "Target", "CVS Pharmacy", "Starbucks",
            "McDonald's", "Shell Gas", "Home Depot", "Best Buy", "Whole Foods",
            "Apple Store", "Netflix", "Uber", "DoorDash", "PayPal",
            # Fraud merchants  
            "fraud_Rippin, Kub and Mann", "fraud_Sporer-Keebler", "fraud_Haley Group"
        ]
        
        self.categories = [
            "grocery_pos", "gas_transport", "misc_net", "grocery_net", "shopping_pos",
            "entertainment", "personal_care", "health_fitness", "travel", "kids_pets"
        ]
        
        self.states = ["NY", "CA", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
        self.genders = ["M", "F"]
        
        self.names = {
            "M": [("John", "Smith"), ("Michael", "Johnson"), ("David", "Williams"), 
                  ("James", "Brown"), ("Robert", "Davis"), ("William", "Miller")],
            "F": [("Mary", "Jones"), ("Jennifer", "Garcia"), ("Linda", "Rodriguez"), 
                  ("Elizabeth", "Wilson"), ("Susan", "Martinez"), ("Jessica", "Anderson")]
        }

    async def setup_connections(self):
        """Setup database, Redis, and HTTP connections"""
        # HTTP session
        self.session = aiohttp.ClientSession()
        
        # Database connection
        try:
            db_url = 'postgresql://frauduser:fraudpass123@postgres:5432/frauddb'
            self.db_engine = create_engine(db_url)
            
            # Create tables if they don't exist
            await self.create_tables()
            logger.info("✅ Database connected and tables ready")
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            
        # Redis connection
        try:
            self.redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")

    async def create_tables(self):
        """Create database tables for storing transaction data"""
        create_sql = """
        CREATE TABLE IF NOT EXISTS realtime_transactions (
            id SERIAL PRIMARY KEY,
            transaction_id VARCHAR(50) UNIQUE NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            cc_num VARCHAR(16),
            merchant VARCHAR(100),
            category VARCHAR(50),
            amount DECIMAL(10,2),
            customer_name VARCHAR(100),
            gender CHAR(1),
            city VARCHAR(50),
            state VARCHAR(2),
            fraud_probability DECIMAL(8,6),
            is_fraud BOOLEAN,
            risk_level VARCHAR(10),
            processing_time_ms DECIMAL(8,2)
        );
        
        CREATE INDEX IF NOT EXISTS idx_realtime_transactions_timestamp 
        ON realtime_transactions(timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_realtime_transactions_fraud 
        ON realtime_transactions(is_fraud);
        """
        
        try:
            with self.db_engine.connect() as conn:
                conn.execute(text(create_sql))
                conn.commit()
        except Exception as e:
            logger.error(f"Error creating tables: {e}")

    def generate_transaction(self) -> Dict:
        """Generate a realistic transaction with controlled fraud probability"""
        # Determine if this should be fraud
        should_be_fraud = random.random() < self.fraud_rate
        
        # Basic transaction data
        gender = random.choice(self.genders)
        first, last = random.choice(self.names[gender])
        state = random.choice(self.states)
        
        # Choose merchant and amount based on fraud flag
        if should_be_fraud:
            # Higher chance of fraud merchants and larger amounts
            if random.random() < 0.7:  # 70% chance to use fraud merchant
                merchant = random.choice([m for m in self.merchants if "fraud_" in m])
            else:
                merchant = random.choice([m for m in self.merchants if "fraud_" not in m])
            amount = random.uniform(500, 8000)  # Larger suspicious amounts
        else:
            # Normal transaction
            merchant = random.choice([m for m in self.merchants if "fraud_" not in m])
            amount = random.uniform(5, 500)  # Normal amounts
        
        # Generate geographic data (simplified US coordinates)
        lat = random.uniform(25.0, 49.0)
        long = random.uniform(-125.0, -66.0)
        
        return {
            "cc_num": f"{random.randint(1000000000000000, 9999999999999999)}",
            "merchant": merchant,
            "category": random.choice(self.categories),
            "amt": round(amount, 2),
            "first": first,
            "last": last,
            "gender": gender,
            "street": f"{random.randint(1, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Elm'])} St",
            "city": f"City{random.randint(1, 100)}",
            "state": state,
            "zip": f"{random.randint(10000, 99999)}",
            "lat": round(lat, 4),
            "long": round(long, 4),
            "city_pop": random.randint(1000, 1000000),
            "job": random.choice(["Engineer", "Teacher", "Doctor", "Manager", "Artist"]),
            "dob": f"{random.randint(1960, 2000)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            "merch_lat": round(lat + random.uniform(-0.1, 0.1), 4),
            "merch_long": round(long + random.uniform(-0.1, 0.1), 4),
            "merch_zipcode": f"{random.randint(10000, 99999)}"
        }

    async def predict_transaction(self, transaction: Dict) -> Dict:
        """Make fraud prediction via ML API"""
        try:
            async with self.session.post(
                f"{self.api_url}/predict",
                json=transaction,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    logger.error(f"API Error {response.status}: {error}")
                    return {"error": f"API returned status {response.status}"}
        except Exception as e:
            logger.error(f"Prediction request failed: {e}")
            return {"error": str(e)}

    async def store_transaction(self, transaction: Dict, prediction: Dict):
        """Store transaction and prediction results in database and Redis"""
        if 'error' in prediction:
            return
            
        try:
            # Generate transaction ID
            transaction_id = prediction.get('transaction_id', f"txn_{int(time.time() * 1000)}")
            
            # Store in database
            insert_sql = """
            INSERT INTO realtime_transactions 
            (transaction_id, cc_num, merchant, category, amount, customer_name, 
             gender, city, state, fraud_probability, is_fraud, risk_level, processing_time_ms)
            VALUES (:tid, :cc_num, :merchant, :category, :amount, :name, 
                    :gender, :city, :state, :fraud_prob, :is_fraud, :risk_level, :proc_time)
            """
            
            customer_name = f"{transaction['first']} {transaction['last']}"
            
            with self.db_engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'tid': transaction_id,
                    'cc_num': transaction['cc_num'][-4:],  # Only last 4 digits
                    'merchant': transaction['merchant'],
                    'category': transaction['category'],
                    'amount': transaction['amt'],
                    'name': customer_name,
                    'gender': transaction['gender'],
                    'city': transaction['city'],
                    'state': transaction['state'],
                    'fraud_prob': prediction.get('fraud_probability', 0),
                    'is_fraud': prediction.get('is_fraud', False),
                    'risk_level': prediction.get('risk_level', 'LOW'),
                    'proc_time': prediction.get('processing_time_ms', 0)
                })
                conn.commit()
            
            # Store recent transactions in Redis for fast dashboard access
            redis_data = {
                'transaction_id': transaction_id,
                'timestamp': datetime.now().isoformat(),
                'merchant': transaction['merchant'],
                'amount': transaction['amt'],
                'fraud_probability': prediction.get('fraud_probability', 0),
                'is_fraud': prediction.get('is_fraud', False),
                'risk_level': prediction.get('risk_level', 'LOW')
            }
            
            if self.redis_client:
                # Store individual transaction
                self.redis_client.setex(
                    f"transaction:{transaction_id}", 
                    3600,  # 1 hour TTL
                    json.dumps(redis_data)
                )
                
                # Add to recent transactions list (keep last 100)
                self.redis_client.lpush("recent_transactions", json.dumps(redis_data))
                self.redis_client.ltrim("recent_transactions", 0, 99)
                
        except Exception as e:
            logger.error(f"Error storing transaction: {e}")

    async def update_dashboard_metrics(self):
        """Update dashboard metrics in database"""
        try:
            # Calculate metrics from recent data
            query = """
            SELECT 
                COUNT(*) as total_predictions,
                COUNT(*) FILTER (WHERE is_fraud = true) as total_frauds,
                AVG(amount) as avg_transaction_amount,
                AVG(fraud_probability) as avg_fraud_probability,
                AVG(processing_time_ms) as avg_processing_time,
                COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '1 hour') as predictions_last_hour,
                COUNT(*) FILTER (WHERE is_fraud = true AND timestamp > NOW() - INTERVAL '1 hour') as frauds_last_hour,
                COUNT(*) FILTER (WHERE risk_level = 'HIGH') as high_risk_count,
                COUNT(*) FILTER (WHERE risk_level = 'MEDIUM') as medium_risk_count,
                COUNT(*) FILTER (WHERE risk_level = 'LOW') as low_risk_count
            FROM realtime_transactions 
            WHERE timestamp > NOW() - INTERVAL '24 hours'
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
                
                if result and result.total_predictions > 0:
                    # Clear and update dashboard metrics
                    conn.execute(text("DELETE FROM dashboard_metrics"))
                    
                    update_sql = """
                    INSERT INTO dashboard_metrics 
                    (total_predictions, total_frauds, avg_fraud_probability, avg_transaction_amount, 
                     avg_processing_time, predictions_last_hour, frauds_last_hour, 
                     high_risk_count, medium_risk_count, low_risk_count)
                    VALUES (:total, :frauds, :prob, :amount, :time, :pred_hour, :fraud_hour,
                            :high, :medium, :low)
                    """
                    
                    conn.execute(text(update_sql), {
                        'total': result.total_predictions,
                        'frauds': result.total_frauds,
                        'prob': float(result.avg_fraud_probability or 0),
                        'amount': float(result.avg_transaction_amount or 0),
                        'time': float(result.avg_processing_time or 0),
                        'pred_hour': result.predictions_last_hour,
                        'fraud_hour': result.frauds_last_hour,
                        'high': result.high_risk_count,
                        'medium': result.medium_risk_count,
                        'low': result.low_risk_count
                    })
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error updating dashboard metrics: {e}")

    def print_transaction_table(self, transaction: Dict, prediction: Dict):
        """Print transaction in table format"""
        if 'error' in prediction:
            print(f"❌ ERROR | {transaction['merchant'][:25]:<25} | ${transaction['amt']:>8.2f} | {prediction.get('error', 'Unknown error')}")
            return
            
        fraud_prob = prediction.get('fraud_probability', 0)
        risk_level = prediction.get('risk_level', 'UNKNOWN')
        is_fraud = prediction.get('is_fraud', False)
        processing_time = prediction.get('processing_time_ms', 0)
        
        # Color coding
        if is_fraud or fraud_prob > 0.7:
            status = "🚨 FRAUD"
            color = "\033[91m"  # Red
        elif fraud_prob > 0.3:
            status = "⚠️  SUSP"
            color = "\033[93m"  # Yellow
        else:
            status = "✅ NORM"
            color = "\033[92m"  # Green
        
        reset_color = "\033[0m"
        
        print(f"{color}{status}{reset_color} | "
              f"{transaction['merchant'][:25]:<25} | "
              f"${transaction['amt']:>8.2f} | "
              f"{fraud_prob:>6.3f} | "
              f"{risk_level:<6} | "
              f"{processing_time:>5.1f}ms | "
              f"{transaction['first']} {transaction['last']}")

    def print_stats(self):
        """Print running statistics"""
        runtime = time.time() - self.stats['start_time']
        fraud_rate = (self.stats['fraud_detected'] / 
                     max(self.stats['total_transactions'], 1) * 100)
        tps = self.stats['total_transactions'] / max(runtime, 1) * 60  # per minute
        
        print(f"\n📊 REALTIME STATS (Runtime: {runtime:.0f}s)")
        print(f"   Transactions: {self.stats['total_transactions']}")
        print(f"   Frauds: {self.stats['fraud_detected']} ({fraud_rate:.1f}%)")
        print(f"   Rate: {tps:.1f} transactions/minute")
        print(f"   Dashboard: http://localhost:8501")

    async def run_continuous_service(self):
        """Run continuous realtime fraud detection service"""
        print("🚀 Starting Integrated Real-time Fraud Detection Service")
        print("   📊 Dashboard: http://localhost:8501")
        print("   🎯 Rate: ~6 transactions/minute")
        print("   🔍 Fraud rate: ~15%")
        print("   Press Ctrl+C to stop\n")
        
        # Print table header
        print("STATUS | MERCHANT                  |   AMOUNT | PROB   | RISK   |  TIME | CUSTOMER")
        print("-" * 90)
        
        await self.setup_connections()
        
        try:
            interval = 60.0 / self.transactions_per_minute  # seconds between transactions
            
            while True:
                start_time = time.time()
                
                # Generate and process transaction
                transaction = self.generate_transaction()
                prediction = await self.predict_transaction(transaction)
                
                # Print transaction details
                self.print_transaction_table(transaction, prediction)
                
                # Store in database and Redis
                await self.store_transaction(transaction, prediction)
                
                # Update statistics
                self.stats['total_transactions'] += 1
                if prediction.get('is_fraud', False) or prediction.get('fraud_probability', 0) > 0.5:
                    self.stats['fraud_detected'] += 1
                
                # Update dashboard metrics every 10 transactions
                if self.stats['total_transactions'] % 10 == 0:
                    await self.update_dashboard_metrics()
                    self.print_stats()
                
                # Wait for next transaction
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print(f"\n⏹️  Service stopped by user")
        finally:
            if self.session:
                await self.session.close()
            self.print_stats()
            print("🏁 Realtime service completed!")

async def main():
    """Main function"""
    service = IntegratedRealtimeService()
    await service.run_continuous_service()

if __name__ == "__main__":
    asyncio.run(main())