# ml-service/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime
import logging
import time
import joblib
import numpy as np
from typing import Dict, Any
from pydantic import BaseModel, Field
import redis
from sqlalchemy import create_engine, text
import os
import json
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for model and connections
model = None
feature_encoder = None
redis_client = None
db_engine = None

# --- Pydantic Models for API ---

class TransactionRequest(BaseModel):
    cc_num: str = Field(..., description="Credit card number")
    merchant: str = Field(..., description="Merchant name")
    category: str = Field(..., description="Transaction category")
    amt: float = Field(..., gt=0, description="Transaction amount")
    first: str = Field(..., description="Customer first name")
    last: str = Field(..., description="Customer last name")
    gender: str = Field(..., description="Customer gender")
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City")
    state: str = Field(..., description="State")
    zip: str = Field(..., description="ZIP code")
    lat: float = Field(..., description="Customer latitude")
    long: float = Field(..., description="Customer longitude")
    city_pop: int = Field(..., description="City population")
    job: str = Field(..., description="Customer job")
    dob: str = Field(..., description="Date of birth")
    merch_lat: float = Field(..., description="Merchant latitude")
    merch_long: float = Field(..., description="Merchant longitude")
    merch_zipcode: str = Field(..., description="Merchant ZIP code")

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    processing_time_ms: float
    model_version: str
    features_used: List[str]

class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionRequest]

class ModelTrainingRequest(BaseModel):
    retrain: bool = True
    model_type: str = "xgboost"
    hyperparameters: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    database_connected: bool
    redis_connected: bool
    uptime_seconds: float

# --- Start of New Pydantic Models for Real-time Dashboard ---

class RealtimeMetrics(BaseModel):
    total_predictions: int
    total_frauds: int
    fraud_rate: float
    avg_amount: float
    avg_processing_time: float
    predictions_last_hour: int
    frauds_last_hour: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int

class RecentPrediction(BaseModel):
    transaction_id: str
    prediction_timestamp: datetime
    amount: float
    merchant_name: str
    category: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    processing_time_ms: int

class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    transaction_count: int
    fraud_count: int
    avg_fraud_prob: float
    avg_amount: float

class RiskDistribution(BaseModel):
    risk_level: str
    count: int
    avg_amount: float
    avg_probability: float

class SimulationControl(BaseModel):
    action: str  # "start", "stop", "status"
    rate: Optional[int] = 10  # transactions per minute
    
# --- End of New Pydantic Models ---


# --- Feature Engineering Functions ---

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points using Haversine formula"""
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def extract_time_features(timestamp: str = None) -> Dict[str, int]:
    """Extract time-based features from timestamp"""
    if timestamp is None:
        dt = datetime.now()
    else:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    return {
        'hour': dt.hour,
        'day_of_week': dt.weekday(),
        'is_weekend': 1 if dt.weekday() >= 5 else 0,
        'is_night': 1 if dt.hour < 6 or dt.hour > 22 else 0
    }

def calculate_age(dob: str) -> int:
    """Calculate age from date of birth"""
    try:
        birth_date = datetime.strptime(dob, '%Y-%m-%d')
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return max(0, min(age, 120))
    except:
        return 35

def create_features(transaction: TransactionRequest) -> np.ndarray:
    """Create engineered features from transaction data matching the enhanced model approach"""
    # Load label encoders
    label_encoders = None
    feature_scaler = None
    try:
        label_encoders_path = '/app/models/trained_models/label_encoders.joblib'
        if os.path.exists(label_encoders_path):
            label_encoders = joblib.load(label_encoders_path)
        
        feature_scaler_path = '/app/models/trained_models/feature_scaler.joblib'
        if os.path.exists(feature_scaler_path):
            feature_scaler = joblib.load(feature_scaler_path)
    except Exception as e:
        logger.warning(f"Failed to load encoders: {e}")
    
    # Create features in the exact order expected by the trained model
    features = []
    
    # 1. merchant - encoded using label encoder
    merchant_encoded = 0
    if label_encoders and 'merchant' in label_encoders:
        try:
            merchant_encoded = label_encoders['merchant'].transform([transaction.merchant])[0]
        except:
            merchant_encoded = 0
    features.append(merchant_encoded)
    
    # 2. category - encoded using label encoder
    category_encoded = 0
    if label_encoders and 'category' in label_encoders:
        try:
            category_encoded = label_encoders['category'].transform([transaction.category])[0]
        except:
            category_encoded = 0
    features.append(category_encoded)
    
    # 3. amt - raw amount
    features.append(transaction.amt)
    
    # 4. gender - encoded using label encoder
    gender_encoded = 0
    if label_encoders and 'gender' in label_encoders:
        try:
            gender_encoded = label_encoders['gender'].transform([transaction.gender])[0]
        except:
            gender_encoded = 1 if transaction.gender.upper() == 'M' else 0
    else:
        gender_encoded = 1 if transaction.gender.upper() == 'M' else 0
    features.append(gender_encoded)
    
    # 5. lat - customer latitude
    features.append(transaction.lat)
    
    # 6. long - customer longitude
    features.append(transaction.long)
    
    # 7. transaction_time - seconds since epoch
    transaction_time = time.time()
    features.append(transaction_time)
    
    # 8. transaction_std - standard deviation of recent transactions (mock for now)
    features.append(50.0)  # Mock std
    
    # 9. transaction_avg - average of recent transactions (mock for now)
    features.append(100.0)  # Mock avg
    
    # 10. age - calculated from DOB
    age = calculate_age(transaction.dob)
    features.append(age)
    
    # Extract time features
    time_features = extract_time_features()
    
    # 11. day - day of month
    dt = datetime.now()
    features.append(dt.day)
    
    # 12. month - month of year
    features.append(dt.month)
    
    # 13. hour - hour of day
    features.append(time_features['hour'])
    
    # Convert to numpy array and apply scaling if available
    feature_array = np.array(features).reshape(1, -1)
    
    # Debug logging
    logger.info(f"Created {len(features)} features: {feature_array.shape}")
    
    if feature_scaler:
        try:
            feature_array = feature_scaler.transform(feature_array)
            logger.info(f"After scaling: {feature_array.shape}")
        except Exception as e:
            logger.warning(f"Failed to apply scaling: {e}")
    
    return feature_array

# --- Startup and Shutdown Events ---

app_start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_encoder, redis_client, db_engine
    logger.info("Starting up Fraud Detection API...")
    
    try:
        redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        redis_client = None
    
    try:
        db_url = os.getenv('DATABASE_URL', 'postgresql://frauduser:fraudpass123@postgres:5432/frauddb')
        db_engine = create_engine(db_url)
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        db_engine = None
    
    try:
        model_path = '/app/models/trained_models/xgboost_fraud_detector.joblib'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Trained model loaded successfully")
        else:
            logger.warning("Model file not found, using dummy model")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            X_dummy = np.random.rand(100, 13)  # Updated to 13 features
            y_dummy = np.random.randint(0, 2, 100)
            model.fit(X_dummy, y_dummy)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
    
    try:
        encoder_path = '/app/models/trained_models/feature_encoder.joblib'
        if os.path.exists(encoder_path):
            feature_encoder = joblib.load(encoder_path)
            logger.info("Feature encoder loaded successfully")
    except Exception as e:
        logger.warning(f"Feature encoder not found: {e}")
        feature_encoder = None
    
    yield
    
    logger.info("Shutting down Fraud Detection API...")
    if redis_client:
        redis_client.close()
    if db_engine:
        db_engine.dispose()

# --- FastAPI App and Middleware ---

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection system for credit card transactions",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
    response.headers["X-Process-Time"] = str(process_time)
    return response

# --- Core API Endpoints ---

@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "Fraud Detection API", "version": "1.0.0", "status": "running", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    uptime = time.time() - app_start_time
    model_loaded = model is not None
    
    db_connected = False
    if db_engine:
        try:
            with db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_connected = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
    
    redis_connected = False
    if redis_client:
        try:
            redis_client.ping()
            redis_connected = True
        except:
            pass
    
    status = "healthy" if all([model_loaded, db_connected, redis_connected]) else "degraded"
    return HealthResponse(status=status, timestamp=datetime.now().isoformat(), model_loaded=model_loaded, database_connected=db_connected, redis_connected=redis_connected, uptime_seconds=uptime)

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest):
    start_time = time.time()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        transaction_id = f"txn_{int(time.time() * 1000)}"
        X = create_features(transaction)  # Now returns np.ndarray directly
        fraud_prob = float(model.predict_proba(X)[0][1])
        is_fraud = fraud_prob > 0.5
        
        if fraud_prob < 0.3: risk_level = "LOW"
        elif fraud_prob < 0.7: risk_level = "MEDIUM"
        else: risk_level = "HIGH"
        
        processing_time = (time.time() - start_time) * 1000
        
        if redis_client:
            try:
                prediction_data = {"transaction_id": transaction_id, "fraud_probability": fraud_prob, "is_fraud": is_fraud, "timestamp": datetime.now().isoformat(), "features": X.tolist()}
                redis_client.setex(f"prediction:{transaction_id}", timedelta(hours=24), json.dumps(prediction_data))
            except Exception as e:
                logger.warning(f"Failed to store prediction in Redis: {e}")
        
        # Feature names from the enhanced model
        feature_names = ['merchant', 'category', 'amt', 'gender', 'lat', 'long', 'transaction_time', 'transaction_std', 'transaction_avg', 'age', 'day', 'month', 'hour']
        
        return PredictionResponse(transaction_id=transaction_id, fraud_probability=fraud_prob, is_fraud=is_fraud, risk_level=risk_level, processing_time_ms=processing_time, model_version="v4.0.0", features_used=feature_names)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    if len(request.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 transactions per batch")
    
    predictions = []
    for transaction in request.transactions:
        try:
            pred_response = await predict_fraud(transaction)
            predictions.append(pred_response)
        except Exception:
            continue
    return {"predictions": predictions, "total": len(predictions)}

@app.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    try:
        metadata_path = '/app/models/trained_models/model_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {"name": "XGBoost Fraud Detector", "version": "v1.0.0", "training_date": "2024-01-01", "accuracy": 0.95}
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")

@app.get("/metrics")
async def get_metrics():
    try:
        metrics = {"uptime_seconds": time.time() - app_start_time, "model_loaded": model is not None, "predictions_made": 0, "last_updated": datetime.now().isoformat()}
        if redis_client:
            pred_count = redis_client.get("metrics:prediction_count")
            if pred_count: metrics["predictions_made"] = int(pred_count)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

# --- Start of New Real-time Dashboard Endpoints ---

@app.get("/realtime/metrics", response_model=RealtimeMetrics)
async def get_realtime_metrics():
    """Get real-time dashboard metrics"""
    if not db_engine:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        query = text("""
        SELECT 
            COUNT(*) as total_predictions,
            COUNT(*) FILTER (WHERE is_fraud = true) as total_frauds,
            COALESCE(ROUND(COUNT(*) FILTER (WHERE is_fraud = true)::DECIMAL / NULLIF(COUNT(*), 0), 4), 0) as fraud_rate,
            COALESCE(ROUND(AVG(amount), 2), 0) as avg_amount,
            COALESCE(ROUND(AVG(processing_time_ms), 2), 0) as avg_processing_time,
            COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '1 hour') as predictions_last_hour,
            COUNT(*) FILTER (WHERE is_fraud = true AND timestamp > NOW() - INTERVAL '1 hour') as frauds_last_hour,
            COUNT(*) FILTER (WHERE risk_level = 'HIGH') as high_risk_count,
            COUNT(*) FILTER (WHERE risk_level = 'MEDIUM') as medium_risk_count,
            COUNT(*) FILTER (WHERE risk_level = 'LOW') as low_risk_count
        FROM realtime_predictions
        """)
        
        with db_engine.connect() as conn:
            result = conn.execute(query).fetchone()
            if result:
                return RealtimeMetrics(**dict(result._mapping))
            else:
                return RealtimeMetrics(total_predictions=0, total_frauds=0, fraud_rate=0.0, avg_amount=0.0, avg_processing_time=0.0, predictions_last_hour=0, frauds_last_hour=0, high_risk_count=0, medium_risk_count=0, low_risk_count=0)
    except Exception as e:
        logger.error(f"Error fetching realtime metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch metrics")

@app.get("/realtime/predictions", response_model=List[RecentPrediction])
async def get_recent_predictions(limit: int = 50, hours: int = 24):
    """Get recent predictions"""
    if not db_engine:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        query = text(f"""
        SELECT 
            transaction_id, timestamp as prediction_timestamp, amount, merchant_name, category, 
            fraud_probability, is_fraud, risk_level, processing_time_ms
        FROM realtime_predictions 
        WHERE timestamp > NOW() - INTERVAL '{hours} hours'
        ORDER BY timestamp DESC 
        LIMIT {limit}
        """)
        
        with db_engine.connect() as conn:
            result = conn.execute(query).fetchall()
            return [RecentPrediction(**dict(row._mapping)) for row in result]
    except Exception as e:
        logger.error(f"Error fetching recent predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch predictions")

@app.get("/realtime/timeseries", response_model=List[TimeSeriesPoint])
async def get_timeseries_data(hours: int = 6, interval: str = "minute"):
    """Get time series data for charts"""
    if not db_engine: raise HTTPException(status_code=503, detail="Database not available")
    if interval not in ["minute", "hour", "day"]: raise HTTPException(status_code=400, detail="Invalid interval")
    
    try:
        query = text(f"""
        SELECT 
            DATE_TRUNC('{interval}', timestamp) as timestamp, COUNT(*) as transaction_count,
            COUNT(*) FILTER (WHERE is_fraud = true) as fraud_count, COALESCE(AVG(fraud_probability), 0) as avg_fraud_prob,
            COALESCE(AVG(amount), 0) as avg_amount
        FROM realtime_predictions 
        WHERE timestamp > NOW() - INTERVAL '{hours} hours'
        GROUP BY timestamp ORDER BY timestamp
        """)
        
        with db_engine.connect() as conn:
            result = conn.execute(query).fetchall()
            return [TimeSeriesPoint(**dict(row._mapping)) for row in result]
    except Exception as e:
        logger.error(f"Error fetching timeseries data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch timeseries data")

@app.get("/realtime/risk-distribution", response_model=List[RiskDistribution])
async def get_risk_distribution(hours: int = 1):
    """Get risk level distribution"""
    if not db_engine: raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        query = text(f"""
        SELECT 
            risk_level, COUNT(*) as count, COALESCE(AVG(amount), 0) as avg_amount,
            COALESCE(AVG(fraud_probability), 0) as avg_probability
        FROM realtime_predictions 
        WHERE timestamp > NOW() - INTERVAL '{hours} hours'
        GROUP BY risk_level ORDER BY avg_probability DESC
        """)
        
        with db_engine.connect() as conn:
            result = conn.execute(query).fetchall()
            return [RiskDistribution(**dict(row._mapping)) for row in result]
    except Exception as e:
        logger.error(f"Error fetching risk distribution: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch risk distribution")

@app.get("/realtime/fraud-alerts")
async def get_fraud_alerts(limit: int = 200): # Increased limit to get more historical data for the map
    """Get recent fraud alerts with geo-coordinates"""
    if not db_engine: raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Modified query to include lat and long for mapping
        query = text(f"""
        SELECT transaction_id, timestamp as prediction_timestamp, amount, merchant_name, category, 
               fraud_probability, risk_level, lat, long
        FROM realtime_predictions 
        WHERE is_fraud = true
        ORDER BY timestamp DESC 
        LIMIT {limit}
        """)
        
        with db_engine.connect() as conn:
            result = conn.execute(query).fetchall()
            return {"alerts": [dict(row._mapping) for row in result]}
    except Exception as e:
        logger.error(f"Error fetching fraud alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch fraud alerts")

@app.post("/realtime/simulation/control")
async def control_simulation(control: SimulationControl):
    """Control simulation service"""
    if not redis_client: raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        control_data = {"action": control.action, "rate": control.rate, "timestamp": datetime.now().isoformat()}
        redis_client.setex("simulation:control", timedelta(minutes=5), json.dumps(control_data))
        
        if control.action == "start": redis_client.set("simulation:status", "running"); message = f"Simulation started at {control.rate} transactions/minute"
        elif control.action == "stop": redis_client.set("simulation:status", "stopped"); message = "Simulation stopped"
        else: status = redis_client.get("simulation:status") or "unknown"; message = f"Simulation status: {status}"
        
        return {"status": "success", "message": message, "control": control_data}
    except Exception as e:
        logger.error(f"Error controlling simulation: {e}")
        raise HTTPException(status_code=500, detail="Failed to control simulation")

@app.get("/realtime/simulation/status")
async def get_simulation_status():
    """Get simulation status"""
    if not redis_client: raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        status = redis_client.get("simulation:status") or "stopped"
        total_predictions = redis_client.get("realtime:total_predictions") or "0"
        fraud_detected = redis_client.get("realtime:fraud_detected") or "0"
        control_data = redis_client.get("simulation:control")
        last_control = json.loads(control_data) if control_data else None
        
        return {"status": status, "total_predictions": int(total_predictions), "fraud_detected": int(fraud_detected), "last_control": last_control}
    except Exception as e:
        logger.error(f"Error getting simulation status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get simulation status")

@app.delete("/realtime/predictions")
async def clear_old_predictions(days: int = 1):
    """Clear old prediction data"""
    if not db_engine: raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        query = text(f"DELETE FROM realtime_predictions WHERE timestamp < NOW() - INTERVAL '{days} days'")
        with db_engine.connect() as conn:
            result = conn.execute(query)
            deleted_count = result.rowcount
            conn.commit()
        return {"status": "success", "message": f"Deleted {deleted_count} old predictions", "deleted_count": deleted_count}
    except Exception as e:
        logger.error(f"Error clearing predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear predictions")

@app.get("/realtime/health")
async def realtime_health_check():
    """Enhanced health check for real-time features"""
    health_status = {"realtime_features": True, "database_connected": False, "redis_connected": False, "simulation_running": False, "recent_predictions": 0, "timestamp": datetime.now().isoformat()}
    
    if db_engine:
        try:
            with db_engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM realtime_predictions WHERE timestamp > NOW() - INTERVAL '1 hour'"))
                health_status["recent_predictions"] = result.scalar() or 0
                health_status["database_connected"] = True
        except Exception:
            pass
    
    if redis_client:
        try:
            redis_client.ping()
            health_status["redis_connected"] = True
            status = redis_client.get("simulation:status")
            health_status["simulation_running"] = status == "running"
        except Exception:
            pass
    
    health_status["status"] = "healthy" if all([health_status["database_connected"], health_status["redis_connected"]]) else "degraded"
    return health_status

# --- Main Execution Block ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)