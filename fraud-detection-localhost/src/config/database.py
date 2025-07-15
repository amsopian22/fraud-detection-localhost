# src/config/database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from .settings import settings

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    echo=settings.DEBUG
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Database models
class Transaction(Base):
    """Transaction data model"""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(255), unique=True, index=True)
    cc_num = Column(String(255))
    merchant = Column(String(255))
    category = Column(String(50))
    amt = Column(Float)
    first_name = Column(String(100))
    last_name = Column(String(100))
    gender = Column(String(1))
    street = Column(String(255))
    city = Column(String(100))
    state = Column(String(2))
    zip = Column(String(10))
    lat = Column(Float)
    long = Column(Float)
    city_pop = Column(Integer)
    job = Column(String(100))
    dob = Column(String(20))  # Store as string for simplicity
    merch_lat = Column(Float)
    merch_long = Column(Float)
    merch_zipcode = Column(String(10))
    unix_time = Column(Integer)
    is_fraud = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    """Prediction results model"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(255), index=True)
    fraud_probability = Column(Float)
    is_fraud_predicted = Column(Boolean)
    risk_level = Column(String(10))
    model_version = Column(String(50))
    processing_time_ms = Column(Float)
    features_used = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelMetrics(Base):
    """Model performance metrics"""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(50))
    accuracy = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    auc_score = Column(Float)
    training_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database dependency
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()