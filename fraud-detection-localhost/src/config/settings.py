# src/config/settings.py
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application configuration settings"""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://frauduser:fraudpass123@postgres:5432/frauddb"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis Configuration
    REDIS_URL: str = "redis://redis:6379"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    
    # Model Configuration
    MODEL_PATH: str = "/app/models"
    MODEL_VERSION: str = "v1.0.0"
    MODEL_RELOAD_INTERVAL: int = 3600  # seconds
    
    # Feature Engineering
    FEATURE_CACHE_TTL: int = 300  # seconds
    MAX_BATCH_SIZE: int = 1000
    
    # Monitoring Configuration
    METRICS_RETENTION_DAYS: int = 30
    ALERT_THRESHOLD_LATENCY: float = 1000.0  # milliseconds
    ALERT_THRESHOLD_ERROR_RATE: float = 0.05  # 5%
    
    # Security Configuration
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Development/Testing
    TESTING: bool = False
    DEBUG: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()