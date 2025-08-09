# src/api/main.py
from fastapi import FastAPI
from src.api.endpoints import prediction, training, monitoring
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="API for real-time fraud detection using an XGBoost model.",
    version="1.0.0"
)

# Include routers
logger.info("Including routers...")
app.include_router(prediction.router, prefix="/predict", tags=["Prediction"])
app.include_router(training.router, prefix="/train", tags=["Training"])
app.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])

@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint for health check."""
    return {"status": "ok", "message": "Welcome to the Fraud Detection API"}

logger.info("Application startup complete.")
