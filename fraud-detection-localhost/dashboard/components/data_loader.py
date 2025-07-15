"""
Data loading utilities for the fraud detection dashboard
"""
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
import streamlit as st

API_BASE_URL = "http://ml-api:8000"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_transaction_data(limit=1000):
    """
    Load transaction data from various sources
    
    Args:
        limit (int): Maximum number of records to load
        
    Returns:
        pd.DataFrame: Transaction data
    """
    try:
        # Try to load actual data files first
        data_path = "/app/data/processed"
        
        if os.path.exists(f"{data_path}/train.parquet"):
            df = pd.read_parquet(f"{data_path}/train.parquet")
            return df.head(limit)
        elif os.path.exists("/app/data/raw/credit_card_transaction_train.csv"):
            df = pd.read_csv("/app/data/raw/credit_card_transaction_train.csv")
            return df.head(limit)
        else:
            # Generate synthetic data as fallback
            return generate_synthetic_data(limit)
            
    except Exception as e:
        st.warning(f"Could not load data files: {e}. Using synthetic data.")
        return generate_synthetic_data(limit)

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic transaction data for testing
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Synthetic transaction data
    """
    np.random.seed(42)
    
    categories = ['grocery_pos', 'gas_transport', 'misc_net', 'entertainment', 'grocery_net', 'misc_pos']
    states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
    
    # Generate transaction amounts with realistic distribution
    amounts = np.random.lognormal(3, 1, n_samples)
    amounts = np.clip(amounts, 0.01, 10000)  # Reasonable range
    
    # Generate timestamps for the last 30 days
    end_time = datetime.now().timestamp()
    start_time = end_time - (30 * 24 * 3600)  # 30 days ago
    timestamps = np.random.uniform(start_time, end_time, n_samples)
    
    sample_data = {
        'amt': amounts,
        'category': np.random.choice(categories, n_samples),
        'state': np.random.choice(states, n_samples),
        'city_pop': np.random.randint(1000, 1000000, n_samples),
        'lat': np.random.uniform(25, 48, n_samples),
        'long': np.random.uniform(-125, -65, n_samples),
        'merch_lat': np.random.uniform(25, 48, n_samples),
        'merch_long': np.random.uniform(-125, -65, n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'unix_time': timestamps.astype(int)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add derived features
    df['hour'] = ((df['unix_time'] % 86400) // 3600).astype(int)
    df['day_of_week'] = ((df['unix_time'] // 86400) % 7).astype(int)
    df['datetime'] = pd.to_datetime(df['unix_time'], unit='s')
    
    # Calculate distance between customer and merchant
    df['distance_km'] = np.sqrt(
        (df['lat'] - df['merch_lat'])**2 + 
        (df['long'] - df['merch_long'])**2
    ) * 111  # Rough conversion to km
    
    return df

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_model_info():
    """
    Get model information from API
    
    Returns:
        dict: Model information or None if unavailable
    """
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_system_health():
    """
    Get system health status from API
    
    Returns:
        dict: Health status or None if unavailable
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None

def get_recent_predictions(limit=100):
    """
    Get recent predictions from API
    
    Args:
        limit (int): Maximum number of predictions to retrieve
        
    Returns:
        list: Recent predictions or empty list if unavailable
    """
    try:
        response = requests.get(f"{API_BASE_URL}/predictions/recent", 
                              params={'limit': limit}, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception:
        return []

def predict_transaction(transaction_data):
    """
    Make a prediction for a single transaction
    
    Args:
        transaction_data (dict): Transaction features
        
    Returns:
        dict: Prediction result or None if failed
    """
    try:
        response = requests.post(f"{API_BASE_URL}/predict", 
                               json=transaction_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None

def batch_predict(transactions):
    """
    Make batch predictions for multiple transactions
    
    Args:
        transactions (list): List of transaction dictionaries
        
    Returns:
        list: Prediction results or empty list if failed
    """
    try:
        response = requests.post(f"{API_BASE_URL}/predict/batch", 
                               json=transactions, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception:
        return []

def load_feature_importance():
    """
    Load feature importance data from model
    
    Returns:
        pd.DataFrame: Feature importance scores
    """
    model_info = get_model_info()
    
    if model_info and 'feature_importance' in model_info:
        importance_data = model_info['feature_importance']
        return pd.DataFrame(list(importance_data.items()), 
                          columns=['Feature', 'Importance'])
    else:
        # Return mock data if not available
        features = [
            'transaction_amount', 'hour_of_day', 'days_since_last_transaction',
            'merchant_category', 'geographic_distance', 'transaction_frequency_1h',
            'avg_amount_last_week', 'day_of_week', 'merchant_risk_score', 'customer_age'
        ]
        
        importance_scores = np.random.exponential(0.1, len(features))
        importance_scores = importance_scores / importance_scores.sum()
        
        return pd.DataFrame({
            'Feature': features,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)

def get_data_quality_metrics():
    """
    Get data quality metrics
    
    Returns:
        dict: Data quality metrics
    """
    try:
        response = requests.get(f"{API_BASE_URL}/data/quality", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            # Return mock metrics if API unavailable
            return {
                'completeness': 0.985,
                'accuracy': 0.972,
                'consistency': 0.991,
                'timeliness': 0.958,
                'validity': 0.967,
                'total_records': 50000,
                'missing_values': 750,
                'duplicate_records': 12,
                'invalid_records': 38
            }
    except Exception:
        return {
            'completeness': 0.985,
            'accuracy': 0.972,
            'consistency': 0.991,
            'timeliness': 0.958,
            'validity': 0.967,
            'total_records': 50000,
            'missing_values': 750,
            'duplicate_records': 12,
            'invalid_records': 38
        }

def validate_transaction_data(transaction):
    """
    Validate transaction data before prediction
    
    Args:
        transaction (dict): Transaction data to validate
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    required_fields = ['amt', 'category', 'state', 'lat', 'long', 'merch_lat', 'merch_long']
    
    for field in required_fields:
        if field not in transaction:
            errors.append(f"Missing required field: {field}")
    
    if 'amt' in transaction:
        if not isinstance(transaction['amt'], (int, float)) or transaction['amt'] <= 0:
            errors.append("Amount must be a positive number")
    
    if 'lat' in transaction:
        if not (-90 <= transaction['lat'] <= 90):
            errors.append("Latitude must be between -90 and 90")
    
    if 'long' in transaction:
        if not (-180 <= transaction['long'] <= 180):
            errors.append("Longitude must be between -180 and 180")
    
    if 'merch_lat' in transaction:
        if not (-90 <= transaction['merch_lat'] <= 90):
            errors.append("Merchant latitude must be between -90 and 90")
    
    if 'merch_long' in transaction:
        if not (-180 <= transaction['merch_long'] <= 180):
            errors.append("Merchant longitude must be between -180 and 180")
    
    return len(errors) == 0, errors

def format_currency(amount):
    """Format amount as currency"""
    return f"${amount:,.2f}"

def format_percentage(value, decimals=1):
    """Format value as percentage"""
    return f"{value * 100:.{decimals}f}%"

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two geographic points in kilometers
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        float: Distance in kilometers
    """
    # Simplified distance calculation (not exact due to Earth's curvature)
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111