# Simple Dashboard for Testing
import streamlit as st
import requests
import time
from datetime import datetime

# --- Configuration ---
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://ml-api:8000"

# --- Simple API Client ---
def api_get(endpoint, params=None):
    """Simple API getter with error handling"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return None

# --- Main Dashboard ---
st.title("🛡️ Fraud Detection Dashboard")

# Health Check
st.header("🩺 System Health")
health = api_get("/health")
if health:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status = health.get('status', 'Unknown')
        if status == 'healthy':
            st.success(f"Status: {status}")
        else:
            st.warning(f"Status: {status}")
    
    with col2:
        model_loaded = health.get('model_loaded', False)
        st.write(f"Model: {'✅' if model_loaded else '❌'}")
    
    with col3:
        db_connected = health.get('database_connected', False)
        st.write(f"Database: {'✅' if db_connected else '❌'}")
    
    with col4:
        redis_connected = health.get('redis_connected', False)
        st.write(f"Cache: {'✅' if redis_connected else '❌'}")
else:
    st.error("❌ Cannot connect to API service")

st.markdown("---")

# Test Prediction
st.header("🔮 Test Prediction")
with st.expander("Make a test prediction"):
    col1, col2 = st.columns(2)
    
    with col1:
        cc_num = st.text_input("Credit Card Number", value="1234567890123456")
        merchant = st.text_input("Merchant", value="test_merchant")
        category = st.selectbox("Category", ["grocery_pos", "gas_transport", "misc_net", "entertainment"])
        amt = st.number_input("Amount", value=50.0, min_value=0.1)
        first = st.text_input("First Name", value="John")
        last = st.text_input("Last Name", value="Doe")
    
    with col2:
        gender = st.selectbox("Gender", ["M", "F"])
        city = st.text_input("City", value="Test City")
        state = st.text_input("State", value="CA")
        zip_code = st.text_input("ZIP", value="12345")
        lat = st.number_input("Latitude", value=40.7128)
        long = st.number_input("Longitude", value=-74.0060)
    
    # Additional fields
    street = st.text_input("Street", value="123 Main St")
    city_pop = st.number_input("City Population", value=50000, min_value=1)
    job = st.text_input("Job", value="Engineer")
    dob = st.date_input("Date of Birth", value=datetime(1980, 1, 1))
    merch_lat = st.number_input("Merchant Latitude", value=40.7580)
    merch_long = st.number_input("Merchant Longitude", value=-73.9855)
    merch_zipcode = st.text_input("Merchant ZIP", value="10001")
    
    if st.button("🎯 Make Prediction"):
        prediction_data = {
            "cc_num": cc_num,
            "merchant": merchant,
            "category": category,
            "amt": amt,
            "first": first,
            "last": last,
            "gender": gender,
            "street": street,
            "city": city,
            "state": state,
            "zip": zip_code,
            "lat": lat,
            "long": long,
            "city_pop": int(city_pop),
            "job": job,
            "dob": dob.strftime("%Y-%m-%d"),
            "merch_lat": merch_lat,
            "merch_long": merch_long,
            "merch_zipcode": merch_zipcode
        }
        
        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=prediction_data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    fraud_prob = result.get('fraud_probability', 0)
                    st.metric("Fraud Probability", f"{fraud_prob:.2%}")
                
                with col2:
                    is_fraud = result.get('is_fraud', False)
                    st.metric("Prediction", "FRAUD" if is_fraud else "LEGITIMATE")
                
                with col3:
                    risk_level = result.get('risk_level', 'UNKNOWN')
                    color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk_level, "⚪")
                    st.metric("Risk Level", f"{color} {risk_level}")
                
                st.json(result)
            else:
                st.error(f"Prediction failed: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

st.markdown("---")

# Basic Metrics
st.header("📊 System Metrics")
try:
    metrics = api_get("/metrics")
    if metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            uptime = metrics.get('uptime_seconds', 0)
            st.metric("Uptime", f"{uptime/3600:.1f} hours")
        
        with col2:
            predictions = metrics.get('predictions_made', 0)
            st.metric("Predictions Made", f"{predictions:,}")
        
        with col3:
            model_loaded = metrics.get('model_loaded', False)
            st.metric("Model Status", "Loaded" if model_loaded else "Not Loaded")
    else:
        st.warning("No metrics available")
except Exception as e:
    st.error(f"Error loading metrics: {e}")

st.markdown("---")

# Model Information
st.header("🤖 Model Information")
try:
    model_info = api_get("/model/info")
    if model_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Name:** {model_info.get('name', 'Unknown')}")
            st.write(f"**Version:** {model_info.get('version', 'Unknown')}")
            st.write(f"**Training Date:** {model_info.get('training_date', 'Unknown')}")
        
        with col2:
            accuracy = model_info.get('accuracy', 0)
            st.metric("Accuracy", f"{accuracy:.2%}" if accuracy else "N/A")
            
            feature_count = model_info.get('feature_count', 0)
            st.metric("Features", feature_count if feature_count else "N/A")
    else:
        st.warning("No model information available")
except Exception as e:
    st.error(f"Error loading model info: {e}")

# Auto-refresh option
st.sidebar.header("⚙️ Settings")
auto_refresh = st.sidebar.toggle("Auto-refresh", value=False)
if auto_refresh:
    refresh_rate = st.sidebar.slider("Refresh rate (seconds)", 5, 60, 30)
    time.sleep(refresh_rate)
    st.rerun()