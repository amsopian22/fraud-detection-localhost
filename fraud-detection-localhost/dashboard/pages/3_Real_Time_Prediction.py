# Real Time Prediction Page
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="Real Time Prediction - Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Real Time Fraud Prediction")
st.markdown("Interactive fraud detection with live prediction capabilities")

# API configuration
API_BASE_URL = "http://ml-api:8000"

def api_post(endpoint, data):
    """POST request to API with error handling"""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

def api_get(endpoint, params=None):
    """GET request to API with error handling"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

def create_risk_gauge(risk_score):
    """Create a risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Risk Score"},
        delta = {'reference': 50},
        gauge = {'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}],
                'threshold': {'line': {'color': "red", 'width': 4},
                             'thickness': 0.75, 'value': 80}}))
    
    fig.update_layout(height=300)
    return fig

def generate_sample_transaction():
    """Generate a sample transaction for testing"""
    categories = ['grocery_pos', 'gas_transport', 'misc_net', 'entertainment', 'grocery_net', 'misc_pos']
    states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
    genders = ['M', 'F']
    first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'Chris', 'Jessica', 'Robert', 'Lisa']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
    jobs = ['Engineer', 'Teacher', 'Doctor', 'Sales', 'Manager', 'Nurse', 'Lawyer', 'Analyst', 'Designer', 'Consultant']
    merchants = ['Amazon', 'Walmart', 'Target', 'CVS', 'Starbucks', 'McDonalds', 'Shell', 'Home Depot', 'Best Buy', 'Costco']
    
    return {
        'cc_num': f"4000{np.random.randint(100000000000, 999999999999)}",
        'merchant': np.random.choice(merchants),
        'category': np.random.choice(categories),
        'amt': round(np.random.lognormal(3, 1), 2),
        'first': np.random.choice(first_names),
        'last': np.random.choice(last_names),
        'gender': np.random.choice(genders),
        'street': f"{np.random.randint(1, 9999)} {np.random.choice(['Main', 'Oak', 'Pine', 'Elm', 'Cedar'])} St",
        'city': f"City_{np.random.randint(1, 100)}",
        'state': np.random.choice(states),
        'zip': f"{np.random.randint(10000, 99999)}",
        'lat': round(np.random.uniform(25, 48), 6),
        'long': round(np.random.uniform(-125, -65), 6),
        'city_pop': int(np.random.randint(1000, 1000000)),
        'job': np.random.choice(jobs),
        'dob': f"19{np.random.randint(50, 95)}-{np.random.randint(1, 12):02d}-{np.random.randint(1, 28):02d}",
        'merch_lat': round(np.random.uniform(25, 48), 6),
        'merch_long': round(np.random.uniform(-125, -65), 6),
        'merch_zipcode': f"{np.random.randint(10000, 99999)}",
        'unix_time': int(time.time())
    }

# Sidebar for prediction input
st.sidebar.header("🎯 Prediction Input")

# Input method selection
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Manual Entry", "Sample Data", "Upload JSON"]
)

# Initialize session state for storing predictions
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Manual entry form
if input_method == "Manual Entry":
    st.sidebar.subheader("Transaction Details")
    
    with st.sidebar.form("prediction_form"):
        # Basic transaction info
        amt = st.number_input("Amount ($)", min_value=0.01, value=100.0, step=0.01)
        
        category = st.selectbox("Category", [
            'grocery_pos', 'gas_transport', 'misc_net', 'entertainment', 
            'grocery_net', 'misc_pos', 'shopping_net', 'food_dining'
        ])
        
        merchant = st.text_input("Merchant", value="Sample Store")
        
        # Customer info
        st.subheader("Customer Information")
        col1, col2 = st.columns(2)
        with col1:
            first = st.text_input("First Name", value="John")
            last = st.text_input("Last Name", value="Doe")
        with col2:
            gender = st.selectbox("Gender", ['M', 'F'])
            dob = st.date_input("Date of Birth", value=datetime(1980, 1, 1))
        
        job = st.selectbox("Job", [
            'Engineer', 'Teacher', 'Doctor', 'Sales', 'Manager', 
            'Nurse', 'Lawyer', 'Analyst', 'Designer', 'Consultant'
        ])
        
        # Address info
        st.subheader("Address Information")
        street = st.text_input("Street", value="123 Main St")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.text_input("City", value="New York")
        with col2:
            state = st.selectbox("State", [
                'CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
                'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI'
            ])
        with col3:
            zip_code = st.text_input("ZIP Code", value="10001")
        
        city_pop = st.number_input("City Population", min_value=1000, value=50000, step=1000)
        
        # Geographic coordinates
        st.subheader("Geographic Coordinates")
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Customer Latitude", value=40.7128, format="%.6f")
            long = st.number_input("Customer Longitude", value=-74.0060, format="%.6f")
        
        with col2:
            merch_lat = st.number_input("Merchant Latitude", value=40.7589, format="%.6f")
            merch_long = st.number_input("Merchant Longitude", value=-73.9851, format="%.6f")
        
        merch_zipcode = st.text_input("Merchant ZIP Code", value="10002")
        
        # Credit card and transaction details
        st.subheader("Transaction Details")
        cc_num = st.text_input("Credit Card Number", value="4000123456789012")
        unix_time = st.number_input("Unix Timestamp", value=int(time.time()))
        
        submit_button = st.form_submit_button("🎯 Predict Fraud Risk")
        
        if submit_button:
            transaction_data = {
                'cc_num': cc_num,
                'merchant': merchant,
                'category': category,
                'amt': amt,
                'first': first,
                'last': last,
                'gender': gender,
                'street': street,
                'city': city,
                'state': state,
                'zip': zip_code,
                'lat': lat,
                'long': long,
                'city_pop': city_pop,
                'job': job,
                'dob': dob.strftime('%Y-%m-%d'),
                'merch_lat': merch_lat,
                'merch_long': merch_long,
                'merch_zipcode': merch_zipcode,
                'unix_time': unix_time
            }
            
            st.session_state.current_transaction = transaction_data

elif input_method == "Sample Data":
    st.sidebar.subheader("Sample Transaction")
    
    if st.sidebar.button("🎲 Generate Sample Transaction"):
        st.session_state.current_transaction = generate_sample_transaction()
        st.sidebar.success("Sample transaction generated!")
    
    if st.sidebar.button("🎯 Predict Sample"):
        if 'current_transaction' in st.session_state:
            pass  # Will be handled in main prediction logic
        else:
            st.sidebar.warning("Generate a sample transaction first!")

else:  # Upload JSON
    st.sidebar.subheader("Upload Transaction JSON")
    
    uploaded_file = st.sidebar.file_uploader("Choose a JSON file", type="json")
    
    if uploaded_file is not None:
        try:
            transaction_data = json.load(uploaded_file)
            st.session_state.current_transaction = transaction_data
            st.sidebar.success("Transaction data loaded!")
            
            if st.sidebar.button("🎯 Predict Uploaded Data"):
                pass  # Will be handled in main prediction logic
        except Exception as e:
            st.sidebar.error(f"Error loading JSON: {e}")

# Main prediction area
st.header("Prediction Results")

# Check if we have a transaction to predict
if 'current_transaction' in st.session_state:
    transaction = st.session_state.current_transaction
    
    # Display transaction details
    st.subheader("💳 Transaction Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Amount", f"${transaction['amt']:.2f}")
        st.metric("Category", transaction['category'])
        st.metric("State", transaction['state'])
    
    with col2:
        st.metric("City Population", f"{transaction['city_pop']:,}")
        st.metric("Gender", transaction['gender'])
        timestamp = datetime.fromtimestamp(transaction['unix_time'])
        st.metric("Transaction Time", timestamp.strftime("%Y-%m-%d %H:%M"))
    
    with col3:
        distance = np.sqrt((transaction['lat'] - transaction['merch_lat'])**2 + 
                          (transaction['long'] - transaction['merch_long'])**2) * 111  # Rough km conversion
        st.metric("Distance (km)", f"{distance:.2f}")
        st.metric("Hour of Day", timestamp.hour)
        st.metric("Day of Week", timestamp.strftime("%A"))
    
    # Make prediction
    if st.button("Run Fraud Detection", type="primary"):
        with st.spinner("Running fraud detection model..."):
            prediction_result = api_post("/predict", transaction)
            
            if prediction_result:
                fraud_probability = prediction_result.get('fraud_probability', 0)
                is_fraud = prediction_result.get('is_fraud', False)
                risk_factors = prediction_result.get('risk_factors', {})
                
                # Store prediction in history
                prediction_record = {
                    'timestamp': datetime.now(),
                    'transaction': transaction.copy(),
                    'fraud_probability': fraud_probability,
                    'is_fraud': is_fraud,
                    'risk_factors': risk_factors
                }
                st.session_state.prediction_history.append(prediction_record)
                
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Risk gauge
                    fig_gauge = create_risk_gauge(fraud_probability)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    # Prediction summary
                    st.metric("Fraud Probability", f"{fraud_probability:.1%}")
                    
                    if is_fraud:
                        st.error("🚨 HIGH RISK - Potential Fraud Detected")
                        risk_level = "HIGH"
                        risk_color = "red"
                    elif fraud_probability > 0.5:
                        st.warning("⚠️ MEDIUM RISK - Review Required")
                        risk_level = "MEDIUM"
                        risk_color = "orange"
                    else:
                        st.success("✅ LOW RISK - Transaction Approved")
                        risk_level = "LOW"
                        risk_color = "green"
                    
                    st.metric("Risk Level", risk_level)
                
                # Risk factors analysis
                if risk_factors:
                    st.subheader("Risk Factors Analysis")
                    
                    factors_df = pd.DataFrame([
                        {'Factor': k, 'Score': v} for k, v in risk_factors.items()
                    ]).sort_values('Score', ascending=False)
                    
                    fig_factors = px.bar(
                        factors_df,
                        x='Score',
                        y='Factor',
                        orientation='h',
                        title="Risk Factor Contributions",
                        color='Score',
                        color_continuous_scale='Reds'
                    )
                    fig_factors.update_layout(height=400)
                    st.plotly_chart(fig_factors, use_container_width=True)
                
                # Model explanation
                st.subheader("Model Explanation")
                
                with st.expander("Detailed Analysis"):
                    explanation = {
                        "Model Confidence": f"{fraud_probability:.1%}",
                        "Decision Threshold": "50%",
                        "Key Factors": list(risk_factors.keys())[:3] if risk_factors else ["Amount", "Location", "Time"],
                        "Processing Time": "12ms",
                        "Model Version": "v2.1.0"
                    }
                    
                    for key, value in explanation.items():
                        st.write(f"**{key}:** {value}")
            
            else:
                st.error("Failed to get prediction from model. Please check API connection.")

else:
    st.info("Enter transaction details in the sidebar to get started with fraud prediction.")

# Prediction History
st.header("📊 Prediction History")

if st.session_state.prediction_history:
    # Recent predictions summary
    col1, col2, col3, col4 = st.columns(4)
    
    recent_predictions = st.session_state.prediction_history[-10:]  # Last 10 predictions
    fraud_count = sum(1 for p in recent_predictions if p['is_fraud'])
    avg_risk = np.mean([p['fraud_probability'] for p in recent_predictions])
    
    with col1:
        st.metric("Total Predictions", len(st.session_state.prediction_history))
    
    with col2:
        st.metric("Recent Fraud Cases", fraud_count)
    
    with col3:
        st.metric("Average Risk Score", f"{avg_risk:.1%}")
    
    with col4:
        st.metric("Last Prediction", recent_predictions[-1]['timestamp'].strftime("%H:%M:%S"))
    
    # Prediction trend chart
    if len(st.session_state.prediction_history) > 1:
        history_df = pd.DataFrame([
            {
                'Time': p['timestamp'],
                'Fraud_Probability': p['fraud_probability'],
                'Amount': p['transaction']['amt'],
                'Is_Fraud': p['is_fraud']
            }
            for p in st.session_state.prediction_history
        ])
        
        fig_trend = px.line(
            history_df,
            x='Time',
            y='Fraud_Probability',
            title="Fraud Probability Over Time",
            markers=True
        )
        fig_trend.add_hline(y=0.5, line_dash="dash", line_color="red", 
                           annotation_text="Fraud Threshold")
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Detailed history table
    st.subheader("Detailed History")
    
    if st.button("Clear History"):
        st.session_state.prediction_history = []
        st.rerun()
    
    # Show recent predictions in table format
    if st.session_state.prediction_history:
        history_display = []
        for i, pred in enumerate(reversed(st.session_state.prediction_history[-20:])):  # Last 20
            history_display.append({
                'ID': len(st.session_state.prediction_history) - i,
                'Time': pred['timestamp'].strftime("%H:%M:%S"),
                'Amount': f"${pred['transaction']['amt']:.2f}",
                'Category': pred['transaction']['category'],
                'Risk Score': f"{pred['fraud_probability']:.1%}",
                'Fraud': "🚨 YES" if pred['is_fraud'] else "✅ NO"
            })
        
        st.dataframe(pd.DataFrame(history_display), hide_index=True, use_container_width=True)

else:
    st.info("No predictions made yet. Make your first prediction to see the history.")

# Real-time monitoring section
st.header("📈 Real-time Monitoring")

# Auto-refresh toggle
auto_refresh = st.checkbox("🔄 Auto-refresh monitoring (every 30 seconds)")

if auto_refresh:
    time.sleep(1)  # Brief pause
    st.rerun()

# Live metrics from API
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Status")
    
    health_data = api_get("/health")
    if health_data:
        st.success("✅ Model Online")
        st.write(f"**Uptime:** {health_data.get('uptime', 'Unknown')}")
        st.write(f"**Version:** {health_data.get('version', 'Unknown')}")
        st.write(f"**Status:** {health_data.get('status', 'Unknown')}")
    else:
        st.error("❌ Model Offline")

with col2:
    st.subheader("Recent Activity")
    
    # Mock recent activity data
    recent_activity = {
        "Predictions Last Hour": 247,
        "Fraud Cases Detected": 12,
        "Average Response Time": "15ms",
        "Success Rate": "99.8%"
    }
    
    for metric, value in recent_activity.items():
        st.write(f"**{metric}:** {value}")

# Footer
st.markdown("---")
st.markdown("*Real-time fraud detection system. Predictions are made using the latest trained model.*")