# Data Overview Page
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import os

# Page config
st.set_page_config(
    page_title="Data Overview - Fraud Detection",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Overview")
st.markdown("Comprehensive analysis of fraud detection data and model performance")

# API configuration
API_BASE_URL = "http://ml-api:8000"

def api_get(endpoint, params=None):
    """Simple API getter with error handling"""
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

def load_sample_data():
    """Load sample data for demonstration"""
    try:
        # Try to load actual data files
        data_path = "/app/data/processed"
        if os.path.exists(f"{data_path}/train.parquet"):
            df = pd.read_parquet(f"{data_path}/train.parquet")
            return df.head(1000)  # Limit for performance
        elif os.path.exists("/app/data/raw/credit_card_transaction_train.csv"):
            df = pd.read_csv("/app/data/raw/credit_card_transaction_train.csv")
            return df.head(1000)
        else:
            # Generate sample data if no files found
            np.random.seed(42)
            n_samples = 1000
            
            categories = ['grocery_pos', 'gas_transport', 'misc_net', 'entertainment', 'grocery_net', 'misc_pos']
            states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
            
            sample_data = {
                'amt': np.random.lognormal(3, 1, n_samples),
                'category': np.random.choice(categories, n_samples),
                'state': np.random.choice(states, n_samples),
                'city_pop': np.random.randint(1000, 1000000, n_samples),
                'lat': np.random.uniform(25, 48, n_samples),
                'long': np.random.uniform(-125, -65, n_samples),
                'merch_lat': np.random.uniform(25, 48, n_samples),
                'merch_long': np.random.uniform(-125, -65, n_samples),
                'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
                'gender': np.random.choice(['M', 'F'], n_samples),
                'unix_time': np.random.randint(1640995200, 1672531200, n_samples)  # 2022 timestamps
            }
            
            df = pd.DataFrame(sample_data)
            df['hour'] = (df['unix_time'] % 86400) // 3600
            df['day_of_week'] = (df['unix_time'] // 86400) % 7
            
            return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
with st.spinner("Loading data..."):
    df = load_sample_data()

if df.empty:
    st.error("No data available to display")
    st.stop()

# Data Summary Section
st.header("📊 Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", f"{len(df):,}")

with col2:
    fraud_count = df['is_fraud'].sum() if 'is_fraud' in df.columns else 0
    st.metric("Fraud Cases", f"{fraud_count:,}")

with col3:
    fraud_rate = (fraud_count / len(df) * 100) if len(df) > 0 else 0
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

with col4:
    avg_amount = df['amt'].mean() if 'amt' in df.columns else 0
    st.metric("Avg Transaction", f"${avg_amount:.2f}")

st.markdown("---")

# Data Distribution Analysis
st.header("📊 Data Distribution Analysis")

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["💰 Amount Analysis", "🌍 Geographic", "⏰ Temporal", "📂 Categorical"])

with tab1:
    st.subheader("Transaction Amount Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Amount distribution histogram
        fig_hist = px.histogram(
            df, 
            x='amt', 
            color='is_fraud' if 'is_fraud' in df.columns else None,
            nbins=50,
            title="Transaction Amount Distribution",
            labels={'amt': 'Transaction Amount ($)', 'count': 'Frequency'},
            color_discrete_map={0: 'lightblue', 1: 'red'} if 'is_fraud' in df.columns else None
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot by fraud status
        if 'is_fraud' in df.columns:
            fig_box = px.box(
                df, 
                y='amt', 
                x='is_fraud',
                title="Amount Distribution by Fraud Status",
                labels={'amt': 'Transaction Amount ($)', 'is_fraud': 'Fraud Status'}
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Fraud status not available in dataset")
    
    # Amount statistics
    st.subheader("Amount Statistics")
    amount_stats = df['amt'].describe()
    stats_df = pd.DataFrame({
        'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
        'Value': [f"{amount_stats['count']:.0f}", f"${amount_stats['mean']:.2f}", 
                 f"${amount_stats['std']:.2f}", f"${amount_stats['min']:.2f}",
                 f"${amount_stats['25%']:.2f}", f"${amount_stats['50%']:.2f}",
                 f"${amount_stats['75%']:.2f}", f"${amount_stats['max']:.2f}"]
    })
    st.dataframe(stats_df, hide_index=True)

with tab2:
    st.subheader("Geographic Distribution")
    
    if all(col in df.columns for col in ['lat', 'long']):
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer locations scatter plot
            fig_geo = px.scatter_mapbox(
                df.sample(min(500, len(df))),  # Sample for performance
                lat='lat',
                lon='long',
                color='is_fraud' if 'is_fraud' in df.columns else None,
                size='amt',
                title="Customer Transaction Locations",
                mapbox_style="carto-positron",
                zoom=3,
                color_discrete_map={0: 'blue', 1: 'red'} if 'is_fraud' in df.columns else None
            )
            fig_geo.update_layout(height=500)
            st.plotly_chart(fig_geo, use_container_width=True)
        
        with col2:
            # State distribution
            if 'state' in df.columns:
                state_counts = df['state'].value_counts().head(10)
                fig_states = px.bar(
                    x=state_counts.index,
                    y=state_counts.values,
                    title="Top 10 States by Transaction Count",
                    labels={'x': 'State', 'y': 'Transaction Count'}
                )
                fig_states.update_layout(height=500)
                st.plotly_chart(fig_states, use_container_width=True)
    else:
        st.info("Geographic data not available in dataset")

with tab3:
    st.subheader("Temporal Patterns")
    
    if 'hour' in df.columns or 'unix_time' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly distribution
            if 'hour' in df.columns:
                hourly_data = df.groupby('hour').agg({
                    'amt': 'count',
                    'is_fraud': 'sum' if 'is_fraud' in df.columns else 'count'
                }).reset_index()
                hourly_data.columns = ['Hour', 'Total_Transactions', 'Fraud_Count']
                
                fig_hourly = go.Figure()
                fig_hourly.add_trace(go.Bar(
                    x=hourly_data['Hour'],
                    y=hourly_data['Total_Transactions'],
                    name='Total Transactions',
                    yaxis='y'
                ))
                
                if 'is_fraud' in df.columns:
                    fig_hourly.add_trace(go.Bar(
                        x=hourly_data['Hour'],
                        y=hourly_data['Fraud_Count'],
                        name='Fraud Cases',
                        yaxis='y2'
                    ))
                
                fig_hourly.update_layout(
                    title="Transactions by Hour of Day",
                    xaxis_title="Hour",
                    yaxis=dict(title="Total Transactions"),
                    yaxis2=dict(title="Fraud Cases", overlaying='y', side='right'),
                    height=400
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Day of week distribution
            if 'day_of_week' in df.columns:
                dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                dow_data = df.groupby('day_of_week').size().reset_index()
                dow_data['day_name'] = dow_data['day_of_week'].map(lambda x: dow_names[x] if x < 7 else 'Unknown')
                
                fig_dow = px.bar(
                    dow_data,
                    x='day_name',
                    y=0,
                    title="Transactions by Day of Week",
                    labels={'day_name': 'Day of Week', 0: 'Transaction Count'}
                )
                fig_dow.update_layout(height=400)
                st.plotly_chart(fig_dow, use_container_width=True)
    else:
        st.info("Temporal data not available in dataset")

with tab4:
    st.subheader("Categorical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            fig_cat = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Transaction Categories"
            )
            fig_cat.update_layout(height=400)
            st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        # Gender distribution
        if 'gender' in df.columns:
            gender_counts = df['gender'].value_counts()
            fig_gender = px.bar(
                x=gender_counts.index,
                y=gender_counts.values,
                title="Gender Distribution",
                labels={'x': 'Gender', 'y': 'Count'}
            )
            fig_gender.update_layout(height=400)
            st.plotly_chart(fig_gender, use_container_width=True)

st.markdown("---")

# Fraud Analysis Section
if 'is_fraud' in df.columns:
    st.header("🚨 Fraud Analysis")
    
    fraud_df = df[df['is_fraud'] == 1]
    normal_df = df[df['is_fraud'] == 0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_fraud_amt = fraud_df['amt'].mean() if len(fraud_df) > 0 else 0
        avg_normal_amt = normal_df['amt'].mean() if len(normal_df) > 0 else 0
        st.metric(
            "Avg Fraud Amount",
            f"${avg_fraud_amt:.2f}",
            delta=f"${avg_fraud_amt - avg_normal_amt:.2f} vs normal"
        )
    
    with col2:
        if 'category' in df.columns and len(fraud_df) > 0:
            top_fraud_cat = fraud_df['category'].mode().iloc[0] if len(fraud_df['category'].mode()) > 0 else "Unknown"
            st.metric("Top Fraud Category", top_fraud_cat)
    
    with col3:
        if 'hour' in df.columns and len(fraud_df) > 0:
            top_fraud_hour = fraud_df['hour'].mode().iloc[0] if len(fraud_df['hour'].mode()) > 0 else "Unknown"
            st.metric("Peak Fraud Hour", f"{top_fraud_hour}:00")
    
    # Fraud patterns
    if len(fraud_df) > 0:
        st.subheader("Fraud Patterns")
        
        # Category-wise fraud rate
        if 'category' in df.columns:
            cat_fraud = df.groupby('category').agg({
                'is_fraud': ['count', 'sum']
            }).round(3)
            cat_fraud.columns = ['Total', 'Fraud_Count']
            cat_fraud['Fraud_Rate'] = (cat_fraud['Fraud_Count'] / cat_fraud['Total'] * 100).round(2)
            cat_fraud = cat_fraud.sort_values('Fraud_Rate', ascending=False)
            
            fig_cat_fraud = px.bar(
                cat_fraud.reset_index(),
                x='category',
                y='Fraud_Rate',
                title="Fraud Rate by Category",
                labels={'category': 'Category', 'Fraud_Rate': 'Fraud Rate (%)'}
            )
            st.plotly_chart(fig_cat_fraud, use_container_width=True)

# Raw Data Sample
st.header("📄 Data Sample")
st.subheader("Raw Data Preview")

# Show sample of the data
sample_size = min(100, len(df))
sample_df = df.sample(sample_size).copy()

# Format for display
if 'amt' in sample_df.columns:
    sample_df['amt'] = sample_df['amt'].apply(lambda x: f"${x:.2f}")

st.dataframe(sample_df, use_container_width=True)

# Data quality information
st.subheader("Data Quality Summary")
quality_info = []
for col in df.columns:
    null_count = df[col].isnull().sum()
    null_pct = (null_count / len(df)) * 100
    unique_count = df[col].nunique()
    
    quality_info.append({
        'Column': col,
        'Data Type': str(df[col].dtype),
        'Null Count': null_count,
        'Null %': f"{null_pct:.2f}%",
        'Unique Values': unique_count,
        'Sample Value': str(df[col].iloc[0]) if len(df) > 0 else 'N/A'
    })

quality_df = pd.DataFrame(quality_info)
st.dataframe(quality_df, hide_index=True, use_container_width=True)

# Model Performance Section
st.header("🤖 Model Information")

model_info = api_get("/model/info")
if model_info:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", model_info.get('model_type', 'Unknown'))
    
    with col2:
        accuracy = model_info.get('accuracy', 0)
        st.metric("Accuracy", f"{accuracy:.2%}" if accuracy else "N/A")
    
    with col3:
        auc = model_info.get('test_auc', 0)
        st.metric("AUC Score", f"{auc:.3f}" if auc else "N/A")
    
    # Model details
    with st.expander("Model Details"):
        st.json(model_info)
else:
    st.info("Model information not available. Make sure the ML API is running.")

# Footer
st.markdown("---")
st.markdown("*Data overview updated in real-time. Refresh page for latest information.*")