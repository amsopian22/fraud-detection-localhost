#!/usr/bin/env python3
"""
Enhanced Real-time Fraud Detection Dashboard
Shows live predictions and simulation data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from sqlalchemy import create_engine, text
import redis
import os

# Page config
st.set_page_config(
    page_title="Real-time Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .fraud-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    
    .normal-transaction {
        background-color: #00cc88;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    
    .warning-transaction {
        background-color: #ffa500;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

class RealtimeDashboard:
    def __init__(self):
        self.db_engine = None
        self.redis_client = None
        self.api_base_url = "http://ml-api:8000"
        self._init_connections()
    
    def _init_connections(self):
        """Initialize database and Redis connections"""
        try:
            # Database
            db_url = os.getenv('DATABASE_URL', 'postgresql://frauduser:fraudpass123@postgres:5432/frauddb')
            self.db_engine = create_engine(db_url)
            
            # Redis
            self.redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
            self.redis_client.ping()
            
        except Exception as e:
            st.error(f"Connection error: {e}")
    
    def get_api_health(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_model_info(self):
        """Get model information"""
        try:
            response = requests.get(f"{self.api_base_url}/model/info", timeout=5)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def get_dashboard_metrics(self):
        """Get dashboard metrics from database"""
        try:
            if not self.db_engine:
                return None
            
            query = "SELECT * FROM dashboard_metrics"
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
                if result:
                    return dict(result._mapping)
                return None
        except Exception as e:
            st.error(f"Database query error: {e}")
            return None
    
    def get_recent_predictions(self, limit=50):
        """Get recent predictions"""
        try:
            if not self.db_engine:
                return pd.DataFrame()
            
            query = f"""
            SELECT 
                transaction_id,
                prediction_timestamp,
                amount,
                merchant_name,
                category,
                fraud_probability,
                is_fraud,
                risk_level,
                processing_time_ms
            FROM realtime_predictions 
            ORDER BY prediction_timestamp DESC 
            LIMIT {limit}
            """
            
            with self.db_engine.connect() as conn:
                df = pd.read_sql(query, conn)
                return df
        except Exception as e:
            st.error(f"Error fetching predictions: {e}")
            return pd.DataFrame()
    
    def get_time_series_data(self, hours=24):
        """Get time series data for charts"""
        try:
            if not self.db_engine:
                return pd.DataFrame()
            
            query = f"""
            SELECT 
                DATE_TRUNC('minute', prediction_timestamp) as time_bucket,
                COUNT(*) as transaction_count,
                COUNT(*) FILTER (WHERE is_fraud = true) as fraud_count,
                AVG(fraud_probability) as avg_fraud_prob,
                AVG(amount) as avg_amount
            FROM realtime_predictions 
            WHERE prediction_timestamp > NOW() - INTERVAL '{hours} hours'
            GROUP BY time_bucket
            ORDER BY time_bucket
            """
            
            with self.db_engine.connect() as conn:
                df = pd.read_sql(query, conn)
                return df
        except Exception as e:
            st.error(f"Error fetching time series data: {e}")
            return pd.DataFrame()
    
    def get_risk_distribution(self):
        """Get risk level distribution"""
        try:
            if not self.db_engine:
                return pd.DataFrame()
            
            query = """
            SELECT 
                risk_level,
                COUNT(*) as count,
                AVG(amount) as avg_amount,
                AVG(fraud_probability) as avg_prob
            FROM realtime_predictions 
            WHERE prediction_timestamp > NOW() - INTERVAL '1 hour'
            GROUP BY risk_level
            ORDER BY avg_prob DESC
            """
            
            with self.db_engine.connect() as conn:
                df = pd.read_sql(query, conn)
                return df
        except Exception:
            return pd.DataFrame()

def main():
    """Main dashboard function"""
    st.title("🔍 Real-time Fraud Detection Dashboard")
    st.markdown("---")
    
    # Initialize dashboard
    dashboard = RealtimeDashboard()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", value=True)
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval", 
        [5, 10, 15, 30, 60], 
        index=1,
        format_func=lambda x: f"{x} seconds"
    )
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # System Status Section
    st.header("🏥 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        api_health = dashboard.get_api_health()
        if api_health:
            st.success("✅ ML API Online")
            uptime = api_health.get('uptime_seconds', 0)
            st.caption(f"Uptime: {uptime/3600:.1f} hours")
        else:
            st.error("❌ ML API Offline")
    
    with col2:
        if dashboard.db_engine:
            st.success("✅ Database Connected")
        else:
            st.error("❌ Database Error")
    
    with col3:
        if dashboard.redis_client:
            try:
                dashboard.redis_client.ping()
                st.success("✅ Redis Connected")
            except:
                st.error("❌ Redis Error")
        else:
            st.error("❌ Redis Error")
    
    with col4:
        model_info = dashboard.get_model_info()
        if model_info:
            st.success("✅ Model Loaded")
            st.caption(f"Version: {model_info.get('version', 'Unknown')}")
        else:
            st.error("❌ Model Error")
    
    # Main Metrics Section
    st.header("📊 Live Metrics")
    
    metrics = dashboard.get_dashboard_metrics()
    
    if metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Predictions",
                f"{metrics.get('total_predictions', 0):,}",
                delta=f"+{metrics.get('predictions_last_hour', 0)}/hr"
            )
        
        with col2:
            fraud_rate = 0
            if metrics.get('total_predictions', 0) > 0:
                fraud_rate = metrics.get('total_frauds', 0) / metrics.get('total_predictions', 1) * 100
            st.metric(
                "Fraud Rate",
                f"{fraud_rate:.2f}%",
                delta=f"{metrics.get('frauds_last_hour', 0)} frauds/hr"
            )
        
        with col3:
            st.metric(
                "Avg Amount",
                f"${metrics.get('avg_transaction_amount', 0):.2f}",
                delta=None
            )
        
        with col4:
            st.metric(
                "Avg Fraud Probability",
                f"{metrics.get('avg_fraud_probability', 0):.4f}",
                delta=None
            )
        
        with col5:
            st.metric(
                "Response Time",
                f"{metrics.get('avg_processing_time', 0):.1f}ms",
                delta=None
            )
    else:
        st.warning("⚠️ No metrics available. Start the simulation service to see real-time data.")
    
    # Risk Distribution
    st.header("⚠️ Risk Level Distribution")
    
    risk_data = dashboard.get_risk_distribution()
    if not risk_data.empty:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Risk level pie chart
            fig_pie = px.pie(
                risk_data, 
                values='count', 
                names='risk_level',
                title="Risk Level Distribution (Last Hour)",
                color='risk_level',
                color_discrete_map={
                    'HIGH': '#ff4b4b',
                    'MEDIUM': '#ffa500', 
                    'LOW': '#00cc88'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Risk metrics table
            st.subheader("Risk Metrics")
            risk_display = risk_data.copy()
            risk_display['avg_amount'] = risk_display['avg_amount'].apply(lambda x: f"${x:.2f}")
            risk_display['avg_prob'] = risk_display['avg_prob'].apply(lambda x: f"{x:.4f}")
            risk_display.columns = ['Risk Level', 'Count', 'Avg Amount', 'Avg Probability']
            st.dataframe(risk_display, hide_index=True)
    
    # Time Series Charts
    st.header("📈 Transaction Trends")
    
    time_series = dashboard.get_time_series_data(hours=6)
    if not time_series.empty:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Transaction Volume', 'Fraud Detection Rate',
                'Average Amount', 'Fraud Probability Trend'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Transaction volume
        fig.add_trace(
            go.Scatter(
                x=time_series['time_bucket'],
                y=time_series['transaction_count'],
                name='Transactions',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Fraud rate
        fraud_rate = (time_series['fraud_count'] / time_series['transaction_count'] * 100).fillna(0)
        fig.add_trace(
            go.Scatter(
                x=time_series['time_bucket'],
                y=fraud_rate,
                name='Fraud Rate %',
                line=dict(color='red')
            ),
            row=1, col=2
        )
        
        # Average amount
        fig.add_trace(
            go.Scatter(
                x=time_series['time_bucket'],
                y=time_series['avg_amount'],
                name='Avg Amount',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        # Fraud probability
        fig.add_trace(
            go.Scatter(
                x=time_series['time_bucket'],
                y=time_series['avg_fraud_prob'],
                name='Avg Fraud Probability',
                line=dict(color='orange')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Transaction Trends (Last 6 Hours)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Predictions Table
    st.header("📋 Recent Predictions")
    
    recent_predictions = dashboard.get_recent_predictions(50)
    if not recent_predictions.empty:
        # Add styling to the dataframe
        def style_prediction_row(row):
            if row['is_fraud']:
                return ['background-color: #ff4b4b; color: white'] * len(row)
            elif row['risk_level'] == 'MEDIUM':
                return ['background-color: #ffa500; color: white'] * len(row)
            else:
                return [''] * len(row)
        
        # Format the data for display
        display_df = recent_predictions.copy()
        display_df['prediction_timestamp'] = pd.to_datetime(display_df['prediction_timestamp']).dt.strftime('%H:%M:%S')
        display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:.2f}")
        display_df['fraud_probability'] = display_df['fraud_probability'].apply(lambda x: f"{x:.4f}")
        display_df['is_fraud'] = display_df['is_fraud'].apply(lambda x: "🚨 FRAUD" if x else "✅ Normal")
        display_df['processing_time_ms'] = display_df['processing_time_ms'].apply(lambda x: f"{x}ms")
        
        # Rename columns for display
        display_df.columns = [
            'Transaction ID', 'Time', 'Amount', 'Merchant', 'Category',
            'Fraud Prob', 'Status', 'Risk Level', 'Processing Time'
        ]
        
        # Display with color coding
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )
        
        # Recent fraud alerts
        fraud_predictions = recent_predictions[recent_predictions['is_fraud'] == True].head(5)
        if not fraud_predictions.empty:
            st.subheader("🚨 Recent Fraud Alerts")
            for _, fraud in fraud_predictions.iterrows():
                st.markdown(
                    f"""
                    <div class="fraud-alert">
                        <strong>FRAUD DETECTED</strong> - {fraud['merchant_name']} | 
                        Amount: ${fraud['amount']:.2f} | 
                        Probability: {fraud['fraud_probability']:.4f} | 
                        Time: {fraud['prediction_timestamp'].strftime('%H:%M:%S')}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("ℹ️ No predictions available yet. Start the simulation service to see real-time predictions.")
    
    # Simulation Controls
    st.header("🎮 Simulation Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Simulation", help="Start the background prediction service"):
            st.info("To start simulation, run: `docker-compose exec ml-api python /app/scripts/realtime_service.py`")
    
    with col2:
        if st.button("🛑 Stop Simulation", help="Stop the background service"):
            st.info("Use Ctrl+C in the terminal running the simulation service")
    
    with col3:
        if st.button("🧹 Clear Data", help="Clear old prediction data"):
            try:
                if dashboard.db_engine:
                    with dashboard.db_engine.connect() as conn:
                        conn.execute(text("DELETE FROM realtime_predictions WHERE prediction_timestamp < NOW() - INTERVAL '1 day'"))
                        conn.commit()
                    st.success("✅ Old data cleared")
            except Exception as e:
                st.error(f"❌ Error clearing data: {e}")
    
    # Instructions
    with st.expander("📚 How to Use This Dashboard"):
        st.markdown("""
        ### Real-time Fraud Detection Dashboard
        
        This dashboard shows live fraud detection predictions in real-time.
        
        **Features:**
        - 🏥 **System Status**: Monitor API, database, and model health
        - 📊 **Live Metrics**: Real-time statistics and KPIs  
        - ⚠️ **Risk Distribution**: Current risk level breakdown
        - 📈 **Trends**: Time-series charts of transaction patterns
        - 📋 **Recent Predictions**: Latest fraud predictions with alerts
        - 🎮 **Controls**: Start/stop simulation and manage data
        
        **To Start:**
        1. Ensure ML model is trained and loaded
        2. Run the simulation service: `docker-compose exec ml-api python /app/scripts/realtime_service.py`
        3. Watch real-time predictions appear in this dashboard
        
        **Color Coding:**
        - 🚨 Red: Fraud detected (high risk)
        - 🟠 Orange: Medium risk
        - ✅ Green: Normal transaction (low risk)
        """
        )
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()