# dashboard/streamlit_app.py
import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://ml-api:8000"

# --- API Client Class ---
class DashboardAPI:
    def __init__(self, base_url):
        self.base_url = base_url

    def _get(self, endpoint, params=None):
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error ({endpoint}): {response.status_code} - {response.text}")
                return None
        except requests.exceptions.RequestException:
            st.error(f"Connection Error: Could not connect to the API at {self.base_url}. Is the service running?")
            return None

    def check_health(self):
        return self._get("/health")

    def get_realtime_metrics(self):
        return self._get("/realtime/metrics")
        
    def get_recent_predictions(self, limit=50):
        return self._get("/realtime/predictions", params={"limit": limit})

    def get_timeseries_data(self, hours=6):
        return self._get("/realtime/timeseries", params={"hours": hours})

    def get_risk_distribution(self, hours=1):
        return self._get("/realtime/risk-distribution", params={"hours": hours})
        
    def get_fraud_alerts(self, limit=10):
        return self._get("/realtime/fraud-alerts")

    def get_simulation_status(self):
        return self._get("/realtime/simulation/status")

    def control_simulation(self, action, rate=10):
        try:
            payload = {"action": action, "rate": rate}
            response = requests.post(f"{self.base_url}/realtime/simulation/control", json=payload, timeout=5)
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")
        return None

    def safe_get_value(self, data, key, default=0):
        """Safely get a value from a dictionary, returning a default if key is missing or data is None."""
        if data and isinstance(data, dict):
            return data.get(key, default)
        return default

# --- Main Dashboard ---
st.title("🛡️ Real-time Fraud Detection Dashboard")

dashboard = DashboardAPI(API_BASE_URL)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Auto-refresh
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_rate = st.slider("Refresh rate (seconds)", 1, 30, 5)

    st.markdown("---")
    
    # Simulation Control
    st.header("🔬 Simulation")
    sim_status = dashboard.get_simulation_status()
    sim_running = sim_status.get('status') == 'running' if sim_status else False

    if st.button("Start Simulation", disabled=sim_running, use_container_width=True):
        dashboard.control_simulation("start")
        st.success("Simulation started!")
        time.sleep(1)
        st.rerun()

    if st.button("Stop Simulation", disabled=not sim_running, use_container_width=True):
        dashboard.control_simulation("stop")
        st.warning("Simulation stopped.")
        time.sleep(1)
        st.rerun()
        
    if sim_status:
        status_color = "green" if sim_running else "red"
        st.markdown(f"**Status:** <span style='color:{status_color};'>{sim_status.get('status', 'unknown').capitalize()}</span>", unsafe_allow_html=True)
        st.write(f"**Total Predictions:** {sim_status.get('total_predictions', 0):,}")
        st.write(f"**Frauds Detected:** {sim_status.get('fraud_detected', 0):,}")

    st.markdown("---")
    
    # Health Status
    st.header("🩺 System Health")
    health = dashboard.check_health()
    if health:
        st.success(f"API Status: {health.get('status', 'Unknown')}")
        st.write(f"Model Loaded: {'✅' if health.get('model_loaded') else '❌'}")
        st.write(f"Database: {'✅' if health.get('database_connected') else '❌'}")
        st.write(f"Redis: {'✅' if health.get('redis_connected') else '❌'}")
    else:
        st.error("API Status: Unreachable")

# --- Main Metrics Section ---
st.header("📊 System Metrics")

metrics = dashboard.get_realtime_metrics()

if metrics:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_pred = dashboard.safe_get_value(metrics, 'total_predictions', 0)
        pred_hour = dashboard.safe_get_value(metrics, 'predictions_last_hour', 0)
        st.metric(
            "Total Predictions", 
            f"{total_pred:,}", 
            delta=f"+{pred_hour}/hr" if pred_hour > 0 else None
        )
    
    with col2:
        fraud_rate = dashboard.safe_get_value(metrics, 'fraud_rate', 0) * 100
        frauds_hour = dashboard.safe_get_value(metrics, 'frauds_last_hour', 0)
        st.metric(
            "Fraud Rate", 
            f"{fraud_rate:.2f}%", 
            delta=f"{frauds_hour} frauds/hr" if frauds_hour > 0 else None
        )
    
    with col3:
        avg_amount = dashboard.safe_get_value(metrics, 'avg_amount', 0)
        st.metric(
            "Avg Amount", 
            f"${avg_amount:.2f}" if avg_amount is not None else "$0.00",
            delta=None
        )
    
    with col4:
        avg_time = dashboard.safe_get_value(metrics, 'avg_processing_time', 0)
        st.metric(
            "Avg Response Time", 
            f"{avg_time:.1f}ms" if avg_time is not None else "0.0ms",
            delta=None
        )
    
    with col5:
        high_risk = dashboard.safe_get_value(metrics, 'high_risk_count', 0)
        st.metric("High Risk", f"{high_risk}", delta=None)
else:
    st.warning("⚠️ No real-time metrics available. This could mean:")
    st.write("- Real-time simulation service is not running")
    st.write("- Database tables are not created or are empty")
    st.write("- API endpoints are not working correctly")

st.markdown("---")

# --- Fraud Hotspot Map ---
st.subheader("📍 Fraud Hotspot Map")
fraud_alerts_data = dashboard.get_fraud_alerts()

if fraud_alerts_data and fraud_alerts_data.get('alerts'):
    alerts = fraud_alerts_data['alerts']
    df_alerts = pd.DataFrame(alerts)
    
    # Ensure lat/long are numeric
    df_alerts['lat'] = pd.to_numeric(df_alerts['lat'], errors='coerce')
    df_alerts['long'] = pd.to_numeric(df_alerts['long'], errors='coerce')
    df_alerts.dropna(subset=['lat', 'long'], inplace=True)

    # Calculate fraud frequency per location
    df_alerts['frequency'] = df_alerts.groupby(['lat', 'long'])['lat'].transform('count')
    
    # Create hover text
    df_alerts['hover_text'] = df_alerts.apply(
        lambda row: f"<b>Location:</b> ({row['lat']:.4f}, {row['long']:.4f})<br>" +
                    f"<b>Fraud Count:</b> {row['frequency']}<br>" +
                    f"<b>Last Merchant:</b> {row['merchant_name']}",
        axis=1
    )

    fig = go.Figure(go.Scattermapbox(
        lat=df_alerts['lat'],
        lon=df_alerts['long'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=df_alerts['frequency'] * 5,  # Scale size by frequency
            color=df_alerts['fraud_probability'],
            colorscale="Reds",
            cmin=0.5,
            cmax=1.0,
            showscale=True,
            colorbar_title="Fraud Prob."
        ),
        text=df_alerts['hover_text'],
        hoverinfo='text'
    ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=3,
        mapbox_center={"lat": df_alerts['lat'].mean(), "lon": df_alerts['long'].mean()},
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No fraud alerts with location data to display on the map.")

st.markdown("---")

# --- Charts and Data Tables ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction & Fraud Trends (Last 6 Hours)")
    timeseries_data = dashboard.get_timeseries_data(hours=6)
    if timeseries_data:
        df_ts = pd.DataFrame(timeseries_data)
        df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_ts['timestamp'], y=df_ts['transaction_count'], mode='lines', name='Total Transactions', yaxis='y1'))
        fig.add_trace(go.Bar(x=df_ts['timestamp'], y=df_ts['fraud_count'], name='Fraud Count', yaxis='y2', marker_color='red', opacity=0.6))

        fig.update_layout(
            yaxis=dict(title='Total Transactions'),
            yaxis2=dict(title='Fraud Count', overlaying='y', side='right'),
            legend=dict(x=0, y=1.1, orientation='h'),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time series data available.")

with col2:
    st.subheader("Risk Level Distribution (Last Hour)")
    risk_data = dashboard.get_risk_distribution(hours=1)
    if risk_data:
        df_risk = pd.DataFrame(risk_data)
        fig = px.pie(df_risk, values='count', names='risk_level', title='Risk Levels',
                     color='risk_level',
                     color_discrete_map={'LOW':'green', 'MEDIUM':'orange', 'HIGH':'red'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No risk distribution data available.")

st.markdown("---")

# --- Recent Predictions Table ---
st.subheader("Recent Predictions")
predictions = dashboard.get_recent_predictions()
if predictions:
    df_preds = pd.DataFrame(predictions)
    # Format for display
    df_preds['prediction_timestamp'] = pd.to_datetime(df_preds['prediction_timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df_preds['amount'] = df_preds['amount'].apply(lambda x: f"${x:,.2f}")
    df_preds['fraud_probability'] = df_preds['fraud_probability'].apply(lambda x: f"{x:.2%}")
    st.dataframe(df_preds[['prediction_timestamp', 'merchant_name', 'category', 'amount', 'risk_level', 'fraud_probability']], use_container_width=True)
else:
    st.info("No recent predictions found.")

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()