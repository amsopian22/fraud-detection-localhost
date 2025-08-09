# Model Monitoring Page
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="Model Monitoring - Fraud Detection",
    page_icon="✅",
    layout="wide"
)

st.title("✅ Model Monitoring & Operations")
st.markdown("Real-time monitoring, model drift detection, and operational insights")

# API configuration
API_BASE_URL = "http://127.0.0.1:8000"

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

# def create_system_health_chart():
#     """Create system health monitoring chart"""
#     # NOTE: This is mock data. The backend does not yet support historical health metrics.
#     times = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
#     cpu_usage = 20 + 30 * np.random.random(len(times))
#     memory_usage = 40 + 20 * np.random.random(len(times))
#     response_time = 10 + 15 * np.random.random(len(times))

#     fig = make_subplots(
#         rows=3, cols=1,
#         subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 'Response Time (ms)'),
#         vertical_spacing=0.1
#     )

#     fig.add_trace(go.Scatter(x=times, y=cpu_usage, mode='lines', name='CPU'), row=1, col=1)
#     fig.add_trace(go.Scatter(x=times, y=memory_usage, mode='lines', name='Memory'), row=2, col=1)
#     fig.add_trace(go.Scatter(x=times, y=response_time, mode='lines', name='Response Time'), row=3, col=1)

#     fig.update_layout(height=600, showlegend=False, title_text="System Health Metrics (Last 24 Hours)")
#     return fig

# def create_prediction_volume_chart():
#     """Create prediction volume monitoring chart"""
#     # NOTE: This is mock data. The backend does not yet support prediction volume metrics.
#     times = pd.date_range(start=datetime.now() - timedelta(hours=12), end=datetime.now(), freq='15min')
#     predictions = np.random.poisson(50, len(times))
#     fraud_detected = np.random.poisson(2, len(times))

#     fig = go.Figure()
#     fig.add_trace(go.Bar(x=times, y=predictions, name='Total Predictions', opacity=0.7))
#     fig.add_trace(go.Bar(x=times, y=fraud_detected, name='Fraud Detected', opacity=0.9))

#     fig.update_layout(
#         title="Prediction Volume (Last 12 Hours)",
#         xaxis_title="Time",
#         yaxis_title="Count",
#         barmode='overlay',
#         height=400
#     )
#     return fig

# def create_model_drift_chart():
#     """Create model drift monitoring chart"""
#     # NOTE: This is mock data. The backend does not yet support model drift metrics.
#     dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')

#     # Simulate drift metrics
#     accuracy_drift = 0.95 + 0.05 * np.sin(np.linspace(0, 2*np.pi, len(dates))) + 0.01 * np.random.random(len(dates))
#     precision_drift = 0.88 + 0.08 * np.sin(np.linspace(0, 2*np.pi, len(dates))) + 0.02 * np.random.random(len(dates))
#     data_drift_score = 0.1 + 0.15 * np.random.random(len(dates))

#     fig = make_subplots(
#         rows=2, cols=1,
#         subplot_titles=('Model Performance Drift', 'Data Drift Score'),
#         vertical_spacing=0.2
#     )

#     fig.add_trace(go.Scatter(x=dates, y=accuracy_drift, mode='lines+markers', name='Accuracy'), row=1, col=1)
#     fig.add_trace(go.Scatter(x=dates, y=precision_drift, mode='lines+markers', name='Precision'), row=1, col=1)
#     fig.add_trace(go.Scatter(x=dates, y=data_drift_score, mode='lines+markers', name='Drift Score', line=dict(color='red')), row=2, col=1)

#     # Add drift threshold line
#     fig.add_hline(y=0.2, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Drift Threshold")

#     fig.update_layout(height=500, title_text="Model and Data Drift Monitoring (Last 30 Days)")
#     return fig

# Auto-refresh functionality
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 120, 30)

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# System Status Overview
st.header("System Status Overview")

col1, col2, col3, col4 = st.columns(4)

# Get system health from API
health_data = api_get("/monitoring/health")
model_info = api_get("/model/info")

with col1:
    if health_data and health_data.get('status') == 'healthy':
        st.success("System Online")
        uptime = health_data.get('uptime', 'Unknown')
        st.metric("Uptime", uptime)
    else:
        st.error("System Issues")
        st.metric("Status", "Offline")

with col2:
    st.info("CPU and memory usage data not yet available.")
    # # Mock CPU and memory data
    # cpu_usage = np.random.uniform(15, 45)
    # memory_usage = np.random.uniform(35, 65)
    # st.metric("CPU Usage", f"{cpu_usage:.1f}%", delta=f"{np.random.uniform(-5, 5):.1f}%")
    # st.metric("Memory Usage", f"{memory_usage:.1f}%", delta=f"{np.random.uniform(-3, 3):.1f}%")

with col3:
    st.info("Prediction metrics not yet available.")
    # # Mock prediction metrics
    # predictions_per_hour = np.random.randint(180, 320)
    # avg_response_time = np.random.uniform(8, 25)
    # st.metric("Predictions/Hour", f"{predictions_per_hour:,}")
    # st.metric("Avg Response Time", f"{avg_response_time:.1f}ms")

with col4:
    st.info("Model performance data not yet available.")
    # # Model performance
    # if model_info:
    #     accuracy = model_info.get('accuracy', 0)
    #     st.metric("Model Accuracy", f"{accuracy:.1%}" if accuracy else "N/A")
    # else:
    #     st.metric("Model Accuracy", "N/A")
    
    # error_rate = np.random.uniform(0.1, 2.5)
    # st.metric("Error Rate", f"{error_rate:.2f}%")

st.markdown("---")

# Real-time Monitoring Dashboard
st.header("Real-time Monitoring")

tab1, tab2, tab3, tab4 = st.tabs(["System Health", "Prediction Volume", "Model Drift", "Alerts"])

with tab1:
    st.subheader("System Health Metrics")
    st.info("NOTE: Historical system health metrics are not yet available from the backend.")
    # System health chart
    # fig_health = create_system_health_chart()
    # st.plotly_chart(fig_health, use_container_width=True)
    
    # Current system metrics table
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Metrics")
        st.info("NOTE: Current metrics are not yet available from the backend.")
        # current_metrics = {
        #     'Metric': ['CPU Usage', 'Memory Usage', 'Disk Usage', 'Network I/O', 'Model Load Time'],
        #     'Value': ['32.4%', '58.7%', '45.2%', '125 MB/s', '850ms'],
        #     'Status': ['=Ã¢ Normal', '=Ã¡ Moderate', '=Ã¢ Normal', '=Ã¢ Normal', '=Ã¢ Normal']
        # }
        # st.dataframe(pd.DataFrame(current_metrics), hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("Service Status")
        st.info("NOTE: Service status data is not yet available from the backend.")
        # service_status = {
        #     'Service': ['ML API', 'Database', 'Redis Cache', 'Load Balancer', 'Monitoring'],
        #     'Status': ['=Ã¢ Running', '=Ã¢ Running', '=Ã¢ Running', '=Ã¢ Running', '=Ã¢ Running'],
        #     'Uptime': ['99.9%', '99.8%', '99.9%', '100%', '99.7%']
        # }
        # st.dataframe(pd.DataFrame(service_status), hide_index=True, use_container_width=True)

with tab2:
    st.subheader("Prediction Volume Analysis")
    st.info("NOTE: Prediction volume metrics are not yet available from the backend.")
    # Prediction volume chart
    # fig_volume = create_prediction_volume_chart()
    # st.plotly_chart(fig_volume, use_container_width=True)
    
    # Volume statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Volume Stats")
        st.info("NOTE: Volume stats are not yet available.")
        # total_today = np.random.randint(5000, 8000)
        # fraud_today = np.random.randint(200, 400)
        # st.metric("Predictions Today", f"{total_today:,}")
        # st.metric("Fraud Detected Today", fraud_today)
        # st.metric("Fraud Rate Today", f"{(fraud_today/total_today)*100:.2f}%")
    
    with col2:
        st.subheader("Timing Stats")
        st.info("NOTE: Timing stats are not yet available.")
        # peak_hour = np.random.randint(9, 17)
        # avg_response = np.random.uniform(10, 20)
        # st.metric("Peak Hour", f"{peak_hour}:00")
        # st.metric("Avg Response Time", f"{avg_response:.1f}ms")
        # st.metric("95th Percentile", f"{avg_response * 1.5:.1f}ms")
    
    with col3:
        st.subheader("Accuracy Stats")
        st.info("NOTE: Accuracy stats are not yet available.")
        # current_accuracy = np.random.uniform(0.94, 0.98)
        # precision = np.random.uniform(0.85, 0.92)
        # st.metric("Current Accuracy", f"{current_accuracy:.1%}")
        # st.metric("Current Precision", f"{precision:.1%}")
        # st.metric("Model Version", "v2.1.0")

with tab3:
    st.subheader("Model and Data Drift Detection")
    st.info("NOTE: Model and data drift metrics are not yet available from the backend.")
    # Drift monitoring chart
    # fig_drift = create_model_drift_chart()
    # st.plotly_chart(fig_drift, use_container_width=True)
    
    # Drift analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Drift Metrics")
        st.info("NOTE: Drift metrics are not yet available.")
        # drift_score = np.random.uniform(0.05, 0.15)
        # feature_drift = np.random.uniform(0.02, 0.12)
        
        # if drift_score > 0.2:
        #     drift_status = "=4 High Drift"
        #     drift_color = "red"
        # elif drift_score > 0.1:
        #     drift_status = "=Ã¡ Moderate Drift"
        #     drift_color = "orange"
        # else:
        #     drift_status = "=Ã¢ Low Drift"
        #     drift_color = "green"
        
        # st.metric("Overall Drift Score", f"{drift_score:.3f}")
        # st.metric("Drift Status", drift_status)
        # st.metric("Feature Drift", f"{feature_drift:.3f}")
        
        # # Recommendation
        # if drift_score > 0.15:
        #     st.warning("Â  Recommend model retraining")
        # else:
        #     st.success(" Model performance stable")
    
    with col2:
        st.subheader("Feature Drift Analysis")
        st.info("NOTE: Feature drift analysis is not yet available.")
        # # Mock feature drift data
        # features = ['transaction_amount', 'hour_of_day', 'merchant_category', 'geographic_distance', 'customer_age']
        # drift_scores = np.random.uniform(0.01, 0.25, len(features))
        
        # feature_drift_df = pd.DataFrame({
        #     'Feature': features,
        #     'Drift Score': drift_scores,
        #     'Status': ['=4 High' if x > 0.2 else '=Ã¡ Medium' if x > 0.1 else '=Ã¢ Low' for x in drift_scores]
        # }).sort_values('Drift Score', ascending=False)
        
        # st.dataframe(feature_drift_df, hide_index=True, use_container_width=True)

with tab4:
    st.subheader("Alert Management")
    st.info("NOTE: Alert management is not yet available from the backend.")
    
    # # Active alerts
    # col1, col2 = st.columns([2, 1])
    
    # with col1:
    #     st.subheader("=Â¨ Active Alerts")
        
    #     # Mock alert data
    #     alerts = [
    #         {
    #             'Time': '14:23:45',
    #             'Severity': '=Ã¡ Medium',
    #             'Component': 'Model Performance',
    #             'Message': 'Accuracy dropped below 95% threshold',
    #             'Action': 'Monitor closely'
    #         },
    #         {
    #             'Time': '13:45:12',
    #             'Severity': '=Ã¢ Low',
    #             'Component': 'System Resources',
    #             'Message': 'Memory usage above 60%',
    #             'Action': 'Auto-scaled'
    #         },
    #         {
    #             'Time': '12:30:33',
    #             'Severity': '=4 High',
    #             'Component': 'Data Quality',
    #             'Message': 'Unusual pattern in transaction data',
    #             'Action': 'Investigation required'
    #         }
    #     ]
        
    #     alerts_df = pd.DataFrame(alerts)
    #     st.dataframe(alerts_df, hide_index=True, use_container_width=True)
    
    # with col2:
    #     st.subheader("=Ã°Å¸â€œÅ  Alert Summary")
        
    #     alert_counts = {
    #         'Critical': 1,
    #         'High': 2,
    #         'Medium': 3,
    #         'Low': 5
    #     }
        
    #     for severity, count in alert_counts.items():
    #         color_map = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green'}
    #         st.metric(f"{severity} Alerts", count)
        
    #     st.markdown("---")
        
    #     # Alert configuration
    #     st.subheader("â„¢ Alert Settings")
        
    #     accuracy_threshold = st.slider("Accuracy Threshold", 0.8, 1.0, 0.95, 0.01)
    #     response_threshold = st.slider("Response Time Threshold (ms)", 10, 100, 50, 5)
    #     drift_threshold = st.slider("Drift Score Threshold", 0.1, 0.5, 0.2, 0.05)
        
    #     if st.button("=Â¾ Save Alert Settings"):
    #         st.success("Alert settings saved!")

# Model Performance Tracking
st.header("Model Performance Tracking")
st.info("NOTE: Model performance tracking is not yet available from the backend.")

# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("<Ã°Å¸â€œÅ  Performance Trends")
    
#     # Create performance trend chart
#     dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
#     accuracy_trend = 0.95 + 0.02 * np.random.random(len(dates))
#     precision_trend = 0.88 + 0.05 * np.random.random(len(dates))
#     recall_trend = 0.91 + 0.03 * np.random.random(len(dates))
    
#     fig_trends = go.Figure()
#     fig_trends.add_trace(go.Scatter(x=dates, y=accuracy_trend, mode='lines+markers', name='Accuracy'))
#     fig_trends.add_trace(go.Scatter(x=dates, y=precision_trend, mode='lines+markers', name='Precision'))
#     fig_trends.add_trace(go.Scatter(x=dates, y=recall_trend, mode='lines+markers', name='Recall'))
    
#     fig_trends.update_layout(
#         title="Performance Metrics (Last 7 Days)",
#         yaxis_title="Score",
#         height=400
#     )
    
#     st.plotly_chart(fig_trends, use_container_width=True)

# with col2:
#     st.subheader("= Model Operations")
    
#     # Model operations panel
#     st.write("**Current Model:** v2.1.0")
#     st.write("**Last Training:** 2024-01-15")
#     st.write("**Next Scheduled Training:** 2024-02-01")
    
#     col_a, col_b = st.columns(2)
    
#     with col_a:
#         if st.button("= Retrain Model"):
#             st.info("Model retraining initiated...")
        
#         if st.button("=Ã°Å¸â€œÅ  Generate Report"):
#             st.info("Performance report generated!")
    
#     with col_b:
#         if st.button("=â‚¬ Deploy New Version"):
#             st.warning("New version deployment started...")
        
#         if st.button("= Rollback Model"):
#             st.error("Model rollback initiated...")

#     # Model versions
#     st.subheader("=Ã¦ Model Versions")
#     versions = [
#         {'Version': 'v2.1.0', 'Status': '=Ã¢ Active', 'Accuracy': '97.5%'},
#         {'Version': 'v2.0.5', 'Status': '=Ã¡ Standby', 'Accuracy': '96.8%'},
#         {'Version': 'v1.9.2', 'Status': '=4 Deprecated', 'Accuracy': '95.2%'}
#     ]
#     st.dataframe(pd.DataFrame(versions), hide_index=True, use_container_width=True)

# Data Quality Monitoring
st.header("Data Quality Monitoring")
st.info("NOTE: Data quality monitoring is not yet available from the backend.")

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.subheader("=Ã°Å¸â€œÅ  Input Data Quality")

#     data_quality_metrics = {
#         'Metric': ['Completeness', 'Accuracy', 'Consistency', 'Timeliness', 'Validity'],
#         'Score': ['98.5%', '97.2%', '99.1%', '95.8%', '96.7%'],
#         'Status': ['=Ã¢', '=Ã¢', '=Ã¢', '=Ã¡', '=Ã¢']
#     }

#     st.dataframe(pd.DataFrame(data_quality_metrics), hide_index=True, use_container_width=True)

# with col2:
#     st.subheader("=Ãˆ Data Volume Trends")

#     # Create data volume trend
#     hours = list(range(24))
#     volumes = [np.random.randint(100, 500) for _ in hours]

#     fig_volume_trend = px.bar(x=hours, y=volumes, title="Hourly Data Volume")
#     fig_volume_trend.update_layout(height=300)
#     st.plotly_chart(fig_volume_trend, use_container_width=True)

# with col3:
#     st.subheader("Â  Data Issues")

#     issues = [
#         "Missing merchant_category: 12 records",
#         "Invalid coordinates: 3 records",
#         "Future timestamps: 1 record",
#         "Negative amounts: 0 records"
#     ]

#     for issue in issues:
#         if "0 records" in issue:
#             st.success(f" {issue}")
#         elif any(word in issue for word in ["1 record", "3 records"]):
#             st.warning(f"Â  {issue}")
#         else:
#             st.error(f"=4 {issue}")

# Footer with last update time
st.markdown("---")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"*Dashboard last updated: {current_time} | Auto-refresh: {'ON' if auto_refresh else 'OFF'}*")