"""
Chart and visualization utilities for the fraud detection dashboard
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_fraud_risk_gauge(risk_score, title="Fraud Risk Score"):
    """
    Create a risk gauge visualization
    
    Args:
        risk_score (float): Risk score between 0 and 1
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Gauge chart
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_confusion_matrix(tp, fp, tn, fn, title="Confusion Matrix"):
    """
    Create confusion matrix heatmap
    
    Args:
        tp, fp, tn, fn (int): Confusion matrix values
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Heatmap chart
    """
    confusion_data = np.array([[tn, fp], [fn, tp]])
    
    fig = go.Figure(data=go.Heatmap(
        z=confusion_data,
        x=['Predicted Normal', 'Predicted Fraud'],
        y=['Actual Normal', 'Actual Fraud'],
        text=[[f'TN: {tn}', f'FP: {fp}'], [f'FN: {fn}', f'TP: {tp}']],
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    
    return fig

def create_roc_curve(fpr=None, tpr=None, auc_score=None, title="ROC Curve"):
    """
    Create ROC curve plot
    
    Args:
        fpr (array): False positive rates
        tpr (array): True positive rates
        auc_score (float): AUC score
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: ROC curve
    """
    if fpr is None or tpr is None:
        # Generate mock ROC curve
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-5 * fpr)
        auc_score = 0.95 if auc_score is None else auc_score
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True,
        height=400
    )
    
    return fig

def create_precision_recall_curve(precision=None, recall=None, avg_precision=None, title="Precision-Recall Curve"):
    """
    Create precision-recall curve
    
    Args:
        precision (array): Precision values
        recall (array): Recall values
        avg_precision (float): Average precision score
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: PR curve
    """
    if precision is None or recall is None:
        # Generate mock PR curve
        recall = np.linspace(0, 1, 100)
        precision = 0.9 - 0.5 * recall + 0.1 * np.random.random(100)
        precision = np.clip(precision, 0, 1)
        avg_precision = 0.87 if avg_precision is None else avg_precision
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR Curve (AP = {avg_precision:.3f})',
        line=dict(color='green', width=3)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        showlegend=True,
        height=400
    )
    
    return fig

def create_feature_importance_chart(features, importance_scores, title="Feature Importance", top_n=10):
    """
    Create feature importance bar chart
    
    Args:
        features (list): Feature names
        importance_scores (list): Importance scores
        title (str): Chart title
        top_n (int): Number of top features to show
        
    Returns:
        plotly.graph_objects.Figure: Bar chart
    """
    # Create DataFrame and sort by importance
    df = pd.DataFrame({
        'Feature': features,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=True).tail(top_n)
    
    fig = px.bar(
        df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=title,
        labels={'Importance': 'Importance Score', 'Feature': 'Features'},
        color='Importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=max(400, top_n * 30))
    return fig

def create_transaction_volume_chart(timestamps, fraud_flags=None, title="Transaction Volume Over Time"):
    """
    Create transaction volume time series chart
    
    Args:
        timestamps (array): Transaction timestamps
        fraud_flags (array): Fraud indicators (optional)
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Time series chart
    """
    # Convert to datetime if needed
    if isinstance(timestamps[0], (int, float)):
        timestamps = pd.to_datetime(timestamps, unit='s')
    
    # Group by hour
    df = pd.DataFrame({'timestamp': timestamps})
    df['hour'] = df['timestamp'].dt.floor('H')
    
    volume_data = df.groupby('hour').size().reset_index(name='count')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=volume_data['hour'],
        y=volume_data['count'],
        mode='lines+markers',
        name='Total Transactions',
        line=dict(color='blue')
    ))
    
    if fraud_flags is not None:
        fraud_df = pd.DataFrame({'timestamp': timestamps, 'is_fraud': fraud_flags})
        fraud_df['hour'] = fraud_df['timestamp'].dt.floor('H')
        fraud_volume = fraud_df[fraud_df['is_fraud'] == 1].groupby('hour').size().reset_index(name='fraud_count')
        
        fig.add_trace(go.Scatter(
            x=fraud_volume['hour'],
            y=fraud_volume['fraud_count'],
            mode='lines+markers',
            name='Fraud Cases',
            line=dict(color='red')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Transaction Count",
        height=400
    )
    
    return fig

def create_amount_distribution_chart(amounts, fraud_flags=None, title="Transaction Amount Distribution"):
    """
    Create transaction amount distribution chart
    
    Args:
        amounts (array): Transaction amounts
        fraud_flags (array): Fraud indicators (optional)
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Histogram
    """
    if fraud_flags is not None:
        df = pd.DataFrame({'amt': amounts, 'is_fraud': fraud_flags})
        df['fraud_status'] = df['is_fraud'].map({0: 'Normal', 1: 'Fraud'})
        
        fig = px.histogram(
            df,
            x='amt',
            color='fraud_status',
            nbins=50,
            title=title,
            labels={'amt': 'Transaction Amount ($)', 'count': 'Frequency'},
            color_discrete_map={'Normal': 'lightblue', 'Fraud': 'red'}
        )
    else:
        fig = px.histogram(
            x=amounts,
            nbins=50,
            title=title,
            labels={'x': 'Transaction Amount ($)', 'y': 'Frequency'}
        )
    
    fig.update_layout(height=400)
    return fig

def create_geographic_scatter(lat, lon, amounts=None, fraud_flags=None, title="Geographic Distribution"):
    """
    Create geographic scatter plot
    
    Args:
        lat (array): Latitude coordinates
        lon (array): Longitude coordinates
        amounts (array): Transaction amounts (optional)
        fraud_flags (array): Fraud indicators (optional)
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Scatter mapbox
    """
    df = pd.DataFrame({'lat': lat, 'lon': lon})
    
    if amounts is not None:
        df['amount'] = amounts
    
    if fraud_flags is not None:
        df['fraud'] = fraud_flags
        df['fraud_status'] = df['fraud'].map({0: 'Normal', 1: 'Fraud'})
        color_col = 'fraud_status'
        color_map = {'Normal': 'blue', 'Fraud': 'red'}
    else:
        color_col = None
        color_map = None
    
    fig = px.scatter_mapbox(
        df.sample(min(1000, len(df))),  # Sample for performance
        lat='lat',
        lon='lon',
        color=color_col,
        size='amount' if amounts is not None else None,
        title=title,
        mapbox_style="carto-positron",
        zoom=3,
        color_discrete_map=color_map
    )
    
    fig.update_layout(height=500)
    return fig

def create_category_distribution_chart(categories, fraud_flags=None, title="Category Distribution"):
    """
    Create category distribution chart
    
    Args:
        categories (array): Transaction categories
        fraud_flags (array): Fraud indicators (optional)
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Pie or bar chart
    """
    if fraud_flags is not None:
        df = pd.DataFrame({'category': categories, 'is_fraud': fraud_flags})
        
        # Calculate fraud rate by category
        cat_stats = df.groupby('category').agg({
            'is_fraud': ['count', 'sum']
        }).round(3)
        cat_stats.columns = ['Total', 'Fraud_Count']
        cat_stats['Fraud_Rate'] = (cat_stats['Fraud_Count'] / cat_stats['Total'] * 100).round(2)
        cat_stats = cat_stats.sort_values('Fraud_Rate', ascending=False).reset_index()
        
        fig = px.bar(
            cat_stats,
            x='category',
            y='Fraud_Rate',
            title=title,
            labels={'category': 'Category', 'Fraud_Rate': 'Fraud Rate (%)'},
            color='Fraud_Rate',
            color_continuous_scale='Reds'
        )
    else:
        category_counts = pd.Series(categories).value_counts()
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title=title
        )
    
    fig.update_layout(height=400)
    return fig

def create_system_metrics_chart(timestamps, cpu_usage, memory_usage, response_times, title="System Metrics"):
    """
    Create system metrics monitoring chart
    
    Args:
        timestamps (array): Timestamp data
        cpu_usage (array): CPU usage percentages
        memory_usage (array): Memory usage percentages
        response_times (array): Response times in ms
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Multi-subplot chart
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('CPU Usage (%)', 'Memory Usage (%)', 'Response Time (ms)'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(go.Scatter(x=timestamps, y=cpu_usage, mode='lines', name='CPU'), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=memory_usage, mode='lines', name='Memory'), row=2, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=response_times, mode='lines', name='Response Time'), row=3, col=1)
    
    fig.update_layout(height=600, showlegend=False, title_text=title)
    return fig

def create_drift_monitoring_chart(dates, accuracy_scores, precision_scores, drift_scores, title="Model Drift Monitoring"):
    """
    Create model drift monitoring chart
    
    Args:
        dates (array): Date timestamps
        accuracy_scores (array): Model accuracy over time
        precision_scores (array): Model precision over time
        drift_scores (array): Data drift scores
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Multi-subplot chart
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Model Performance', 'Data Drift Score'),
        vertical_spacing=0.2
    )
    
    fig.add_trace(go.Scatter(x=dates, y=accuracy_scores, mode='lines+markers', name='Accuracy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=precision_scores, mode='lines+markers', name='Precision'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=drift_scores, mode='lines+markers', name='Drift Score', 
                            line=dict(color='red')), row=2, col=1)
    
    # Add drift threshold line
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", row=2, col=1)
    
    fig.update_layout(height=500, title_text=title)
    return fig

def create_hourly_pattern_chart(hours, transaction_counts, fraud_counts=None, title="Hourly Transaction Patterns"):
    """
    Create hourly transaction pattern chart
    
    Args:
        hours (array): Hour of day (0-23)
        transaction_counts (array): Transaction counts per hour
        fraud_counts (array): Fraud counts per hour (optional)
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Bar chart
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hours,
        y=transaction_counts,
        name='Total Transactions',
        opacity=0.7
    ))
    
    if fraud_counts is not None:
        fig.add_trace(go.Bar(
            x=hours,
            y=fraud_counts,
            name='Fraud Cases',
            opacity=0.9
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Hour of Day",
        yaxis_title="Transaction Count",
        barmode='overlay' if fraud_counts is not None else 'group',
        height=400
    )
    
    return fig