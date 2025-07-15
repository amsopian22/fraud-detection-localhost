"""
Metrics calculation and monitoring utilities for the fraud detection dashboard
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_prob (array): Prediction probabilities (optional)
        
    Returns:
        dict: Dictionary of classification metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate additional metrics if probabilities are provided
    auc_score = None
    if y_prob is not None:
        auc_score = calculate_auc_score(y_true, y_prob)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'auc_score': auc_score
    }

def calculate_auc_score(y_true, y_prob):
    """
    Calculate AUC score using trapezoidal rule
    
    Args:
        y_true (array): True labels
        y_prob (array): Prediction probabilities
        
    Returns:
        float: AUC score
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Sort by probability in descending order
    desc_score_indices = np.argsort(y_prob)[::-1]
    y_prob_sorted = y_prob[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    
    # Calculate TPR and FPR for different thresholds
    tpr_list = []
    fpr_list = []
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Random classifier performance
    
    tp = 0
    fp = 0
    
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / n_pos
        fpr = fp / n_neg
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    # Calculate AUC using trapezoidal rule
    fpr_array = np.array([0] + fpr_list)
    tpr_array = np.array([0] + tpr_list)
    
    auc = np.trapz(tpr_array, fpr_array)
    return auc

def calculate_fraud_rate_metrics(df, time_column='datetime', fraud_column='is_fraud'):
    """
    Calculate fraud rate metrics over different time periods
    
    Args:
        df (pd.DataFrame): Transaction data
        time_column (str): Name of datetime column
        fraud_column (str): Name of fraud indicator column
        
    Returns:
        dict: Fraud rate metrics by time period
    """
    df = df.copy()
    
    # Ensure datetime column
    if df[time_column].dtype == 'object' or df[time_column].dtype == 'int64':
        df[time_column] = pd.to_datetime(df[time_column])
    
    # Calculate overall metrics
    total_transactions = len(df)
    total_fraud = df[fraud_column].sum()
    overall_fraud_rate = total_fraud / total_transactions if total_transactions > 0 else 0
    
    # Calculate metrics by hour
    df['hour'] = df[time_column].dt.hour
    hourly_metrics = df.groupby('hour').agg({
        fraud_column: ['count', 'sum']
    })
    hourly_metrics.columns = ['total_transactions', 'fraud_count']
    hourly_metrics['fraud_rate'] = hourly_metrics['fraud_count'] / hourly_metrics['total_transactions']
    
    # Calculate metrics by day of week
    df['day_of_week'] = df[time_column].dt.dayofweek
    daily_metrics = df.groupby('day_of_week').agg({
        fraud_column: ['count', 'sum']
    })
    daily_metrics.columns = ['total_transactions', 'fraud_count']
    daily_metrics['fraud_rate'] = daily_metrics['fraud_count'] / daily_metrics['total_transactions']
    
    return {
        'overall': {
            'total_transactions': total_transactions,
            'total_fraud': int(total_fraud),
            'fraud_rate': overall_fraud_rate
        },
        'hourly': hourly_metrics.to_dict('index'),
        'daily': daily_metrics.to_dict('index')
    }

def calculate_amount_statistics(amounts, fraud_flags=None):
    """
    Calculate transaction amount statistics
    
    Args:
        amounts (array): Transaction amounts
        fraud_flags (array): Fraud indicators (optional)
        
    Returns:
        dict: Amount statistics
    """
    amounts = np.array(amounts)
    
    stats = {
        'count': len(amounts),
        'mean': np.mean(amounts),
        'median': np.median(amounts),
        'std': np.std(amounts),
        'min': np.min(amounts),
        'max': np.max(amounts),
        'q25': np.percentile(amounts, 25),
        'q75': np.percentile(amounts, 75)
    }
    
    if fraud_flags is not None:
        fraud_flags = np.array(fraud_flags)
        fraud_amounts = amounts[fraud_flags == 1]
        normal_amounts = amounts[fraud_flags == 0]
        
        if len(fraud_amounts) > 0:
            stats['fraud_mean'] = np.mean(fraud_amounts)
            stats['fraud_median'] = np.median(fraud_amounts)
            stats['fraud_std'] = np.std(fraud_amounts)
        else:
            stats['fraud_mean'] = 0
            stats['fraud_median'] = 0
            stats['fraud_std'] = 0
        
        if len(normal_amounts) > 0:
            stats['normal_mean'] = np.mean(normal_amounts)
            stats['normal_median'] = np.median(normal_amounts)
            stats['normal_std'] = np.std(normal_amounts)
        else:
            stats['normal_mean'] = 0
            stats['normal_median'] = 0
            stats['normal_std'] = 0
    
    return stats

def calculate_geographic_statistics(lat, lon, fraud_flags=None):
    """
    Calculate geographic distribution statistics
    
    Args:
        lat (array): Latitude coordinates
        lon (array): Longitude coordinates
        fraud_flags (array): Fraud indicators (optional)
        
    Returns:
        dict: Geographic statistics
    """
    lat = np.array(lat)
    lon = np.array(lon)
    
    # Calculate geographic bounds
    stats = {
        'lat_min': np.min(lat),
        'lat_max': np.max(lat),
        'lat_mean': np.mean(lat),
        'lon_min': np.min(lon),
        'lon_max': np.max(lon),
        'lon_mean': np.mean(lon),
        'geographic_spread': np.sqrt(np.var(lat) + np.var(lon))
    }
    
    if fraud_flags is not None:
        fraud_flags = np.array(fraud_flags)
        fraud_lat = lat[fraud_flags == 1]
        fraud_lon = lon[fraud_flags == 1]
        
        if len(fraud_lat) > 0:
            stats['fraud_lat_mean'] = np.mean(fraud_lat)
            stats['fraud_lon_mean'] = np.mean(fraud_lon)
            stats['fraud_geographic_spread'] = np.sqrt(np.var(fraud_lat) + np.var(fraud_lon))
        else:
            stats['fraud_lat_mean'] = 0
            stats['fraud_lon_mean'] = 0
            stats['fraud_geographic_spread'] = 0
    
    return stats

def calculate_model_drift_score(recent_predictions, historical_baseline):
    """
    Calculate model drift score based on prediction distributions
    
    Args:
        recent_predictions (array): Recent prediction probabilities
        historical_baseline (array): Historical prediction probabilities
        
    Returns:
        float: Drift score (0 = no drift, 1 = maximum drift)
    """
    recent_predictions = np.array(recent_predictions)
    historical_baseline = np.array(historical_baseline)
    
    if len(recent_predictions) == 0 or len(historical_baseline) == 0:
        return 0.0
    
    # Calculate distribution differences using KL divergence approximation
    recent_mean = np.mean(recent_predictions)
    historical_mean = np.mean(historical_baseline)
    
    recent_std = np.std(recent_predictions)
    historical_std = np.std(historical_baseline)
    
    # Simple drift score based on mean and std differences
    mean_diff = abs(recent_mean - historical_mean)
    std_diff = abs(recent_std - historical_std)
    
    # Normalize to 0-1 scale
    drift_score = min(1.0, (mean_diff + std_diff) / 2)
    
    return drift_score

def calculate_data_quality_score(df, required_columns=None):
    """
    Calculate data quality score
    
    Args:
        df (pd.DataFrame): Input data
        required_columns (list): Required column names
        
    Returns:
        dict: Data quality metrics
    """
    if required_columns is None:
        required_columns = ['amt', 'category', 'state', 'lat', 'long']
    
    total_rows = len(df)
    
    if total_rows == 0:
        return {
            'completeness': 0.0,
            'validity': 0.0,
            'consistency': 0.0,
            'overall_score': 0.0
        }
    
    # Completeness: percentage of non-null values
    completeness_scores = []
    for col in required_columns:
        if col in df.columns:
            non_null_ratio = df[col].notna().sum() / total_rows
            completeness_scores.append(non_null_ratio)
    
    completeness = np.mean(completeness_scores) if completeness_scores else 0.0
    
    # Validity: percentage of values within expected ranges
    validity_scores = []
    
    # Check amount validity (positive values)
    if 'amt' in df.columns:
        valid_amounts = (df['amt'] > 0) & (df['amt'] < 100000)  # Reasonable range
        validity_scores.append(valid_amounts.sum() / total_rows)
    
    # Check coordinate validity
    if 'lat' in df.columns:
        valid_lat = (df['lat'] >= -90) & (df['lat'] <= 90)
        validity_scores.append(valid_lat.sum() / total_rows)
    
    if 'long' in df.columns:
        valid_lon = (df['long'] >= -180) & (df['long'] <= 180)
        validity_scores.append(valid_lon.sum() / total_rows)
    
    validity = np.mean(validity_scores) if validity_scores else 1.0
    
    # Consistency: no duplicate records
    duplicate_ratio = df.duplicated().sum() / total_rows
    consistency = 1.0 - duplicate_ratio
    
    # Overall score
    overall_score = (completeness + validity + consistency) / 3
    
    return {
        'completeness': completeness,
        'validity': validity,
        'consistency': consistency,
        'overall_score': overall_score,
        'total_records': total_rows,
        'missing_values': total_rows - df.dropna().shape[0],
        'duplicate_records': df.duplicated().sum()
    }

def calculate_system_health_score(cpu_usage, memory_usage, response_times):
    """
    Calculate system health score
    
    Args:
        cpu_usage (array): CPU usage percentages
        memory_usage (array): Memory usage percentages
        response_times (array): Response times in milliseconds
        
    Returns:
        dict: System health metrics
    """
    cpu_usage = np.array(cpu_usage)
    memory_usage = np.array(memory_usage)
    response_times = np.array(response_times)
    
    # Calculate CPU health (lower is better, penalty for high usage)
    cpu_score = np.clip(1.0 - (np.mean(cpu_usage) / 100), 0, 1)
    
    # Calculate memory health
    memory_score = np.clip(1.0 - (np.mean(memory_usage) / 100), 0, 1)
    
    # Calculate response time health (penalty for slow responses)
    avg_response_time = np.mean(response_times)
    response_score = np.clip(1.0 - (avg_response_time / 1000), 0, 1)  # Normalize to seconds
    
    # Overall health score
    overall_score = (cpu_score + memory_score + response_score) / 3
    
    return {
        'cpu_score': cpu_score,
        'memory_score': memory_score,
        'response_score': response_score,
        'overall_score': overall_score,
        'avg_cpu_usage': np.mean(cpu_usage),
        'avg_memory_usage': np.mean(memory_usage),
        'avg_response_time': avg_response_time,
        'max_response_time': np.max(response_times),
        'p95_response_time': np.percentile(response_times, 95)
    }

def calculate_business_impact_metrics(transactions_df, fraud_column='is_fraud', amount_column='amt'):
    """
    Calculate business impact metrics from fraud detection
    
    Args:
        transactions_df (pd.DataFrame): Transaction data
        fraud_column (str): Fraud indicator column
        amount_column (str): Transaction amount column
        
    Returns:
        dict: Business impact metrics
    """
    total_transactions = len(transactions_df)
    total_amount = transactions_df[amount_column].sum()
    
    fraud_transactions = transactions_df[transactions_df[fraud_column] == 1]
    fraud_amount = fraud_transactions[amount_column].sum()
    
    # Calculate losses prevented (assuming all fraud was detected)
    losses_prevented = fraud_amount
    
    # Calculate fraud rate
    fraud_rate = len(fraud_transactions) / total_transactions if total_transactions > 0 else 0
    
    # Average fraud amount
    avg_fraud_amount = fraud_transactions[amount_column].mean() if len(fraud_transactions) > 0 else 0
    avg_normal_amount = transactions_df[transactions_df[fraud_column] == 0][amount_column].mean()
    
    return {
        'total_transactions': total_transactions,
        'total_transaction_volume': total_amount,
        'fraud_transactions': len(fraud_transactions),
        'fraud_volume': fraud_amount,
        'fraud_rate': fraud_rate,
        'losses_prevented': losses_prevented,
        'avg_fraud_amount': avg_fraud_amount,
        'avg_normal_amount': avg_normal_amount,
        'fraud_amount_ratio': fraud_amount / total_amount if total_amount > 0 else 0
    }

def generate_alert_conditions(metrics_dict, thresholds=None):
    """
    Generate alerts based on metric thresholds
    
    Args:
        metrics_dict (dict): Dictionary of calculated metrics
        thresholds (dict): Alert thresholds
        
    Returns:
        list: List of alert dictionaries
    """
    if thresholds is None:
        thresholds = {
            'accuracy_min': 0.95,
            'precision_min': 0.85,
            'recall_min': 0.80,
            'fraud_rate_max': 0.10,
            'response_time_max': 100,  # ms
            'cpu_usage_max': 80,  # %
            'memory_usage_max': 85,  # %
            'drift_score_max': 0.20
        }
    
    alerts = []
    current_time = datetime.now()
    
    # Check model performance alerts
    if 'accuracy' in metrics_dict and metrics_dict['accuracy'] < thresholds['accuracy_min']:
        alerts.append({
            'timestamp': current_time,
            'severity': 'HIGH',
            'component': 'Model Performance',
            'metric': 'accuracy',
            'value': metrics_dict['accuracy'],
            'threshold': thresholds['accuracy_min'],
            'message': f"Model accuracy {metrics_dict['accuracy']:.1%} below threshold {thresholds['accuracy_min']:.1%}"
        })
    
    if 'precision' in metrics_dict and metrics_dict['precision'] < thresholds['precision_min']:
        alerts.append({
            'timestamp': current_time,
            'severity': 'MEDIUM',
            'component': 'Model Performance',
            'metric': 'precision',
            'value': metrics_dict['precision'],
            'threshold': thresholds['precision_min'],
            'message': f"Model precision {metrics_dict['precision']:.1%} below threshold {thresholds['precision_min']:.1%}"
        })
    
    # Check system alerts
    if 'avg_response_time' in metrics_dict and metrics_dict['avg_response_time'] > thresholds['response_time_max']:
        alerts.append({
            'timestamp': current_time,
            'severity': 'MEDIUM',
            'component': 'System Performance',
            'metric': 'response_time',
            'value': metrics_dict['avg_response_time'],
            'threshold': thresholds['response_time_max'],
            'message': f"Average response time {metrics_dict['avg_response_time']:.1f}ms above threshold {thresholds['response_time_max']}ms"
        })
    
    return alerts