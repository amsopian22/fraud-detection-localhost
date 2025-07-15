
# src/models/model_evaluation.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, 
    confusion_matrix, classification_report, average_precision_score
)
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and monitoring"""
    
    def __init__(self):
        self.evaluation_history = []
        
    def evaluate_binary_classifier(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
        """Comprehensive evaluation of binary classifier"""
        
        logger.info(f"Evaluating {model_name}...")
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'average_precision': average_precision_score(y_true, y_prob)
        }
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Additional metrics
        metrics.update({
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
        })
        
        # Business metrics for fraud detection
        total_fraud_amount = 1000000  # Simulated total potential fraud amount
        detected_fraud_amount = metrics['recall'] * total_fraud_amount
        false_alarm_cost = metrics['false_positive_rate'] * 100000  # Cost of investigating false positives
        
        metrics.update({
            'detected_fraud_amount': detected_fraud_amount,
            'false_alarm_cost': false_alarm_cost,
            'net_benefit': detected_fraud_amount - false_alarm_cost
        })
        
        # Store evaluation
        evaluation_record = {
            'model_name': model_name,
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics,
            'sample_size': len(y_true),
            'fraud_rate': y_true.mean()
        }
        self.evaluation_history.append(evaluation_record)
        
        logger.info(f"Evaluation complete for {model_name}")
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str = "Model") -> go.Figure:
        """Plot interactive confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Legitimate', 'Predicted Fraud'],
            y=['Actual Legitimate', 'Actual Fraud'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400
        )
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      model_name: str = "Model") -> go.Figure:
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   model_name: str = "Model") -> go.Figure:
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'{model_name} (AP = {avg_precision:.3f})',
            line=dict(color='green', width=2)
        ))
        
        # Baseline (random classifier)
        baseline = y_true.mean()
        fig.add_hline(y=baseline, line_dash="dash", line_color="red",
                     annotation_text=f"Baseline = {baseline:.3f}")
        
        fig.update_layout(
            title=f'Precision-Recall Curve - {model_name}',
            xaxis_title='Recall',
            yaxis_title='Precision',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_threshold_analysis(self, y_true: np.ndarray, y_prob: np.ndarray,
                              model_name: str = "Model") -> go.Figure:
        """Analyze performance across different thresholds"""
        thresholds = np.arange(0.1, 1.0, 0.05)
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_prob > threshold).astype(int)
            
            if len(np.unique(y_pred_thresh)) > 1:
                precision_scores.append(precision_score(y_true, y_pred_thresh, zero_division=0))
                recall_scores.append(recall_score(y_true, y_pred_thresh, zero_division=0))
                f1_scores.append(f1_score(y_true, y_pred_thresh, zero_division=0))
            else:
                precision_scores.append(0)
                recall_scores.append(0)
                f1_scores.append(0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=thresholds, y=precision_scores, name='Precision', mode='lines'))
        fig.add_trace(go.Scatter(x=thresholds, y=recall_scores, name='Recall', mode='lines'))
        fig.add_trace(go.Scatter(x=thresholds, y=f1_scores, name='F1-Score', mode='lines'))
        
        # Find optimal threshold (max F1)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        fig.add_vline(x=optimal_threshold, line_dash="dash", line_color="black",
                     annotation_text=f"Optimal Threshold = {optimal_threshold:.2f}")
        
        fig.update_layout(
            title=f'Threshold Analysis - {model_name}',
            xaxis_title='Threshold',
            yaxis_title='Score',
            height=500,
            showlegend=True
        )
        
        return fig, optimal_threshold
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     model_name: str = "Model") -> pd.DataFrame:
        """Generate detailed classification report"""
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Convert to DataFrame for better display
        df_report = pd.DataFrame(report).transpose()
        df_report = df_report.round(4)
        
        return df_report