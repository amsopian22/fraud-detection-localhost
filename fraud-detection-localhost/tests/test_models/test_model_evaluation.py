# tests/test_models/test_model_evaluation.py
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from models.model_evaluation import ModelEvaluator
from utils.data_quality import DataQualityChecker

class TestModelEvaluation:
    """Test cases for model evaluation utilities"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.y_true = np.random.randint(0, 2, 1000)
        self.y_prob = np.random.random(1000)
        self.y_pred = (self.y_prob > 0.5).astype(int)
        self.evaluator = ModelEvaluator()
    
    def test_binary_classifier_evaluation(self):
        """Test binary classifier evaluation"""
        metrics = self.evaluator.evaluate_binary_classifier(
            self.y_true, self.y_pred, self.y_prob, "Test Model"
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
            'true_positives', 'true_negatives', 'false_positives', 'false_negatives',
            'specificity', 'sensitivity'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Invalid type for {metric}"
        
        # Validate metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_evaluation_history(self):
        """Test evaluation history tracking"""
        initial_count = len(self.evaluator.evaluation_history)
        
        self.evaluator.evaluate_binary_classifier(
            self.y_true, self.y_pred, self.y_prob, "Model 1"
        )
        
        self.evaluator.evaluate_binary_classifier(
            self.y_true, self.y_pred, self.y_prob, "Model 2"
        )
        
        assert len(self.evaluator.evaluation_history) == initial_count + 2
        assert self.evaluator.evaluation_history[-1]['model_name'] == "Model 2"
    
    def test_classification_report_generation(self):
        """Test classification report generation"""
        report_df = self.evaluator.generate_classification_report(
            self.y_true, self.y_pred, "Test Model"
        )
        
        assert isinstance(report_df, pd.DataFrame)
        assert len(report_df) > 0
        assert 'precision' in report_df.columns
        assert 'recall' in report_df.columns
        assert 'f1-score' in report_df.columns

class TestDataQuality:
    """Test cases for data quality assessment"""
    
    def setup_method(self):
        """Setup test data with known quality issues"""
        self.checker = DataQualityChecker()
        
        # Create test data with various quality issues
        self.good_data = pd.DataFrame({
            'amt': [10.0, 50.0, 100.0, 25.0],
            'cc_num': ['1234567890123456', '9876543210987654', '1111222233334444', '5555666677778888'],
            'lat': [40.7128, 34.0522, 41.8781, 39.9526],
            'long': [-74.0060, -118.2437, -87.6298, -75.1652],
            'city_pop': [1000000, 500000, 2000000, 800000]
        })
        
        self.bad_data = pd.DataFrame({
            'amt': [10.0, -50.0, 15000.0, None],  # Negative and extreme values
            'cc_num': ['123', '9876543210987654', None, '5555666677778888'],  # Too short and missing
            'lat': [40.7128, 95.0, 41.8781, None],  # Out of range and missing
            'long': [-74.0060, -200.0, -87.6298, -75.1652],  # Out of range
            'city_pop': [1000000, -500, 70000000, 800000]  # Negative and extreme values
        })
    
    def test_missing_values_check(self):
        """Test missing values detection"""
        result = self.checker.check_missing_values(self.bad_data)
        
        assert 'column_analysis' in result
        assert 'overall_missing_percentage' in result
        assert result['overall_missing_percentage'] > 0
        
        # Check specific columns with missing values
        assert result['column_analysis']['amt']['missing_count'] > 0
        assert result['column_analysis']['cc_num']['missing_count'] > 0
    
    def test_outlier_detection(self):
        """Test outlier detection"""
        result = self.checker.check_outliers(self.bad_data)
        
        # Should detect outliers in amt and city_pop
        assert 'amt' in result
        assert 'city_pop' in result
        
        # Extreme values should be detected as outliers
        assert result['amt']['outlier_count'] > 0
        assert result['city_pop']['outlier_count'] > 0
    
    def test_business_rules_validation(self):
        """Test business rules validation"""
        violations = self.checker.check_business_rules(self.bad_data)
        
        # Should detect violations in multiple columns
        assert len(violations) > 0
        assert 'amt' in violations  # Negative values
        assert 'lat' in violations  # Out of range values
    
    def test_quality_score_calculation(self):
        """Test overall quality score calculation"""
        good_report = self.checker.check_data_quality(self.good_data)
        bad_report = self.checker.check_data_quality(self.bad_data)
        
        # Good data should have higher quality score
        assert good_report['overall_quality_score'] > bad_report['overall_quality_score']
        assert good_report['overall_quality_score'] > 80  # Should be high quality
        assert bad_report['overall_quality_score'] < 60   # Should be low quality
    
    def test_quality_recommendations(self):
        """Test quality recommendations generation"""
        bad_report = self.checker.check_data_quality(self.bad_data)
        recommendations = self.checker.get_quality_recommendations(bad_report)
        
        assert len(recommendations) > 0
        assert any("missing values" in rec.lower() for rec in recommendations)
        assert any("business rule" in rec.lower() for rec in recommendations)