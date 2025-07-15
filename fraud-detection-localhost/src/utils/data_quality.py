# src/utils/data_quality.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class DataQualityChecker:
    """Data quality assessment and validation"""
    
    def __init__(self):
        self.quality_rules = self.define_quality_rules()
        
    def define_quality_rules(self) -> Dict[str, Dict[str, Any]]:
        """Define data quality rules for fraud detection data"""
        return {
            'amt': {
                'min_value': 0.01,
                'max_value': 10000,
                'required': True,
                'type': 'numeric'
            },
            'cc_num': {
                'required': True,
                'min_length': 13,
                'max_length': 19,
                'type': 'string'
            },
            'lat': {
                'min_value': -90,
                'max_value': 90,
                'required': True,
                'type': 'numeric'
            },
            'long': {
                'min_value': -180,
                'max_value': 180,
                'required': True,
                'type': 'numeric'
            },
            'city_pop': {
                'min_value': 1,
                'max_value': 50000000,
                'required': True,
                'type': 'numeric'
            }
        }
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        logger.info("Starting data quality assessment...")
        
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_value_analysis': self.check_missing_values(df),
            'outlier_analysis': self.check_outliers(df),
            'duplicate_analysis': self.check_duplicates(df),
            'data_type_analysis': self.check_data_types(df),
            'business_rule_violations': self.check_business_rules(df),
            'overall_quality_score': 0.0
        }
        
        # Calculate overall quality score
        quality_report['overall_quality_score'] = self.calculate_quality_score(quality_report)
        
        logger.info(f"Data quality assessment complete. Score: {quality_report['overall_quality_score']:.2f}")
        return quality_report
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values"""
        missing_analysis = {}
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            
            missing_analysis[column] = {
                'missing_count': int(missing_count),
                'missing_percentage': round(missing_percentage, 2)
            }
        
        # Overall missing data statistics
        total_missing = sum([info['missing_count'] for info in missing_analysis.values()])
        total_cells = len(df) * len(df.columns)
        overall_missing_percentage = (total_missing / total_cells) * 100
        
        return {
            'column_analysis': missing_analysis,
            'total_missing_values': total_missing,
            'overall_missing_percentage': round(overall_missing_percentage, 2),
            'columns_with_missing': [col for col, info in missing_analysis.items() 
                                   if info['missing_count'] > 0]
        }
    
    def check_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for outliers in numerical columns"""
        outlier_analysis = {}
        
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numerical_columns:
            if column in df.columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                
                outlier_analysis[column] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': round((len(outliers) / len(df)) * 100, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2),
                    'q1': round(Q1, 2),
                    'q3': round(Q3, 2)
                }
        
        return outlier_analysis
    
    def check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate records"""
        duplicate_analysis = {
            'total_duplicates': int(df.duplicated().sum()),
            'duplicate_percentage': round((df.duplicated().sum() / len(df)) * 100, 2),
            'unique_rows': int(len(df) - df.duplicated().sum())
        }
        
        # Check for duplicates in key identifier columns
        if 'trans_num' in df.columns:
            trans_num_duplicates = df['trans_num'].duplicated().sum()
            duplicate_analysis['trans_num_duplicates'] = int(trans_num_duplicates)
        
        if 'cc_num' in df.columns:
            # Check for suspicious patterns in credit card numbers
            cc_value_counts = df['cc_num'].value_counts()
            suspicious_cc = cc_value_counts[cc_value_counts > 100]  # More than 100 transactions per card
            duplicate_analysis['suspicious_cc_patterns'] = len(suspicious_cc)
        
        return duplicate_analysis
    
    def check_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data types and format consistency"""
        type_analysis = {}
        
        for column in df.columns:
            dtype = str(df[column].dtype)
            type_analysis[column] = {
                'current_type': dtype,
                'non_null_count': int(df[column].count()),
                'unique_values': int(df[column].nunique())
            }
            
            # Check if numeric columns have string values
            if dtype == 'object':
                try:
                    pd.to_numeric(df[column], errors='raise')
                    type_analysis[column]['could_be_numeric'] = True
                except:
                    type_analysis[column]['could_be_numeric'] = False
        
        return type_analysis
    
    def check_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check business rule violations"""
        violations = {}
        
        for column, rules in self.quality_rules.items():
            if column in df.columns:
                column_violations = []
                
                # Check required fields
                if rules.get('required', False):
                    null_count = df[column].isnull().sum()
                    if null_count > 0:
                        column_violations.append(f"Missing values: {null_count}")
                
                # Check numeric ranges
                if rules.get('type') == 'numeric' and column in df.select_dtypes(include=[np.number]).columns:
                    if 'min_value' in rules:
                        below_min = (df[column] < rules['min_value']).sum()
                        if below_min > 0:
                            column_violations.append(f"Values below minimum: {below_min}")
                    
                    if 'max_value' in rules:
                        above_max = (df[column] > rules['max_value']).sum()
                        if above_max > 0:
                            column_violations.append(f"Values above maximum: {above_max}")
                
                # Check string lengths
                if rules.get('type') == 'string' and column in df.select_dtypes(include=['object']).columns:
                    if 'min_length' in rules:
                        below_min_length = (df[column].str.len() < rules['min_length']).sum()
                        if below_min_length > 0:
                            column_violations.append(f"Strings too short: {below_min_length}")
                    
                    if 'max_length' in rules:
                        above_max_length = (df[column].str.len() > rules['max_length']).sum()
                        if above_max_length > 0:
                            column_violations.append(f"Strings too long: {above_max_length}")
                
                if column_violations:
                    violations[column] = column_violations
        
        return violations
    
    def calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        
        # Deduct points for missing values
        missing_percentage = quality_report['missing_value_analysis']['overall_missing_percentage']
        score -= min(missing_percentage * 2, 30)  # Max 30 points deduction
        
        # Deduct points for duplicates
        duplicate_percentage = quality_report['duplicate_analysis']['duplicate_percentage']
        score -= min(duplicate_percentage * 1.5, 20)  # Max 20 points deduction
        
        # Deduct points for business rule violations
        violation_count = len(quality_report['business_rule_violations'])
        score -= min(violation_count * 5, 25)  # Max 25 points deduction
        
        # Deduct points for excessive outliers
        outlier_columns = quality_report['outlier_analysis']
        high_outlier_columns = [col for col, info in outlier_columns.items() 
                               if info['outlier_percentage'] > 10]
        score -= min(len(high_outlier_columns) * 10, 25)  # Max 25 points deduction
        
        return max(score, 0.0)
    
    def get_quality_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        # Missing value recommendations
        missing_analysis = quality_report['missing_value_analysis']
        if missing_analysis['overall_missing_percentage'] > 5:
            recommendations.append("High percentage of missing values detected. Consider imputation strategies.")
        
        # Duplicate recommendations
        duplicate_analysis = quality_report['duplicate_analysis']
        if duplicate_analysis['duplicate_percentage'] > 1:
            recommendations.append("Duplicate records found. Review data collection process.")
        
        # Business rule violation recommendations
        violations = quality_report['business_rule_violations']
        if violations:
            recommendations.append("Business rule violations detected. Data validation needed.")
        
        # Outlier recommendations
        outlier_analysis = quality_report['outlier_analysis']
        high_outlier_columns = [col for col, info in outlier_analysis.items() 
                               if info['outlier_percentage'] > 15]
        if high_outlier_columns:
            recommendations.append(f"High outlier percentage in columns: {', '.join(high_outlier_columns)}")
        
        # Overall score recommendations
        if quality_report['overall_quality_score'] < 70:
            recommendations.append("Overall data quality is below acceptable threshold. Comprehensive data cleaning required.")
        elif quality_report['overall_quality_score'] < 85:
            recommendations.append("Data quality is acceptable but could be improved.")
        
        return recommendations