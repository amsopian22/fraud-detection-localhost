import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    """Create features for fraud detection"""
    features = df.copy()
    
    # Basic features
    features['amt_log'] = np.log1p(features['amt'])
    features['city_pop_log'] = np.log1p(features['city_pop'])
    
    # Geographic distance between customer and merchant (vectorized)
    def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
        from math import radians
        lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 6371  # Earth radius in km
    
    print("Calculating distances...")
    features['customer_merchant_distance'] = haversine_distance_vectorized(
        features['lat'], features['long'], features['merch_lat'], features['merch_long']
    )
    
    # Time features
    print("Processing time features...")
    features['trans_datetime'] = pd.to_datetime(features['trans_date_trans_time'])
    features['hour'] = features['trans_datetime'].dt.hour
    features['day_of_week'] = features['trans_datetime'].dt.dayofweek
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    features['is_night'] = ((features['hour'] < 6) | (features['hour'] > 22)).astype(int)
    
    # Age calculation
    print("Calculating customer age...")
    features['dob'] = pd.to_datetime(features['dob'], errors='coerce')
    features['customer_age'] = (datetime.now() - features['dob']).dt.days / 365.25
    features['customer_age'] = features['customer_age'].fillna(features['customer_age'].median())
    
    # Categorical encoding
    print("Encoding categorical features...")
    le_gender = LabelEncoder()
    le_category = LabelEncoder()
    le_state = LabelEncoder()
    features['gender_encoded'] = le_gender.fit_transform(features['gender'])
    features['category_encoded'] = le_category.fit_transform(features['category'])
    features['state_encoded'] = le_state.fit_transform(features['state'])
    
    return features

def train_model():
    """Train fraud detection model"""
    print("Loading training data...")
    
    # Load data from CSV files
    train_data = pd.read_csv('/app/data/raw/credit_card_transaction_train.csv')
    test_data = pd.read_csv('/app/data/raw/credit_card_transaction_test.csv')
    
    # Split test data into validation and test sets
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42, stratify=test_data['is_fraud'])
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Fraud rate in training: {train_data['is_fraud'].mean():.3f}")
    print(f"Fraud rate in validation: {val_data['is_fraud'].mean():.3f}")
    
    # Create features
    print("Creating features...")
    train_features = create_features(train_data)
    val_features = create_features(val_data)
    test_features = create_features(test_data)
    
    # Select feature columns
    feature_cols = [
        'amt', 'amt_log', 'city_pop_log', 'customer_merchant_distance',
        'hour', 'day_of_week', 'is_weekend', 'is_night',
        'customer_age', 'gender_encoded', 'category_encoded', 'state_encoded'
    ]
    
    X_train = train_features[feature_cols]
    y_train = train_features['is_fraud']
    X_val = val_features[feature_cols]
    y_val = val_features['is_fraud']
    X_test = test_features[feature_cols]
    y_test = test_features['is_fraud']
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Features: {feature_cols}")
    
    # Train multiple models for comparison
    models = {}
    
    # 1. XGBoost model (optimized for high recall)
    print("Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=6,
        tree_method='hist',
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.4,
        random_state=42,
        scale_pos_weight=150,  # Increased to emphasize fraud class (was 99)
        eval_metric='logloss',
        # Additional parameters for better recall
        min_child_weight=1,    # Allow more specialized splits
        gamma=0,               # No pruning to capture more fraud patterns
        reg_alpha=0.1,         # Light L1 regularization
        reg_lambda=1.0         # L2 regularization
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # 2. Random Forest model (balanced approach)
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced_subsample',  # Handle imbalanced data
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    
    # 3. LightGBM model (fast and efficient)
    print("Training LightGBM model...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.4,
        class_weight='balanced',
        random_state=42,
        is_unbalance=True,  # Handle imbalanced data
        min_child_samples=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        verbose=-1  # Suppress output
    )
    lgb_model.fit(X_train, y_train)
    models['LightGBM'] = lgb_model
    
    # Evaluate all models
    print("\n" + "="*60)
    print("EVALUATING ALL MODELS")
    print("="*60)
    
    model_results = {}
    best_model = None
    best_score = 0
    
    for model_name, model in models.items():
        print(f"\n=== {model_name} Model Results ===")
        
        # Make predictions
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Calculate AUC scores
        val_auc = roc_auc_score(y_val, val_prob)
        test_auc = roc_auc_score(y_test, test_prob)
        
        print(f"Validation AUC: {val_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        # Find optimal threshold for high recall (>=0.95)
        precision, recall, thresholds = precision_recall_curve(y_val, val_prob)
        
        # Find threshold that gives recall >= 0.95
        high_recall_indices = recall >= 0.95
        if np.any(high_recall_indices):
            best_idx = np.where(high_recall_indices)[0][0]
            optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.1
            optimal_precision = precision[best_idx]
            optimal_recall = recall[best_idx]
            
            print(f"Optimal threshold for recall ≥ 0.95: {optimal_threshold:.4f}")
            print(f"At this threshold - Precision: {optimal_precision:.4f}, Recall: {optimal_recall:.4f}")
            
            # Apply optimal threshold
            val_pred_optimized = (val_prob >= optimal_threshold).astype(int)
            test_pred_optimized = (test_prob >= optimal_threshold).astype(int)
        else:
            print("No threshold found that achieves recall >= 0.95")
            optimal_threshold = 0.1
            val_pred_optimized = (val_prob >= optimal_threshold).astype(int)
            test_pred_optimized = (test_prob >= optimal_threshold).astype(int)
            optimal_recall = 0.0
        
        # Calculate optimized recall
        optimized_val_recall = ((val_pred_optimized == 1) & (y_val == 1)).sum() / (y_val == 1).sum() if (y_val == 1).sum() > 0 else 0.0
        optimized_test_recall = ((test_pred_optimized == 1) & (y_test == 1)).sum() / (y_test == 1).sum() if (y_test == 1).sum() > 0 else 0.0
        
        print(f"Validation Classification Report (Default):")
        print(classification_report(y_val, val_pred))
        
        print(f"High Recall Results (Threshold: {optimal_threshold:.4f}):")
        print(classification_report(y_val, val_pred_optimized))
        
        # Store results
        model_results[model_name] = {
            'model': model,
            'val_auc': val_auc,
            'test_auc': test_auc,
            'optimal_threshold': optimal_threshold,
            'optimized_val_recall': optimized_val_recall,
            'optimized_test_recall': optimized_test_recall,
            'val_prob': val_prob,
            'test_prob': test_prob,
            'val_pred_optimized': val_pred_optimized,
            'test_pred_optimized': test_pred_optimized
        }
        
        # Determine best model based on validation AUC and recall
        combined_score = val_auc + optimized_val_recall  # Combine AUC and recall
        if combined_score > best_score:
            best_score = combined_score
            best_model = model_name
    
    # Select best model for saving
    print(f"\n" + "="*60)
    print(f"BEST MODEL: {best_model}")
    print(f"Combined Score (AUC + Recall): {best_score:.4f}")
    print("="*60)
    
    # Use best model for final evaluation and saving
    model = model_results[best_model]['model']
    val_prob = model_results[best_model]['val_prob']
    test_prob = model_results[best_model]['test_prob']
    optimal_threshold = model_results[best_model]['optimal_threshold']
    val_pred_optimized = model_results[best_model]['val_pred_optimized']
    test_pred_optimized = model_results[best_model]['test_pred_optimized']
    val_auc = model_results[best_model]['val_auc']
    test_auc = model_results[best_model]['test_auc']
    
    print(f"\n=== Final Results for Best Model ({best_model}) ===")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Get default predictions for the best model
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    print("\n=== Validation Classification Report (Default Threshold) ===")
    print(classification_report(y_val, val_pred))
    
    print("\n=== Test Classification Report (Default Threshold) ===")
    print(classification_report(y_test, test_pred))
    
    print(f"\n=== High Recall Results (Threshold: {optimal_threshold:.4f}) ===")
    print("Validation Set:")
    print(classification_report(y_val, val_pred_optimized))
    
    print("Test Set:")
    print(classification_report(y_test, test_pred_optimized))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Feature Importance ===")
    print(feature_importance)
    
    # Save model
    print("Saving model...")
    os.makedirs('/app/models/trained_models', exist_ok=True)
    
    joblib.dump(model, '/app/models/trained_models/xgboost_fraud_detector.joblib')
    
    # Save feature names
    joblib.dump(feature_cols, '/app/models/trained_models/feature_names.joblib')
    
    # Save model metadata
    metadata = {
        'model_name': f'{best_model} Fraud Detector',
        'best_model_type': best_model,
        'version': 'v2.0.0',
        'training_date': datetime.now().isoformat(),
        'features': feature_cols,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'models_compared': list(models.keys()),
        'model_selection_score': float(best_score),
        'metrics': {
            'validation_auc': float(val_auc),
            'test_auc': float(test_auc),
            'val_accuracy': float((val_pred == y_val).mean()),
            'test_accuracy': float((test_pred == y_test).mean()),
            'val_precision': float(((val_pred == 1) & (y_val == 1)).sum() / (val_pred == 1).sum()) if (val_pred == 1).sum() > 0 else 0.0,
            'val_recall': float(((val_pred == 1) & (y_val == 1)).sum() / (y_val == 1).sum()) if (y_val == 1).sum() > 0 else 0.0,
            'test_precision': float(((test_pred == 1) & (y_test == 1)).sum() / (test_pred == 1).sum()) if (test_pred == 1).sum() > 0 else 0.0,
            'test_recall': float(((test_pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum()) if (y_test == 1).sum() > 0 else 0.0,
            'optimal_threshold_for_high_recall': float(optimal_threshold),
            'optimized_val_precision': float(((val_pred_optimized == 1) & (y_val == 1)).sum() / (val_pred_optimized == 1).sum()) if (val_pred_optimized == 1).sum() > 0 else 0.0,
            'optimized_val_recall': float(((val_pred_optimized == 1) & (y_val == 1)).sum() / (y_val == 1).sum()) if (y_val == 1).sum() > 0 else 0.0,
            'optimized_test_precision': float(((test_pred_optimized == 1) & (y_test == 1)).sum() / (test_pred_optimized == 1).sum()) if (test_pred_optimized == 1).sum() > 0 else 0.0,
            'optimized_test_recall': float(((test_pred_optimized == 1) & (y_test == 1)).sum() / (y_test == 1).sum()) if (y_test == 1).sum() > 0 else 0.0
        },
        'feature_importance': feature_importance.to_dict('records'),
        'all_model_results': {name: {'val_auc': float(results['val_auc']), 'test_auc': float(results['test_auc']), 'optimized_val_recall': float(results['optimized_val_recall'])} for name, results in model_results.items()}
    }
    
    with open('/app/models/trained_models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Model training completed successfully!")
    print("Model saved to: /app/models/trained_models/")
    
    return model, metadata

if __name__ == "__main__":
    model, metadata = train_model()
