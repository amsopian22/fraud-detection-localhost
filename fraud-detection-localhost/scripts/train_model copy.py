import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
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
    
    # Geographic distance between customer and merchant
    def haversine_distance(lat1, lon1, lat2, lon2):
        from math import radians, cos, sin, asin, sqrt
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return c * 6371  # Earth radius in km
    
    features['customer_merchant_distance'] = features.apply(
        lambda x: haversine_distance(x['lat'], x['long'], x['merch_lat'], x['merch_long']), axis=1
    )
    
    # Time features
    features['trans_datetime'] = pd.to_datetime(features['trans_date_trans_time'])
    features['hour'] = features['trans_datetime'].dt.hour
    features['day_of_week'] = features['trans_datetime'].dt.dayofweek
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    features['is_night'] = ((features['hour'] < 6) | (features['hour'] > 22)).astype(int)
    
    # Age calculation
    features['dob'] = pd.to_datetime(features['dob'], errors='coerce')
    features['customer_age'] = (datetime.now() - features['dob']).dt.days / 365.25
    features['customer_age'] = features['customer_age'].fillna(features['customer_age'].median())
    
    # Categorical encoding
    le = LabelEncoder()
    features['gender_encoded'] = le.fit_transform(features['gender'])
    features['category_encoded'] = le.fit_transform(features['category'])
    features['state_encoded'] = le.fit_transform(features['state'])
    
    return features

def train_model():
    """Train fraud detection model"""
    print("Loading training data...")
    
    # Load data
    train_data = pd.read_parquet('/app/data/processed/train.parquet')
    val_data = pd.read_parquet('/app/data/processed/validation.parquet')
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Fraud rate in training: {train_data['is_fraud'].mean():.3f}")
    
    # Create features
    print("Creating features...")
    train_features = create_features(train_data)
    val_features = create_features(val_data)
    
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
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Features: {feature_cols}")
    
    # Train XGBoost model
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    train_auc = roc_auc_score(y_train, train_prob)
    val_auc = roc_auc_score(y_val, val_prob)
    
    print("\n=== Training Results ===")
    print(f"Training AUC: {train_auc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    
    print("\n=== Validation Classification Report ===")
    print(classification_report(y_val, val_pred))
    
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
    
    #joblib.dump(model, '/app/models/trained_models/xgboost_fraud_detector.joblib')
    
    # Save feature names
    #joblib.dump(feature_cols, '/app/models/trained_models/feature_names.joblib')
    
    joblib.dump(model, '/app/models/xgboost_fraud_detector.joblib')
    joblib.dump(feature_cols, '/app/models/feature_names.joblib')
    
    # Save model metadata
    metadata = {
        'model_name': 'XGBoost Fraud Detector',
        'version': 'v1.0.0',
        'training_date': datetime.now().isoformat(),
        'features': feature_cols,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'metrics': {
            'train_auc': float(train_auc),
            'validation_auc': float(val_auc),
            'accuracy': float((val_pred == y_val).mean()),
            'precision': float(((val_pred == 1) & (y_val == 1)).sum() / (val_pred == 1).sum()) if (val_pred == 1).sum() > 0 else 0.0,
            'recall': float(((val_pred == 1) & (y_val == 1)).sum() / (y_val == 1).sum()) if (y_val == 1).sum() > 0 else 0.0
        },
        'feature_importance': feature_importance.to_dict('records')
    }
    
    with open('/app/models/trained_models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Model training completed successfully!")
    print("Model saved to: /app/models/trained_models/")
    
    return model, metadata

if __name__ == "__main__":
    model, metadata = train_model()
