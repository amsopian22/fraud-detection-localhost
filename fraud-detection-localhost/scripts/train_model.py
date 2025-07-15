import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def reduce_mem_usage(df):
    """Optimize dataframe memory usage"""
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def create_features_original(df):
    """Original feature engineering approach with haversine distance"""
    features = df.copy()
    
    # Basic features
    features['amt_log'] = np.log1p(features['amt'])
    features['city_pop_log'] = np.log1p(features['city_pop'])
    
    # Geographic distance between customer and merchant (vectorized)
    def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
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

def create_features_enhanced(df):
    """Enhanced feature engineering based on notebook's 96% recall method"""
    print("Starting enhanced feature engineering...")
    features = df.copy()
    
    # Step 1: Sort by card number and transaction time for proper time-based features
    features['trans_date_trans_time'] = pd.to_datetime(features['trans_date_trans_time'])
    features = features.sort_values(by=['cc_num', 'trans_date_trans_time'])
    
    # Step 2: Time-based features per card (KEY FOR 96% RECALL)
    print("Creating transaction time features...")
    features['transaction_time'] = features.groupby('cc_num')['trans_date_trans_time'].diff().dt.total_seconds() / 3600
    features['transaction_std'] = features.groupby('cc_num')['amt'].transform('std')
    features['transaction_avg'] = features.groupby('cc_num')['amt'].transform('mean')
    
    # Step 3: Age calculation (2025 reference as in notebook)
    print("Calculating customer age...")
    features['dob'] = pd.to_datetime(features['dob'], errors='coerce')
    features['age'] = 2025 - features['dob'].dt.year
    
    # Step 4: Temporal features
    print("Creating temporal features...")
    features['day'] = features['trans_date_trans_time'].dt.day
    features['month'] = features['trans_date_trans_time'].dt.month
    features['hour'] = features['trans_date_trans_time'].dt.hour
    
    # Step 5: Drop unnecessary columns (as done in notebook)
    to_drop = ['Unnamed: 0', 'first', 'last', 'merch_lat', 'merch_long', 'job', 'dob', 
               'street', 'city', 'state', 'zip', 'city_pop', 'merch_zipcode', 
               'unix_time', 'trans_num', 'cc_num', 'trans_date_trans_time']
    
    # Only drop columns that exist
    to_drop_existing = [col for col in to_drop if col in features.columns]
    features = features.drop(columns=to_drop_existing)
    
    print(f"Features after processing: {list(features.columns)}")
    print(f"Final feature shape: {features.shape}")
    
    return features

def train_model():
    """Comprehensive training with both original and enhanced approaches"""
    print("=" * 80)
    print("COMPREHENSIVE FRAUD DETECTION MODEL TRAINING")
    print("=" * 80)
    
    # Configuration - choose approach
    USE_ENHANCED_APPROACH = True  # Set False for original approach
    USE_UNDERSAMPLING = True      # Set False to skip undersampling
    
    print(f"Approach: {'Enhanced (96% Recall Method)' if USE_ENHANCED_APPROACH else 'Original (Multi-Model)'}")
    print(f"Undersampling: {'Enabled' if USE_UNDERSAMPLING else 'Disabled'}")
    
    # Load and optimize memory
    print("\nLoading training data...")
    train_data = pd.read_csv('/app/data/raw/credit_card_transaction_train.csv')
    test_data = pd.read_csv('/app/data/raw/credit_card_transaction_test.csv')
    
    print("Optimizing memory usage...")
    train_data = reduce_mem_usage(train_data)
    test_data = reduce_mem_usage(test_data)
    
    # Split test data into validation and test sets
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42, stratify=test_data['is_fraud'])
    
    print(f"\nTraining data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Fraud rate in training: {train_data['is_fraud'].mean():.3f}")
    print(f"Fraud rate in validation: {val_data['is_fraud'].mean():.3f}")
    
    # Feature Engineering
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    if USE_ENHANCED_APPROACH:
        train_features = create_features_enhanced(train_data)
        val_features = create_features_enhanced(val_data)
        test_features = create_features_enhanced(test_data)
        
        # Enhanced categorical encoding
        print("Encoding categorical features...")
        cat_features = ['merchant', 'category', 'gender']
        label_encoders = {}
        
        for col in cat_features:
            if col in train_features.columns:
                le = LabelEncoder()
                train_features[col] = le.fit_transform(train_features[col])
                # Apply same encoding to validation and test, handle unseen values
                for data in [val_features, test_features]:
                    if col in data.columns:
                        # Handle unseen categories by setting them to -1
                        unseen_mask = ~data[col].isin(le.classes_)
                        data[col] = data[col].map(dict(zip(le.classes_, le.transform(le.classes_)))).fillna(-1)
                label_encoders[col] = le
        
        # Handle imbalanced data with undersampling if enabled
        if USE_UNDERSAMPLING:
            print("Applying undersampling to balance data...")
            X_temp = train_features.drop(['is_fraud'], axis=1)
            y_temp = train_features['is_fraud']
            
            # Use 20% fraud ratio as in notebook
            rus = RandomUnderSampler(sampling_strategy=0.2, random_state=42)
            X_train_resampled, y_train_resampled = rus.fit_resample(X_temp, y_temp)
            
            # Convert back to dataframes
            train_balanced = pd.concat([
                pd.DataFrame(X_train_resampled, columns=X_temp.columns), 
                pd.DataFrame(y_train_resampled, columns=['is_fraud'])
            ], axis=1)
            
            print(f"After undersampling - fraud rate: {train_balanced['is_fraud'].mean():.4f}")
            print(f"Training samples after undersampling: {len(train_balanced)}")
            
            X_train = train_balanced.drop(['is_fraud'], axis=1)
            y_train = train_balanced['is_fraud']
        else:
            X_train = train_features.drop(['is_fraud'], axis=1)
            y_train = train_features['is_fraud']
        
        X_val = val_features.drop(['is_fraud'], axis=1)
        y_val = val_features['is_fraud']
        X_test = test_features.drop(['is_fraud'], axis=1)
        y_test = test_features['is_fraud']
        
        # Normalization (MinMaxScaler for enhanced approach)
        print("Normalizing features with MinMaxScaler...")
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]
        X_val = X_val.dropna()
        y_val = y_val.loc[X_val.index]
        X_test = X_test.dropna()
        y_test = y_test.loc[X_test.index]
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
    else:
        # Original approach
        train_features = create_features_original(train_data)
        val_features = create_features_original(val_data)
        test_features = create_features_original(test_data)
        
        # Original feature selection
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
        
        # Normalization (StandardScaler for original approach)
        print("Normalizing features with StandardScaler...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    print(f"Final feature matrix shape: {X_train_scaled.shape}")
    print(f"Features: {list(X_train_scaled.columns)}")
    
    # Model Training
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    models = {}
    
    # 1. XGBoost model with enhanced or original configuration
    print("Training XGBoost model...")
    if USE_ENHANCED_APPROACH:
        xgb_model = xgb.XGBClassifier(
            n_estimators=2000,           # As in notebook
            max_depth=6,                 # As in notebook  
            learning_rate=0.02,          # As in notebook
            subsample=0.8,               # As in notebook
            colsample_bytree=0.4,        # As in notebook
            scale_pos_weight=99,         # Exact value from notebook
            tree_method='hist',          # As in notebook
            eval_metric='auc',           # As in notebook
            missing=-1,                  # As in notebook
            random_state=42
        )
    else:
        xgb_model = xgb.XGBClassifier(
            n_estimators=2000,
            max_depth=6,
            tree_method='hist',
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.4,
            random_state=42,
            scale_pos_weight=150,  # Higher for original approach
            eval_metric='logloss',
            min_child_weight=1,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
    
    xgb_model.fit(X_train_scaled, y_train)
    models['XGBoost'] = xgb_model
    
    # 2. Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    models['RandomForest'] = rf_model
    
    # 3. LightGBM model
    print("Training LightGBM model...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.4,
        class_weight='balanced',
        random_state=42,
        is_unbalance=True,
        min_child_samples=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        verbose=-1
    )
    lgb_model.fit(X_train_scaled, y_train)
    models['LightGBM'] = lgb_model
    
    # Model Evaluation
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    model_results = {}
    best_model = None
    best_score = 0
    
    for model_name, model in models.items():
        print(f"\n=== {model_name} Model Results ===")
        
        # Make predictions
        val_prob = model.predict_proba(X_val_scaled)[:, 1]
        test_prob = model.predict_proba(X_test_scaled)[:, 1]
        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Calculate AUC scores
        val_auc = roc_auc_score(y_val, val_prob)
        test_auc = roc_auc_score(y_test, test_prob)
        
        print(f"Validation AUC: {val_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        # Enhanced threshold optimization
        if USE_ENHANCED_APPROACH and model_name == 'XGBoost':
            # Apply notebook's successful threshold of 0.4 for XGBoost
            threshold_04 = 0.4  # Notebook's key threshold for 96% recall
            val_pred_04 = (val_prob >= threshold_04).astype(int)
            test_pred_04 = (test_prob >= threshold_04).astype(int)
            
            # Calculate recall with 0.4 threshold
            val_recall_04 = ((val_pred_04 == 1) & (y_val == 1)).sum() / (y_val == 1).sum() if (y_val == 1).sum() > 0 else 0.0
            test_recall_04 = ((test_pred_04 == 1) & (y_test == 1)).sum() / (y_test == 1).sum() if (y_test == 1).sum() > 0 else 0.0
            
            print(f"🎯 SPECIAL: Recall with 0.4 threshold - Val: {val_recall_04:.4f}, Test: {test_recall_04:.4f}")
            
            # Use 0.4 threshold for final evaluation
            val_pred_optimized = val_pred_04
            test_pred_optimized = test_pred_04
            optimal_threshold = threshold_04
        else:
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
    
    # Final Results
    print(f"\n" + "=" * 60)
    print(f"BEST MODEL: {best_model}")
    print(f"Combined Score (AUC + Recall): {best_score:.4f}")
    print("=" * 60)
    
    # Use best model for final evaluation and saving
    model = model_results[best_model]['model']
    val_prob = model_results[best_model]['val_prob']
    test_prob = model_results[best_model]['test_prob']
    optimal_threshold = model_results[best_model]['optimal_threshold']
    val_pred_optimized = model_results[best_model]['val_pred_optimized']
    test_pred_optimized = model_results[best_model]['test_pred_optimized']
    val_auc = model_results[best_model]['val_auc']
    test_auc = model_results[best_model]['test_auc']
    
    # Get default predictions for the best model
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    
    print(f"\n=== Final Results for Best Model ({best_model}) ===")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
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
        'feature': list(X_train_scaled.columns),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Feature Importance ===")
    print(feature_importance)
    
    # Save model and artifacts
    print("\nSaving model and artifacts...")
    os.makedirs('/app/models/trained_models', exist_ok=True)
    
    joblib.dump(model, '/app/models/trained_models/xgboost_fraud_detector.joblib')
    joblib.dump(list(X_train_scaled.columns), '/app/models/trained_models/feature_names.joblib')
    joblib.dump(scaler, '/app/models/trained_models/feature_scaler.joblib')
    
    if USE_ENHANCED_APPROACH:
        joblib.dump(label_encoders, '/app/models/trained_models/label_encoders.joblib')
    
    # Save comprehensive metadata
    metadata = {
        'model_name': f'{best_model} Comprehensive Fraud Detector',
        'best_model_type': best_model,
        'version': 'v4.0.0',
        'training_date': datetime.now().isoformat(),
        'approach': 'Enhanced' if USE_ENHANCED_APPROACH else 'Original',
        'undersampling_enabled': USE_UNDERSAMPLING,
        'features': list(X_train_scaled.columns),
        'training_samples': len(X_train_scaled),
        'validation_samples': len(X_val_scaled),
        'test_samples': len(X_test_scaled),
        'models_compared': list(models.keys()),
        'model_selection_score': float(best_score),
        'preprocessing': {
            'memory_optimization': True,
            'undersampling_ratio': 0.2 if USE_UNDERSAMPLING else None,
            'normalization': 'MinMaxScaler' if USE_ENHANCED_APPROACH else 'StandardScaler',
            'categorical_encoding': 'LabelEncoder'
        },
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