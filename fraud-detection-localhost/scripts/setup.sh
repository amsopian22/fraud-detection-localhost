#!/bin/bash
# scripts/setup.sh

set -e

echo "🔍 Setting up Fraud Detection System..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/{raw,processed,features,samples}
mkdir -p models/{trained_models,model_registry}
mkdir -p logs
mkdir -p sql
mkdir -p monitoring

# Create environment file
echo "🔧 Creating environment file..."
cat > .env << EOF
# Database Configuration
POSTGRES_DB=frauddb
POSTGRES_USER=frauduser
POSTGRES_PASSWORD=fraudpass123
DATABASE_URL=postgresql://frauduser:fraudpass123@postgres:5432/frauddb

# Redis Configuration
REDIS_URL=redis://redis:6379

# API Configuration
API_URL=http://ml-api:8000
LOG_LEVEL=INFO

# Model Configuration
MODEL_PATH=/app/models
MODEL_VERSION=v1.0.0

# Dashboard Configuration
DASHBOARD_PORT=8501
JUPYTER_TOKEN=fraudtoken123
EOF

# Create SQL initialization file
echo "🗄️ Creating database schema..."
cat > sql/init.sql << 'EOF'
-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    cc_num VARCHAR(255),
    merchant VARCHAR(255),
    category VARCHAR(50),
    amt DECIMAL(10,2),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    gender CHAR(1),
    street VARCHAR(255),
    city VARCHAR(100),
    state CHAR(2),
    zip VARCHAR(10),
    lat DECIMAL(10,6),
    long DECIMAL(10,6),
    city_pop INTEGER,
    job VARCHAR(100),
    dob DATE,
    merch_lat DECIMAL(10,6),
    merch_long DECIMAL(10,6),
    merch_zipcode VARCHAR(10),
    unix_time BIGINT,
    is_fraud BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) NOT NULL,
    fraud_probability DECIMAL(5,4),
    is_fraud_predicted BOOLEAN,
    risk_level VARCHAR(10),
    model_version VARCHAR(50),
    processing_time_ms DECIMAL(8,3),
    features_used TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);

-- Create model_metrics table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_score DECIMAL(5,4),
    training_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(created_at);
CREATE INDEX IF NOT EXISTS idx_transactions_fraud ON transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions(risk_level);

-- Insert sample model metrics
INSERT INTO model_metrics (model_version, accuracy, precision_score, recall_score, f1_score, auc_score, training_date)
VALUES ('v1.0.0', 0.9500, 0.9000, 0.8500, 0.8750, 0.9200, CURRENT_TIMESTAMP)
ON CONFLICT DO NOTHING;
EOF

# Create prometheus configuration
echo "📊 Creating monitoring configuration..."
mkdir -p monitoring
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'fraud-api'
    static_configs:
      - targets: ['ml-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 60s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 60s
EOF

# Create sample data generator
echo "📊 Creating sample data generator..."
cat > scripts/generate_sample_data.py << 'EOF'
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_sample_transactions(n_samples=1000, fraud_rate=0.05):
    """Generate synthetic transaction data"""
    
    categories = ['grocery_pos', 'gas_transport', 'misc_net', 'grocery_net', 'entertainment', 'misc_pos']
    states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
    genders = ['M', 'F']
    jobs = ['Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Artist', 'Manager', 'Analyst', 'Developer', 'Designer', 'Consultant']
    
    # Generate base data
    data = []
    for i in range(n_samples):
        # Customer info
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'Chris', 'Amy', 'Robert', 'Emma']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
        
        # Transaction timestamp
        base_time = datetime.now() - timedelta(days=30)
        transaction_time = base_time + timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        # Generate transaction
        transaction = {
            'trans_date_trans_time': transaction_time.strftime('%Y-%m-%d %H:%M:%S'),
            'cc_num': f"{random.randint(1000000000000000, 9999999999999999)}",
            'merchant': f"merchant_{random.randint(1000, 9999)}",
            'category': random.choice(categories),
            'amt': round(random.uniform(5.0, 1000.0), 2),
            'first': random.choice(first_names),
            'last': random.choice(last_names),
            'gender': random.choice(genders),
            'street': f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Elm', 'Cedar'])} St",
            'city': f"City_{random.randint(1, 100)}",
            'state': random.choice(states),
            'zip': f"{random.randint(10000, 99999)}",
            'lat': round(random.uniform(25.0, 49.0), 4),
            'long': round(random.uniform(-125.0, -66.0), 4),
            'city_pop': random.randint(1000, 500000),
            'job': random.choice(jobs),
            'dob': f"{random.randint(1950, 2000)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            'trans_num': f"{random.randint(100000000000000000000000000000000, 999999999999999999999999999999999):032x}",
            'unix_time': int(transaction_time.timestamp()),
            'merch_lat': round(random.uniform(25.0, 49.0), 4),
            'merch_long': round(random.uniform(-125.0, -66.0), 4),
            'merch_zipcode': f"{random.randint(10000, 99999)}"
        }
        
        # Determine if fraud (based on some rules)
        is_fraud = False
        if random.random() < fraud_rate:
            is_fraud = True
            # Make fraudulent transactions more suspicious
            transaction['amt'] = round(random.uniform(500.0, 2000.0), 2)  # Higher amounts
            # Distance between customer and merchant
            distance = abs(transaction['lat'] - transaction['merch_lat']) + abs(transaction['long'] - transaction['merch_long'])
            if distance < 1.0:  # Make them far apart
                transaction['merch_lat'] = round(random.uniform(25.0, 49.0), 4)
                transaction['merch_long'] = round(random.uniform(-125.0, -66.0), 4)
        
        transaction['is_fraud'] = 1 if is_fraud else 0
        data.append(transaction)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating sample fraud detection data...")
    
    # Generate datasets
    train_data = generate_sample_transactions(5000, 0.05)
    val_data = generate_sample_transactions(1000, 0.05)
    test_data = generate_sample_transactions(500, 0.05)
    
    # Create output directory
    os.makedirs('/app/data/processed', exist_ok=True)
    os.makedirs('/app/data/raw', exist_ok=True)
    
    # Save datasets
    train_data.to_csv('/app/data/raw/train_data.csv', index=False)
    val_data.to_csv('/app/data/raw/val_data.csv', index=False)
    test_data.to_csv('/app/data/raw/test_data.csv', index=False)
    
    # Save as parquet for better performance
    train_data.to_parquet('/app/data/processed/train.parquet', index=False)
    val_data.to_parquet('/app/data/processed/validation.parquet', index=False)
    test_data.to_parquet('/app/data/processed/test.parquet', index=False)
    
    print(f"Generated datasets:")
    print(f"  Training: {len(train_data)} transactions ({train_data['is_fraud'].sum()} fraud)")
    print(f"  Validation: {len(val_data)} transactions ({val_data['is_fraud'].sum()} fraud)")
    print(f"  Test: {len(test_data)} transactions ({test_data['is_fraud'].sum()} fraud)")
    print("Data saved to /app/data/")
EOF

# Create model training script
echo "🤖 Creating model training script..."
cat > scripts/train_model.py << 'EOF'
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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
    
    joblib.dump(model, '/app/models/trained_models/xgboost_fraud_detector.joblib')
    
    # Save feature names
    joblib.dump(feature_cols, '/app/models/trained_models/feature_names.joblib')
    
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
EOF

# Create main README
echo "📚 Creating documentation..."
cat > README.md << 'EOF'
# Fraud Detection System - Localhost Edition

A complete end-to-end fraud detection system designed to run on localhost with 8GB RAM and M3 processor.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- 8GB RAM minimum
- 10GB free disk space

### Setup and Run

1. **Clone and setup:**
```bash
git clone <repository_url>
cd fraud-detection-localhost
chmod +x scripts/setup.sh
./scripts/setup.sh
```

2. **Start all services:**
```bash
docker-compose up -d
```

3. **Generate sample data and train model:**
```bash
# Generate synthetic data
docker-compose exec ml-api python scripts/generate_sample_data.py

# Train initial model
docker-compose exec ml-api python scripts/train_model.py
```

### Access Points

- **API Documentation**: http://localhost:8000/docs
- **Streamlit Dashboard**: http://localhost:8501
- **Jupyter Lab**: http://localhost:8888 (token: fraudtoken123)
- **Prometheus Monitoring**: http://localhost:9090

### Usage

1. **Real-time Prediction**: Use the Streamlit dashboard or API endpoints
2. **Model Training**: Run training scripts or use Jupyter notebooks
3. **Monitoring**: View metrics in Prometheus and Streamlit dashboard

### API Examples

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cc_num": "1234567890123456",
    "merchant": "test_merchant",
    "category": "grocery_pos",
    "amt": 50.0,
    "first": "John",
    "last": "Doe",
    "gender": "M",
    "street": "123 Main St",
    "city": "Test City",
    "state": "CA",
    "zip": "12345",
    "lat": 40.7128,
    "long": -74.0060,
    "city_pop": 50000,
    "job": "Engineer",
    "dob": "1980-01-01",
    "merch_lat": 40.7580,
    "merch_long": -73.9855,
    "merch_zipcode": "10001"
  }'
```

## Architecture

- **ML API**: FastAPI with XGBoost model
- **Dashboard**: Streamlit for visualization
- **Database**: PostgreSQL for data storage
- **Cache**: Redis for caching predictions
- **Monitoring**: Prometheus for metrics
- **Development**: Jupyter Lab for experimentation

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in docker-compose.yml
2. **Memory issues**: Reduce batch sizes in training scripts
3. **Model not loading**: Check if model files exist in models/trained_models/

### Logs

```bash
# View API logs
docker-compose logs ml-api

# View all logs
docker-compose logs

# View specific service logs
docker-compose logs [service_name]
```

### Reset System

```bash
# Stop and remove all containers
docker-compose down -v

# Remove all data (be careful!)
sudo rm -rf data/ models/ logs/

# Restart fresh
./scripts/setup.sh
docker-compose up -d
```

## Development

### Adding New Features

1. Modify code in `src/` directory
2. Test in Jupyter notebooks
3. Update API endpoints in `ml-service/app.py`
4. Add dashboard components in `dashboard/streamlit_app.py`

### Model Improvements

1. Use `notebooks/` for experimentation
2. Modify training scripts in `scripts/`
3. Update feature engineering in model training

### Contributing

1. Fork the repository
2. Create feature branch
3. Test changes thoroughly
4. Submit pull request

## License

MIT License - see LICENSE file for details.
EOF

echo "✅ Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Run: docker-compose up -d"
echo "2. Generate data: docker-compose exec ml-api python scripts/generate_sample_data.py"
echo "3. Train model: docker-compose exec ml-api python scripts/train_model.py"
echo "4. Open dashboard: http://localhost:8501"
echo ""
echo "For more information, see README.md"
