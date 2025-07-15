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
