-- sql/realtime_schema.sql
-- Database schema for real-time fraud detection

-- Table untuk menyimpan prediksi real-time
CREATE TABLE IF NOT EXISTS realtime_predictions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) UNIQUE NOT NULL,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Transaction details
    amount DECIMAL(10,2) NOT NULL,
    merchant_name VARCHAR(255),
    category VARCHAR(100),
    customer_location VARCHAR(255),
    merchant_location VARCHAR(255),
    
    -- Prediction results
    fraud_probability DECIMAL(5,4) NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    confidence_score DECIMAL(5,4),
    
    -- Model info
    model_version VARCHAR(50),
    processing_time_ms INTEGER,
    
    -- Raw transaction data (JSON)
    transaction_data JSONB,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_realtime_predictions_timestamp 
ON realtime_predictions(prediction_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_realtime_predictions_fraud 
ON realtime_predictions(is_fraud);

CREATE INDEX IF NOT EXISTS idx_realtime_predictions_risk 
ON realtime_predictions(risk_level);

CREATE INDEX IF NOT EXISTS idx_realtime_predictions_amount 
ON realtime_predictions(amount);

-- Table untuk tracking simulation stats
CREATE TABLE IF NOT EXISTS simulation_stats (
    id SERIAL PRIMARY KEY,
    date DATE DEFAULT CURRENT_DATE,
    total_transactions INTEGER DEFAULT 0,
    total_frauds INTEGER DEFAULT 0,
    fraud_rate DECIMAL(5,4) DEFAULT 0,
    avg_amount DECIMAL(10,2) DEFAULT 0,
    avg_processing_time_ms INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- View untuk dashboard metrics
CREATE OR REPLACE VIEW dashboard_metrics AS
SELECT 
    COUNT(*) as total_predictions,
    COUNT(*) FILTER (WHERE is_fraud = true) as total_frauds,
    ROUND(AVG(fraud_probability), 4) as avg_fraud_probability,
    ROUND(AVG(amount), 2) as avg_transaction_amount,
    ROUND(AVG(processing_time_ms), 2) as avg_processing_time,
    COUNT(*) FILTER (WHERE prediction_timestamp > NOW() - INTERVAL '1 hour') as predictions_last_hour,
    COUNT(*) FILTER (WHERE is_fraud = true AND prediction_timestamp > NOW() - INTERVAL '1 hour') as frauds_last_hour,
    COUNT(*) FILTER (WHERE risk_level = 'HIGH') as high_risk_count,
    COUNT(*) FILTER (WHERE risk_level = 'MEDIUM') as medium_risk_count,
    COUNT(*) FILTER (WHERE risk_level = 'LOW') as low_risk_count
FROM realtime_predictions;

-- View untuk recent predictions
CREATE OR REPLACE VIEW recent_predictions AS
SELECT 
    transaction_id,
    prediction_timestamp,
    amount,
    merchant_name,
    category,
    fraud_probability,
    is_fraud,
    risk_level,
    processing_time_ms
FROM realtime_predictions 
ORDER BY prediction_timestamp DESC 
LIMIT 100;

-- Function untuk cleanup old data (keep last 7 days)
CREATE OR REPLACE FUNCTION cleanup_old_predictions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM realtime_predictions 
    WHERE prediction_timestamp < NOW() - INTERVAL '7 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function untuk update simulation stats
CREATE OR REPLACE FUNCTION update_simulation_stats()
RETURNS VOID AS $$
BEGIN
    INSERT INTO simulation_stats (
        date, 
        total_transactions, 
        total_frauds, 
        fraud_rate, 
        avg_amount,
        avg_processing_time_ms
    )
    SELECT 
        CURRENT_DATE,
        COUNT(*),
        COUNT(*) FILTER (WHERE is_fraud = true),
        ROUND(COUNT(*) FILTER (WHERE is_fraud = true)::DECIMAL / NULLIF(COUNT(*), 0), 4),
        ROUND(AVG(amount), 2),
        ROUND(AVG(processing_time_ms))
    FROM realtime_predictions 
    WHERE prediction_timestamp::date = CURRENT_DATE
    ON CONFLICT (date) DO UPDATE SET
        total_transactions = EXCLUDED.total_transactions,
        total_frauds = EXCLUDED.total_frauds,
        fraud_rate = EXCLUDED.fraud_rate,
        avg_amount = EXCLUDED.avg_amount,
        avg_processing_time_ms = EXCLUDED.avg_processing_time_ms,
        last_updated = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;