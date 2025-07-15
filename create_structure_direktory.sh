#!/bin/bash

# Root project folder
mkdir -p realtime_fraud_detection
cd realtime_fraud_detection

# Top-level files
touch docker-compose.yml requirements.txt .env

# data_producer
mkdir -p data_producer
touch data_producer/__init__.py data_producer/transaction_producer.py data_producer/data_generator.py

# flink_processor
mkdir -p flink_processor
touch flink_processor/__init__.py flink_processor/fraud_detector.py flink_processor/feature_engineering.py flink_processor/kafka_connectors.py

# model_api
mkdir -p model_api/models model_api/services model_api/utils
touch model_api/__init__.py model_api/main.py
touch model_api/models/fraud_model.py model_api/models/model_registry.py
touch model_api/services/prediction_service.py model_api/services/elasticsearch_service.py
touch model_api/utils/feature_store.py model_api/utils/validators.py

# mlops
mkdir -p mlops
touch mlops/__init__.py mlops/training_pipeline.py mlops/model_deployment.py mlops/data_lake.py mlops/monitoring.py

# dashboard
mkdir -p dashboard/src/components dashboard/src/services
touch dashboard/package.json dashboard/src/App.js
touch dashboard/src/components/RealTimeDashboard.js dashboard/src/components/MLOpsMonitoring.js dashboard/src/components/FraudAlerts.js
touch dashboard/src/services/kafkaService.js dashboard/src/services/elasticsearchService.js dashboard/src/services/mlflowService.js

# configs
mkdir -p configs/kafka configs/flink configs/elasticsearch configs/mlflow

# scripts
mkdir -p scripts
touch scripts/setup.sh scripts/start_services.sh scripts/deploy_models.sh

echo "✅ Project structure created successfully!"