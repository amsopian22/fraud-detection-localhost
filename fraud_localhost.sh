#!/bin/bash
# filepath: generate_structure.sh

set -e

mkdir -p fraud-detection-localhost/{data/{raw,processed,features,samples},src/{config,data_processing,feature_engineering,models,api/{endpoints,models,middleware},utils},ml-service,dashboard/{pages,components},jupyter,notebooks,models/{trained_models,model_registry},sql,monitoring/grafana/dashboards,scripts,tests/{test_api,test_models,test_features},logs,docs}

touch fraud-detection-localhost/README.md
touch fraud-detection-localhost/docker-compose.yml
touch fraud-detection-localhost/.env
touch fraud-detection-localhost/.gitignore
touch fraud-detection-localhost/requirements.txt

# Data files
touch fraud-detection-localhost/data/raw/fraud_data.csv
touch fraud-detection-localhost/data/processed/train.parquet
touch fraud-detection-localhost/data/processed/validation.parquet
touch fraud-detection-localhost/data/processed/test.parquet
touch fraud-detection-localhost/data/features/feature_store.parquet
touch fraud-detection-localhost/data/samples/sample_transactions.json

# src/config
touch fraud-detection-localhost/src/__init__.py
touch fraud-detection-localhost/src/config/__init__.py
touch fraud-detection-localhost/src/config/settings.py
touch fraud-detection-localhost/src/config/database.py

# src/data_processing
touch fraud-detection-localhost/src/data_processing/__init__.py
touch fraud-detection-localhost/src/data_processing/ingestion.py
touch fraud-detection-localhost/src/data_processing/preprocessing.py
touch fraud-detection-localhost/src/data_processing/validation.py

# src/feature_engineering
touch fraud-detection-localhost/src/feature_engineering/__init__.py
touch fraud-detection-localhost/src/feature_engineering/core_features.py
touch fraud-detection-localhost/src/feature_engineering/geo_features.py
touch fraud-detection-localhost/src/feature_engineering/temporal_features.py
touch fraud-detection-localhost/src/feature_engineering/aggregation_features.py

# src/models
touch fraud-detection-localhost/src/models/__init__.py
touch fraud-detection-localhost/src/models/base_model.py
touch fraud-detection-localhost/src/models/xgboost_model.py
touch fraud-detection-localhost/src/models/random_forest_model.py
touch fraud-detection-localhost/src/models/model_evaluation.py

# src/api
touch fraud-detection-localhost/src/api/__init__.py
touch fraud-detection-localhost/src/api/main.py
touch fraud-detection-localhost/src/api/endpoints/__init__.py
touch fraud-detection-localhost/src/api/endpoints/prediction.py
touch fraud-detection-localhost/src/api/endpoints/training.py
touch fraud-detection-localhost/src/api/endpoints/monitoring.py
touch fraud-detection-localhost/src/api/models/__init__.py
touch fraud-detection-localhost/src/api/models/request_models.py
touch fraud-detection-localhost/src/api/models/response_models.py
touch fraud-detection-localhost/src/api/middleware/__init__.py
touch fraud-detection-localhost/src/api/middleware/logging.py
touch fraud-detection-localhost/src/api/middleware/monitoring.py

# src/utils
touch fraud-detection-localhost/src/utils/__init__.py
touch fraud-detection-localhost/src/utils/helpers.py
touch fraud-detection-localhost/src/utils/metrics.py
touch fraud-detection-localhost/src/utils/visualization.py

# ml-service
touch fraud-detection-localhost/ml-service/Dockerfile
touch fraud-detection-localhost/ml-service/requirements.txt
touch fraud-detection-localhost/ml-service/app.py
touch fraud-detection-localhost/ml-service/model_service.py

# dashboard
touch fraud-detection-localhost/dashboard/Dockerfile
touch fraud-detection-localhost/dashboard/requirements.txt
touch fraud-detection-localhost/dashboard/streamlit_app.py
touch fraud-detection-localhost/dashboard/pages/1_Data_Overview.py
touch fraud-detection-localhost/dashboard/pages/2_Model_Performance.py
touch fraud-detection-localhost/dashboard/pages/3_Real_Time_Prediction.py
touch fraud-detection-localhost/dashboard/pages/4_Model_Monitoring.py
touch fraud-detection-localhost/dashboard/components/__init__.py
touch fraud-detection-localhost/dashboard/components/charts.py
touch fraud-detection-localhost/dashboard/components/metrics.py
touch fraud-detection-localhost/dashboard/components/data_loader.py

# jupyter
touch fraud-detection-localhost/jupyter/Dockerfile
touch fraud-detection-localhost/jupyter/requirements.txt

# notebooks
touch fraud-detection-localhost/notebooks/01_data_exploration.ipynb
touch fraud-detection-localhost/notebooks/02_feature_engineering.ipynb
touch fraud-detection-localhost/notebooks/03_model_training.ipynb
touch fraud-detection-localhost/notebooks/04_model_evaluation.ipynb
touch fraud-detection-localhost/notebooks/05_model_interpretation.ipynb

# models
touch fraud-detection-localhost/models/trained_models/xgboost_fraud_detector.joblib
touch fraud-detection-localhost/models/trained_models/feature_encoder.joblib
touch fraud-detection-localhost/models/trained_models/model_metadata.json
touch fraud-detection-localhost/models/model_registry/model_versions.json

# sql
touch fraud-detection-localhost/sql/init.sql
touch fraud-detection-localhost/sql/create_tables.sql
touch fraud-detection-localhost/sql/sample_data.sql

# monitoring
touch fraud-detection-localhost/monitoring/prometheus.yml
touch fraud-detection-localhost/monitoring/grafana/dashboards/fraud_detection.json

# scripts
touch fraud-detection-localhost/scripts/setup.sh
touch fraud-detection-localhost/scripts/train_model.py
touch fraud-detection-localhost/scripts/generate_sample_data.py
touch fraud-detection-localhost/scripts/run_tests.py

# tests
touch fraud-detection-localhost/tests/__init__.py
touch fraud-detection-localhost/tests/test_api/__init__.py
touch fraud-detection-localhost/tests/test_api/test_prediction.py
touch fraud-detection-localhost/tests/test_api/test_training.py
touch fraud-detection-localhost/tests/test_models/__init__.py
touch fraud-detection-localhost/tests/test_models/test_xgboost.py
touch fraud-detection-localhost/tests/test_features/__init__.py
touch fraud-detection-localhost/tests/test_features/test_feature_engineering.py

# logs
touch fraud-detection-localhost/logs/api.log
touch fraud-detection-localhost/logs/training.log
touch fraud-detection-localhost/logs/monitoring.log

# docs
touch fraud-detection-localhost/docs/api_documentation.md
touch fraud-detection-localhost/docs/model_documentation.md
touch fraud-detection-localhost/docs/deployment_guide.md
touch fraud-detection-localhost/docs/troubleshooting.md

echo "Project structure created successfully."