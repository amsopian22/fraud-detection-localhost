# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Primary Commands (via Makefile)
- `make setup` - Setup development environment and start all services
- `make test` - Run all tests in Docker containers
- `make test-unit` - Run unit tests only
- `make test-integration` - Run integration tests only  
- `make test-api` - Run API tests only
- `make lint` - Run flake8 and pylint code linting
- `make format` - Format code with black and isort
- `make generate-data` - Generate synthetic transaction data
- `make train-model` - Train the fraud detection model
- `make shell` - Open bash shell in ml-api container

### Docker Operations
- `docker-compose up -d` - Start all services
- `docker-compose logs -f ml-api` - View API logs
- `docker-compose exec ml-api python -m pytest tests/ -v` - Run tests directly

### Service Access Points
- API Documentation: http://localhost:8000/docs
- Streamlit Dashboard: http://localhost:8501
- Jupyter Lab: http://localhost:8888 (token: fraudtoken123)
- Prometheus Monitoring: http://localhost:9090

## Architecture Overview

This is a containerized fraud detection system with these core components:

### Services (docker-compose.yml)
- **ml-api**: FastAPI service (`ml-service/app.py`) - Main fraud detection API with XGBoost model
- **dashboard**: Streamlit visualization service (`dashboard/streamlit_app.py`)
- **postgres**: PostgreSQL database for storing predictions and metadata
- **redis**: Cache and message queue for real-time features
- **jupyter**: Development environment with notebooks
- **prometheus**: Metrics collection (optional)

### Code Structure
- **src/**: Core Python modules organized by functionality
  - `api/`: FastAPI endpoints and middleware
  - `config/`: Application settings and database configuration
  - `data_processing/`: Data ingestion, preprocessing, and batch processing
  - `feature_engineering/`: Feature creation modules (temporal, geo, aggregation)
  - `models/`: Model classes, evaluation, and registry
  - `utils/`: Helper utilities and monitoring
- **ml-service/app.py**: Main FastAPI application (500+ lines) - primary entry point
- **scripts/**: Training, data generation, and deployment scripts
- **tests/**: Test suite with pytest configuration
- **notebooks/**: Jupyter notebooks for experimentation
- **data/**: Raw, processed data and feature store

### Key Features
- Real-time fraud prediction with `/predict` endpoint
- Batch processing via `/predict/batch`
- Real-time dashboard endpoints (`/realtime/*`)
- Feature engineering with distance calculations, time features, and encoding
- Model registry and metadata tracking
- Comprehensive monitoring and health checks

### Model & Data Flow
1. Transactions come in via API endpoints
2. Feature engineering creates derived features (distance, time, categorical encoding)
3. XGBoost model predicts fraud probability
4. Results stored in PostgreSQL and cached in Redis
5. Dashboard displays real-time metrics and visualizations

### Testing
- Uses pytest with coverage reporting (80% minimum)
- Tests marked with: unit, integration, slow, api, model
- Run in Docker containers to match production environment

### Configuration
- Settings in `src/config/settings.py` using Pydantic
- Environment variables for database/redis connections
- Model paths and hyperparameters configurable