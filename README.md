1. Environment Setup
- Clone repository: git clone <repository_url>
- cd fraud-detection-localhost

# Create environment file
cp .env.example .env

# Build and start services
docker-compose up -d
2. Access Points

API Documentation: http://localhost:8000/docs
Dashboard: http://localhost:8501
Jupyter Lab: http://localhost:8888 (token: fraudtoken123)
Prometheus: http://localhost:9090
Database: localhost:5432

3. Initial Data Setup
bash# Generate sample data
docker-compose exec ml-api python scripts/generate_sample_data.py

# Train initial model
docker-compose exec ml-api python scripts/train_model.py
