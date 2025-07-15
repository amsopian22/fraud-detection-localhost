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
