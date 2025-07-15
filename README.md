# Environment Setup

1. **Clone repository**

```bash
git clone <repository_url>
cd fraud-detection-localhost
```

2. **Create environment file**

```bash
cp .env.example .env
```

3. **Build and start services**

```bash
docker-compose up -d
```

---

# Access Points

- **API Documentation:** http://localhost:8000/docs
- **Dashboard:** http://localhost:8501
- **Jupyter Lab:** http://localhost:8888 (token: fraudtoken123)
- **Prometheus:** http://localhost:9090
- **Database:** localhost:5432

---

# Initial Data Setup

1. **Generate sample data**

```bash
docker-compose exec ml-api python scripts/generate_sample_data.py
```

2. **Train initial model**

```bash
docker-compose exec ml-api python scripts/train_model.py
```
