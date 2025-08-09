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
2. **Train data fraud (credit_card)**

```bash
docker-compose exec ml-api python scripts/train_model.py
```

3. **Realtime Generate sample transaction**

```bash
docker-compose exec ml-api python scripts/realtime_service.py
```

---

# Local Development

For running the services locally without Docker:

1. **Install dependencies**

   It is recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   pip install -r fraud-detection-localhost/dashboard/requirements.txt
   ```

2. **Run the ML API**

   From the root of the project, run:

   ```bash
   PYTHONPATH=. uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```

   The `PYTHONPATH=.` is important to ensure the API can find the `src` modules. The API will be available at `http://127.0.0.1:8000`.

3. **Run the Streamlit Dashboard**

   In a separate terminal, from the root of the project, run:

   ```bash
   streamlit run fraud-detection-localhost/dashboard/streamlit_app.py
   ```

   The dashboard will be available at `http://localhost:8501`.
