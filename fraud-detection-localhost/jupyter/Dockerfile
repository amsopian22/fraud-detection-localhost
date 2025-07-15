# jupyter/Dockerfile
FROM jupyter/scipy-notebook:latest

USER root

# Install additional packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

USER $NB_UID

# Copy requirements
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional ML packages
RUN pip install --no-cache-dir \
    xgboost==1.7.4 \
    lightgbm==3.3.5 \
    shap==0.41.0 \
    optuna==3.1.0

# Set working directory
WORKDIR /home/jovyan/work

# Expose port
EXPOSE 8888