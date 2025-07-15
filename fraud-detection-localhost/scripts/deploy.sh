#!/bin/bash
# Deployment script for different environments

set -e

ENVIRONMENT=${1:-localhost}
FORCE_REBUILD=${2:-false}

echo "🚀 Deploying Fraud Detection System to $ENVIRONMENT"
echo "=================================================="

# Load configuration
if [ ! -f "deployment_config.yaml" ]; then
    echo "❌ deployment_config.yaml not found"
    exit 1
fi

# Function to extract config values (requires yq)
get_config() {
    if command -v yq &> /dev/null; then
        yq eval ".environments.$ENVIRONMENT.$1" deployment_config.yaml
    else
        echo "⚠️ yq not found, using defaults"
        echo "docker-compose.yml"
    fi
}

# Get environment-specific configuration
COMPOSE_FILE=$(get_config "docker_compose_file")
ENV_FILE=$(get_config "env_file")
HEALTH_URL=$(get_config "health_check_url")

echo "📋 Configuration:"
echo "  Environment: $ENVIRONMENT"
echo "  Compose File: $COMPOSE_FILE"
echo "  Env File: $ENV_FILE"
echo "  Health Check: $HEALTH_URL"

# Pre-deployment checks
echo -e "\n🔍 Pre-deployment checks..."

# Check if compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "❌ Docker Compose file not found: $COMPOSE_FILE"
    exit 1
fi

# Check if env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "⚠️ Environment file not found: $ENV_FILE"
    echo "Creating default environment file..."
    cp .env.example "$ENV_FILE"
fi

# Backup (for production/staging)
if [ "$ENVIRONMENT" != "localhost" ]; then
    echo -e "\n💾 Creating backup..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_dir="./backups/${ENVIRONMENT}_${timestamp}"
    mkdir -p "$backup_dir"
    
    # Backup database
    if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "Up"; then
        echo "Backing up database..."
        docker-compose -f "$COMPOSE_FILE" exec -T postgres \
            pg_dump -U frauduser frauddb > "$backup_dir/database.sql"
    fi
    
    # Backup models
    if docker-compose -f "$COMPOSE_FILE" ps ml-api | grep -q "Up"; then
        echo "Backing up models..."
        docker cp $(docker-compose -f "$COMPOSE_FILE" ps -q ml-api):/app/models "$backup_dir/"
    fi
    
    echo "✅ Backup created: $backup_dir"
fi

# Stop existing services
echo -e "\n⏹️ Stopping existing services..."
docker-compose -f "$COMPOSE_FILE" down

# Build images (if force rebuild or production)
if [ "$FORCE_REBUILD" = "true" ] || [ "$ENVIRONMENT" = "production" ]; then
    echo -e "\n🔨 Building images..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache
fi

# Start services
echo -e "\n▶️ Starting services..."
if [ -f "$ENV_FILE" ]; then
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
else
    docker-compose -f "$COMPOSE_FILE" up -d
fi

# Wait for services to be ready
echo -e "\n⏳ Waiting for services to be ready..."
sleep 30

# Health check
echo -e "\n🏥 Performing health check..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    echo "Attempt $attempt/$max_attempts..."
    
    if curl -f -s "$HEALTH_URL" > /dev/null; then
        echo "✅ Health check passed!"
        break
    fi
    
    if [ $attempt -eq $max_attempts ]; then
        echo "❌ Health check failed after $max_attempts attempts"
        echo "Checking service logs..."
        docker-compose -f "$COMPOSE_FILE" logs --tail=20
        exit 1
    fi
    
    sleep 10
    ((attempt++))
done

# Post-deployment tasks
echo -e "\n🔧 Post-deployment tasks..."

# Run database migrations (if needed)
if [ "$ENVIRONMENT" != "localhost" ]; then
    echo "Running database migrations..."
    docker-compose -f "$COMPOSE_FILE" exec -T ml-api \
        python -c "
import os
from sqlalchemy import create_engine
from src.config.database import Base

engine = create_engine(os.getenv('DATABASE_URL'))
Base.metadata.create_all(engine)
print('Database schema updated')
" || echo "⚠️ Migration script not found or failed"
fi

# Warm up model
echo "Warming up model..."
curl -s -X POST "$HEALTH_URL/../predict" \
    -H "Content-Type: application/json" \
    -d '{
        "cc_num": "1234567890123456",
        "merchant": "warmup_merchant",
        "category": "grocery_pos",
        "amt": 50.0,
        "first": "Warmup",
        "last": "Test",
        "gender": "M",
        "street": "123 Test St",
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
    }' > /dev/null && echo "✅ Model warmed up" || echo "⚠️ Model warmup failed"

# Final verification
echo -e "\n✅ Deployment verification..."
docker-compose -f "$COMPOSE_FILE" ps

echo -e "\n🎉 Deployment to $ENVIRONMENT completed successfully!"
echo "📊 Access points:"
if [ "$ENVIRONMENT" = "localhost" ]; then
    echo "  API: http://localhost:8000/docs"
    echo "  Dashboard: http://localhost:8501"
    echo "  Jupyter: http://localhost:8888 (token: fraudtoken123)"
    echo "  Monitoring: http://localhost:9090"
else
    echo "  API: $HEALTH_URL"
    echo "  Dashboard: https://dashboard.frauddetection.com"
    echo "  Monitoring: https://monitoring.frauddetection.com"
fi

echo -e "\n📝 Next steps:"
echo "  1. Verify all services are running correctly"
echo "  2. Test API endpoints"
echo "  3. Check monitoring dashboards"
echo "  4. Review logs for any issues"

if [ "$ENVIRONMENT" != "localhost" ]; then
    echo "  5. Update DNS records if needed"
    echo "  6. Configure SSL certificates"
    echo "  7. Set up monitoring alerts"
fi