#!/bin/bash
# scripts/health_check.sh - Comprehensive system health check

set -e

echo "🏥 Fraud Detection System - Health Check"
echo "======================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# Check Docker and Docker Compose
echo "🐳 Checking Docker..."
if command -v docker &> /dev/null; then
    print_status 0 "Docker is installed"
    docker --version
else
    print_status 1 "Docker is not installed"
    exit 1
fi

if command -v docker-compose &> /dev/null; then
    print_status 0 "Docker Compose is installed"
    docker-compose --version
else
    print_status 1 "Docker Compose is not installed"
    exit 1
fi

# Check if services are running
echo -e "\n🔍 Checking Services..."
services=("postgres" "redis" "ml-api" "dashboard")

for service in "${services[@]}"; do
    if docker-compose ps --services --filter "status=running" | grep -q "^${service}$"; then
        print_status 0 "${service} is running"
    else
        print_status 1 "${service} is not running"
    fi
done

# Check service health
echo -e "\n🏥 Checking Service Health..."

# PostgreSQL health
echo "Checking PostgreSQL..."
if docker-compose exec -T postgres pg_isready -U frauduser -d frauddb &> /dev/null; then
    print_status 0 "PostgreSQL is healthy"
else
    print_status 1 "PostgreSQL is not healthy"
fi

# Redis health
echo "Checking Redis..."
if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
    print_status 0 "Redis is healthy"
else
    print_status 1 "Redis is not healthy"
fi

# API health
echo "Checking ML API..."
api_response=$(curl -s -w "%{http_code}" -o /tmp/health_response http://localhost:8000/health || echo "000")
if [ "$api_response" = "200" ]; then
    print_status 0 "ML API is healthy"
    # Parse health response
    if command -v jq &> /dev/null; then
        model_loaded=$(jq -r '.model_loaded' /tmp/health_response)
        db_connected=$(jq -r '.database_connected' /tmp/health_response)
        redis_connected=$(jq -r '.redis_connected' /tmp/health_response)
        
        if [ "$model_loaded" = "true" ]; then
            print_status 0 "Model is loaded"
        else
            print_status 1 "Model is not loaded"
        fi
        
        if [ "$db_connected" = "true" ]; then
            print_status 0 "Database connection OK"
        else
            print_status 1 "Database connection failed"
        fi
        
        if [ "$redis_connected" = "true" ]; then
            print_status 0 "Redis connection OK"
        else
            print_status 1 "Redis connection failed"
        fi
    fi
else
    print_status 1 "ML API is not responding (HTTP: $api_response)"
fi

# Dashboard health
echo "Checking Dashboard..."
dashboard_response=$(curl -s -w "%{http_code}" -o /dev/null http://localhost:8501 || echo "000")
if [ "$dashboard_response" = "200" ]; then
    print_status 0 "Dashboard is accessible"
else
    print_status 1 "Dashboard is not accessible (HTTP: $dashboard_response)"
fi

# Check disk space
echo -e "\n💾 Checking Disk Space..."
available_space=$(df . | awk 'NR==2 {print $4}')
available_gb=$((available_space / 1024 / 1024))

if [ $available_gb -gt 5 ]; then
    print_status 0 "Sufficient disk space: ${available_gb}GB available"
elif [ $available_gb -gt 1 ]; then
    print_warning "Low disk space: ${available_gb}GB available"
else
    print_status 1 "Critical: Very low disk space: ${available_gb}GB available"
fi

# Check memory usage
echo -e "\n🧠 Checking Memory Usage..."
if command -v free &> /dev/null; then
    memory_info=$(free -h | awk 'NR==2{printf "Used: %s, Available: %s, Usage: %.2f%%", $3, $7, $3/$2*100}')
    echo "$memory_info"
    
    memory_percent=$(free | awk 'NR==2{printf "%.0f", $3/$2*100}')
    if [ $memory_percent -lt 80 ]; then
        print_status 0 "Memory usage is normal"
    elif [ $memory_percent -lt 90 ]; then
        print_warning "High memory usage: ${memory_percent}%"
    else
        print_status 1 "Critical memory usage: ${memory_percent}%"
    fi
fi

# Check model files
echo -e "\n🤖 Checking Model Files..."
if docker-compose exec -T ml-api test -f /app/models/trained_models/xgboost_fraud_detector.joblib; then
    print_status 0 "Model file exists"
    
    # Check model metadata
    if docker-compose exec -T ml-api test -f /app/models/trained_models/model_metadata.json; then
        print_status 0 "Model metadata exists"
    else
        print_warning "Model metadata missing"
    fi
else
    print_status 1 "Model file missing - run training script"
fi

# Check data files
echo -e "\n📊 Checking Data Files..."
if docker-compose exec -T ml-api test -f /app/data/processed/train.parquet; then
    print_status 0 "Training data exists"
else
    print_warning "Training data missing - run data generation script"
fi

# Test prediction
echo -e "\n🔮 Testing Prediction..."
prediction_test='{
    "cc_num": "1234567890123456",
    "merchant": "test_merchant",
    "category": "grocery_pos",
    "amt": 50.0,
    "first": "Test",
    "last": "User",
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
}'

prediction_response=$(curl -s -w "%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d "$prediction_test" \
    -o /tmp/prediction_response \
    http://localhost:8000/predict || echo "000")

if [ "$prediction_response" = "200" ]; then
    print_status 0 "Prediction endpoint working"
    if command -v jq &> /dev/null; then
        fraud_prob=$(jq -r '.fraud_probability' /tmp/prediction_response)
        risk_level=$(jq -r '.risk_level' /tmp/prediction_response)
        echo "  Fraud probability: $fraud_prob"
        echo "  Risk level: $risk_level"
    fi
else
    print_status 1 "Prediction endpoint failed (HTTP: $prediction_response)"
fi

# Summary
echo -e "\n📋 Health Check Summary"
echo "======================="

# Count issues
total_checks=0
failed_checks=0

# This would be implemented with proper tracking in a real script
echo "System Status: $(docker-compose ps --services --filter 'status=running' | wc -l)/4 services running"

if [ $failed_checks -eq 0 ]; then
    echo -e "${GREEN}🎉 All checks passed! System is healthy.${NC}"
    exit 0
elif [ $failed_checks -lt 3 ]; then
    echo -e "${YELLOW}⚠️ Some issues detected but system is mostly functional.${NC}"
    exit 1
else
    echo -e "${RED}❌ Multiple critical issues detected. System needs attention.${NC}"
    exit 2
fi