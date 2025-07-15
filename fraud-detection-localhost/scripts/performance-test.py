#!/bin/bash
# Performance testing script

echo "⚡ Fraud Detection Performance Test"
echo "=================================="

API_URL=${1:-http://localhost:8000}
CONCURRENT_REQUESTS=${2:-10}
TOTAL_REQUESTS=${3:-100}

echo "Configuration:"
echo "  API URL: $API_URL"
echo "  Concurrent Requests: $CONCURRENT_REQUESTS"
echo "  Total Requests: $TOTAL_REQUESTS"

# Test data
TEST_TRANSACTION='{
    "cc_num": "1234567890123456",
    "merchant": "test_merchant_perf",
    "category": "grocery_pos",
    "amt": 75.50,
    "first": "Performance",
    "last": "Test",
    "gender": "M",
    "street": "123 Perf St",
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

# Single request test
echo -e "\n📊 Single Request Performance:"
curl -w "@curl-format.txt" -s -o /dev/null \
    -X POST "$API_URL/predict" \
    -H "Content-Type: application/json" \
    -d "$TEST_TRANSACTION"

# Concurrent requests test
echo -e "\n🔥 Load Test Results:"
if command -v ab &> /dev/null; then
    # Use Apache Bench if available
    echo "Using Apache Bench (ab)..."
    
    # Save test data to temp file
    echo "$TEST_TRANSACTION" > /tmp/test_data.json
    
    ab -n "$TOTAL_REQUESTS" -c "$CONCURRENT_REQUESTS" \
       -p /tmp/test_data.json \
       -T "application/json" \
       "$API_URL/predict"
    
    rm /tmp/test_data.json
    
elif command -v hey &> /dev/null; then
    # Use hey if available
    echo "Using hey..."
    hey -n "$TOTAL_REQUESTS" -c "$CONCURRENT_REQUESTS" \
        -m POST \
        -H "Content-Type: application/json" \
        -d "$TEST_TRANSACTION" \
        "$API_URL/predict"
        
else
    echo "⚠️ No load testing tool found (ab or hey)"
    echo "Installing hey..."
    
    # Simple concurrent test with curl
    echo "Running simple concurrent test..."
    
    start_time=$(date +%s)
    
    for i in $(seq 1 $CONCURRENT_REQUESTS); do
        (
            for j in $(seq 1 $((TOTAL_REQUESTS / CONCURRENT_REQUESTS))); do
                curl -s -o /dev/null \
                    -X POST "$API_URL/predict" \
                    -H "Content-Type: application/json" \
                    -d "$TEST_TRANSACTION"
            done
        ) &
    done
    
    wait
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "Completed $TOTAL_REQUESTS requests in ${duration}s"
    echo "Average: $((TOTAL_REQUESTS / duration)) requests/second"
fi

# Memory and CPU usage during test
echo -e "\n📈 Resource Usage:"
if command -v docker &> /dev/null; then
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
fi

echo -e "\n✅ Performance test completed!"