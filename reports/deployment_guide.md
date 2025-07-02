# 🚀 Deployment Guide - Food Delivery Time Prediction API

Complete deployment guide for production environments with monitoring, scaling, and maintenance procedures.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Environment Configuration](#environment-configuration)
- [Monitoring and Health Checks](#monitoring-and-health-checks)
- [Scaling and Load Balancing](#scaling-and-load-balancing)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **Docker**: Version 20.10 or higher
- **Memory**: Minimum 2GB RAM (4GB recommended for production)
- **Storage**: At least 5GB available space
- **Network**: Port 8000 available (or custom port)

### Required Tools

```bash
# Docker installation (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Verify installation
docker --version
docker compose --version
```

## Docker Deployment

### 1. Basic Docker Setup

```bash
# Clone the repository
git clone <repository-url>
cd ds_tech_task

# Build the Docker image
docker build -t food-delivery-api:latest .

# Run the container
docker run -d \
  --name food-delivery-container \
  -p 8000:8000 \
  --restart unless-stopped \
  food-delivery-api:latest
```

### 2. Docker Compose Deployment

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  food-delivery-api:
    build: .
    container_name: food-delivery-api
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - PYTHONPATH=/app:/app/model_pipeline
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    container_name: food-delivery-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - food-delivery-api
    restart: unless-stopped

volumes:
  logs:
```

Deploy with Docker Compose:

```bash
# Start services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### 3. Advanced Docker Configuration

#### Multi-stage Dockerfile (Optimized)

```dockerfile
# Build stage
FROM python:3.12-slim-bullseye as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim-bullseye

WORKDIR /app

# Copy dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV PYTHONPATH=/app:/app/model_pipeline \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Production Deployment

### 1. Cloud Deployment (AWS ECS)

#### ECS Task Definition

```json
{
  "family": "food-delivery-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "food-delivery-api",
      "image": "your-ecr-repo/food-delivery-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/food-delivery-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 30
      }
    }
  ]
}
```

### 2. Kubernetes Deployment

#### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: food-delivery-api
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: food-delivery-api
  template:
    metadata:
      labels:
        app: food-delivery-api
    spec:
      containers:
      - name: food-delivery-api
        image: food-delivery-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: food-delivery-api-service
  namespace: production
spec:
  selector:
    app: food-delivery-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Environment Configuration

### Environment Variables

Create `.env` file for environment-specific configurations:

```bash
# Application Settings
APP_NAME=Food Delivery Time Prediction API
APP_VERSION=1.0.0
LOG_LEVEL=INFO
DEBUG=false

# API Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Model Configuration
MODEL_PATH=/app/models/random_forest.joblib
PREPROCESSOR_PATH=/app/models/preprocessor.joblib
MODEL_VERSION=1.0.0

# Security
SECRET_KEY=your-secret-key-here
API_KEY_HEADER=X-API-Key

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090

# Database (if using persistent storage)
DATABASE_URL=sqlite:///./data/food_delivery.db

# External Services
SENTRY_DSN=your-sentry-dsn
NEW_RELIC_LICENSE_KEY=your-newrelic-key
```

### Configuration Management

```python
# config.py updates for production
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Food Delivery Time Prediction API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Model settings
    model_path: str = "/app/models/random_forest.joblib"
    preprocessor_path: str = "/app/models/preprocessor.joblib"
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "fallback-secret-key")
    api_key_header: str = "X-API-Key"
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    class Config:
        env_file = ".env"

settings = Settings()
```

## Monitoring and Health Checks

### 1. Health Check Endpoints

The API provides comprehensive health monitoring:

```bash
# Basic health check
curl -X GET "http://localhost:8000/health"

# Detailed health information
curl -X GET "http://localhost:8000/health?detailed=true"

# Model-specific health
curl -X GET "http://localhost:8000/model/health"

# System metrics
curl -X GET "http://localhost:8000/metrics"
```

### 2. Logging Configuration

#### Structured Logging Setup

```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
            
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/app.log')
    ]
)
```

### 3. Monitoring with Prometheus

#### Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration')
ACTIVE_PREDICTIONS = Gauge('active_predictions', 'Number of active predictions')
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total model predictions')

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            # Process request
            await self.app(scope, receive, send)
            
            # Record metrics
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
            REQUEST_COUNT.labels(
                method=scope["method"],
                endpoint=scope["path"],
                status="200"  # Simplified
            ).inc()
```

### 4. Log Aggregation with ELK Stack

#### Docker Compose with ELK

```yaml
version: '3.8'

services:
  food-delivery-api:
    build: .
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    depends_on:
      - elasticsearch

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:7.15.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

## Scaling and Load Balancing

### 1. Horizontal Scaling

#### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml food-delivery-stack

# Scale service
docker service scale food-delivery-stack_api=5
```

#### Kubernetes Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: food-delivery-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: food-delivery-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 2. Load Balancing with NGINX

#### NGINX Configuration

```nginx
upstream food_delivery_api {
    least_conn;
    server api1:8000 weight=1 max_fails=3 fail_timeout=30s;
    server api2:8000 weight=1 max_fails=3 fail_timeout=30s;
    server api3:8000 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://food_delivery_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Health check
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
    }
    
    location /health {
        access_log off;
        proxy_pass http://food_delivery_api;
    }
}
```

## Security Considerations

### 1. Container Security

```dockerfile
# Security-hardened Dockerfile
FROM python:3.12-slim-bullseye

# Create non-root user early
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Install security updates
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy and install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and change ownership
COPY . .
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Remove unnecessary packages
RUN pip uninstall -y pip setuptools

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. API Security

#### Rate Limiting Implementation

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, data: DeliveryRequest):
    # Implementation
    pass
```

#### API Key Authentication

```python
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

@app.post("/predict")
async def predict(data: DeliveryRequest, api_key: str = Depends(verify_api_key)):
    # Implementation
    pass
```

### 3. HTTPS Configuration

#### SSL Certificate with Let's Encrypt

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Container Won't Start

```bash
# Check container logs
docker logs food-delivery-container

# Common fixes:
# - Port already in use
sudo lsof -i :8000
sudo kill -9 <PID>

# - Permission issues
sudo chown -R $USER:$USER .
```

#### 2. High Memory Usage

```bash
# Monitor container resources
docker stats food-delivery-container

# Optimize container memory
docker run -d \
  --name food-delivery-container \
  --memory="1g" \
  --memory-swap="1g" \
  -p 8000:8000 \
  food-delivery-api
```

#### 3. API Performance Issues

```bash
# Check API metrics
curl -X GET "http://localhost:8000/metrics"

# Monitor response times
time curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 5.0, "weather": "Clear", "traffic_level": "Medium", "vehicle_type": "Scooter", "preparation_time_min": 15.0, "courier_experience_yrs": 3.0}'
```

### Debugging Commands

```bash
# Container inspection
docker inspect food-delivery-container

# Execute commands inside container
docker exec -it food-delivery-container /bin/bash

# View real-time logs
docker logs -f food-delivery-container

# Network troubleshooting
docker network ls
docker network inspect bridge
```

## Maintenance

### 1. Regular Updates

```bash
# Update application
docker pull food-delivery-api:latest
docker stop food-delivery-container
docker rm food-delivery-container
docker run -d --name food-delivery-container -p 8000:8000 food-delivery-api:latest

# Clean up old images
docker image prune -f
docker system prune -f
```

### 2. Backup Strategies

```bash
# Backup application data
docker exec food-delivery-container tar czf - /app/data | gzip > backup-$(date +%Y%m%d).tar.gz

# Backup configuration
cp docker-compose.yml backup/
cp .env backup/
```

### 3. Health Monitoring Script

```bash
#!/bin/bash
# health_monitor.sh

API_URL="http://localhost:8000/health"
SLACK_WEBHOOK="your-slack-webhook-url"

response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL)

if [ $response != "200" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"🚨 Food Delivery API is down! HTTP Status: '$response'"}' \
        $SLACK_WEBHOOK
    
    # Restart container
    docker restart food-delivery-container
fi
```

### 4. Performance Monitoring

```bash
# Monitor API performance
#!/bin/bash
# performance_monitor.sh

while true; do
    response_time=$(curl -o /dev/null -s -w "%{time_total}" http://localhost:8000/health)
    echo "$(date): Response time: ${response_time}s"
    
    if (( $(echo "$response_time > 5.0" | bc -l) )); then
        echo "⚠️ High response time detected: ${response_time}s"
    fi
    
    sleep 30
done
```

## Rollback Procedures

### Quick Rollback

```bash
# Tag current version before deployment
docker tag food-delivery-api:latest food-delivery-api:backup-$(date +%Y%m%d)

# If rollback needed
docker stop food-delivery-container
docker rm food-delivery-container
docker run -d --name food-delivery-container -p 8000:8000 food-delivery-api:backup-20250701
```

### Zero-Downtime Deployment

```bash
# Blue-Green deployment script
#!/bin/bash

# Start new version (green)
docker run -d --name food-delivery-green -p 8001:8000 food-delivery-api:latest

# Health check on green
if curl -f http://localhost:8001/health; then
    # Switch traffic (update load balancer)
    # Update NGINX upstream to point to :8001
    
    # Stop old version (blue)
    docker stop food-delivery-container
    docker rm food-delivery-container
    
    # Rename green to main
    docker stop food-delivery-green
    docker run -d --name food-delivery-container -p 8000:8000 food-delivery-api:latest
    docker rm food-delivery-green
else
    echo "Health check failed, keeping old version"
    docker stop food-delivery-green
    docker rm food-delivery-green
fi
```

---

## 📞 Support and Contacts

- **Technical Issues**: Create an issue in the repository
- **Production Support**: contact-production@yourcompany.com
- **Security Issues**: security@yourcompany.com

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [NGINX Configuration Guide](https://nginx.org/en/docs/)
