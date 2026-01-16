# Deployment Guide

**🚀 Live Demo**: https://multilingual-sentiment-analysis.streamlit.app/
**📖 Repository**: https://github.com/midlaj-muhammed/Multilingual-Sentiment-Analysis-Tool

This guide covers various deployment options for the Multilingual Sentiment Analysis Tool.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Production Considerations](#production-considerations)
5. [Monitoring and Logging](#monitoring-and-logging)

## Local Development

### Prerequisites

- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- GPU (optional, for better performance)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd multilingual-sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your settings
```

### Running Services

#### API Server

```bash
# Start FastAPI server
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Web Interface

```bash
# Start Streamlit app
streamlit run app/frontend/streamlit_app.py --server.port 8501
```

#### Both Services

```bash
# Terminal 1: API Server
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Web Interface  
streamlit run app/frontend/streamlit_app.py --server.port 8501
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=False
      - LOG_LEVEL=INFO
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    depends_on:
      - redis
    command: uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 4

  frontend:
    build: .
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
    command: streamlit run app/frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
      - frontend

volumes:
  redis_data:
```

### Building and Running

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d

# Scale API service
docker-compose up --scale api=3

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## Cloud Deployment

### AWS Deployment

#### Using AWS ECS

```yaml
# ecs-task-definition.json
{
  "family": "sentiment-analysis",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/sentiment-analysis:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DEBUG",
          "value": "False"
        },
        {
          "name": "LOG_LEVEL", 
          "value": "INFO"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/sentiment-analysis",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Deployment Script

```bash
#!/bin/bash

# Build and push Docker image
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-west-2.amazonaws.com

docker build -t sentiment-analysis .
docker tag sentiment-analysis:latest your-account.dkr.ecr.us-west-2.amazonaws.com/sentiment-analysis:latest
docker push your-account.dkr.ecr.us-west-2.amazonaws.com/sentiment-analysis:latest

# Update ECS service
aws ecs update-service --cluster sentiment-cluster --service sentiment-service --force-new-deployment
```

### Google Cloud Platform

#### Using Cloud Run

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/sentiment-analysis', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/sentiment-analysis']
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'sentiment-analysis'
      - '--image'
      - 'gcr.io/$PROJECT_ID/sentiment-analysis'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--memory'
      - '2Gi'
      - '--cpu'
      - '2'
      - '--max-instances'
      - '10'
```

#### Deploy Command

```bash
# Deploy to Cloud Run
gcloud run deploy sentiment-analysis \
  --image gcr.io/PROJECT_ID/sentiment-analysis \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10
```

### Azure Deployment

#### Using Container Instances

```bash
# Create resource group
az group create --name sentiment-rg --location eastus

# Deploy container
az container create \
  --resource-group sentiment-rg \
  --name sentiment-analysis \
  --image your-registry.azurecr.io/sentiment-analysis:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables DEBUG=False LOG_LEVEL=INFO \
  --dns-name-label sentiment-analysis-api
```

## Production Considerations

### Performance Optimization

#### Model Optimization

```python
# Use quantized models for faster inference
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

# Quantize model (CPU)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Use TensorRT for GPU optimization (NVIDIA GPUs)
# Requires additional setup
```

#### Caching Strategy

```python
# Redis configuration for production
REDIS_CONFIG = {
    "host": "redis-cluster.cache.amazonaws.com",
    "port": 6379,
    "db": 0,
    "max_connections": 20,
    "retry_on_timeout": True,
    "socket_keepalive": True,
    "socket_keepalive_options": {}
}
```

### Security

#### API Security

```python
# Add API key authentication
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != "your-secret-api-key":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials
```

#### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/analyze/")
@limiter.limit("10/minute")
async def analyze_text(request: Request, ...):
    # Endpoint implementation
    pass
```

### Load Balancing

#### Nginx Configuration

```nginx
upstream sentiment_api {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location /api/ {
        proxy_pass http://sentiment_api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    location / {
        proxy_pass http://frontend:8501/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### Database Integration

```python
# PostgreSQL for storing analysis results
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True)
    text = Column(String)
    sentiment = Column(String)
    confidence = Column(Float)
    language = Column(String)
    created_at = Column(DateTime)

# Database connection
engine = create_engine("postgresql://user:password@localhost/sentiment_db")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

## Monitoring and Logging

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

@app.middleware("http")
async def add_prometheus_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    logger.info(
        "request_started",
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host
    )
    
    response = await call_next(request)
    duration = time.time() - start_time
    
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=duration
    )
    
    return response
```

### Health Checks

```python
@app.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    # Check if model is loaded and dependencies are available
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analysis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment-analysis
  template:
    metadata:
      labels:
        app: sentiment-analysis
    spec:
      containers:
      - name: api
        image: sentiment-analysis:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: sentiment-analysis-service
spec:
  selector:
    app: sentiment-analysis
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```
