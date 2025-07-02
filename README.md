# Food Delivery Time Prediction API

ML-powered API for predicting food delivery times based on various factors like distance, weather, and traffic conditions.

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [🐳 Docker Deployment (Recommended)](#-docker-deployment-recommended)
  - [🔧 Local Development](#-local-development)
- [🚀 API Endpoints](#-api-endpoints)
  - [Core Endpoints](#core-endpoints)
  - [Model Management](#model-management)
  - [Example Usage](#example-usage)
- [🧪 Testing](#-testing)
  - [Run the Test Suite](#run-the-test-suite)
  - [Manual Testing](#manual-testing)
  - [Docker Testing](#docker-testing)
- [📋 Deployment Guide](#-deployment-guide)
- [Dataset Overview](#dataset-overview)
  - [🎯 Core Delivery Metrics](#-core-delivery-metrics)
  - [🌤️ Environmental Factors](#️-environmental-factors)
  - [🚚 Operational Variables](#-operational-variables)
  - [📈 Data Quality & Characteristics](#-data-quality--characteristics)
- [🔍 Part I: SQL Business Intelligence](#-part-i-sql-business-intelligence-)
- [🤖 Part II: ML Pipeline & Analysis](#-part-ii-ml-pipeline--analysis-)
- [🚀 Part III: Production FastAPI Service](#-part-iii-production-fastapi-service-)
- [Getting Started with Development](#getting-started-with-development)
  - [Prerequisites](#prerequisites)
  - [Development Setup](#development-setup)
  - [Production Deployment Options](#production-deployment-options)

---

## Project Structure

```
ds_tech_task/
┣━━ 📊 data/
┃   ┣━━ Food_Delivery_Times.csv          # Raw delivery dataset (10K+ records)
┃   └━━ food_delivery.db                 # SQLite database with structured tables
┣━━ 🔍 sql/
┃   ┣━━ sql_queries.sql                  # Business intelligence queries & analysis
┃   └━━ sql_insights.md                  # Key business insights & recommendations
┣━━ 📓 notebooks/
┃   ┣━━ create_sql_tables.ipynb          # Database setup & data ingestion
┃   ┣━━ exploratory_data_analysis.ipynb  # Comprehensive EDA with visualizations
┃   └━━ modelo_completo_pipeline.ipynb   # Complete model pipeline notebook
┣━━ 🤖 model_pipeline/                   # Complete ML pipeline architecture
┃   ┣━━ data_preprocessing.py            # Feature engineering & data preparation
┃   ┣━━ model_training.py                # Model training with hyperparameter tuning
┃   ┣━━ model_evaluation.py              # Performance evaluation & validation
┃   ┣━━ prediction_service.py            # Core prediction engine
┃   └━━ utils/
┃       └━━ model_utils.py               # ML utilities & model persistence
┣━━ 📋 reports/
┃   ┣━━ EDA_report.md                    # Detailed exploratory analysis findings
┃   ┣━━ model_notes.md                   # Model selection & tuning methodology
┃   ┣━━ explainability.md                # Feature importance & SHAP analysis
┃   ┣━━ error_insights.md                # Error patterns & failure analysis
┃   ┣━━ strategic_reflections.md         # Strategic insights & business impact
┃   └━━ next_steps.md                    # Future improvements roadmap
┣━━ 🚀 src/                              # Production-ready FastAPI service
┃   ┣━━ main.py                          # FastAPI app with all REST endpoints
┃   ┣━━ config.py                        # Environment & configuration management
┃   ┣━━ middleware.py                    # Custom middleware (logging, rate limiting)
┃   ┣━━ utils.py                         # API utilities & error handling
┃   ┣━━ test_api.py                      # Comprehensive test suite
┃   └━━ client_example.py                # Example client for API integration
┣━━ � Deployment
┃   ┣━━ Dockerfile                       # Container configuration
┃   ┣━━ requirements.txt                 # Python dependencies
┗━━ ⚙️ Configuration
    ┣━━ pyproject.toml                   # Project configuration & dependencies
    └━━ uv.lock                          # Dependency lock file
```

## Quick Start

### 🐳 Docker Deployment (Recommended)

```bash
# Clone and navigate to the project
git clone <repository-url>
cd ds_tech_task

# Build and run with Docker
docker build -t food-delivery-api .
docker run -d -p 8000:8000 --name food-delivery-container food-delivery-api

# Check if the container is running
docker ps

# View logs (optional)
docker logs food-delivery-container
```

### 🔧 Local Development

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the development server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

- **API Documentation**: `http://localhost:8000/docs`
- **Alternative Docs**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

## 🚀 API Endpoints

### Core Endpoints
- `POST /predict` - Get delivery time prediction
- `GET /health` - Health check and system status
- `GET /docs` - Interactive Swagger UI documentation
- `GET /redoc` - Alternative API documentation

### Model Management
- `GET /model/info` - Model metadata and performance metrics
- `GET /metrics` - Real-time API usage metrics

### Example Usage

#### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

#### Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "distance_km": 5.2,
  "weather": "Clear",
  "traffic_level": "Medium",
  "vehicle_type": "Scooter",
  "preparation_time_min": 15.0,
  "courier_experience_yrs": 3.5
}'
```

#### Check Model Information
```bash
curl -X GET "http://localhost:8000/model/info"
```

#### View API Metrics
```bash
curl -X GET "http://localhost:8000/metrics"
```

#### Python Client Example
```python
import requests

# Make a prediction
response = requests.post("http://localhost:8000/predict", json={
    "distance_km": 5.2,
    "weather": "Clear",
    "traffic_level": "Medium",
    "vehicle_type": "Scooter",
    "preparation_time_min": 15.0,
    "courier_experience_yrs": 3.5
})

result = response.json()
print(f"Estimated delivery time: {result['predicted_delivery_time']} minutes")
```

## 🧪 Testing

### Run the Test Suite
```bash
# Install test dependencies if not already installed
pip install pytest pytest-asyncio httpx

# Run all tests
pytest src/test_api.py -v

# Run specific test categories
pytest src/test_api.py::test_health_endpoint -v
pytest src/test_api.py::test_predict_endpoint -v
```

### Manual Testing
```bash
# Test with client example
python src/client_example.py

# Interactive testing with Swagger UI
open http://localhost:8000/docs
```

### Docker Testing
```bash
# Test the API in Docker container
docker run -d -p 8000:8000 --name test-container food-delivery-api

# Wait a moment for startup, then test
sleep 5
curl http://localhost:8000/health

# Cleanup
docker stop test-container && docker rm test-container
```

## 📋 Deployment Guide

For production deployment, scaling, monitoring, and security considerations, see the comprehensive deployment guide:

**[📖 Detailed Deployment Guide](reports/deployment_guide.md)**

This guide includes:
- **Docker & Docker Compose** setup
- **Cloud deployment** (AWS, GCP, Azure)
- **Kubernetes** configurations
- **Monitoring & logging** setup
- **Security** best practices
- **Performance optimization**
- **CI/CD** pipeline examples
## Dataset Overview

The dataset contains **10,000+ delivery records** with comprehensive delivery information across multiple dimensions:

### 🎯 Core Delivery Metrics
- **Order_ID**: Unique identifier for each delivery transaction
- **Distance_km**: Delivery distance in kilometers (0.5 - 50 km range)
- **Delivery_Time_min**: Target variable - actual delivery time in minutes

### 🌤️ Environmental Factors
- **Weather**: Weather conditions affecting delivery
  - Clear, Rainy, Snowy, Foggy, Windy
- **Traffic_Level**: Real-time traffic conditions
  - Low, Medium, High
- **Time_of_Day**: Delivery time periods
  - Morning, Afternoon, Evening, Night

### 🚚 Operational Variables
- **Vehicle_Type**: Delivery vehicle types
  - Car, Bike, Scooter (each with different speed/capacity characteristics)
- **Preparation_Time_min**: Restaurant food preparation time
- **Courier_Experience_yrs**: Delivery person experience level (0-20 years)

### 📈 Data Quality & Characteristics
- **Completeness**: 100% data completeness across all features
- **Distribution**: Balanced representation across categories
- **Time Range**: Multi-seasonal data covering various weather patterns
- **Geographic Coverage**: Urban delivery scenarios with varying distances
- **Business Context**: Real-world delivery scenarios with operational constraints

## 🔍 Part I: SQL Business Intelligence ✅

**Comprehensive business analysis** addressing delivery delay root causes through strategic SQL queries:

**Deliverables:**
- `sql/sql_queries.sql` - 10+ business-focused analytical queries
- `sql/sql_insights.md` - Actionable insights and recommendations

**Business Impact:**
- Customer area performance analysis and delay hotspot identification
- Traffic pattern correlation with delivery performance
- Courier performance evaluation and experience impact assessment
- Revenue optimization opportunities through operational efficiency

## 🤖 Part II: ML Pipeline & Analysis ✅

**End-to-end machine learning pipeline** with production-ready architecture:

**Core Components:**
- `model_pipeline/data_preprocessing.py` - Feature engineering & data preparation
- `model_pipeline/model_training.py` - Multi-algorithm training with hyperparameter tuning
- `model_pipeline/model_evaluation.py` - Comprehensive performance evaluation
- `model_pipeline/prediction_service.py` - Production prediction engine

**Analysis & Reporting:**
- `notebooks/exploratory_data_analysis.ipynb` - Interactive EDA with visualizations
- `reports/EDA_report.md` - Detailed exploratory findings
- `reports/model_notes.md` - Model selection methodology and performance metrics
- `reports/explainability.md` - SHAP analysis and feature importance insights
- `reports/error_insights.md` - Failure pattern analysis and mitigation strategies

**Technical Highlights:**
- Multiple ML algorithms (Random Forest, XGBoost, Linear Regression) with cross-validation
- Advanced feature engineering with categorical encoding and scaling
- Comprehensive model evaluation with MAE, RMSE, R² metrics
- SHAP-based explainability for stakeholder communication

## 🚀 Part III: Production FastAPI Service ✅

**Enterprise-grade REST API** with comprehensive production features:

**Core API Features:**
- `src/main.py` - FastAPI application with comprehensive endpoints
- Single & batch prediction endpoints with Pydantic validation
- Health monitoring, metrics collection, and model information endpoints
- Interactive OpenAPI/Swagger documentation

**Production Infrastructure:**
- `src/middleware.py` - Custom middleware for logging, rate limiting, and metrics
- `src/config.py` - Environment-based configuration management
- `src/utils.py` - Error handling, input sanitization, and security utilities
- `src/test_api.py` - Comprehensive test suite with unit and integration tests

## Getting Started with Development

### Prerequisites
- Python 3.12+
- SQLite3
- Docker (optional, for containerized deployment)
- Git

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ds_tech_task
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run development server**
   ```bash
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Production Deployment Options

For detailed production deployment instructions, see [`reports/deployment_guide.md`](reports/deployment_guide.md)

**Quick production setup:**
```bash
# Docker deployment
docker build -t food-delivery-api .
docker run -d -p 8000:8000 food-delivery-api

# Docker Compose (recommended)
docker-compose up -d --build
```