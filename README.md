# Food Delivery Time Prediction API

ML-powered API for predicting food delivery times based on various factors like distance, weather, and traffic conditions.

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

```bash
# Clone the repository
git clone https://github.com/yourusername/ds_tech_task.git
cd ds_tech_task

# Build and run with Docker
docker build -t food-delivery-api .
docker run -p 8000:8000 \
    -v $(pwd)/data:/app/data:ro \
    -v $(pwd)/models:/app/models:ro \
    food-delivery-api
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `POST /predict`: Get delivery time prediction
- `GET /health`: Health check endpoint
- `GET /docs`: Interactive API documentation

## Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the development server:
```bash
uvicorn src.main:app --reload
```

## Testing

```bash
pytest src/test_api.py
```
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

## Getting Started

### Prerequisites
- Python 3.10+
- SQLite3
- Docker (optional, for containerized deployment)
- Git

### Local Setup

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

### Docker Setup

1. **Build Docker image**
   ```bash
   docker build -t food-delivery-predictor .
   ```

2. **Run container**
   ```bash
   docker run -p 8000:8000 -v $(pwd)/data:/app/data food-delivery-predictor
   ```

3. **Using Docker Compose (recommended)**
   ```bash
   docker-compose up --build
   ```

### Quick Start with Docker
```bash
# One-command setup
git clone <repository-url> && cd ds_tech_task && docker-compose up --build
```

### Quick Start

```bash
# Option 1: Quick startup script
./start_api.sh

# Option 2: Manual setup
pip install -r requirements.txt
uvicorn src.main:app --reload

# Option 3: Docker deployment  
docker-compose up --build
```

### API Usage Examples

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "distance_km": 5.2,
       "weather": "Clear", 
       "traffic_level": "Medium",
       "vehicle_type": "Scooter",
       "preparation_time_min": 15.0,
       "courier_experience_yrs": 3.0
     }'

# Interactive API documentation
open http://localhost:8000/docs
```

### Python Client Usage
```python
# FastAPI service example
import requests

response = requests.post("http://localhost:8000/predict", json={
    'distance_km': 5.2,
    'weather': 'Clear',
    'traffic_level': 'Medium',
    'vehicle_type': 'Scooter',
    'preparation_time_min': 15,
    'courier_experience_yrs': 3
})

estimated_time = response.json()['predicted_delivery_time']
```