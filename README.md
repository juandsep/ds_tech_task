# Food Delivery Time Prediction Platform

A food delivery platform operates in multiple urban regions. Lately, they've received complaints about delivery delays — but the causes are unclear. This project investigates the delivery delay issues, builds a system to predict delivery time, and provides actionable insights to help the Operations team respond more intelligently.

## Project Architecture

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
┃   └━━ exploratory_data_analysis.ipynb # Comprehensive EDA with visualizations
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
┃   └━━ error_insights.md                # Error patterns & failure analysis
┣━━ 🚀 src/                              # Production-ready FastAPI service
┃   ┣━━ main.py                          # FastAPI app with all REST endpoints
┃   ┣━━ config.py                        # Environment & configuration management
┃   ┣━━ middleware.py                    # Custom middleware (logging, rate limiting)
┃   ┣━━ utils.py                         # API utilities & error handling
┃   ┣━━ test_api.py                      # Comprehensive test suite
┃   └━━ client_example.py                # Example client for API integration
┣━━ 🐳 Deployment & Infrastructure
┃   ┣━━ Dockerfile                       # Multi-stage container configuration
┃   ┣━━ docker-compose.yml               # Service orchestration with nginx
┃   ┣━━ nginx.conf                       # Reverse proxy & load balancing config
┃   ┣━━ requirements.txt                 # Python dependencies specification
┃   ┣━━ start_api.sh                     # Quick development startup script
┃   └━━ .env.example                     # Environment variables template
┣━━ 📝 Documentation & Strategy
┃   ┣━━ strategic_reflections.md         # Strategic insights & business impact
┃   ┣━━ next_steps.md                    # Roadmap & future enhancements
┃   └━━ README.md                        # Complete project documentation
┗━━ ⚙️ Configuration Files
    ┣━━ pyproject.toml                   # Project configuration & dependencies
    ┣━━ uv.lock                          # Dependency lock file
    └━━ .gitignore                       # Version control exclusions
```

### 🏗️ Architecture Layers

**Data Layer (📊)**
- Raw CSV data ingestion and SQLite database management
- Structured data storage with optimized queries for analysis

**Analysis Layer (🔍📓)**
- SQL-based business intelligence and operational insights  
- Interactive Jupyter notebooks for exploratory data analysis
- Statistical analysis and data visualization

**ML Pipeline Layer (🤖)**
- Modular preprocessing with feature engineering
- Model training with cross-validation and hyperparameter optimization
- Comprehensive evaluation with multiple metrics and error analysis
- Production-ready prediction service with monitoring

**API Layer (🚀)**
- FastAPI REST service with OpenAPI documentation
- Authentication, rate limiting, and security middleware
- Health checks, metrics collection, and monitoring endpoints
- Comprehensive testing and client integration examples

**Infrastructure Layer (🐳)**
- Docker containerization with multi-stage builds
- Service orchestration with nginx reverse proxy
- Environment-based configuration management
- Production deployment automation

**Documentation Layer (📝)**
- Business strategy and stakeholder communication
- Technical documentation and API guides
- Implementation roadmap and future planning

## 📊 Dataset Overview

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

**Deployment Ready:**
- Docker containerization with multi-stage builds
- Docker Compose orchestration with nginx reverse proxy
- Environment configuration templates and startup scripts
- Production security features (CORS, rate limiting, input validation)

**API Capabilities:**
- Real-time delivery time predictions with confidence intervals
- Batch processing for high-throughput scenarios
- Feature explanation and model interpretability
- Performance monitoring and operational metrics

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

## 🎯 Key Features & Capabilities

### 📊 Business Intelligence & Analytics
- **SQL-driven insights** into delivery performance patterns and delay root causes
- **Operational KPIs** tracking with actionable recommendations for operations teams
- **Customer area analysis** identifying high-delay zones and optimization opportunities
- **Revenue impact assessment** through delivery efficiency improvements

### 🤖 Advanced Machine Learning
- **Multi-algorithm ensemble** with Random Forest, XGBoost, and Linear Regression models
- **Hyperparameter optimization** using cross-validation and grid search techniques
- **Feature engineering pipeline** with categorical encoding, scaling, and interaction terms
- **Model explainability** using SHAP values and feature importance analysis
- **Error pattern analysis** with segmentation and mitigation strategies

### 🚀 Production-Ready API Architecture
- **RESTful API design** with FastAPI and OpenAPI/Swagger documentation
- **Scalable prediction service** supporting both single and batch predictions
- **Enterprise security** with authentication, rate limiting, and input validation
- **Comprehensive monitoring** with health checks, metrics, and performance tracking
- **Container orchestration** with Docker and nginx reverse proxy support

### 📈 Operational Excellence
- **End-to-end testing** with unit tests, integration tests, and load testing
- **Configuration management** with environment-based deployments
- **Error handling & logging** with structured logging and error tracking
- **Documentation standards** with technical and business stakeholder communication
- **Deployment automation** with Docker Compose and production-ready configurations

### 🎯 Business Impact
- **Delivery time optimization** through predictive analytics and route planning
- **Customer satisfaction improvement** via accurate delivery time estimates
- **Operational cost reduction** through resource optimization and efficiency gains
- **Strategic decision support** with data-driven insights for business growth

## 🏆 Technical Excellence

This project demonstrates enterprise-level data science practices including:

- **Software Engineering Standards**: Modular code architecture, comprehensive testing, and documentation
- **ML Operations (MLOps)**: Model versioning, monitoring, and production deployment pipelines  
- **Security & Compliance**: Input validation, authentication, and secure API design
- **Scalability & Performance**: Optimized for high-throughput production environments
- **Business Communication**: Clear stakeholder reports with actionable insights 