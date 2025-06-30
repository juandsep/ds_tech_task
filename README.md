# Food Delivery Time Prediction Platform

A food delivery platform operates in multiple urban regions. Lately, they've received complaints about delivery delays — but the causes are unclear. This project investigates the delivery delay issues, builds a system to predict delivery time, and provides actionable insights to help the Operations team respond more intelligently.

## Project Architecture

```
ds_tech_task/
├── data/
│   ├── Food_Delivery_Times.csv          # Raw delivery data
│   └── food_delivery.db                 # SQLite database
├── sql/
│   ├── sql_queries.sql                  # Business analysis queries
│   └── sql_insights.md                  # Additional business insights
├── notebooks/
│   └── create_sql_tables.ipynb          # Database setup notebook
├── model_pipeline/                      # End-to-end ML pipeline
│   ├── data_preprocessing.py            # Data cleaning and feature engineering
│   ├── model_training.py                # Model training and validation
│   ├── model_evaluation.py              # Model performance evaluation
│   ├── prediction_service.py            # Prediction API service
│   └── utils/                           # Utility functions
├── reports/
│   ├── EDA_report.md                    # Exploratory data analysis findings
│   ├── model_notes.md                   # Modeling methodology and decisions
│   ├── explainability.md                # Feature importance and model insights
│   └── error_insights.md                # Model failure analysis
├── strategic_reflections.md             # Strategic thinking and communication
├── next_steps.md                        # API prototype and future enhancements  
├── src/                                 # Production-ready FastAPI service
│   ├── main.py                          # FastAPI application with all endpoints
│   ├── config.py                        # Configuration management
│   ├── middleware.py                    # Custom middleware for logging/monitoring
│   ├── utils.py                         # Utility functions and error handling
│   ├── test_api.py                      # Comprehensive test suite
│   └── client_example.py                # Example API client
├── Dockerfile                           # Container configuration
├── docker-compose.yml                   # Multi-service orchestration
├── nginx.conf                           # Reverse proxy configuration  
├── requirements.txt                     # Python dependencies
├── start_api.sh                         # Quick startup script
├── .env.example                         # Environment configuration template
└── README.md                            # This file
```

## Dataset Overview

The dataset contains delivery information with the following key features:

**Core Delivery Metrics:**
- Order_ID: Unique identifier for each delivery
- Distance_km: Delivery distance in kilometers
- Delivery_Time_min: Target variable (actual delivery time)

**Environmental Factors:**
- Weather: Weather conditions (Clear, Rainy, Snowy, Foggy, Windy)
- Traffic_Level: Traffic conditions (Low, Medium, High)
- Time_of_Day: Delivery time period (Morning, Afternoon, Evening, Night)

**Operational Variables:**
- Vehicle_Type: Delivery vehicle (Car, Bike, Scooter)
- Preparation_Time_min: Restaurant preparation time
- Courier_Experience_yrs: Delivery person experience level

## Part I: SQL Queries

Comprehensive business analysis addressing delivery delay issues through SQL queries (`sql/sql_queries.sql` and `sql/sql_insights.md`). Investigates customer areas, traffic patterns, delivery personnel performance, and revenue optimization opportunities to provide actionable insights for the Operations team.

## Part II: Exploration & Modeling

End-to-end machine learning pipeline (`model_pipeline/`) that predicts delivery times using environmental factors, operational variables, and delivery metrics. Includes data preprocessing, model training with multiple algorithms, performance evaluation, and comprehensive reporting on model behavior and failure patterns.

## Part III: FastAPI Service ✅

Production-ready FastAPI service (`src/`) that provides real-time delivery time predictions through REST endpoints. Features include:

- **REST API**: Single and batch prediction endpoints with comprehensive validation
- **Authentication**: JWT and API key support with middleware integration  
- **Monitoring**: Health checks, metrics collection, and performance tracking
- **Security**: CORS, rate limiting, input sanitization, and trusted host validation
- **Testing**: Comprehensive test suite with unit tests and load testing
- **Documentation**: OpenAPI/Swagger integration with interactive docs

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

## Key Features

- **Comprehensive Business Analysis**: SQL-based insights into delivery performance patterns
- **Production-Ready ML Pipeline**: End-to-end modeling with proper software engineering practices
- **Actionable Intelligence**: Clear recommendations for operations team optimization
- **Strategic Thinking**: Addresses real-world deployment challenges and business considerations
- **Scalable Architecture**: Designed for production deployment and cross-regional expansion

## Contributing

This project follows best practices for data science and machine learning projects, including proper documentation, modular code structure, and comprehensive testing strategies for production deployment. 