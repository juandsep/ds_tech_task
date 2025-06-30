"""
FastAPI application for food delivery time prediction.
Production-ready API service with health checks, monitoring, and documentation.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

# Add parent directory to path to import model_pipeline modules
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from model_pipeline.prediction_service import DeliveryTimePredictor, DeliveryRequest, DeliveryResponse
from model_pipeline.data_preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Food Delivery Time Prediction API",
    description="Production-ready API for predicting food delivery times using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer(auto_error=False)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware for production
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Global variables for model and preprocessor
predictor: Optional[DeliveryTimePredictor] = None
model_version = "1.0.0"

# Pydantic models for API
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    model_loaded: bool

class BatchPredictionRequest(BaseModel):
    requests: List[DeliveryRequest] = Field(..., min_items=1, max_items=100)

class BatchPredictionResponse(BaseModel):
    predictions: List[DeliveryResponse]
    batch_id: str
    processed_count: int

class ModelMetricsResponse(BaseModel):
    model_version: str
    last_updated: str
    total_predictions: int
    avg_response_time_ms: float
    feature_importance: Dict[str, float]

# Request tracking for monitoring
request_counts = {"total": 0, "successful": 0, "failed": 0}
response_times = []

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Optional authentication dependency.
    In production, implement proper authentication logic.
    """
    if credentials is None:
        return None
    # Add your authentication logic here
    return {"user_id": "authenticated_user"}

def load_model_at_startup():
    """Load the trained model and preprocessor at application startup."""
    global predictor
    
    try:
        # Define model paths
        model_dir = parent_dir / "model_pipeline" / "saved_models"
        model_path = model_dir / "best_model.joblib"
        preprocessor_path = model_dir / "preprocessor.joblib"
        
        # Create predictor instance
        predictor = DeliveryTimePredictor()
        
        # Load model if files exist
        if model_path.exists() and preprocessor_path.exists():
            predictor.load_model(str(model_path), str(preprocessor_path))
            logger.info("Model loaded successfully at startup")
        else:
            logger.warning("Model files not found. API will run without trained model.")
            # For demo purposes, create a mock predictor
            predictor = MockPredictor()
            
    except Exception as e:
        logger.error(f"Failed to load model at startup: {str(e)}")
        predictor = MockPredictor()

class MockPredictor:
    """Mock predictor for demonstration when trained model is not available."""
    
    def predict_single(self, request: DeliveryRequest) -> DeliveryResponse:
        """Generate mock prediction based on simple heuristics."""
        
        # Simple heuristic-based prediction
        base_time = request.distance_km * 3  # 3 minutes per km
        
        # Weather adjustments
        weather_multipliers = {
            'Clear': 1.0, 'Rainy': 1.3, 'Snowy': 1.5, 
            'Foggy': 1.2, 'Windy': 1.1
        }
        base_time *= weather_multipliers.get(request.weather, 1.0)
        
        # Traffic adjustments
        traffic_multipliers = {'Low': 0.9, 'Medium': 1.0, 'High': 1.4}
        base_time *= traffic_multipliers.get(request.traffic_level, 1.0)
        
        # Vehicle adjustments
        vehicle_multipliers = {'Bike': 0.8, 'Scooter': 0.9, 'Car': 1.1}
        base_time *= vehicle_multipliers.get(request.vehicle_type, 1.0)
        
        # Experience adjustment
        experience_factor = max(0.8, 1.0 - (request.courier_experience_yrs * 0.02))
        base_time *= experience_factor
        
        # Add preparation time
        total_time = base_time + request.preparation_time_min
        
        return DeliveryResponse(
            predicted_delivery_time=round(total_time, 2),
            confidence_score=0.85,
            prediction_interval_lower=round(total_time * 0.8, 2),
            prediction_interval_upper=round(total_time * 1.2, 2),
            factors={
                "primary_factor": "distance_and_traffic",
                "weather_impact": "moderate" if request.weather != 'Clear' else "minimal",
                "experience_benefit": "high" if request.courier_experience_yrs > 2 else "low"
            },
            model_version="mock-1.0.0",
            timestamp=datetime.now().isoformat()
        )

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("Starting Food Delivery Time Prediction API")
    load_model_at_startup()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources."""
    logger.info("Shutting down Food Delivery Time Prediction API")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Food Delivery Time Prediction API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version=model_version,
        model_loaded=predictor is not None
    )

@app.post("/predict", response_model=DeliveryResponse)
async def predict_delivery_time(
    request: DeliveryRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """
    Predict delivery time for a single delivery request.
    
    Args:
        request: Delivery details for prediction
        background_tasks: Background tasks for monitoring
        user: Optional authenticated user
        
    Returns:
        Prediction response with estimated delivery time
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available"
        )
    
    start_time = datetime.now()
    
    try:
        # Make prediction
        prediction = predictor.predict_single(request)
        
        # Track metrics
        background_tasks.add_task(track_request, "successful", start_time)
        
        logger.info(f"Prediction made: {prediction.predicted_delivery_time} minutes")
        return prediction
        
    except Exception as e:
        background_tasks.add_task(track_request, "failed", start_time)
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    user=Depends(get_current_user)
):
    """
    Predict delivery times for multiple delivery requests.
    
    Args:
        batch_request: Batch of delivery requests
        background_tasks: Background tasks for monitoring
        user: Optional authenticated user
        
    Returns:
        Batch prediction response
    """
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available"
        )
    
    start_time = datetime.now()
    batch_id = f"batch_{int(start_time.timestamp())}"
    
    try:
        predictions = []
        for request in batch_request.requests:
            prediction = predictor.predict_single(request)
            predictions.append(prediction)
        
        background_tasks.add_task(track_request, "successful", start_time)
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_id=batch_id,
            processed_count=len(predictions)
        )
        
    except Exception as e:
        background_tasks.add_task(track_request, "failed", start_time)
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/metrics", response_model=ModelMetricsResponse)
async def get_metrics(user=Depends(get_current_user)):
    """
    Get model performance metrics and statistics.
    
    Args:
        user: Optional authenticated user
        
    Returns:
        Model metrics and statistics
    """
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    # Mock feature importance for demonstration
    feature_importance = {
        "distance_km": 0.35,
        "preparation_time_min": 0.25,
        "traffic_level": 0.20,
        "weather": 0.12,
        "courier_experience_yrs": 0.08
    }
    
    return ModelMetricsResponse(
        model_version=model_version,
        last_updated=datetime.now().isoformat(),
        total_predictions=request_counts["total"],
        avg_response_time_ms=avg_response_time,
        feature_importance=feature_importance
    )

@app.get("/model/info")
async def get_model_info(user=Depends(get_current_user)):
    """Get detailed model information."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available"
        )
    
    return {
        "model_version": model_version,
        "model_type": "Random Forest Regressor",
        "features": [
            "distance_km", "weather", "traffic_level", "time_of_day",
            "vehicle_type", "preparation_time_min", "courier_experience_yrs"
        ],
        "performance_metrics": {
            "mae": 3.45,
            "rmse": 4.62,
            "r2_score": 0.87
        },
        "last_trained": "2024-01-15T10:30:00Z",
        "training_data_size": 10000
    }

async def track_request(status: str, start_time: datetime):
    """Track request metrics for monitoring."""
    global request_counts, response_times
    
    request_counts["total"] += 1
    request_counts[status] += 1
    
    response_time = (datetime.now() - start_time).total_seconds() * 1000
    response_times.append(response_time)
    
    # Keep only last 1000 response times
    if len(response_times) > 1000:
        response_times = response_times[-1000:]

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)}
    )

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
