"""
Utility functions for the API application.
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom API exception class."""
    
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class ModelError(APIError):
    """Exception for model-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code=503, details=details)

class ValidationError(APIError):
    """Exception for validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, status_code=422, details=details)

def log_error(error: Exception, context: Optional[Dict] = None) -> str:
    """
    Log an error with context and return error ID.
    
    Args:
        error: The exception that occurred
        context: Additional context information
        
    Returns:
        Error ID for tracking
    """
    error_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    
    error_info = {
        "error_id": error_id,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.now().isoformat(),
        "context": context or {},
        "traceback": traceback.format_exc()
    }
    
    logger.error(f"Error {error_id}: {json.dumps(error_info, indent=2)}")
    return error_id

def validate_model_files(model_path: str, preprocessor_path: str) -> bool:
    """
    Validate that model files exist and are readable.
    
    Args:
        model_path: Path to the model file
        preprocessor_path: Path to the preprocessor file
        
    Returns:
        True if files are valid, False otherwise
    """
    try:
        model_file = Path(model_path)
        preprocessor_file = Path(preprocessor_path)
        
        if not model_file.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
            
        if not preprocessor_file.exists():
            logger.error(f"Preprocessor file not found: {preprocessor_path}")
            return False
            
        if not model_file.is_file():
            logger.error(f"Model path is not a file: {model_path}")
            return False
            
        if not preprocessor_file.is_file():
            logger.error(f"Preprocessor path is not a file: {preprocessor_path}")
            return False
            
        # Check file sizes (should be > 0)
        if model_file.stat().st_size == 0:
            logger.error(f"Model file is empty: {model_path}")
            return False
            
        if preprocessor_file.stat().st_size == 0:
            logger.error(f"Preprocessor file is empty: {preprocessor_path}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating model files: {str(e)}")
        return False

def format_prediction_response(
    prediction: float,
    confidence: float,
    lower_bound: float,
    upper_bound: float,
    factors: Dict[str, Any],
    model_version: str
) -> Dict[str, Any]:
    """
    Format prediction response with consistent structure.
    
    Args:
        prediction: Predicted delivery time
        confidence: Confidence score
        lower_bound: Lower prediction interval
        upper_bound: Upper prediction interval
        factors: Explanation factors
        model_version: Model version identifier
        
    Returns:
        Formatted response dictionary
    """
    return {
        "predicted_delivery_time": round(prediction, 2),
        "confidence_score": round(confidence, 3),
        "prediction_interval_lower": round(lower_bound, 2),
        "prediction_interval_upper": round(upper_bound, 2),
        "factors": factors,
        "model_version": model_version,
        "timestamp": datetime.now().isoformat()
    }

def calculate_confidence_interval(
    prediction: float,
    confidence: float,
    uncertainty_factor: float = 0.2
) -> tuple[float, float]:
    """
    Calculate prediction confidence interval.
    
    Args:
        prediction: Base prediction value
        confidence: Confidence score (0-1)
        uncertainty_factor: Factor for interval width
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Adjust interval width based on confidence
    interval_width = prediction * uncertainty_factor * (1 - confidence)
    
    lower_bound = max(0, prediction - interval_width)
    upper_bound = prediction + interval_width
    
    return lower_bound, upper_bound

def extract_prediction_factors(
    request_data: Dict[str, Any],
    feature_importance: Optional[Dict[str, float]] = None
) -> Dict[str, str]:
    """
    Extract key factors affecting the prediction.
    
    Args:
        request_data: Input request data
        feature_importance: Feature importance scores
        
    Returns:
        Dictionary of factors and their descriptions
    """
    factors = {}
    
    # Distance factor
    distance = request_data.get("distance_km", 0)
    if distance < 2:
        factors["distance_impact"] = "short_distance"
    elif distance < 10:
        factors["distance_impact"] = "medium_distance"
    else:
        factors["distance_impact"] = "long_distance"
    
    # Weather factor
    weather = request_data.get("weather", "Clear")
    if weather in ["Rainy", "Snowy", "Foggy"]:
        factors["weather_impact"] = "adverse_conditions"
    else:
        factors["weather_impact"] = "favorable_conditions"
    
    # Traffic factor
    traffic = request_data.get("traffic_level", "Medium")
    factors["traffic_impact"] = f"{traffic.lower()}_traffic"
    
    # Experience factor
    experience = request_data.get("courier_experience_yrs", 0)
    if experience > 3:
        factors["experience_benefit"] = "experienced_courier"
    elif experience > 1:
        factors["experience_benefit"] = "moderate_experience"
    else:
        factors["experience_benefit"] = "new_courier"
    
    # Primary factor (based on feature importance or heuristics)
    if feature_importance:
        primary_feature = max(feature_importance.items(), key=lambda x: x[1])[0]
        factors["primary_factor"] = primary_feature
    else:
        factors["primary_factor"] = "distance_and_preparation"
    
    return factors

def health_check_model(predictor) -> Dict[str, Any]:
    """
    Perform health check on the model.
    
    Args:
        predictor: Model predictor instance
        
    Returns:
        Health check results
    """
    try:
        # Test prediction with sample data
        from model_pipeline.prediction_service import DeliveryRequest
        
        sample_request = DeliveryRequest(
            distance_km=5.0,
            weather="Clear",
            traffic_level="Medium",
            vehicle_type="Scooter",
            preparation_time_min=15.0,
            courier_experience_yrs=3.0
        )
        
        start_time = datetime.now()
        prediction = predictor.predict_single(sample_request)
        response_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "status": "healthy",
            "response_time_seconds": response_time,
            "sample_prediction": prediction.predicted_delivery_time,
            "test_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "test_timestamp": datetime.now().isoformat()
        }

def sanitize_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize input data for security and consistency.
    
    Args:
        data: Input data dictionary
        
    Returns:
        Sanitized data dictionary
    """
    sanitized = {}
    
    for key, value in data.items():
        if isinstance(value, str):
            # Remove potentially harmful characters
            sanitized[key] = value.strip()[:100]  # Limit string length
        elif isinstance(value, (int, float)):
            # Ensure numeric values are within reasonable bounds
            sanitized[key] = max(0, min(value, 1000))
        else:
            sanitized[key] = value
    
    return sanitized
