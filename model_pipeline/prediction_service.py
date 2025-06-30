"""
Prediction service for delivery time estimation.
FastAPI-ready service for real-time predictions.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Any, Union
from pydantic import BaseModel, validator
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor
from utils.model_utils import ModelPersistence, ModelMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeliveryRequest(BaseModel):
    """
    Pydantic model for delivery prediction request.
    """
    distance_km: float
    weather: str
    traffic_level: str
    time_of_day: str = None
    vehicle_type: str
    preparation_time_min: float
    courier_experience_yrs: float
    
    @validator('distance_km')
    def validate_distance(cls, v):
        if v <= 0 or v > 50:
            raise ValueError('Distance must be between 0 and 50 km')
        return v
    
    @validator('weather')
    def validate_weather(cls, v):
        valid_weather = ['Clear', 'Rainy', 'Snowy', 'Foggy', 'Windy']
        if v not in valid_weather:
            raise ValueError(f'Weather must be one of: {valid_weather}')
        return v
    
    @validator('traffic_level')
    def validate_traffic(cls, v):
        valid_traffic = ['Low', 'Medium', 'High']
        if v not in valid_traffic:
            raise ValueError(f'Traffic level must be one of: {valid_traffic}')
        return v
    
    @validator('vehicle_type')
    def validate_vehicle(cls, v):
        valid_vehicles = ['Car', 'Bike', 'Scooter']
        if v not in valid_vehicles:
            raise ValueError(f'Vehicle type must be one of: {valid_vehicles}')
        return v
    
    @validator('preparation_time_min')
    def validate_prep_time(cls, v):
        if v <= 0 or v > 120:
            raise ValueError('Preparation time must be between 0 and 120 minutes')
        return v
    
    @validator('courier_experience_yrs')
    def validate_experience(cls, v):
        if v < 0 or v > 20:
            raise ValueError('Courier experience must be between 0 and 20 years')
        return v

class DeliveryResponse(BaseModel):
    """
    Pydantic model for delivery prediction response.
    """
    predicted_delivery_time: float
    confidence_score: float
    prediction_interval_lower: float
    prediction_interval_upper: float
    factors: Dict[str, str]
    model_version: str
    timestamp: str

class PredictionService:
    """
    Production-ready delivery time prediction service.
    """
    
    def __init__(self, model=None, preprocessor=None, model_path: str = None, preprocessor_path: str = None):
        """
        Initialize the prediction service.
        
        Args:
            model: Pre-trained model instance
            preprocessor: Fitted preprocessor instance
            model_path: Path to the trained model (if loading from file)
            preprocessor_path: Path to the fitted preprocessor (if loading from file)
        """
        self.model = model
        self.preprocessor = preprocessor
        self.model_metadata = None
        self.feature_names = None
        self.monitor = ModelMonitor()
        
        if model_path and preprocessor_path:
            self.load_model(model_path, preprocessor_path)
    
    def load_model(self, model_path: str, preprocessor_path: str, metadata_path: str = None):
        """
        Load the trained model and preprocessor.
        
        Args:
            model_path: Path to the model file
            preprocessor_path: Path to the preprocessor file
            metadata_path: Path to the metadata file
        """
        try:
            # Load model
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.model_metadata = model_data.get('config', {})
            else:
                self.model = model_data
                self.model_metadata = {}
            
            # Load preprocessor
            if hasattr(preprocessor_path, 'endswith') and preprocessor_path.endswith('.pkl'):
                self.preprocessor = joblib.load(preprocessor_path)
            else:
                self.preprocessor = DataPreprocessor()
                self.preprocessor.load_preprocessor(preprocessor_path)
                
            if hasattr(self.preprocessor, 'get_feature_names'):
                self.feature_names = self.preprocessor.get_feature_names()
            
            # Load additional metadata if available  
            if metadata_path:
                additional_metadata = joblib.load(metadata_path)
                self.model_metadata.update(additional_metadata)
            
            logger.info("Model and preprocessor loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, request: DeliveryRequest) -> Dict[str, Any]:
        """
        Make a simple prediction (compatible with notebook usage).
        
        Args:
            request: Delivery request data
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded before prediction")
        
        try:
            # Convert request to DataFrame
            input_data = pd.DataFrame([request.dict()])
            
            # Handle Time_of_Day None values
            if 'time_of_day' in input_data.columns and input_data['time_of_day'].isna().any():
                input_data['time_of_day'] = input_data['time_of_day'].fillna('Evening')
            
            # Add dummy columns if needed for preprocessing compatibility
            if 'Order_ID' not in input_data.columns:
                input_data['Order_ID'] = 999999
            if 'Delivery_Time_min' not in input_data.columns:
                input_data['Delivery_Time_min'] = 0  # Will be ignored in transform
            
            # Rename columns to match training data format
            column_mapping = {
                'distance_km': 'Distance_km',
                'weather': 'Weather', 
                'traffic_level': 'Traffic_Level',
                'time_of_day': 'Time_of_Day',
                'vehicle_type': 'Vehicle_Type',
                'preparation_time_min': 'Preparation_Time_min',
                'courier_experience_yrs': 'Courier_Experience_yrs'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in input_data.columns:
                    input_data[new_col] = input_data[old_col]
                    input_data.drop(columns=[old_col], inplace=True)
            
            # Transform the data - simple approach for notebook compatibility
            feature_columns = ['Distance_km', 'Weather', 'Traffic_Level', 'Time_of_Day', 
                             'Vehicle_Type', 'Preparation_Time_min', 'Courier_Experience_yrs']
            
            X = input_data[feature_columns].copy()
            
            # If preprocessor has transform method, use it; otherwise, use simple preprocessing
            if hasattr(self.preprocessor, 'transform') and hasattr(self.preprocessor, 'preprocessor'):
                X_processed = self.preprocessor.transform(X)
            else:
                # Simple preprocessing fallback
                from sklearn.preprocessing import LabelEncoder, StandardScaler
                X_processed = X.copy()
                
                # Handle categorical variables
                categorical_cols = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
                for col in categorical_cols:
                    if col in X_processed.columns:
                        le = LabelEncoder()
                        # Fit with common values to avoid unseen category errors
                        if col == 'Weather':
                            le.fit(['Clear', 'Rainy', 'Snowy', 'Foggy', 'Windy'])
                        elif col == 'Traffic_Level':
                            le.fit(['Low', 'Medium', 'High'])
                        elif col == 'Time_of_Day':
                            le.fit(['Morning', 'Afternoon', 'Evening', 'Night'])
                        elif col == 'Vehicle_Type':
                            le.fit(['Car', 'Bike', 'Scooter'])
                        
                        X_processed[col] = le.transform(X_processed[col])
                
                X_processed = X_processed.values
            
            # Make prediction
            prediction = self.model.predict(X_processed)[0]
            
            # Calculate simple confidence score
            confidence = 0.85  # Default confidence for now
            
            # Create simple response
            response = {
                'prediction': float(prediction),
                'confidence': confidence,
                'model_version': '1.0',
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def predict_single(self, request: DeliveryRequest) -> DeliveryResponse:
        """
        Make a single delivery time prediction.
        
        Args:
            request: Delivery request data
            
        Returns:
            Prediction response
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model and preprocessor must be loaded before prediction")
        
        try:
            # Convert request to DataFrame
            input_data = pd.DataFrame([request.dict()])
            
            # Add dummy Order_ID and Delivery_Time_min for preprocessing
            input_data['Order_ID'] = 999999
            input_data['Delivery_Time_min'] = 0  # Will be ignored in transform
            
            # Rename columns to match training data format
            column_mapping = {
                'distance_km': 'Distance_km',
                'weather': 'Weather',
                'traffic_level': 'Traffic_Level',
                'time_of_day': 'Time_of_Day',
                'vehicle_type': 'Vehicle_Type',
                'preparation_time_min': 'Preparation_Time_min',
                'courier_experience_yrs': 'Courier_Experience_yrs'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in input_data.columns:
                    input_data[new_col] = input_data[old_col]
                    input_data.drop(columns=[old_col], inplace=True)
            
            # Transform the data
            X_processed = self.preprocessor.transform(input_data)
            
            # Make prediction
            prediction = self.model.predict(X_processed)[0]
            
            # Calculate confidence score
            confidence = self._calculate_confidence(X_processed, prediction)
            
            # Calculate prediction interval
            lower_bound, upper_bound = self._calculate_prediction_interval(
                X_processed, prediction, confidence
            )
            
            # Analyze prediction factors
            factors = self._analyze_prediction_factors(request, prediction)
            
            # Create response
            response = DeliveryResponse(
                predicted_delivery_time=round(prediction, 2),
                confidence_score=round(confidence, 3),
                prediction_interval_lower=round(lower_bound, 2),
                prediction_interval_upper=round(upper_bound, 2),
                factors=factors,
                model_version=self.model_metadata.get('version', '1.0'),
                timestamp=pd.Timestamp.now().isoformat()
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def predict_batch(self, requests: List[DeliveryRequest]) -> List[DeliveryResponse]:
        """
        Make batch predictions.
        
        Args:
            requests: List of delivery requests
            
        Returns:
            List of prediction responses
        """
        return [self.predict_single(request) for request in requests]
    
    def _calculate_confidence(self, X_processed: np.ndarray, prediction: float) -> float:
        """
        Calculate prediction confidence score.
        
        Args:
            X_processed: Processed input features
            prediction: Model prediction
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Use ensemble model variance if available
            if hasattr(self.model, 'estimators_'):
                predictions = np.array([
                    estimator.predict(X_processed)[0] 
                    for estimator in self.model.estimators_
                ])
                variance = np.var(predictions)
                confidence = 1 / (1 + variance)
            else:
                # For single models, use a heuristic based on feature values
                # This is a simplified approach - in production, you might want
                # to use uncertainty quantification methods
                confidence = 0.85  # Default confidence
            
            return max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0
            
        except Exception:
            return 0.5  # Default confidence if calculation fails
    
    def _calculate_prediction_interval(self, X_processed: np.ndarray, 
                                     prediction: float, confidence: float) -> tuple:
        """
        Calculate prediction interval.
        
        Args:
            X_processed: Processed input features
            prediction: Model prediction
            confidence: Confidence score
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Simple approach: use confidence to determine interval width
        # In production, you might want to use more sophisticated methods
        # like quantile regression or bootstrapping
        
        base_uncertainty = 5.0  # Base uncertainty in minutes
        uncertainty = base_uncertainty * (2 - confidence)  # Higher uncertainty for lower confidence
        
        lower_bound = max(0, prediction - uncertainty)
        upper_bound = prediction + uncertainty
        
        return lower_bound, upper_bound
    
    def _analyze_prediction_factors(self, request: DeliveryRequest, 
                                  prediction: float) -> Dict[str, str]:
        """
        Analyze factors contributing to the prediction.
        
        Args:
            request: Original request
            prediction: Model prediction
            
        Returns:
            Dictionary of factors and their impact
        """
        factors = {}
        
        # Distance impact
        if request.distance_km < 2:
            factors['distance'] = 'Very short distance - reduces delivery time'
        elif request.distance_km > 10:
            factors['distance'] = 'Long distance - increases delivery time'
        else:
            factors['distance'] = 'Moderate distance - average impact'
        
        # Weather impact
        if request.weather in ['Rainy', 'Snowy', 'Foggy']:
            factors['weather'] = f'{request.weather} weather - may increase delivery time'
        else:
            factors['weather'] = f'{request.weather} weather - favorable conditions'
        
        # Traffic impact
        if request.traffic_level == 'High':
            factors['traffic'] = 'High traffic - increases delivery time'
        elif request.traffic_level == 'Low':
            factors['traffic'] = 'Low traffic - reduces delivery time'
        else:
            factors['traffic'] = 'Medium traffic - moderate impact'
        
        # Vehicle impact
        vehicle_speeds = {'Car': 'fastest', 'Scooter': 'moderate', 'Bike': 'slowest'}
        factors['vehicle'] = f'{request.vehicle_type} - {vehicle_speeds.get(request.vehicle_type, "moderate")} option'
        
        # Experience impact
        if request.courier_experience_yrs > 5:
            factors['experience'] = 'Experienced courier - may reduce delivery time'
        elif request.courier_experience_yrs < 1:
            factors['experience'] = 'New courier - may increase delivery time'
        else:
            factors['experience'] = 'Moderately experienced courier'
        
        return factors
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the prediction service.
        
        Returns:
            Health status dictionary
        """
        status = {
            'status': 'healthy',
            'model_loaded': self.model is not None,
            'preprocessor_loaded': self.preprocessor is not None,
            'model_type': type(self.model).__name__ if self.model else None,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'model_version': self.model_metadata.get('version', 'unknown')
        }
        
        # Test prediction with dummy data
        try:
            test_request = DeliveryRequest(
                distance_km=5.0,
                weather='Clear',
                traffic_level='Medium',
                vehicle_type='Scooter',
                preparation_time_min=15.0,
                courier_experience_yrs=3.0
            )
            
            test_response = self.predict_single(test_request)
            status['test_prediction'] = 'passed'
            status['test_prediction_time'] = test_response.predicted_delivery_time
            
        except Exception as e:
            status['status'] = 'unhealthy'
            status['test_prediction'] = 'failed'
            status['error'] = str(e)
        
        return status
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        if self.model is None:
            return {'error': 'No model loaded'}
        
        info = {
            'model_type': type(self.model).__name__,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'model_metadata': self.model_metadata,
            'feature_names': self.feature_names[:10] if self.feature_names else []  # First 10 features
        }
        
        # Add model-specific parameters if available
        if hasattr(self.model, 'get_params'):
            info['model_parameters'] = self.model.get_params()
        
        return info

def main():
    """Example usage of the prediction service."""
    # Create a test request
    test_request = DeliveryRequest(
        distance_km=7.5,
        weather='Clear',
        traffic_level='Medium',
        vehicle_type='Scooter',
        preparation_time_min=20.0,
        courier_experience_yrs=2.5
    )
    
    # Initialize predictor (would normally load from saved files)
    predictor = PredictionService()
    
    # For testing without actual model files
    print("Test request created:")
    print(test_request.dict())
    
    # Health check
    health = predictor.health_check()
    print("\nHealth check (without loaded model):")
    print(health)

if __name__ == "__main__":
    main()
