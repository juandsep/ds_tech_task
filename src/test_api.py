"""
Test suite for the FastAPI application.
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from src.main import app

class TestFoodDeliveryAPI:
    """Test cases for the Food Delivery Time Prediction API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "model_loaded" in data
    
    def test_predict_endpoint_valid_request(self, client):
        """Test prediction endpoint with valid request."""
        request_data = {
            "distance_km": 5.2,
            "weather": "Clear",
            "traffic_level": "Medium",
            "time_of_day": "Evening",
            "vehicle_type": "Scooter",
            "preparation_time_min": 15.0,
            "courier_experience_yrs": 3.0
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predicted_delivery_time" in data
        assert "confidence_score" in data
        assert "prediction_interval_lower" in data
        assert "prediction_interval_upper" in data
        assert "factors" in data
        assert "model_version" in data
        assert "timestamp" in data
    
    def test_predict_endpoint_invalid_distance(self, client):
        """Test prediction endpoint with invalid distance."""
        request_data = {
            "distance_km": -1.0,  # Invalid negative distance
            "weather": "Clear",
            "traffic_level": "Medium",
            "vehicle_type": "Scooter",
            "preparation_time_min": 15.0,
            "courier_experience_yrs": 3.0
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422
    
    def test_predict_endpoint_invalid_weather(self, client):
        """Test prediction endpoint with invalid weather."""
        request_data = {
            "distance_km": 5.2,
            "weather": "Sunny",  # Invalid weather option
            "traffic_level": "Medium",
            "vehicle_type": "Scooter",
            "preparation_time_min": 15.0,
            "courier_experience_yrs": 3.0
        }
        
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422
    
    def test_batch_prediction(self, client):
        """Test batch prediction endpoint."""
        batch_data = {
            "requests": [
                {
                    "distance_km": 5.2,
                    "weather": "Clear",
                    "traffic_level": "Medium",
                    "vehicle_type": "Scooter",
                    "preparation_time_min": 15.0,
                    "courier_experience_yrs": 3.0
                },
                {
                    "distance_km": 8.1,
                    "weather": "Rainy",
                    "traffic_level": "High",
                    "vehicle_type": "Car",
                    "preparation_time_min": 20.0,
                    "courier_experience_yrs": 5.0
                }
            ]
        }
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "batch_id" in data
        assert "processed_count" in data
        assert len(data["predictions"]) == 2
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_version" in data
        assert "total_predictions" in data
        assert "avg_response_time_ms" in data
        assert "feature_importance" in data
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_version" in data
        assert "model_type" in data
        assert "features" in data
        assert "performance_metrics" in data
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        """Test handling of concurrent requests."""
        request_data = {
            "distance_km": 5.2,
            "weather": "Clear",
            "traffic_level": "Medium",
            "vehicle_type": "Scooter",
            "preparation_time_min": 15.0,
            "courier_experience_yrs": 3.0
        }
        
        # Send multiple concurrent requests
        tasks = []
        for _ in range(10):
            task = async_client.post("/predict", json=request_data)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
    
    def test_request_validation_edge_cases(self, client):
        """Test edge cases for request validation."""
        # Test maximum distance
        request_data = {
            "distance_km": 50.0,  # Maximum allowed
            "weather": "Clear",
            "traffic_level": "Medium",
            "vehicle_type": "Scooter",
            "preparation_time_min": 15.0,
            "courier_experience_yrs": 3.0
        }
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        # Test maximum preparation time
        request_data["preparation_time_min"] = 120.0  # Maximum allowed
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        
        # Test maximum courier experience
        request_data["courier_experience_yrs"] = 20.0  # Maximum allowed
        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

class TestModelPredictionLogic:
    """Test the prediction logic and model behavior."""
    
    def test_prediction_consistency(self, client=TestClient(app)):
        """Test that same inputs produce consistent outputs."""
        request_data = {
            "distance_km": 5.2,
            "weather": "Clear",
            "traffic_level": "Medium",
            "vehicle_type": "Scooter",
            "preparation_time_min": 15.0,
            "courier_experience_yrs": 3.0
        }
        
        # Make multiple requests with same data
        responses = []
        for _ in range(3):
            response = client.post("/predict", json=request_data)
            responses.append(response.json())
        
        # All predictions should be identical
        first_prediction = responses[0]["predicted_delivery_time"]
        for response in responses[1:]:
            assert response["predicted_delivery_time"] == first_prediction
    
    def test_prediction_bounds(self, client=TestClient(app)):
        """Test that predictions are within reasonable bounds."""
        request_data = {
            "distance_km": 1.0,
            "weather": "Clear",
            "traffic_level": "Low",
            "vehicle_type": "Bike",
            "preparation_time_min": 5.0,
            "courier_experience_yrs": 5.0
        }
        
        response = client.post("/predict", json=request_data)
        data = response.json()
        
        # Prediction should be positive and reasonable
        assert data["predicted_delivery_time"] > 0
        assert data["predicted_delivery_time"] < 180  # Less than 3 hours
        
        # Confidence intervals should make sense
        assert data["prediction_interval_lower"] <= data["predicted_delivery_time"]
        assert data["prediction_interval_upper"] >= data["predicted_delivery_time"]

if __name__ == "__main__":
    pytest.main([__file__])
