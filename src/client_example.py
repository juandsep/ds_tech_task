"""
Example client for testing the Food Delivery Time Prediction API.
"""

import requests
import json
import time
from typing import Dict, List

class FoodDeliveryAPIClient:
    """Client for interacting with the Food Delivery API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict_delivery_time(self, delivery_data: Dict) -> Dict:
        """
        Predict delivery time for a single delivery.
        
        Args:
            delivery_data: Dictionary with delivery details
            
        Returns:
            Prediction response
        """
        response = self.session.post(
            f"{self.base_url}/predict",
            json=delivery_data
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, delivery_requests: List[Dict]) -> Dict:
        """
        Predict delivery times for multiple deliveries.
        
        Args:
            delivery_requests: List of delivery data dictionaries
            
        Returns:
            Batch prediction response
        """
        batch_data = {"requests": delivery_requests}
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            json=batch_data
        )
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> Dict:
        """Get API metrics and statistics."""
        response = self.session.get(f"{self.base_url}/metrics")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        response = self.session.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()

def main():
    """Example usage of the API client."""
    
    # Initialize client
    client = FoodDeliveryAPIClient()
    
    print("🍕 Food Delivery Time Prediction API Client")
    print("=" * 50)
    
    try:
        # Health check
        print("1. Checking API health...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        print()
        
        # Single prediction
        print("2. Making single prediction...")
        delivery_data = {
            "distance_km": 5.2,
            "weather": "Clear",
            "traffic_level": "Medium",
            "time_of_day": "Evening",
            "vehicle_type": "Scooter",
            "preparation_time_min": 15.0,
            "courier_experience_yrs": 3.0
        }
        
        prediction = client.predict_delivery_time(delivery_data)
        print(f"   Predicted time: {prediction['predicted_delivery_time']} minutes")
        print(f"   Confidence: {prediction['confidence_score']:.2%}")
        print(f"   Range: {prediction['prediction_interval_lower']:.1f} - {prediction['prediction_interval_upper']:.1f} minutes")
        print()
        
        # Batch prediction
        print("3. Making batch prediction...")
        batch_requests = [
            {
                "distance_km": 3.1,
                "weather": "Rainy",
                "traffic_level": "High",
                "vehicle_type": "Car",
                "preparation_time_min": 20.0,
                "courier_experience_yrs": 5.0
            },
            {
                "distance_km": 8.5,
                "weather": "Clear",
                "traffic_level": "Low",
                "vehicle_type": "Bike",
                "preparation_time_min": 10.0,
                "courier_experience_yrs": 2.0
            }
        ]
        
        batch_result = client.predict_batch(batch_requests)
        print(f"   Processed {batch_result['processed_count']} requests")
        for i, pred in enumerate(batch_result['predictions']):
            print(f"   Request {i+1}: {pred['predicted_delivery_time']} minutes")
        print()
        
        # Get metrics
        print("4. Fetching API metrics...")
        metrics = client.get_metrics()
        print(f"   Total predictions: {metrics['total_predictions']}")
        print(f"   Average response time: {metrics['avg_response_time_ms']:.2f}ms")
        print()
        
        # Get model info
        print("5. Getting model information...")
        model_info = client.get_model_info()
        print(f"   Model version: {model_info['model_version']}")
        print(f"   Model type: {model_info['model_type']}")
        print(f"   Features: {len(model_info['features'])}")
        print()
        
        # Performance test
        print("6. Running performance test...")
        start_time = time.time()
        
        for i in range(10):
            test_data = {
                "distance_km": 5.0 + i * 0.5,
                "weather": "Clear",
                "traffic_level": "Medium",
                "vehicle_type": "Scooter",
                "preparation_time_min": 15.0,
                "courier_experience_yrs": 3.0
            }
            client.predict_delivery_time(test_data)
        
        total_time = time.time() - start_time
        print(f"   Completed 10 predictions in {total_time:.2f} seconds")
        print(f"   Average: {(total_time/10)*1000:.1f}ms per prediction")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running:")
        print("   uvicorn src.main:app --reload")
        print("   or")
        print("   docker-compose up")
    
    except requests.exceptions.HTTPError as e:
        print(f"❌ API Error: {e}")
        if hasattr(e.response, 'text'):
            try:
                error_detail = json.loads(e.response.text)
                print(f"   Details: {error_detail}")
            except:
                print(f"   Response: {e.response.text}")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
