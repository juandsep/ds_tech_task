"""
Configuration settings for the FastAPI application.
"""

import os
from typing import List
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    app_name: str = "Food Delivery Time Prediction API"
    version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    allowed_hosts: List[str] = ["*"]
    cors_origins: List[str] = ["*"]
    
    # Model Configuration
    model_path: str = "../model_pipeline/saved_models/best_model.joblib"
    preprocessor_path: str = "../model_pipeline/saved_models/preprocessor.joblib"
    model_reload_interval: int = 3600  # seconds
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Monitoring
    enable_metrics: bool = True
    metrics_retention_days: int = 30
    
    # Database (for request logging, if needed)
    database_url: str = "sqlite:///./requests.db"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()
