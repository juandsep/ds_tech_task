#!/bin/bash

# Startup script for the Food Delivery Time Prediction API

set -e

echo "🚀 Starting Food Delivery Time Prediction API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📋 Installing dependencies..."
pip install -r requirements.txt

# Check if model files exist
if [ ! -f "model_pipeline/saved_models/best_model.joblib" ]; then
    echo "🤖 Training model (model files not found)..."
    cd model_pipeline
    python model_training.py
    cd ..
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment configuration..."
    cp .env.example .env
    echo "📝 Please review and update .env file with your configuration"
fi

# Create logs directory
mkdir -p logs

# Run tests
echo "🧪 Running tests..."
pytest src/test_api.py -v

# Start the API
echo "🌟 Starting API server..."
echo "📖 API Documentation: http://localhost:8000/docs"
echo "❤️ Health Check: http://localhost:8000/health"
echo "🔍 Metrics: http://localhost:8000/metrics"

uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
