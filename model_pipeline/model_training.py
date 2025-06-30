"""
Model training pipeline for delivery time prediction.
Implements multiple algorithms with hyperparameter tuning and validation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
import time
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor
from utils.model_utils import ModelEvaluator, ModelSelector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeliveryTimePredictor:
    """
    Comprehensive model training pipeline for delivery time prediction.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the predictor with configuration.
        
        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = config or self._get_default_config()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.evaluator = ModelEvaluator()
        self.selector = ModelSelector()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for model training."""
        return {
            'cv_folds': 5,
            'scoring': 'neg_mean_absolute_error',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1,
            'test_size': 0.2,
            'models_to_train': ['linear', 'ridge', 'random_forest', 'xgboost', 'lightgbm'],
            'hyperparameter_tuning': True,
            'quick_train': False  # Set to True for faster training with fewer parameters
        }
    
    def get_model_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Define models and their hyperparameter grids.
        
        Returns:
            Dictionary of model definitions
        """
        if self.config['quick_train']:
            # Simplified parameters for quick training
            return {
                'linear': {
                    'model': LinearRegression(),
                    'params': {}
                },
                'ridge': {
                    'model': Ridge(random_state=self.config['random_state']),
                    'params': {'alpha': [0.1, 1.0, 10.0]}
                },
                'random_forest': {
                    'model': RandomForestRegressor(random_state=self.config['random_state']),
                    'params': {
                        'n_estimators': [50, 100],
                        'max_depth': [10, None]
                    }
                },
                'xgboost': {
                    'model': xgb.XGBRegressor(random_state=self.config['random_state']),
                    'params': {
                        'n_estimators': [50, 100],
                        'max_depth': [3, 6]
                    }
                }
            }
        else:
            # Full parameter grids for comprehensive training
            return {
                'linear': {
                    'model': LinearRegression(),
                    'params': {}
                },
                'ridge': {
                    'model': Ridge(random_state=self.config['random_state']),
                    'params': {
                        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                    }
                },
                'lasso': {
                    'model': Lasso(random_state=self.config['random_state']),
                    'params': {
                        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                    }
                },
                'random_forest': {
                    'model': RandomForestRegressor(random_state=self.config['random_state']),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 15, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor(random_state=self.config['random_state']),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.2]
                    }
                },
                'xgboost': {
                    'model': xgb.XGBRegressor(random_state=self.config['random_state']),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                },
                'lightgbm': {
                    'model': lgb.LGBMRegressor(random_state=self.config['random_state'], verbose=-1),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'num_leaves': [31, 50, 100]
                    }
                },
                'svr': {
                    'model': SVR(),
                    'params': {
                        'C': [0.1, 1, 10],
                        'gamma': ['scale', 'auto'],
                        'epsilon': [0.01, 0.1, 1.0]
                    }
                }
            }
    
    def train_single_model(self, model_name: str, model_def: Dict[str, Any], 
                          X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model
            model_def: Model definition dictionary
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary with trained model and metadata
        """
        logger.info(f"Training {model_name}...")
        start_time = time.time()
        
        model = model_def['model']
        param_grid = model_def['params']
        
        if param_grid and self.config['hyperparameter_tuning']:
            # Hyperparameter tuning with GridSearchCV
            grid_search = GridSearchCV(
                model, 
                param_grid,
                cv=self.config['cv_folds'],
                scoring=self.config['scoring'],
                n_jobs=self.config['n_jobs'],
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
            
        else:
            # Train with default parameters
            model.fit(X_train, y_train)
            best_model = model
            best_params = {}
            
            # Get cross-validation score
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config['cv_folds'],
                scoring=self.config['scoring'],
                n_jobs=self.config['n_jobs']
            )
            cv_score = cv_scores.mean()
        
        training_time = time.time() - start_time
        
        result = {
            'model': best_model,
            'best_params': best_params,
            'cv_score': cv_score,
            'training_time': training_time
        }
        
        logger.info(f"{model_name} training completed in {training_time:.2f}s. CV Score: {cv_score:.4f}")
        
        return result
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Train all specified models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Starting training of all models...")
        logger.info(f'Configuration: {self.config}')
        model_definitions = self.get_model_definitions()

        trained_models = {}
        
        for model_name in self.config['models_to_train']:
            if model_name in model_definitions:
                try:
                    result = self.train_single_model(
                        model_name, 
                        model_definitions[model_name], 
                        X_train, 
                        y_train
                    )
                    trained_models[model_name] = result
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {str(e)}")
                    continue
            else:
                logger.warning(f"Model {model_name} not found in definitions")
        
        return trained_models
    
    def evaluate_models(self, models: Dict[str, Dict[str, Any]], 
                       X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test set.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        evaluation_results = {}
        
        for model_name, model_info in models.items():
            model = model_info['model']
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(y_test, y_pred)
            metrics['cv_score'] = model_info['cv_score']
            metrics['training_time'] = model_info['training_time']
            
            evaluation_results[model_name] = metrics
            
            logger.info(f"{model_name} - MAE: {metrics['mae']:.3f}, "
                       f"RMSE: {metrics['rmse']:.3f}, R2: {metrics['r2']:.3f}")
        
        return evaluation_results
    
    def select_best_model(self, evaluation_results: Dict[str, Dict[str, float]], 
                         models: Dict[str, Dict[str, Any]]) -> Tuple[str, Any]:
        """
        Select the best model based on evaluation metrics.
        
        Args:
            evaluation_results: Dictionary of evaluation results
            models: Dictionary of trained models
            
        Returns:
            Tuple of (best_model_name, best_model)
        """
        best_model_name = self.selector.select_best_model(evaluation_results)
        best_model = models[best_model_name]['model']
        
        logger.info(f"Best model selected: {best_model_name}")
        
        return best_model_name, best_model
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract feature importance from the model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_)
        else:
            # Models without feature importance
            return pd.DataFrame({'feature': feature_names, 'importance': [0] * len(feature_names)})
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def train_pipeline(self, data_path: str) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            data_path: Path to the training data
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting training pipeline...")
        
        # Data preprocessing
        self.preprocessor = DataPreprocessor()
        df = self.preprocessor.load_data(data_path)
        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(df)
        
        # Train models
        trained_models = self.train_all_models(X_train, y_train)
        self.models = trained_models
        
        # Evaluate models
        evaluation_results = self.evaluate_models(trained_models, X_test, y_test)
        
        # Select best model
        self.best_model_name, self.best_model = self.select_best_model(evaluation_results, trained_models)
        
        # Feature importance
        feature_names = self.preprocessor.get_feature_names()
        feature_importance = self.get_feature_importance(self.best_model, feature_names)
        
        # Prepare results
        results = {
            'evaluation_results': evaluation_results,
            'best_model_name': self.best_model_name,
            'feature_importance': feature_importance,
            'test_predictions': self.best_model.predict(X_test),
            'test_actuals': y_test,
            'model_params': trained_models[self.best_model_name]['best_params']
        }
        
        logger.info("Training pipeline completed successfully")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the best model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        return self.best_model.predict(X)
    
    def save_model(self, model_path: str, preprocessor_path: str):
        """
        Save the trained model and preprocessor.
        
        Args:
            model_path: Path to save the model
            preprocessor_path: Path to save the preprocessor
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        # Save model
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'config': self.config
        }
        joblib.dump(model_data, model_path)
        
        # Save preprocessor
        self.preprocessor.save_preprocessor(preprocessor_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    def load_model(self, model_path: str, preprocessor_path: str):
        """
        Load a trained model and preprocessor.
        
        Args:
            model_path: Path to the saved model
            preprocessor_path: Path to the saved preprocessor
        """
        # Load model
        model_data = joblib.load(model_path)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.config = model_data['config']
        
        # Load preprocessor
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load_preprocessor(preprocessor_path)
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Preprocessor loaded from {preprocessor_path}")

def main():
    """Main function for training the delivery time prediction model."""
    # Configuration
    config = {
        'models_to_train': ['linear', 'ridge', 'random_forest', 'xgboost'],
        'quick_train': True,  # Set to False for full training
        'hyperparameter_tuning': True
    }
    
    # Initialize predictor
    predictor = DeliveryTimePredictor(config)
    
    # Train the pipeline
    results = predictor.train_pipeline('../data/Food_Delivery_Times.csv')
    
    # Print results
    print("\n=== TRAINING RESULTS ===")
    print(f"Best model: {results['best_model_name']}")
    print("\nModel Performance:")
    for model_name, metrics in results['evaluation_results'].items():
        print(f"{model_name}: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}")
    
    print("\nTop 10 Important Features:")
    print(results['feature_importance'].head(10))
    
    # Save model
    import os
    os.makedirs('../models', exist_ok=True)
    predictor.save_model('../models/delivery_time_model.pkl', '../models/preprocessor.pkl')
    
    print("\nModel saved successfully!")

if __name__ == "__main__":
    main()
