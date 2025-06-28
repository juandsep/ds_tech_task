"""
Model utilities for delivery time prediction.
Shared utilities used across the modeling pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
from typing import Dict, List, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Model evaluation utilities.
    """
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

class ModelSelector:
    """
    Model selection utilities.
    """
    
    def select_best_model(self, evaluation_results: Dict[str, Dict[str, float]], 
                         criteria: str = 'mae') -> str:
        """
        Select the best model based on criteria.
        
        Args:
            evaluation_results: Dictionary of model evaluation results
            criteria: Selection criteria
            
        Returns:
            Name of the best model
        """
        if criteria in ['mae', 'rmse']:
            return min(evaluation_results.keys(), 
                      key=lambda x: evaluation_results[x][criteria])
        elif criteria == 'r2':
            return max(evaluation_results.keys(), 
                      key=lambda x: evaluation_results[x][criteria])
        else:
            raise ValueError(f"Unknown criteria: {criteria}")

class ConfigManager:
    """
    Configuration management utilities.
    """
    
    @staticmethod
    def get_default_preprocessing_config() -> Dict[str, Any]:
        """Get default preprocessing configuration."""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'handle_outliers': True,
            'outlier_method': 'iqr',
            'outlier_factor': 1.5,
            'scale_target': False,
            'create_features': True
        }
    
    @staticmethod
    def get_default_training_config() -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            'cv_folds': 5,
            'scoring': 'neg_mean_absolute_error',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1,
            'models_to_train': ['linear', 'ridge', 'random_forest', 'xgboost'],
            'hyperparameter_tuning': True,
            'quick_train': False
        }

class DataValidator:
    """
    Data validation utilities.
    """
    
    @staticmethod
    def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that the DataFrame has required columns.
        
        Args:
            df: Input DataFrame
            required_columns: List of required column names
            
        Returns:
            True if valid, False otherwise
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality metrics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data quality metrics
        """
        quality_report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Check for outliers in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outliers[col] = outlier_count
            
        quality_report['outliers'] = outliers
        
        return quality_report

class FeatureEngineer:
    """
    Feature engineering utilities.
    """
    
    @staticmethod
    def create_time_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """
        Create time-based features from datetime column.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            
        Returns:
            DataFrame with time features
        """
        df_copy = df.copy()
        
        if datetime_col in df_copy.columns:
            df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
            
            df_copy['hour'] = df_copy[datetime_col].dt.hour
            df_copy['day_of_week'] = df_copy[datetime_col].dt.dayofweek
            df_copy['month'] = df_copy[datetime_col].dt.month
            df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
            
        return df_copy
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between specified pairs.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of feature pairs to create interactions
            
        Returns:
            DataFrame with interaction features
        """
        df_copy = df.copy()
        
        for feature1, feature2 in feature_pairs:
            if feature1 in df_copy.columns and feature2 in df_copy.columns:
                interaction_name = f"{feature1}_{feature2}_interaction"
                
                if df_copy[feature1].dtype in ['object'] and df_copy[feature2].dtype in ['object']:
                    # Categorical-categorical interaction
                    df_copy[interaction_name] = df_copy[feature1] + '_' + df_copy[feature2]
                elif df_copy[feature1].dtype in ['object'] or df_copy[feature2].dtype in ['object']:
                    # Categorical-numerical interaction (groupby mean)
                    if df_copy[feature1].dtype in ['object']:
                        cat_col, num_col = feature1, feature2
                    else:
                        cat_col, num_col = feature2, feature1
                    
                    group_means = df_copy.groupby(cat_col)[num_col].transform('mean')
                    df_copy[interaction_name] = group_means
                else:
                    # Numerical-numerical interaction
                    df_copy[interaction_name] = df_copy[feature1] * df_copy[feature2]
        
        return df_copy

class ModelPersistence:
    """
    Model persistence utilities.
    """
    
    @staticmethod
    def save_model_artifacts(model: Any, preprocessor: Any, metadata: Dict[str, Any], 
                           model_path: str, preprocessor_path: str, metadata_path: str):
        """
        Save all model artifacts.
        
        Args:
            model: Trained model
            preprocessor: Fitted preprocessor
            metadata: Model metadata
            model_path: Path to save model
            preprocessor_path: Path to save preprocessor
            metadata_path: Path to save metadata
        """
        joblib.dump(model, model_path)
        joblib.dump(preprocessor, preprocessor_path)
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model artifacts saved:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Preprocessor: {preprocessor_path}")
        logger.info(f"  Metadata: {metadata_path}")
    
    @staticmethod
    def load_model_artifacts(model_path: str, preprocessor_path: str, 
                           metadata_path: str) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Load all model artifacts.
        
        Args:
            model_path: Path to model file
            preprocessor_path: Path to preprocessor file
            metadata_path: Path to metadata file
            
        Returns:
            Tuple of (model, preprocessor, metadata)
        """
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        metadata = joblib.load(metadata_path)
        
        logger.info(f"Model artifacts loaded:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Preprocessor: {preprocessor_path}")
        logger.info(f"  Metadata: {metadata_path}")
        
        return model, preprocessor, metadata

class ModelMonitor:
    """
    Model monitoring utilities.
    """
    
    @staticmethod
    def detect_data_drift(reference_data: pd.DataFrame, new_data: pd.DataFrame, 
                         threshold: float = 0.1) -> Dict[str, bool]:
        """
        Detect data drift between reference and new data.
        
        Args:
            reference_data: Reference dataset
            new_data: New dataset to compare
            threshold: Threshold for drift detection
            
        Returns:
            Dictionary indicating drift for each feature
        """
        drift_results = {}
        
        for col in reference_data.columns:
            if col in new_data.columns:
                if reference_data[col].dtype in ['object']:
                    # Categorical drift using distribution comparison
                    ref_dist = reference_data[col].value_counts(normalize=True)
                    new_dist = new_data[col].value_counts(normalize=True)
                    
                    # Calculate total variation distance
                    all_categories = set(ref_dist.index) | set(new_dist.index)
                    tvd = 0.5 * sum(abs(ref_dist.get(cat, 0) - new_dist.get(cat, 0)) 
                                   for cat in all_categories)
                    
                    drift_results[col] = tvd > threshold
                    
                else:
                    # Numerical drift using mean and std comparison
                    ref_mean, ref_std = reference_data[col].mean(), reference_data[col].std()
                    new_mean, new_std = new_data[col].mean(), new_data[col].std()
                    
                    mean_drift = abs(ref_mean - new_mean) / ref_std > threshold
                    std_drift = abs(ref_std - new_std) / ref_std > threshold
                    
                    drift_results[col] = mean_drift or std_drift
        
        return drift_results
    
    @staticmethod
    def calculate_prediction_confidence(model: Any, X: np.ndarray) -> np.ndarray:
        """
        Calculate prediction confidence scores.
        
        Args:
            model: Trained model
            X: Input features
            
        Returns:
            Confidence scores
        """
        if hasattr(model, 'predict_proba'):
            # For models with probability prediction
            probas = model.predict_proba(X)
            confidence = np.max(probas, axis=1)
        elif hasattr(model, 'decision_function'):
            # For models with decision function
            scores = model.decision_function(X)
            confidence = np.abs(scores)
        else:
            # For regression models, use prediction variance if available
            if hasattr(model, 'predict') and hasattr(model, 'estimators_'):
                # Ensemble models
                predictions = np.array([estimator.predict(X) for estimator in model.estimators_])
                confidence = 1 / (1 + np.std(predictions, axis=0))
            else:
                # Default: equal confidence for all predictions
                confidence = np.ones(X.shape[0])
        
        return confidence

def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )

def main():
    """Example usage of model utilities."""
    # Example data validation
    data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randn(100)
    })
    
    validator = DataValidator()
    is_valid = validator.validate_data(data, ['feature1', 'feature2', 'target'])
    quality_report = validator.check_data_quality(data)
    
    print(f"Data is valid: {is_valid}")
    print(f"Quality report: {quality_report}")

if __name__ == "__main__":
    main()
