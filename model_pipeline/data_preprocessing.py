"""
Data preprocessing pipeline for food delivery time prediction.
Handles feature engineering, encoding, and data validation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for delivery time prediction.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Configuration dictionary for preprocessing parameters
        """
        self.config = config or self._get_default_config()
        self.preprocessor = None
        self.feature_names = None
        self.target_scaler = StandardScaler()
        self.is_fitted = False
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for preprocessing."""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'handle_outliers': True,
            'outlier_method': 'iqr',
            'outlier_factor': 1.5,
            'scale_target': False,
            'create_features': True
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file with validation.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded and validated DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            
            # Validate required columns
            required_cols = ['Delivery_Time_min']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        df_clean = df.copy()
        
        # Check for missing values
        missing_summary = df_clean.isnull().sum()
        if missing_summary.sum() > 0:
            logger.info("Missing values found:")
            logger.info(missing_summary[missing_summary > 0])
            
            # Handle missing values based on column type
            for col in df_clean.columns:
                if df_clean[col].isnull().sum() > 0:
                    if df_clean[col].dtype in ['object']:
                        # Categorical: fill with mode
                        mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                        df_clean[col].fillna(mode_val, inplace=True)
                    else:
                        # Numerical: fill with median
                        median_val = df_clean[col].median()
                        df_clean[col].fillna(median_val, inplace=True)
                        
        return df_clean
    
    def detect_outliers(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Detect outliers using IQR method.
        
        Args:
            df: Input DataFrame
            column: Column name to check for outliers
            
        Returns:
            Boolean series indicating outliers
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        factor = self.config['outlier_factor']
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        return outliers
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in numerical columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled outliers
        """
        if not self.config['handle_outliers']:
            return df
            
        df_clean = df.copy()
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col != 'Order_ID':  # Skip ID columns
                outliers = self.detect_outliers(df_clean, col)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.info(f"Found {outlier_count} outliers in {col}")
                    
                    # Cap outliers at 95th and 5th percentiles
                    lower_cap = df_clean[col].quantile(0.05)
                    upper_cap = df_clean[col].quantile(0.95)
                    
                    df_clean.loc[df_clean[col] < lower_cap, col] = lower_cap
                    df_clean.loc[df_clean[col] > upper_cap, col] = upper_cap
                    
        return df_clean
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        if not self.config['create_features']:
            return df
            
        df_features = df.copy()
        
        # Speed feature (if distance is available)
        if 'Distance_km' in df.columns:
            # Avoid division by zero
            df_features['Speed_kmh'] = np.where(
                df_features['Delivery_Time_min'] > 0,
                (df_features['Distance_km'] / df_features['Delivery_Time_min']) * 60,
                0
            )
            
        # Experience categories
        if 'Courier_Experience_yrs' in df.columns:
            df_features['Experience_Level'] = pd.cut(
                df_features['Courier_Experience_yrs'], 
                bins=[0, 1, 3, 5, float('inf')], 
                labels=['Novice', 'Beginner', 'Experienced', 'Expert']
            ).astype(str)
            
        # Distance categories
        if 'Distance_km' in df.columns:
            df_features['Distance_Category'] = pd.cut(
                df_features['Distance_km'],
                bins=[0, 2, 5, 10, 15, float('inf')],
                labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            ).astype(str)
            
        # Time efficiency
        if 'Preparation_Time_min' in df.columns:
            df_features['Prep_to_Total_Ratio'] = np.where(
                df_features['Delivery_Time_min'] > 0,
                df_features['Preparation_Time_min'] / df_features['Delivery_Time_min'],
                0
            )
            
        # Weather-Traffic interaction
        if 'Weather' in df.columns and 'Traffic_Level' in df.columns:
            df_features['Weather_Traffic'] = df_features['Weather'] + '_' + df_features['Traffic_Level']
            
        logger.info(f"Created features. New shape: {df_features.shape}")
        return df_features
    
    def prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, target)
        """
        # Target variable
        target = df['Delivery_Time_min'].copy()
        
        # Features (exclude target and ID columns)
        exclude_cols = ['Delivery_Time_min', 'Order_ID']
        features = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        
        return features, target
    
    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Build the preprocessing pipeline.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Configured ColumnTransformer
        """
        # Identify column types
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Numerical features: {numerical_features}")
        logger.info(f"Categorical features: {categorical_features}")
        
        # Define transformers
        transformers = []
        
        if numerical_features:
            numerical_transformer = Pipeline([
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_transformer, numerical_features))
        
        if categorical_features:
            categorical_transformer = Pipeline([
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        return preprocessor
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Starting data preprocessing...")
        
        # Data cleaning steps
        df_clean = self.handle_missing_values(df)
        df_clean = self.handle_outliers(df_clean)
        df_features = self.create_features(df_clean)
        
        # Prepare features and target
        X, y = self.prepare_features_target(df_features)
        
        # Build and fit preprocessor
        self.preprocessor = self.build_preprocessor(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'], 
            random_state=self.config['random_state']
        )
        
        # Fit and transform
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Store feature names
        self._store_feature_names(X)
        
        # Scale target if requested
        if self.config['scale_target']:
            y_train = self.target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
            y_test = self.target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
        
        self.is_fitted = True
        
        logger.info(f"Preprocessing complete. Train shape: {X_train_processed.shape}, Test shape: {X_test_processed.shape}")
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed feature array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Apply same transformations as training
        df_clean = self.handle_missing_values(df)
        df_features = self.create_features(df_clean)
        X, _ = self.prepare_features_target(df_features)
        
        # Transform
        X_processed = self.preprocessor.transform(X)
        
        return X_processed
    
    def _store_feature_names(self, X: pd.DataFrame):
        """Store feature names for later use."""
        # Get feature names from the transformer
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            self.feature_names = self.preprocessor.get_feature_names_out()
        else:
            # Fallback for older versions
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    def get_feature_names(self) -> list:
        """Get the names of the processed features."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return self.feature_names
    
    def save_preprocessor(self, file_path: str):
        """Save the fitted preprocessor to disk."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        save_data = {
            'preprocessor': self.preprocessor,
            'target_scaler': self.target_scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_data, file_path)
        logger.info(f"Preprocessor saved to {file_path}")
    
    def load_preprocessor(self, file_path: str):
        """Load a fitted preprocessor from disk."""
        save_data = joblib.load(file_path)
        
        self.preprocessor = save_data['preprocessor']
        self.target_scaler = save_data['target_scaler']
        self.feature_names = save_data['feature_names']
        self.config = save_data['config']
        self.is_fitted = save_data['is_fitted']
        
        logger.info(f"Preprocessor loaded from {file_path}")

def main():
    """Main function for testing the preprocessor."""
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load and process data
    df = preprocessor.load_data('../data/Food_Delivery_Times.csv')
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Feature names: {len(preprocessor.get_feature_names())}")
    
    # Save preprocessor
    preprocessor.save_preprocessor('../models/preprocessor.pkl')

if __name__ == "__main__":
    main()
