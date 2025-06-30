"""
Model evaluation utilities for delivery time prediction.
Provides comprehensive evaluation metrics and visualization tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    mean_absolute_percentage_error
)
from scipy import stats
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation toolkit.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'mean_residual': np.mean(y_pred - y_true),
            'std_residual': np.std(y_pred - y_true)
        }
        
        return metrics
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Actual vs Predicted") -> plt.Figure:
        """
        Create prediction scatter plot.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, color='blue', s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Formatting
        ax.set_xlabel('Actual Delivery Time (minutes)')
        ax.set_ylabel('Predicted Delivery Time (minutes)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add metrics text
        metrics = self.calculate_metrics(y_true, y_pred)
        metrics_text = f"MAE: {metrics['mae']:.2f}\nRMSE: {metrics['rmse']:.2f}\nR²: {metrics['r2']:.3f}"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
        """
        Create residual plots for model diagnosis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            matplotlib Figure
        """
        residuals = y_pred - y_true
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, color='blue')
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs Actual
        axes[0, 1].scatter(y_true, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Actual')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot of residuals
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                               top_n: int = 15) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_importance: DataFrame with feature importance
            top_n: Number of top features to show
            
        Returns:
            matplotlib Figure
        """
        # Get top features
        top_features = feature_importance.head(top_n)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def analyze_errors_by_segments(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  segment_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Analyze model errors by different segments.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            segment_data: DataFrame with segmentation variables
            
        Returns:
            Dictionary of error analysis by segments
        """
        residuals = y_pred - y_true
        abs_errors = np.abs(residuals)
        
        # Create analysis DataFrame
        analysis_df = segment_data.copy()
        analysis_df['actual'] = y_true
        analysis_df['predicted'] = y_pred
        analysis_df['residual'] = residuals
        analysis_df['abs_error'] = abs_errors
        
        results = {}
        
        # Analyze by categorical columns
        categorical_cols = analysis_df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in analysis_df.columns:
                segment_analysis = analysis_df.groupby(col).agg({
                    'abs_error': ['mean', 'std', 'count'],
                    'residual': ['mean', 'std'],
                    'actual': 'mean',
                    'predicted': 'mean'
                }).round(3)
                
                segment_analysis.columns = [f'{col[1]}_{col[0]}' for col in segment_analysis.columns]
                results[col] = segment_analysis
        
        return results
    
    def create_error_heatmap(self, error_analysis: Dict[str, pd.DataFrame]) -> plt.Figure:
        """
        Create heatmap of errors by segments.
        
        Args:
            error_analysis: Error analysis by segments
            
        Returns:
            matplotlib Figure
        """
        if not error_analysis:
            return None
        
        # Create subplots for each segment
        n_segments = len(error_analysis)
        fig, axes = plt.subplots(1, n_segments, figsize=(5*n_segments, 6))
        
        if n_segments == 1:
            axes = [axes]
        
        for i, (segment_name, segment_data) in enumerate(error_analysis.items()):
            # Get mean absolute error column
            mae_col = [col for col in segment_data.columns if 'mean_abs_error' in col]
            if mae_col:
                data_to_plot = segment_data[mae_col[0]].values.reshape(-1, 1)
                
                sns.heatmap(data_to_plot, 
                           yticklabels=segment_data.index,
                           xticklabels=[segment_name],
                           annot=True, fmt='.2f', 
                           cmap='Reds', ax=axes[i])
                axes[i].set_title(f'Mean Absolute Error by {segment_name}')
        
        plt.tight_layout()
        return fig
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  feature_importance: pd.DataFrame = None,
                                  segment_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            feature_importance: Feature importance DataFrame
            segment_data: Data for segment analysis
            
        Returns:
            Dictionary with evaluation results and plots
        """
        logger.info("Generating evaluation report...")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Create plots
        plots = {}
        plots['predictions'] = self.plot_predictions(y_true, y_pred)
        plots['residuals'] = self.plot_residuals(y_true, y_pred)
        
        if feature_importance is not None:
            plots['feature_importance'] = self.plot_feature_importance(feature_importance)
        
        # Error analysis by segments
        error_analysis = None
        if segment_data is not None:
            error_analysis = self.analyze_errors_by_segments(y_true, y_pred, segment_data)
            if error_analysis:
                plots['error_heatmap'] = self.create_error_heatmap(error_analysis)
        
        # Compile report
        report = {
            'metrics': metrics,
            'plots': plots,
            'error_analysis': error_analysis,
            'summary': self._create_summary(metrics, error_analysis)
        }
        
        logger.info("Evaluation report generated successfully")
        return report
    
    def _create_summary(self, metrics: Dict[str, float], 
                       error_analysis: Dict[str, pd.DataFrame] = None) -> str:
        """
        Create a text summary of the evaluation.
        
        Args:
            metrics: Model metrics
            error_analysis: Error analysis by segments
            
        Returns:
            Text summary
        """
        summary = f"""
        MODEL EVALUATION SUMMARY
        ========================
        
        Overall Performance:
        - Mean Absolute Error: {metrics['mae']:.2f} minutes
        - Root Mean Square Error: {metrics['rmse']:.2f} minutes
        - R-squared Score: {metrics['r2']:.3f}
        - Mean Absolute Percentage Error: {metrics['mape']:.2f}%
        
        Residual Analysis:
        - Mean Residual: {metrics['mean_residual']:.2f} minutes
        - Standard Deviation of Residuals: {metrics['std_residual']:.2f} minutes
        
        Model Quality Assessment:
        """
        
        # Model quality assessment
        if metrics['r2'] > 0.8:
            summary += "- Excellent model performance (R² > 0.8)\n"
        elif metrics['r2'] > 0.6:
            summary += "- Good model performance (R² > 0.6)\n"
        elif metrics['r2'] > 0.4:
            summary += "- Moderate model performance (R² > 0.4)\n"
        else:
            summary += "- Poor model performance (R² ≤ 0.4)\n"
        
        if abs(metrics['mean_residual']) < 1:
            summary += "- Model shows minimal bias (|mean residual| < 1 minute)\n"
        else:
            summary += f"- Model shows bias (mean residual = {metrics['mean_residual']:.2f} minutes)\n"
        
        # Segment analysis summary
        if error_analysis:
            summary += "\nSegment Analysis:\n"
            for segment_name, segment_data in error_analysis.items():
                mae_col = [col for col in segment_data.columns if 'mean_abs_error' in col]
                if mae_col:
                    best_segment = segment_data[mae_col[0]].idxmin()
                    worst_segment = segment_data[mae_col[0]].idxmax()
                    summary += f"- {segment_name}: Best = {best_segment}, Worst = {worst_segment}\n"
        
        return summary

class ModelSelector:
    """
    Model selection utilities.
    """
    
    def __init__(self):
        """Initialize the selector."""
        pass
    
    def select_best_model(self, evaluation_results: Dict[str, Dict[str, float]], 
                         criteria: str = 'mae') -> str:
        """
        Select the best model based on specified criteria.
        
        Args:
            evaluation_results: Dictionary of model evaluation results
            criteria: Criteria for selection ('mae', 'rmse', 'r2')
            
        Returns:
            Name of the best model
        """
        if criteria in ['mae', 'rmse']:
            # Lower is better
            best_model = min(evaluation_results.keys(), 
                           key=lambda x: evaluation_results[x][criteria])
        elif criteria == 'r2':
            # Higher is better
            best_model = max(evaluation_results.keys(), 
                           key=lambda x: evaluation_results[x][criteria])
        else:
            raise ValueError(f"Unknown criteria: {criteria}")
        
        return best_model
    
    def rank_models(self, evaluation_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Rank models based on multiple criteria.
        
        Args:
            evaluation_results: Dictionary of model evaluation results
            
        Returns:
            DataFrame with model rankings
        """
        # Convert to DataFrame
        df = pd.DataFrame(evaluation_results).T
        
        # Calculate ranks (1 = best)
        df['mae_rank'] = df['mae'].rank()
        df['rmse_rank'] = df['rmse'].rank()
        df['r2_rank'] = df['r2'].rank(ascending=False)
        
        # Calculate average rank
        df['avg_rank'] = (df['mae_rank'] + df['rmse_rank'] + df['r2_rank']) / 3
        
        # Sort by average rank
        df = df.sort_values('avg_rank')
        
        return df

def main():
    """Example usage of the evaluation utilities."""
    # Generate sample data for testing
    np.random.seed(42)
    y_true = np.random.normal(30, 10, 1000)
    y_pred = y_true + np.random.normal(0, 3, 1000)  # Add some noise
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    print("Metrics:", metrics)
    
    # Create plots
    fig1 = evaluator.plot_predictions(y_true, y_pred)
    fig2 = evaluator.plot_residuals(y_true, y_pred)
    
    plt.show()

if __name__ == "__main__":
    main()
