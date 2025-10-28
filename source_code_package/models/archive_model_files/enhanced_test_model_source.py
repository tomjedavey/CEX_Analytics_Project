#!/usr/bin/env python3
"""
Enhanced AS_1 Linear Regression Testing Module

This module provides comprehensive functionality for testing AS_1 linear regression models
across multiple datasets. It includes detailed evaluation metrics, prediction analysis,
residual analysis, and comparative testing capabilities.

Author: Tom Davey
Date: July 2025
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    mean_absolute_percentage_error, explained_variance_score
)

# Configure warnings to reduce noise during testing
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*n_init.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*normalize.*')

# Add source_code_package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


# ================================
# CONFIGURATION LOADING
# ================================

def load_as1_testing_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load AS_1 testing configuration from YAML file.
    
    This function reads the AS_1 configuration file and extracts testing-specific
    parameters that control the model testing process.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to the config YAML file. If None, uses default config_AS_1.yaml.
        
    Returns:
    --------
    dict
        Dictionary containing AS_1 testing configuration parameters
        
    Example:
    --------
    >>> config = load_as1_testing_config()
    >>> features = config['features']['independent_variables']
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../../config/config_AS_1.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ValueError(f"Error loading config from {config_path}: {e}")


# ================================
# LOGGING SETUP
# ================================

def setup_testing_logger(config_path: str, config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging for AS_1 testing process.
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file
    config : dict
        Configuration dictionary containing logging parameters
        
    Returns:
    --------
    logging.Logger
        Configured logger instance for testing process
    """
    log_path = config.get('output', {}).get('logs_path', 'data/logs/AS_1_testing.log')
    
    # Modify log path for testing
    log_path = log_path.replace('training', 'testing')
    
    # Make log path absolute if it's relative
    if not os.path.isabs(log_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        log_path = os.path.join(project_root, log_path)
    
    # Ensure log directory exists
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create unique logger name
    dataset_name = Path(config_path).stem.replace('config_AS_1_', '')
    logger_name = f'AS1_tester_{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # Setup logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ================================
# MODEL LOADING AND VALIDATION
# ================================

def load_trained_as1_model(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Tuple[object, Optional[object]]:
    """
    Load the trained AS_1 model and scaler from disk.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing model path
    logger : logging.Logger, optional
        Logger instance for output messages
        
    Returns:
    --------
    tuple
        - model : sklearn model object
            Trained linear regression model
        - scaler : StandardScaler or None
            Fitted scaler object, None if scaling not applied
    """
    model_path = config.get('output', {}).get('model_path', 'linear_regression_model.pkl')
    
    if not os.path.isabs(model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        model_path = os.path.join(project_root, model_path)
    
    if not os.path.exists(model_path):
        error_msg = f"Model file not found: {model_path}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"ERROR: {error_msg}")
        raise FileNotFoundError(error_msg)
    
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data.get('scaler', None)
        
        if logger:
            logger.info(f"Model loaded successfully from: {model_path}")
        else:
            print(f"Model loaded successfully from: {model_path}")
        
        return model, scaler
        
    except Exception as e:
        error_msg = f"Error loading model: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(f"ERROR: {error_msg}")
        raise


def load_test_data_for_evaluation(config: Dict[str, Any], scaler: Optional[object], 
                                logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Index]:
    """
    Load and prepare test data for model evaluation.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    scaler : StandardScaler or None
        Fitted scaler object from training
    logger : logging.Logger, optional
        Logger instance for output messages
        
    Returns:
    --------
    tuple
        - X_test_raw : pd.DataFrame
            Raw test features
        - y_test : pd.Series
            Test target values
        - X_test_processed : np.ndarray
            Processed test features (scaled if applicable)
        - test_indices : pd.Index
            Test sample indices
    """
    if logger:
        logger.info("Loading test data...")
    else:
        print("Loading test data...")
    
    # Load data
    data_path = config['data']['processed_data_path']
    if not os.path.isabs(data_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        data_path = os.path.join(project_root, data_path)
    
    df = pd.read_csv(data_path)
    
    # Drop ID columns (typically WALLET or similar identifier columns)
    id_columns = ['WALLET', 'wallet', 'ID', 'id', 'index']
    for col in id_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            if logger:
                logger.info(f"Dropped ID column: {col}")
    
    # Get features and target
    features = config['features']['independent_variables']
    target_col = config['features']['dependent_variable']
    
    X = df[features]
    y = df[target_col]
    
    # Reproduce the exact same train/test split used in training
    split_cfg = config.get('train_test_split', {})
    test_size = split_cfg.get('test_size', 0.2)
    random_state = split_cfg.get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Apply scaling if it was used during training
    if scaler is not None:
        if logger:
            logger.info("Applying scaling to test data...")
        else:
            print("Applying scaling to test data...")
        X_test_processed = scaler.transform(X_test)
    else:
        X_test_processed = X_test.values
    
    test_indices = X_test.index
    
    return X_test, y_test, X_test_processed, test_indices


# ================================
# EVALUATION METRICS CALCULATION
# ================================

def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for model performance.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
        
    Returns:
    --------
    dict
        Dictionary containing comprehensive evaluation metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred),
        'mean_residual': np.mean(y_true - y_pred),
        'std_residual': np.std(y_true - y_pred),
        'max_residual': np.max(np.abs(y_true - y_pred)),
        'q95_residual': np.percentile(np.abs(y_true - y_pred), 95),
        'q75_residual': np.percentile(np.abs(y_true - y_pred), 75),
        'median_residual': np.median(np.abs(y_true - y_pred))
    }
    
    # Calculate MAPE only if no zero values in y_true
    if np.all(y_true != 0):
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
    else:
        metrics['mape'] = None
        
    # Add relative metrics
    if np.std(y_true) > 0:
        metrics['normalized_rmse'] = metrics['rmse'] / np.std(y_true)
        metrics['cv_rmse'] = metrics['rmse'] / np.mean(y_true) if np.mean(y_true) != 0 else None
    else:
        metrics['normalized_rmse'] = None
        metrics['cv_rmse'] = None
    
    return metrics


def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Perform detailed residual analysis.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
        
    Returns:
    --------
    dict
        Dictionary containing residual analysis results
    """
    residuals = y_true - y_pred
    
    def calculate_skewness(data: np.ndarray) -> float:
        """Calculate skewness of data."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.sum(((data - mean) / std) ** 3) / n
    
    def calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.sum(((data - mean) / std) ** 4) / n - 3
    
    analysis = {
        'residual_stats': {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'q25': float(np.percentile(residuals, 25)),
            'median': float(np.median(residuals)),
            'q75': float(np.percentile(residuals, 75)),
            'skewness': float(calculate_skewness(residuals)),
            'kurtosis': float(calculate_kurtosis(residuals))
        },
        'outlier_analysis': {
            'outlier_threshold_iqr': None,
            'n_outliers_iqr': 0,
            'outlier_threshold_std': None,
            'n_outliers_std': 0
        }
    }
    
    # IQR-based outlier detection
    q25, q75 = np.percentile(residuals, [25, 75])
    iqr = q75 - q25
    outlier_threshold_iqr = 1.5 * iqr
    outliers_iqr = np.abs(residuals - np.median(residuals)) > outlier_threshold_iqr
    
    analysis['outlier_analysis']['outlier_threshold_iqr'] = float(outlier_threshold_iqr)
    analysis['outlier_analysis']['n_outliers_iqr'] = int(np.sum(outliers_iqr))
    
    # Standard deviation-based outlier detection
    outlier_threshold_std = 2 * np.std(residuals)
    outliers_std = np.abs(residuals - np.mean(residuals)) > outlier_threshold_std
    
    analysis['outlier_analysis']['outlier_threshold_std'] = float(outlier_threshold_std)
    analysis['outlier_analysis']['n_outliers_std'] = int(np.sum(outliers_std))
    
    return analysis


def analyze_feature_impact(X_test: pd.DataFrame, model: object) -> Dict[str, Any]:
    """
    Analyze feature impact on predictions.
    
    Parameters:
    -----------
    X_test : pd.DataFrame
        Test features
    model : sklearn model
        Trained model with coefficients
        
    Returns:
    --------
    dict
        Dictionary containing feature impact analysis
    """
    feature_analysis = {}
    
    # Get feature coefficients
    if hasattr(model, 'coef_'):
        coefficients = dict(zip(X_test.columns, model.coef_))
        feature_analysis['coefficients'] = coefficients
        
        # Rank features by absolute coefficient value
        feature_importance = sorted(
            coefficients.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        feature_analysis['feature_importance_ranking'] = feature_importance
    
    # Calculate feature statistics for test set
    feature_stats = {}
    for feature in X_test.columns:
        feature_stats[feature] = {
            'mean': float(X_test[feature].mean()),
            'std': float(X_test[feature].std()),
            'min': float(X_test[feature].min()),
            'max': float(X_test[feature].max()),
            'range': float(X_test[feature].max() - X_test[feature].min())
        }
    
    feature_analysis['feature_statistics'] = feature_stats
    
    return feature_analysis


# ================================
# MAIN TESTING FUNCTION
# ================================

def test_as1_linear_regression(config_path: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Test AS_1 linear regression model comprehensively.
    
    This is the main testing function that orchestrates the entire testing process,
    including model loading, data preparation, evaluation, and analysis.
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration YAML file
    logger : logging.Logger, optional
        Logger instance for output messages
        
    Returns:
    --------
    dict
        Dictionary containing comprehensive test results and analysis
    """
    start_time = datetime.now()
    
    # Load configuration
    config = load_as1_testing_config(config_path)
    
    # Setup logging if not provided
    if logger is None:
        logger = setup_testing_logger(config_path, config)
    
    logger.info("Starting comprehensive model testing...")
    
    try:
        # Load trained model
        model, scaler = load_trained_as1_model(config, logger)
        
        # Load test data
        X_test_raw, y_test, X_test_processed, test_indices = load_test_data_for_evaluation(config, scaler, logger)
        
        logger.info(f"Test set size: {len(y_test)} samples")
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        logger.info("Predictions completed, calculating metrics...")
        
        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(y_test.values, y_pred)
        
        # Perform residual analysis
        residual_analysis = analyze_residuals(y_test.values, y_pred)
        
        # Analyze feature impact
        feature_analysis = analyze_feature_impact(X_test_raw, model)
        
        # Compile test results
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'config_path': config_path,
            'test_samples': len(y_test),
            'features_used': list(X_test_raw.columns),
            'target_variable': config['features']['dependent_variable'],
            'scaling_applied': scaler is not None,
            'testing_time_seconds': (datetime.now() - start_time).total_seconds(),
            'performance_metrics': metrics,
            'residual_analysis': residual_analysis,
            'feature_analysis': feature_analysis,
            'prediction_summary': {
                'y_true_mean': float(np.mean(y_test.values)),
                'y_true_std': float(np.std(y_test.values)),
                'y_pred_mean': float(np.mean(y_pred)),
                'y_pred_std': float(np.std(y_pred)),
                'prediction_range': {
                    'y_true_min': float(np.min(y_test.values)),
                    'y_true_max': float(np.max(y_test.values)),
                    'y_pred_min': float(np.min(y_pred)),
                    'y_pred_max': float(np.max(y_pred))
                }
            }
        }
        
        # Log key results
        logger.info(f"Test R²: {metrics['r2']:.4f}")
        logger.info(f"Test RMSE: {metrics['rmse']:.4f}")
        logger.info(f"Test MAE: {metrics['mae']:.4f}")
        
        # Save detailed results
        save_test_results(y_test, y_pred, test_indices, config, logger)
        save_test_metrics(test_results, config, logger)
        
        return test_results
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise


# ================================
# RESULTS SAVING FUNCTIONS
# ================================

def save_test_results(y_test: pd.Series, y_pred: np.ndarray, test_indices: pd.Index, 
                     config: Dict[str, Any], logger: Optional[logging.Logger] = None):
    """Save detailed test results to CSV."""
    results_path = config.get('output', {}).get('test_results_path', 'data/scores/AS_1_test_results.csv')
    
    if not os.path.isabs(results_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        results_path = os.path.join(project_root, results_path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'index': test_indices,
        'y_true': y_test.values,
        'y_pred': y_pred,
        'residual': y_test.values - y_pred,
        'abs_residual': np.abs(y_test.values - y_pred),
        'percent_error': ((y_test.values - y_pred) / y_test.values * 100) if np.all(y_test.values != 0) else np.nan
    })
    
    results_df.to_csv(results_path, index=False)
    
    if logger:
        logger.info(f"Test results saved to: {results_path}")
    else:
        print(f"Test results saved to: {results_path}")


def save_test_metrics(test_results: Dict[str, Any], config: Dict[str, Any], logger: Optional[logging.Logger] = None):
    """
    Save test metrics to a JSON file with automatic path handling.
    
    This function saves model testing results to a JSON file, automatically
    handling path resolution and directory creation. The output path is derived
    from the configuration with automatic modification for test-specific naming.
    
    Parameters:
    -----------
    test_results : Dict[str, Any]
        Dictionary containing test results and metrics to save
    config : Dict[str, Any]
        Configuration dictionary containing output path settings
    logger : logging.Logger, optional
        Logger instance for output messages. If None, uses print statements
    
    Returns:
    --------
    None
        Saves results to file and logs/prints confirmation
    
    Notes:
    ------
    - Automatically converts 'metrics' to 'test_metrics' in filename
    - Creates output directory if it doesn't exist
    - Handles both absolute and relative paths
    - Uses project root detection for relative paths
    
    Example:
    --------
    >>> test_results = {'accuracy': 0.85, 'precision': 0.80}
    >>> config = {'output': {'metrics_path': 'data/scores/AS_1_metrics.json'}}
    >>> save_test_metrics(test_results, config)
    Test metrics saved to: /path/to/project/data/scores/AS_1_test_metrics.json
    """
    metrics_path = config.get('output', {}).get('metrics_path', 'data/scores/AS_1_metrics.json')
    
    # Modify path for test metrics
    metrics_path = metrics_path.replace('metrics', 'test_metrics')
    
    if not os.path.isabs(metrics_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        metrics_path = os.path.join(project_root, metrics_path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    with open(metrics_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    if logger:
        logger.info(f"Test metrics saved to: {metrics_path}")
    else:
        print(f"Test metrics saved to: {metrics_path}")


# ================================
# BATCH PROCESSING FUNCTIONS
# ================================

def test_model_from_config(config_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Test AS_1 model from configuration file.
    
    Args:
        config_path: Path to configuration YAML file
        verbose: Whether to enable verbose logging
        
    Returns:
        Dictionary containing test results and metrics
    """
    try:
        if verbose:
            print(f"Testing AS_1 model with config: {config_path}")
        
        results = test_as1_linear_regression(config_path)
        
        if verbose:
            print(f"✅ Testing completed successfully")
            print(f"   Test R²: {results['performance_metrics']['r2']:.4f}")
            print(f"   Test RMSE: {results['performance_metrics']['rmse']:.4f}")
            print(f"   Test MAE: {results['performance_metrics']['mae']:.4f}")
        
        return {
            'success': True,
            'config_path': config_path,
            'test_results': results,
            'error': None
        }
        
    except Exception as e:
        error_msg = f"Testing failed for {config_path}: {str(e)}"
        if verbose:
            print(f"❌ {error_msg}")
        
        return {
            'success': False,
            'config_path': config_path,
            'test_results': None,
            'error': error_msg
        }


def test_multiple_models_batch(config_paths: List[str], verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Test multiple AS_1 models in batch.
    
    Args:
        config_paths: List of configuration file paths
        verbose: Whether to enable verbose logging
        
    Returns:
        Dictionary mapping config paths to test results
    """
    results = {}
    
    if verbose:
        print(f"AS_1 BATCH MODEL TESTING")
        print("=" * 50)
        print(f"Testing {len(config_paths)} models...")
        print()
    
    for i, config_path in enumerate(config_paths, 1):
        if verbose:
            dataset_name = Path(config_path).stem.replace('config_AS_1_', '')
            print(f"[{i}/{len(config_paths)}] Testing model for: {dataset_name}")
        
        result = test_model_from_config(config_path, verbose=False)
        results[config_path] = result
        
        if verbose:
            if result['success']:
                metrics = result['test_results']['performance_metrics']
                print(f"   ✅ Success - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
            else:
                print(f"   ❌ Failed - {result['error']}")
            print()
    
    # Summary
    successful = sum(1 for r in results.values() if r['success'])
    if verbose:
        print(f"BATCH TESTING SUMMARY")
        print(f"Successful: {successful}/{len(config_paths)}")
        
        if successful > 0:
            print("\nModel Performance Summary:")
            for config_path, result in results.items():
                if result['success']:
                    dataset_name = Path(config_path).stem.replace('config_AS_1_', '')
                    metrics = result['test_results']['performance_metrics']
                    print(f"  {dataset_name}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
    
    return results


# ================================
# EXAMPLE USAGE AND TESTING
# ================================

if __name__ == "__main__":
    # Example single model testing
    config_path = "/Users/tomdavey/Documents/GitHub/CEX_Analytics_Project/source_code_package/config/config_AS_1_full_dataset.yaml"
    
    print("Testing AS_1 Enhanced Testing Module")
    print("=" * 40)
    
    try:
        # Test configuration loading
        print("1. Testing configuration loading...")
        config = load_as1_testing_config(config_path)
        print(f"✅ Config loaded successfully")
        
        # Test single model testing
        print("2. Testing single model testing...")
        result = test_model_from_config(config_path, verbose=True)
        print(f"Testing result: {result['success']}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    print("\nTesting completed.")
