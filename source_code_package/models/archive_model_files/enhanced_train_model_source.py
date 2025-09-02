#!/usr/bin/env python3
"""
Enhanced AS_1 Linear Regression Training Module

This module provides comprehensive functionality for training AS_1 linear regression models
across multiple datasets (full dataset, cluster-specific datasets). It includes enhanced
logging, metrics tracking, model validation, and comparative analysis capabilities.

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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Configure warnings to reduce noise during training
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*n_init.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*normalize.*')

# Add source_code_package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


# ================================
# CONFIGURATION LOADING
# ================================

def load_as1_training_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load AS_1 training configuration from YAML file.
    
    This function reads the AS_1 configuration file and extracts training-specific
    parameters that control the model training process.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to the config YAML file. If None, uses default config_AS_1.yaml.
        
    Returns:
    --------
    dict
        Dictionary containing AS_1 training configuration parameters including:
        - data paths: processed data, output paths
        - features: independent and dependent variables
        - model parameters: fit_intercept, regularization settings
        - preprocessing: scaling options
        - train_test_split: split parameters
        
    Example:
    --------
    >>> config = load_as1_training_config()
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


def validate_as1_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate AS_1 training configuration parameters and provide warnings/recommendations.
    
    This function checks the validity of AS_1 training parameters and provides guidance
    on parameter selection based on best practices.
    
    Parameters:
    -----------
    config : dict
        AS_1 training configuration dictionary
        
    Returns:
    --------
    dict
        Dictionary containing validation results, warnings, and recommendations
        
    Example:
    --------
    >>> config = load_as1_training_config()
    >>> validation = validate_as1_training_config(config)
    >>> print(validation['warnings'])
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'recommendations': [],
        'errors': []
    }
    
    # Check required sections
    required_sections = ['data', 'features', 'train_test_split', 'output']
    for section in required_sections:
        if section not in config:
            validation_results['errors'].append(f"Missing required section: {section}")
            validation_results['valid'] = False
    
    # Check data paths
    if 'data' in config:
        if 'processed_data_path' not in config['data']:
            validation_results['errors'].append("Missing processed_data_path in data section")
            validation_results['valid'] = False
    
    # Check features
    if 'features' in config:
        if 'independent_variables' not in config['features']:
            validation_results['errors'].append("Missing independent_variables in features section")
            validation_results['valid'] = False
        elif not config['features']['independent_variables']:
            validation_results['warnings'].append("No independent variables specified")
        
        if 'dependent_variable' not in config['features']:
            validation_results['errors'].append("Missing dependent_variable in features section")
            validation_results['valid'] = False
    
    # Check train/test split
    if 'train_test_split' in config:
        test_size = config['train_test_split'].get('test_size', 0.2)
        if test_size <= 0 or test_size >= 1:
            validation_results['errors'].append(f"Invalid test_size: {test_size}. Must be between 0 and 1")
            validation_results['valid'] = False
        elif test_size < 0.1:
            validation_results['warnings'].append(f"Very small test size ({test_size}). Consider increasing for better validation")
        elif test_size > 0.5:
            validation_results['warnings'].append(f"Large test size ({test_size}). Consider reducing for more training data")
    
    # Add recommendations
    if not validation_results['errors']:
        validation_results['recommendations'].append("Configuration appears valid for AS_1 training")
        if config.get('preprocessing', {}).get('use_scaling', False):
            validation_results['recommendations'].append("Feature scaling is enabled - good for linear regression")
        else:
            validation_results['recommendations'].append("Consider enabling feature scaling for better model performance")
    
    return validation_results


# ================================
# LOGGING SETUP
# ================================

def setup_training_logger(config_path: str, config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging for AS_1 training process.
    
    This function configures logging for the training process, creating both file
    and console handlers for comprehensive logging output.
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file (used for logger naming)
    config : dict
        Configuration dictionary containing logging parameters
        
    Returns:
    --------
    logging.Logger
        Configured logger instance for training process
        
    Example:
    --------
    >>> config = load_as1_training_config()
    >>> logger = setup_training_logger(config_path, config)
    >>> logger.info("Training started")
    """
    log_path = config.get('output', {}).get('logs_path', 'data/logs/AS_1_training.log')
    
    # Make log path absolute if it's relative
    if not os.path.isabs(log_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        log_path = os.path.join(project_root, log_path)
    
    # Ensure log directory exists
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create unique logger name based on config
    dataset_name = Path(config_path).stem.replace('config_AS_1_', '')
    logger_name = f'AS1_trainer_{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
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
# DATA VALIDATION AND PREPARATION
# ================================

def validate_training_data_paths(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> bool:
    """
    Validate that required data paths exist for training.
    
    This function checks if the processed data file specified in the configuration
    exists and is accessible for training.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing data paths
    logger : logging.Logger, optional
        Logger instance for output messages
        
    Returns:
    --------
    bool
        True if all required data paths exist, False otherwise
        
    Example:
    --------
    >>> config = load_as1_training_config()
    >>> valid = validate_training_data_paths(config)
    >>> print(f"Data paths valid: {valid}")
    """
    data_path = config['data']['processed_data_path']
    
    if not os.path.isabs(data_path):
        # Try relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        data_path = os.path.join(project_root, data_path)
    
    if not os.path.exists(data_path):
        if logger:
            logger.error(f"Processed data file not found: {data_path}")
        else:
            print(f"ERROR: Processed data file not found: {data_path}")
        return False
        
    if logger:
        logger.info(f"Data validation passed: {data_path}")
    else:
        print(f"Data validation passed: {data_path}")
    return True


def load_and_prepare_training_data(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load and prepare training data from configuration.
    
    This function loads the processed data, extracts features and target variables,
    and applies the train/test split as specified in the configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing data and feature specifications
    logger : logging.Logger, optional
        Logger instance for output messages
        
    Returns:
    --------
    tuple
        - X_train : pd.DataFrame
            Training features
        - X_test : pd.DataFrame  
            Testing features
        - y_train : pd.Series
            Training target variable
        - y_test : pd.Series
            Testing target variable
            
    Example:
    --------
    >>> config = load_as1_training_config()
    >>> X_train, X_test, y_train, y_test = load_and_prepare_training_data(config)
    >>> print(f"Training samples: {len(X_train)}")
    """
    if logger:
        logger.info("Loading and preparing data...")
    else:
        print("Loading and preparing data...")
    
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
    
    if logger:
        logger.info(f"Loaded dataset with shape: {df.shape}")
    else:
        print(f"Loaded dataset with shape: {df.shape}")
    
    # Get features and target
    features = config['features']['independent_variables']
    target_col = config['features']['dependent_variable']
    
    # Validate features exist
    missing_features = set(features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")
    
    if target_col not in df.columns:
        raise ValueError(f"Target variable '{target_col}' not found in dataset")
    
    X = df[features]
    y = df[target_col]
    
    if logger:
        logger.info(f"Features selected: {len(features)} variables")
        logger.info(f"Target variable: {target_col}")
    else:
        print(f"Features selected: {len(features)} variables")
        print(f"Target variable: {target_col}")
    
    # Apply train/test split
    split_cfg = config.get('train_test_split', {})
    test_size = split_cfg.get('test_size', 0.2)
    random_state = split_cfg.get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    if logger:
        logger.info(f"Train/test split: {len(X_train)} train, {len(X_test)} test samples")
    else:
        print(f"Train/test split: {len(X_train)} train, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test


def apply_feature_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame, config_path: str, 
                         config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[StandardScaler]]:
    """
    Apply feature scaling if configured in the training setup.
    
    This function applies StandardScaler to the features if scaling is enabled
    in the configuration, ensuring consistent preprocessing.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Testing features  
    config_path : str
        Path to configuration file
    config : dict
        Configuration dictionary
    logger : logging.Logger, optional
        Logger instance for output messages
        
    Returns:
    --------
    tuple
        - X_train_scaled : pd.DataFrame
            Scaled training features
        - X_test_scaled : pd.DataFrame
            Scaled testing features
        - scaler : StandardScaler or None
            Fitted scaler object, None if scaling not applied
            
    Example:
    --------
    >>> X_train_scaled, X_test_scaled, scaler = apply_feature_scaling(X_train, X_test, config_path, config)
    >>> print(f"Scaling applied: {scaler is not None}")
    """
    use_scaling = config.get('preprocessing', {}).get('use_scaling', False)
    
    if not use_scaling:
        if logger:
            logger.info("Feature scaling disabled - using raw features")
        else:
            print("Feature scaling disabled - using raw features")
        return X_train, X_test, None
    
    if logger:
        logger.info("Applying feature scaling...")
    else:
        print("Applying feature scaling...")
    
    # Apply StandardScaler to our specific features only
    scaler = StandardScaler()
    
    # Fit on training data and transform both sets
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    if logger:
        logger.info("Feature scaling applied successfully")
        logger.info(f"Scaled {len(X_train.columns)} feature columns: {list(X_train.columns)}")
    else:
        print("Feature scaling applied successfully")
        print(f"Scaled {len(X_train.columns)} feature columns: {list(X_train.columns)}")
    
    return X_train_scaled, X_test_scaled, scaler


# ================================
# MODEL TRAINING FUNCTIONALITY  
# ================================

def train_as1_linear_regression(config_path: str, validate_config: bool = True, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Train AS_1 linear regression model with comprehensive functionality.
    
    This is the main training function that orchestrates the entire training process,
    including data loading, preprocessing, model training, evaluation, and saving.
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration YAML file
    validate_config : bool, default=True
        Whether to validate configuration parameters before training
    logger : logging.Logger, optional
        Logger instance for output messages
        
    Returns:
    --------
    dict
        Dictionary containing comprehensive training results and metrics including:
        - Performance metrics (R², RMSE, MAE for train and test sets)
        - Feature importance (coefficients)
        - Training metadata (sample counts, timing, etc.)
        - Model artifacts information
        
    Example:
    --------
    >>> config_path = "config_AS_1_full_dataset.yaml"
    >>> results = train_as1_linear_regression(config_path)
    >>> print(f"Test R²: {results['test_r2']:.4f}")
    """
    start_time = datetime.now()
    
    # Load configuration
    config = load_as1_training_config(config_path)
    
    # Setup logging
    if logger is None:
        logger = setup_training_logger(config_path, config)
    
    logger.info("Starting AS_1 model training...")
    
    try:
        # Validate configuration if requested
        if validate_config:
            validation = validate_as1_training_config(config)
            if not validation['valid']:
                raise ValueError(f"Invalid AS_1 training configuration: {validation['errors']}")
            
            # Print warnings and recommendations
            for warning in validation['warnings']:
                logger.warning(warning)
            for rec in validation['recommendations']:
                logger.info(f"RECOMMENDATION: {rec}")
        
        # Validate data paths
        if not validate_training_data_paths(config, logger):
            raise ValueError("Data validation failed")
        
        # Load and prepare data
        X_train, X_test, y_train, y_test = load_and_prepare_training_data(config, logger)
        
        # Apply scaling if configured
        X_train_final, X_test_final, scaler = apply_feature_scaling(X_train, X_test, config_path, config, logger)
        
        # Initialize and train model
        model_config = config.get('model', {})
        fit_intercept = model_config.get('fit_intercept', True)
        
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_train_final, y_train)
        
        logger.info("Model training completed")
        
        # Calculate training metrics
        y_train_pred = model.predict(X_train_final)
        y_test_pred = model.predict(X_test_final)
        
        training_metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': list(X_train.columns),
            'target_variable': config['features']['dependent_variable'],
            'scaling_applied': scaler is not None,
            'training_time_seconds': (datetime.now() - start_time).total_seconds()
        }
        
        # Add feature importance (coefficients)
        feature_importance = {}
        for i, feature in enumerate(X_train.columns):
            feature_importance[feature] = float(model.coef_[i])
        
        training_metrics['feature_coefficients'] = feature_importance
        training_metrics['intercept'] = float(model.intercept_)
        
        # Log key metrics
        logger.info(f"Training R²: {training_metrics['train_r2']:.4f}")
        logger.info(f"Test R²: {training_metrics['test_r2']:.4f}")
        logger.info(f"Training RMSE: {training_metrics['train_rmse']:.4f}")
        logger.info(f"Test RMSE: {training_metrics['test_rmse']:.4f}")
        
        # Save model and metrics
        save_trained_model(model, scaler, config_path, config, logger)
        save_training_metrics(training_metrics, config_path, config, logger)
        
        return training_metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


# ================================
# MODEL AND METRICS SAVING
# ================================

def save_trained_model(model: LinearRegression, scaler: Optional[StandardScaler], 
                      config_path: str, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
    """
    Save the trained model and scaler to disk.
    
    This function saves the trained linear regression model along with the scaler
    and metadata for later use in testing and deployment.
    
    Parameters:
    -----------
    model : LinearRegression
        Trained linear regression model
    scaler : StandardScaler or None
        Fitted scaler object, None if scaling not applied
    config_path : str
        Path to configuration file
    config : dict
        Configuration dictionary
    logger : logging.Logger, optional
        Logger instance for output messages
        
    Example:
    --------
    >>> save_trained_model(model, scaler, config_path, config)
    """
    model_path = config.get('output', {}).get('model_path', 'linear_regression_model.pkl')
    
    if not os.path.isabs(model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        model_path = os.path.join(project_root, model_path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'config_path': config_path,
        'training_timestamp': datetime.now().isoformat(),
        'features': config['features']['independent_variables'],
        'target': config['features']['dependent_variable']
    }
    
    joblib.dump(model_data, model_path)
    
    if logger:
        logger.info(f"Model saved to: {model_path}")
    else:
        print(f"Model saved to: {model_path}")


def save_training_metrics(training_metrics: Dict[str, Any], config_path: str, 
                         config: Dict[str, Any], logger: Optional[logging.Logger] = None):
    """
    Save training metrics to JSON file.
    
    This function saves comprehensive training metrics and metadata to a JSON file
    for later analysis and comparison.
    
    Parameters:
    -----------
    training_metrics : dict
        Dictionary containing training metrics and results
    config_path : str
        Path to configuration file
    config : dict
        Configuration dictionary
    logger : logging.Logger, optional
        Logger instance for output messages
        
    Example:
    --------
    >>> save_training_metrics(metrics, config_path, config)
    """
    metrics_path = config.get('output', {}).get('metrics_path', 'data/scores/AS_1_metrics.json')
    
    if not os.path.isabs(metrics_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        metrics_path = os.path.join(project_root, metrics_path)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    # Add metadata
    metrics_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'config_path': config_path,
        'dataset_info': {
            'raw_data_path': config['data']['raw_data_path'],
            'processed_data_path': config['data']['processed_data_path']
        },
        'metrics': training_metrics
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_with_metadata, f, indent=2)
    
    if logger:
        logger.info(f"Metrics saved to: {metrics_path}")
    else:
        print(f"Metrics saved to: {metrics_path}")


# ================================
# BATCH PROCESSING FUNCTIONS
# ================================
def train_model_from_config(config_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Train AS_1 model from configuration file.
    
    This function serves as the main entry point for training AS_1 models from a
    configuration file, providing a simple interface for the enhanced training functionality.
    
    Args:
        config_path: Path to configuration YAML file
        verbose: Whether to enable verbose logging
        
    Returns:
        Dictionary containing training results and metrics
        
    Example:
    --------
    >>> result = train_model_from_config("config_AS_1_full_dataset.yaml")
    >>> print(f"Success: {result['success']}")
    """
    try:
        if verbose:
            print(f"Training AS_1 model with config: {config_path}")
        
        # Train the model using the main training function
        metrics = train_as1_linear_regression(config_path, validate_config=True)
        
        if verbose:
            print(f"✅ Training completed successfully")
            print(f"   Test R²: {metrics['test_r2']:.4f}")
            print(f"   Test RMSE: {metrics['test_rmse']:.4f}")
        
        # Load config to get model path
        config = load_as1_training_config(config_path)
        model_path = config.get('output', {}).get('model_path')
        
        return {
            'success': True,
            'config_path': config_path,
            'metrics': metrics,
            'model_path': model_path,
            'error': None
        }
        
    except Exception as e:
        error_msg = f"Training failed for {config_path}: {str(e)}"
        if verbose:
            print(f"❌ {error_msg}")
        
        return {
            'success': False,
            'config_path': config_path,
            'metrics': None,
            'model_path': None,
            'error': error_msg
        }


def train_multiple_models_batch(config_paths: List[str], verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple AS_1 models in batch.
    
    This function processes multiple configuration files to train AS_1 models
    for different datasets in a batch operation.
    
    Args:
        config_paths: List of configuration file paths
        verbose: Whether to enable verbose logging
        
    Returns:
        Dictionary mapping config paths to training results
        
    Example:
    --------
    >>> config_paths = ["config_AS_1_full_dataset.yaml", "config_AS_1_cluster_0.yaml"]
    >>> results = train_multiple_models_batch(config_paths)
    >>> print(f"Successful: {sum(1 for r in results.values() if r['success'])}")
    """
    results = {}
    
    if verbose:
        print(f"AS_1 BATCH MODEL TRAINING")
        print("=" * 50)
        print(f"Training {len(config_paths)} models...")
        print()
    
    for i, config_path in enumerate(config_paths, 1):
        if verbose:
            dataset_name = Path(config_path).stem.replace('config_AS_1_', '')
            print(f"[{i}/{len(config_paths)}] Training model for: {dataset_name}")
        
        result = train_model_from_config(config_path, verbose=False)
        results[config_path] = result
        
        if verbose:
            if result['success']:
                print(f"   ✅ Success - R²: {result['metrics']['test_r2']:.4f}")
            else:
                print(f"   ❌ Failed - {result['error']}")
            print()
    
    # Summary
    successful = sum(1 for r in results.values() if r['success'])
    if verbose:
        print(f"BATCH TRAINING SUMMARY")
        print(f"Successful: {successful}/{len(config_paths)}")
        
        if successful > 0:
            print("\nModel Performance Summary:")
            for config_path, result in results.items():
                if result['success']:
                    dataset_name = Path(config_path).stem.replace('config_AS_1_', '')
                    metrics = result['metrics']
                    print(f"  {dataset_name}: R²={metrics['test_r2']:.4f}, RMSE={metrics['test_rmse']:.4f}")
    
    return results


# ================================
# EXAMPLE USAGE AND TESTING
# ================================

if __name__ == "__main__":
    # Example single model training
    config_path = "/Users/tomdavey/Documents/GitHub/MLProject1/source_code_package/config/config_AS_1_full_dataset.yaml"
    
    print("Testing AS_1 Enhanced Training Module")
    print("=" * 40)
    
    try:
        # Test configuration loading
        print("1. Testing configuration loading...")
        config = load_as1_training_config(config_path)
        print(f"✅ Config loaded successfully")
        
        # Test configuration validation
        print("2. Testing configuration validation...")
        validation = validate_as1_training_config(config)
        print(f"✅ Config validation: {'PASSED' if validation['valid'] else 'FAILED'}")
        
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        # Test single model training
        print("3. Testing single model training...")
        result = train_model_from_config(config_path, verbose=True)
        print(f"Training result: {result['success']}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    print("\nTesting completed.")
