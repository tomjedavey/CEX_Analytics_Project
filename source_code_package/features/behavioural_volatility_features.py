#!/usr/bin/env python3
"""
Behavioural Volatility Score Feature Engineering Module

This module provides functionality to calculate behavioural volatility scores for cryptocurrency 
wallets based on three main components:

1. Financial Volatility (35%): USD_TRANSFER_STDDEV / AVG_TRANSFER_USD
2. Activity Volatility (40%): Composite of CV, variance ratio, and Gini coefficient of activity patterns
3. Exploration Volatility (25%): Exploration intensity based on diversity metrics relative to activity rate

The behavioural volatility score measures the inconsistency and unpredictability of wallet behaviour
across financial, activity, and exploration dimensions.

Author: Tom Davey
Date: August 2025
"""

import pandas as pd
import numpy as np
import os
import yaml
from typing import Optional, Dict, Tuple, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.stats import zscore


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to configuration file. If None, uses default config.
        
    Returns:
    --------
    Dict
        Configuration dictionary
    """
    if config_path is None:
        # Default path relative to this file
        current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, '..', 'config', 'config_behavioural_volatility.yaml')
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def improved_component_normalization(
    values: pd.Series, 
    method: str = 'robust_scaler',
    percentile_cap: Optional[float] = None
) -> pd.Series:
    """
    Apply improved normalization to behavioural volatility components.
    
    Parameters:
    -----------
    values : pd.Series
        Raw component values to normalize
    method : str
        Normalization method ('robust_scaler', 'zscore', 'log_zscore')
    percentile_cap : float, optional
        Percentile value (0-100) to cap outliers at before normalization
        
    Returns:
    --------
    pd.Series
        Normalized values in range [0, 1]
    """
    # Remove any infinite or missing values
    clean_values = values.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(clean_values) == 0:
        return pd.Series(index=values.index, data=0.0)
    
    # Apply percentile capping if specified
    if percentile_cap is not None:
        cap_value = np.percentile(clean_values, percentile_cap)
        clean_values = clean_values.clip(upper=cap_value)
    
    # Apply the specified normalization method
    if method == 'robust_scaler':
        # Use RobustScaler (median and IQR-based scaling)
        scaler = RobustScaler()
        normalized = scaler.fit_transform(clean_values.values.reshape(-1, 1)).flatten()
        # Clip to ensure positive values and apply sqrt transformation
        normalized = np.clip(normalized, 0, None)
        normalized = np.sqrt(normalized) / np.sqrt(normalized.max()) if normalized.max() > 0 else normalized
        
    elif method == 'zscore':
        # Z-score normalization with capping at ±3
        normalized = zscore(clean_values)
        normalized = np.clip(normalized, -3, 3)
        # Shift to [0, 6] and normalize to [0, 1]
        normalized = (normalized + 3) / 6
        
    elif method == 'log_zscore':
        # Log transformation followed by z-score
        log_values = np.log1p(clean_values)  # log(1 + x) to handle zeros
        normalized = zscore(log_values)
        normalized = np.clip(normalized, -3, 3)
        # Shift to [0, 6] and normalize to [0, 1]
        normalized = (normalized + 3) / 6
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Create result series with original index
    result = pd.Series(index=values.index, data=0.0)
    result.loc[clean_values.index] = normalized
    
    return result


def calculate_financial_volatility(df: pd.DataFrame, 
                                 numerator_col: str = "USD_TRANSFER_STDDEV",
                                 denominator_col: str = "AVG_TRANSFER_USD",
                                 epsilon: float = 1e-8) -> pd.Series:
    """
    Calculate financial volatility score.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    numerator_col : str
        Column containing transfer standard deviation
    denominator_col : str
        Column containing average transfer amount
    epsilon : float
        Small value to avoid division by zero
        
    Returns:
    --------
    pd.Series
        Financial volatility scores
    """
    # Ensure epsilon is a float
    epsilon = float(epsilon)
    
    # Convert to numeric and handle missing values
    numerator = pd.to_numeric(df[numerator_col], errors='coerce').fillna(0)
    denominator = pd.to_numeric(df[denominator_col], errors='coerce').fillna(0)
    
    # Avoid division by zero
    financial_volatility = np.where(
        denominator.values > epsilon,
        numerator.values / denominator.values,
        0.0
    )
    
    return pd.Series(financial_volatility, index=df.index)


def calculate_coefficient_of_variance(activity_counts: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calculate coefficient of variance for activity counts.
    
    Parameters:
    -----------
    activity_counts : np.ndarray
        Array of activity counts
    epsilon : float
        Small value to avoid division by zero
        
    Returns:
    --------
    float
        Coefficient of variance
    """
    activity_counts = np.array(activity_counts, dtype=np.float64)
    mean_activity = np.mean(activity_counts)
    std_activity = np.std(activity_counts)
    
    if mean_activity > epsilon:
        return std_activity / mean_activity
    else:
        return 0.0


def calculate_variance_ratio_from_uniform(activity_counts: np.ndarray) -> float:
    """
    Calculate variance ratio from uniform distribution.
    
    Parameters:
    -----------
    activity_counts : np.ndarray
        Array of activity counts
        
    Returns:
    --------
    float
        Variance ratio from uniform distribution
    """
    activity_counts = np.array(activity_counts, dtype=np.float64)
    n_categories = len(activity_counts)
    
    if n_categories <= 1:
        return 0.0
    
    # Actual variance
    actual_variance = np.var(activity_counts)
    
    # Expected variance for uniform distribution
    total_events = np.sum(activity_counts)
    if total_events == 0:
        return 0.0
    
    expected_mean = total_events / n_categories
    expected_variance = ((n_categories - 1) * expected_mean**2) / n_categories
    
    if expected_variance > 0:
        return actual_variance / expected_variance
    else:
        return 0.0


def calculate_gini_coefficient(activity_counts: np.ndarray) -> float:
    """
    Calculate Gini coefficient for activity distribution.
    
    Parameters:
    -----------
    activity_counts : np.ndarray
        Array of activity counts
        
    Returns:
    --------
    float
        Gini coefficient
    """
    activity_counts = np.array(activity_counts, dtype=np.float64)
    
    # Remove negative values and sort
    activity_counts = activity_counts[activity_counts >= 0]
    if len(activity_counts) == 0:
        return 0.0
    
    activity_counts = np.sort(activity_counts)
    n = len(activity_counts)
    
    if n == 1 or np.sum(activity_counts) == 0:
        return 0.0
    
    # Calculate Gini coefficient
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * activity_counts)) / (n * np.sum(activity_counts)) - (n + 1) / n
    
    return max(0.0, gini)  # Ensure non-negative


def calculate_activity_volatility(df: pd.DataFrame, 
                                event_columns: List[str],
                                weights: Dict[str, float]) -> pd.Series:
    """
    Calculate activity volatility score as composite of CV, variance ratio, and Gini coefficient.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    event_columns : List[str]
        List of event column names
    weights : Dict[str, float]
        Weights for sub-components
        
    Returns:
    --------
    pd.Series
        Activity volatility scores
    """
    activity_volatility_scores = []
    
    for idx, row in df.iterrows():
        # Get activity counts for this wallet
        activity_counts = [row[col] for col in event_columns]
        activity_counts = np.array(activity_counts, dtype=np.float64)
        
        # Calculate sub-components
        cv = calculate_coefficient_of_variance(activity_counts)
        variance_ratio = calculate_variance_ratio_from_uniform(activity_counts)
        gini = calculate_gini_coefficient(activity_counts)
        
        # Composite activity volatility score
        activity_volatility = (
            weights['coefficient_of_variance'] * cv +
            weights['variance_ratio'] * variance_ratio +
            weights['gini_coefficient'] * gini
        )
        
        activity_volatility_scores.append(activity_volatility)
    
    return pd.Series(activity_volatility_scores, index=df.index)


def calculate_exploration_volatility(df: pd.DataFrame,
                                   diversity_columns: List[str],
                                   activity_rate_column: str = "TX_PER_MONTH",
                                   apply_sqrt_transform: bool = True,
                                   epsilon: float = 1e-8) -> pd.Series:
    """
    Calculate exploration volatility score based on diversity metrics and activity rate.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    diversity_columns : List[str]
        List of diversity column names
    activity_rate_column : str
        Column containing activity rate (transactions per month)
    apply_sqrt_transform : bool
        Whether to apply square root transformation
    epsilon : float
        Small value to avoid division by zero
        
    Returns:
    --------
    pd.Series
        Exploration volatility scores
    """
    # Ensure epsilon is a float
    epsilon = float(epsilon)
    
    # Calculate average diversity - convert to numeric and handle missing values
    diversity_data = df[diversity_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
    average_diversity = diversity_data.mean(axis=1)
    
    # Get activity rate - convert to numeric and handle missing values
    activity_rate = pd.to_numeric(df[activity_rate_column], errors='coerce').fillna(0)
    
    # Calculate exploration intensity
    exploration_intensity = np.where(
        activity_rate.values > epsilon,
        average_diversity.values / activity_rate.values,
        0.0
    )
    
    # Apply square root transformation if specified
    if apply_sqrt_transform:
        exploration_intensity = np.sqrt(np.maximum(0, exploration_intensity))
    
    return pd.Series(exploration_intensity, index=df.index)


def calculate_behavioural_volatility_score(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Calculate the complete behavioural volatility score.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    config : Dict
        Configuration dictionary
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with behavioural volatility components and final score
    """
    df_result = df.copy()
    
    # Extract configuration
    component_weights = config['features']['component_weights']
    activity_weights = config['features']['activity_volatility_weights']
    event_columns = config['activity_events']['event_columns']
    financial_config = config['financial_volatility']
    exploration_config = config['exploration_volatility']
    epsilon = config['features']['epsilon']
    
    print("Calculating Behavioural Volatility Score components...")
    
    # 1. Financial Volatility (35%)
    print("  1. Calculating Financial Volatility component...")
    financial_volatility_raw = calculate_financial_volatility(
        df_result,
        financial_config['numerator_column'],
        financial_config['denominator_column'],
        epsilon
    )
    
    # 2. Activity Volatility (40%)
    print("  2. Calculating Activity Volatility component...")
    activity_volatility_raw = calculate_activity_volatility(
        df_result,
        event_columns,
        activity_weights
    )
    
    # 3. Exploration Volatility (25%)
    print("  3. Calculating Exploration Volatility component...")
    exploration_volatility_raw = calculate_exploration_volatility(
        df_result,
        exploration_config['diversity_columns'],
        exploration_config['activity_rate_column'],
        exploration_config['apply_sqrt_transform'],
        epsilon
    )
    
    # Apply improved normalization if configured
    normalization_method = config['features'].get('normalization_method', 'none')
    use_improved_normalization = (normalization_method == 'improved' and 
                                 'improved_normalization' in config['features'])
    
    if use_improved_normalization:
        print("  Applying improved component normalization...")
        improved_config = config['features']['improved_normalization']
        
        # Apply component-specific normalization
        financial_config = improved_config.get('financial_volatility', {})
        activity_config = improved_config.get('activity_volatility', {})
        exploration_config = improved_config.get('exploration_volatility', {})
        
        # Apply improved normalization to each component with component-specific settings
        df_result['FINANCIAL_VOLATILITY'] = improved_component_normalization(
            financial_volatility_raw, 
            financial_config.get('method', 'robust_scaler'),
            financial_config.get('percentile_cap', None) if financial_config.get('apply_percentile_cap', False) else None
        )
        df_result['ACTIVITY_VOLATILITY'] = improved_component_normalization(
            activity_volatility_raw,
            activity_config.get('method', 'zscore'),
            None  # Z-score method doesn't use percentile capping
        )
        df_result['EXPLORATION_VOLATILITY'] = improved_component_normalization(
            exploration_volatility_raw,
            exploration_config.get('method', 'log_zscore'),
            None  # Log z-score method doesn't use percentile capping
        )
    else:
        # Use original normalization
        df_result['FINANCIAL_VOLATILITY'] = financial_volatility_raw
        df_result['ACTIVITY_VOLATILITY'] = activity_volatility_raw
        df_result['EXPLORATION_VOLATILITY'] = exploration_volatility_raw
    
    # 4. Calculate composite score
    print("  4. Computing composite Behavioural Volatility Score...")
    df_result['BEHAVIOURAL_VOLATILITY_SCORE_RAW'] = (
    component_weights['financial_volatility'] * df_result['FINANCIAL_VOLATILITY'] +
    component_weights['activity_volatility'] * df_result['ACTIVITY_VOLATILITY'] +
    component_weights['exploration_volatility'] * df_result['EXPLORATION_VOLATILITY']
    )
    
    # Apply final transformation if using improved normalization
    if use_improved_normalization:
        final_config = config['features']['improved_normalization'].get('final_transformation', {})
        if final_config.get('apply_sqrt', True):
            print("  5. Applying final square root transformation...")
            df_result['BEHAVIOURAL_VOLATILITY_SCORE'] = np.sqrt(df_result['BEHAVIOURAL_VOLATILITY_SCORE_RAW'])
        else:
            df_result['BEHAVIOURAL_VOLATILITY_SCORE'] = df_result['BEHAVIOURAL_VOLATILITY_SCORE_RAW']
    else:
        # 5. Apply normalization if specified (original method)
        if config['features']['normalize_score']:
            print("  5. Applying score normalization...")
            normalization_method = config['features']['normalization_method']
            
            if normalization_method == "min_max":
                scaler = MinMaxScaler()
                df_result['BEHAVIOURAL_VOLATILITY_SCORE'] = scaler.fit_transform(
                    df_result[['BEHAVIOURAL_VOLATILITY_SCORE_RAW']]
                ).flatten()
            elif normalization_method == "z_score":
                scaler = StandardScaler()
                df_result['BEHAVIOURAL_VOLATILITY_SCORE'] = scaler.fit_transform(
                    df_result[['BEHAVIOURAL_VOLATILITY_SCORE_RAW']]
                ).flatten()
            else:
                df_result['BEHAVIOURAL_VOLATILITY_SCORE'] = df_result['BEHAVIOURAL_VOLATILITY_SCORE_RAW']
        else:
            df_result['BEHAVIOURAL_VOLATILITY_SCORE'] = df_result['BEHAVIOURAL_VOLATILITY_SCORE_RAW']
    
    print("Behavioural Volatility Score calculation completed!")
    
    return df_result


def behavioural_volatility_pipeline(input_path: Optional[str] = None,
                                 output_path: Optional[str] = None,
                                 config_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete pipeline for behavioural volatility score calculation.
    
    Parameters:
    -----------
    input_path : str, optional
        Path to input CSV file
    output_path : str, optional
        Path to output CSV file
    config_path : str, optional
        Path to configuration file
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict]
        Processed dataframe and configuration used
    """
    # Load configuration
    config = load_config(config_path)
    
    # Use paths from config if not provided
    if input_path is None:
        input_path = config['data']['input_path']
    if output_path is None:
        output_path = config['data']['output_path']
    
    print(f"Loading data from: {input_path}")
    
    # Load input data
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Validate required columns exist
    required_columns = (
        config['activity_events']['event_columns'] +
        [config['financial_volatility']['numerator_column'],
         config['financial_volatility']['denominator_column']] +
        config['exploration_volatility']['diversity_columns'] +
        [config['exploration_volatility']['activity_rate_column']]
    )
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Calculate behavioural volatility score
    df_result = calculate_behavioural_volatility_score(df, config)
    
    # Save results
    print(f"Saving results to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_result.to_csv(output_path, index=False)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("BEHAVIOURAL VOLATILITY SCORE SUMMARY")
    print("="*80)
    
    print(f"Dataset shape: {df_result.shape}")
    print(f"Behavioural Volatility Score statistics:")
    print(f"  Mean: {df_result['BEHAVIOURAL_VOLATILITY_SCORE'].mean():.6f}")
    print(f"  Median: {df_result['BEHAVIOURAL_VOLATILITY_SCORE'].median():.6f}")
    print(f"  Std: {df_result['BEHAVIOURAL_VOLATILITY_SCORE'].std():.6f}")
    print(f"  Min: {df_result['BEHAVIOURAL_VOLATILITY_SCORE'].min():.6f}")
    print(f"  Max: {df_result['BEHAVIOURAL_VOLATILITY_SCORE'].max():.6f}")
    
    print(f"\nComponent Score Statistics:")
    print(f"Financial Volatility - Mean: {df_result['FINANCIAL_VOLATILITY'].mean():.6f}, Std: {df_result['FINANCIAL_VOLATILITY'].std():.6f}")
    print(f"Activity Volatility - Mean: {df_result['ACTIVITY_VOLATILITY'].mean():.6f}, Std: {df_result['ACTIVITY_VOLATILITY'].std():.6f}")
    print(f"Exploration Volatility - Mean: {df_result['EXPLORATION_VOLATILITY'].mean():.6f}, Std: {df_result['EXPLORATION_VOLATILITY'].std():.6f}")
    
    print(f"\nFeature engineering completed successfully!")
    print(f"Output saved to: {output_path}")
    
    return df_result, config


if __name__ == "__main__":
    # Run the pipeline with default configuration
    df_result, config = behavioural_volatility_pipeline()
