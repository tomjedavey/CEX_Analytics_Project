#!/usr/bin/env python3
"""
Shannon Entropy Feature Engineering Module for Cross Domain Engagement Score

This module provides functionality to calculate Shannon entropy scores for cryptocurrency wallet 
cross-domain engagement based on event count proportions. Shannon entropy measures the engagement
diversity across different blockchain event types and domains.

The entropy calculation:
H(X) = -Σ(i=1 to n) pi × log₂(pi)

Where pi is the proportion of each event type, and the result is normalized to [0,1] scale
by dividing by the maximum possible entropy (log₂(number_of_categories)).

Author: Tom Davey
Date: August 2025
"""

import pandas as pd
import numpy as np
import os
import yaml
from typing import Optional, Dict, Tuple, List


def calculate_event_proportions(df: pd.DataFrame, event_columns: List[str]) -> pd.DataFrame:
    """
    Calculate proportions for each event type per wallet.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with event counts
    event_columns : List[str]
        List of event column names
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with proportion columns
    """
    df_result = df.copy()
    
    # Calculate total events per wallet (sum across all event types)
    df_result['TOTAL_EVENTS'] = df[event_columns].sum(axis=1)
    
    # Calculate proportions for each event type
    proportion_columns = []
    for col in event_columns:
        prop_col = f"{col}_PROPORTION"
        # Avoid division by zero - set proportion to 0 where total events is 0
        df_result[prop_col] = np.where(
            df_result['TOTAL_EVENTS'] > 0,
            df_result[col].astype(float) / df_result['TOTAL_EVENTS'].astype(float),
            0.0
        )
        proportion_columns.append(prop_col)
    
    return df_result, proportion_columns


def calculate_shannon_entropy(proportions: np.ndarray, filter_zeros: bool = True) -> float:
    """
    Calculate Shannon entropy for a given set of proportions.
    
    Parameters:
    -----------
    proportions : np.ndarray
        Array of proportions (should sum to 1)
    filter_zeros : bool
        Whether to filter out zero proportions before calculation
        
    Returns:
    --------
    float
        Shannon entropy value
    """
    # Ensure we have a proper numpy array with float dtype
    proportions = np.array(proportions, dtype=np.float64)
    
    if filter_zeros:
        # Filter out zero proportions
        proportions = proportions[proportions > 0]
    
    if len(proportions) == 0:
        return 0.0
    
    # Calculate Shannon entropy: H(X) = -Σ(pi × log₂(pi))
    # Use np.log2 with proper numpy arrays
    log_proportions = np.log2(proportions)
    entropy = -np.sum(proportions * log_proportions)
    
    return float(entropy)


def normalize_entropy(entropy: float, max_categories: int) -> float:
    """
    Normalize entropy to [0,1] scale.
    
    Parameters:
    -----------
    entropy : float
        Raw Shannon entropy value
    max_categories : int
        Maximum number of categories (for calculating max possible entropy)
        
    Returns:
    --------
    float
        Normalized entropy score [0,1]
    """
    if max_categories <= 1:
        return 0.0
    
    max_entropy = np.log2(max_categories)
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Ensure the result is between 0 and 1
    return max(0.0, min(1.0, normalized))


def calculate_cross_domain_engagement_score(df: pd.DataFrame, 
                                          event_columns: List[str],
                                          filter_zeros: bool = True,
                                          normalize: bool = True) -> pd.DataFrame:
    """
    Calculate cross-domain engagement scores using Shannon entropy for each wallet.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing event count data
    event_columns : List[str]
        List of event column names
    filter_zeros : bool
        Whether to filter out zero proportions before entropy calculation
    normalize : bool
        Whether to normalize entropy to [0,1] scale
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with activity diversity score added
    """
    if len(event_columns) == 0:
        raise ValueError("No event columns provided")
    
    print(f"Calculating cross-domain engagement for {len(event_columns)} event types:")
    for col in event_columns:
        print(f"  - {col}")
    
    # Calculate proportions
    df_result, proportion_columns = calculate_event_proportions(df, event_columns)
    
    print(f"Calculated proportions for {len(proportion_columns)} event types")
    
    # Calculate Shannon entropy for each wallet
    entropy_scores = []
    zero_activity_count = 0
    
    for idx, row in df_result.iterrows():
        # Get proportions for this wallet
        proportions = row[proportion_columns].values
        
        # Check if wallet has any activity
        if row['TOTAL_EVENTS'] == 0:
            entropy_scores.append(0.0)
            zero_activity_count += 1
        else:
            # Calculate entropy
            entropy = calculate_shannon_entropy(proportions, filter_zeros=filter_zeros)
            
            # Normalize if requested
            if normalize:
                # For normalization, use the number of non-zero event types for this wallet
                if filter_zeros:
                    active_categories = len(proportions[proportions > 0])
                else:
                    active_categories = len(event_columns)
                
                entropy = normalize_entropy(entropy, active_categories)
            
            entropy_scores.append(entropy)
    
    # Add entropy score to dataframe
    df_result['CROSS_DOMAIN_ENGAGEMENT_SCORE'] = entropy_scores
    
    print(f"\nActivity diversity calculation complete:")
    print(f"  Wallets processed: {len(df_result)}")
    print(f"  Wallets with zero activity: {zero_activity_count}")
    print(f"  Mean diversity score: {np.mean(entropy_scores):.4f}")
    print(f"  Median diversity score: {np.median(entropy_scores):.4f}")
    print(f"  Min diversity score: {np.min(entropy_scores):.4f}")
    print(f"  Max diversity score: {np.max(entropy_scores):.4f}")
    
    return df_result


def cross_domain_engagement_pipeline(data_path: Optional[str] = None,
                              config_path: Optional[str] = None,
                              output_path: Optional[str] = None,
                              save_results: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete pipeline for activity diversity feature engineering using Shannon entropy.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to input CSV file. If None, uses default from config.
    config_path : str, optional  
        Path to configuration file. If None, uses default config.
    output_path : str, optional
        Path to save output CSV file. If None, uses default from config.
    save_results : bool
        Whether to save results to file
        
    Returns:
    --------
    tuple
        (processed_dataframe, processing_info)
    """
    print("Starting Cross Domain Engagement Score Feature Engineering Pipeline...")
    
    # Load configuration
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../config/config_cross_domain_engagement.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from: {config_path}")
    except FileNotFoundError:
        # Use default configuration if file not found
        print("No configuration file found, using default settings")
        config = {
            'data': {
                'input_path': 'data/raw_data/new_raw_data_polygon.csv',
                'output_path': 'data/processed_data/cross_domain_engagement_features.csv'
            },
            'features': {
                'event_suffix': 'EVENTS',
                'filter_zeros': True,
                'normalize_entropy': True
            }
        }
    
    # Determine data path
    if data_path is None:
        config_data = config.get('cross_domain_engagement', config)  # Support both new and old config structure
        data_path = config_data.get('input_file', 'data/processed_data/new_raw_data_polygon.csv')
        # Convert relative path to absolute
        if not os.path.isabs(data_path):
            current_dir = os.path.dirname(__file__)
            while not os.path.exists(os.path.join(current_dir, 'pyproject.toml')):
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:
                    raise FileNotFoundError('Could not find project root (pyproject.toml)')
                current_dir = parent_dir
            project_root = current_dir
            data_path = os.path.join(project_root, data_path)
            data_path = os.path.normpath(data_path)
    
    # Load input data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input data file not found: {data_path}")
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    original_shape = df.shape
    print(f"Original data shape: {original_shape}")
    
    # Extract feature configuration
    config_data = config.get('cross_domain_engagement', config)  # Support both new and old config structure
    
    # Use hardcoded event columns from config
    event_columns = config_data.get('event_columns', [])
    if len(event_columns) == 0:
        raise ValueError("No event columns specified in configuration")
    
    print(f"Using {len(event_columns)} hardcoded event columns from config:")
    for col in sorted(event_columns):
        print(f"  - {col}")
    
    # Legacy configuration support for other features
    feature_config = config_data.get('features', {})
    filter_zeros = feature_config.get('filter_zeros', True)
    normalize_entropy = feature_config.get('normalize_entropy', True)

    # Apply cross domain engagement calculation
    df_processed = calculate_cross_domain_engagement_score(
        df=df,
        event_columns=event_columns,
        filter_zeros=filter_zeros,
        normalize=normalize_entropy
    )
    
    # Prepare output path
    if output_path is None:
        config_data = config.get('cross_domain_engagement', config)  # Support both new and old config structure
        output_path = config_data.get('output_file', 'data/processed_data/cross_domain_engagement_features.csv')
        if not os.path.isabs(output_path):
            current_dir = os.path.dirname(__file__)
            while not os.path.exists(os.path.join(current_dir, 'pyproject.toml')):
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:
                    raise FileNotFoundError('Could not find project root (pyproject.toml)')
                current_dir = parent_dir
            project_root = current_dir
            output_path = os.path.join(project_root, output_path)
            output_path = os.path.normpath(output_path)
    
    # Save results if requested
    if save_results:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        df_processed.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    
    # Calculate additional features created
    new_features = ['TOTAL_EVENTS', 'CROSS_DOMAIN_ENGAGEMENT_SCORE']
    proportion_features = [f"{col}_PROPORTION" for col in event_columns]
    new_features.extend(proportion_features)
    
    # Prepare processing info
    processing_info = {
        'original_shape': original_shape,
        'processed_shape': df_processed.shape,
        'event_columns_found': event_columns,
        'new_features_added': new_features,
        'settings': {
            'filter_zeros': filter_zeros,
            'normalize_entropy': normalize_entropy,
            'event_columns_used': event_columns
        },
        'statistics': {
            'mean_diversity_score': df_processed['CROSS_DOMAIN_ENGAGEMENT_SCORE'].mean(),
            'median_diversity_score': df_processed['CROSS_DOMAIN_ENGAGEMENT_SCORE'].median(),
            'std_diversity_score': df_processed['CROSS_DOMAIN_ENGAGEMENT_SCORE'].std(),
            'min_diversity_score': df_processed['CROSS_DOMAIN_ENGAGEMENT_SCORE'].min(),
            'max_diversity_score': df_processed['CROSS_DOMAIN_ENGAGEMENT_SCORE'].max()
        },
        'input_file': data_path,
        'output_file': output_path if save_results else None
    }
    
    print("\nActivity Diversity Feature Engineering Pipeline completed!")
    print(f"Added {len(new_features)} new features")
    
    return df_processed, processing_info


if __name__ == "__main__":
    """Test the activity diversity feature engineering functionality."""
    print("Activity Diversity Feature Engineering - Test Run")
    print("=" * 60)
    
    try:
        df_result, info = cross_domain_engagement_pipeline()
        print("✅ Pipeline completed successfully!")
        print(f"📊 Processed {info['processed_shape'][0]:,} wallets")
        print(f"🎯 Added {len(info['new_features_added'])} new features")
        print(f"📈 Mean engagement score: {info['statistics']['mean_diversity_score']:.4f}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
