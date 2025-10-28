#!/usr/bin/env python3
"""
Revenue Proxy Feature Engineering Module

This module provides functionality to calculate the REVENUE_SCORE_PROXY feature
for cryptocurrency wallet analysis. The revenue proxy score estimates potential
revenue contribution of a wallet based on activity patterns.

Author: Tom Davey
Date: August 2025
"""

import pandas as pd
import numpy as np
import os
import yaml
from typing import Optional, Dict, Tuple, List


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that all required columns are present in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to validate
    required_columns : List[str]
        List of required column names
        
    Returns:
    --------
    tuple
        (is_valid, missing_columns)
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns


def handle_missing_values(df: pd.DataFrame, columns: List[str], fill_method: str = 'zero') -> pd.DataFrame:
    """
    Handle missing values in specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to handle missing values for
    fill_method : str
        Method to fill missing values ('zero', 'mean', 'median')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    df_processed = df.copy()
    
    for col in columns:
        if col in df_processed.columns:
            if fill_method == 'zero':
                df_processed[col] = df_processed[col].fillna(0)
            elif fill_method == 'mean':
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            elif fill_method == 'median':
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed


def calculate_revenue_score_proxy(df: pd.DataFrame, 
                                 weights: Optional[Dict[str, float]] = None,
                                 handle_missing: bool = True,
                                 fill_method: str = 'zero') -> pd.DataFrame:
    """
    Calculate the REVENUE_SCORE_PROXY for each wallet based on the formula:
    
    REVENUE_SCORE_PROXY = 0.4 * AVG_TRANSFER_USD * TX_PER_MONTH + 
                         0.35 * (DEX_EVENTS + DEFI_EVENTS) * AVG_TRANSFER_USD + 
                         0.25 * BRIDGE_TOTAL_VOLUME_USD
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing wallet activity data
    weights : Dict[str, float], optional
        Custom weights for formula components. Keys: 'transaction_activity', 'dex_defi_activity', 'bridge_activity'
    handle_missing : bool
        Whether to handle missing values
    fill_method : str
        Method for handling missing values ('zero', 'mean', 'median')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with REVENUE_SCORE_PROXY column added
    """
    # Define required columns for the calculation
    required_columns = [
        'AVG_TRANSFER_USD',
        'TX_PER_MONTH', 
        'DEX_EVENTS',
        'DEFI_EVENTS',
        'BRIDGE_TOTAL_VOLUME_USD'
    ]
    
    # Validate required columns
    is_valid, missing_columns = validate_required_columns(df, required_columns)
    if not is_valid:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Create a copy to avoid modifying original dataframe
    df_result = df.copy()
    
    # Handle missing values if requested
    if handle_missing:
        df_result = handle_missing_values(df_result, required_columns, fill_method)
        print(f"Handled missing values using method: {fill_method}")
    
    # Set default weights if not provided
    if weights is None:
        weights = {
            'transaction_activity': 0.4,    # Weight for AVG_TRANSFER_USD * TX_PER_MONTH
            'dex_defi_activity': 0.35,      # Weight for (DEX_EVENTS + DEFI_EVENTS) * AVG_TRANSFER_USD
            'bridge_activity': 0.25         # Weight for BRIDGE_TOTAL_VOLUME_USD
        }
    
    print("Calculating REVENUE_SCORE_PROXY components...")
    
    # Calculate components of the revenue proxy score
    # Component 1: Transaction activity (frequency * average value)
    transaction_component = (df_result['AVG_TRANSFER_USD'] * df_result['TX_PER_MONTH'])
    
    # Component 2: DEX/DeFi activity impact on transfer value
    dex_defi_component = ((df_result['DEX_EVENTS'] + df_result['DEFI_EVENTS']) * 
                         df_result['AVG_TRANSFER_USD'])
    
    # Component 3: Bridge activity (direct volume measure)
    bridge_component = df_result['BRIDGE_TOTAL_VOLUME_USD']
    
    # Calculate final revenue proxy score
    df_result['REVENUE_SCORE_PROXY'] = (
        weights['transaction_activity'] * transaction_component +
        weights['dex_defi_activity'] * dex_defi_component +
        weights['bridge_activity'] * bridge_component
    )
    
    # Add component columns for analysis (optional)
    # Note: These store the unweighted raw components for analytical purposes
    df_result['REVENUE_PROXY_TRANSACTION_COMPONENT'] = transaction_component
    df_result['REVENUE_PROXY_DEX_DEFI_COMPONENT'] = dex_defi_component  
    df_result['REVENUE_PROXY_BRIDGE_COMPONENT'] = bridge_component
    
    print(f"REVENUE_SCORE_PROXY calculated for {len(df_result)} wallets")
    print(f"Score statistics:")
    print(f"  Mean: ${df_result['REVENUE_SCORE_PROXY'].mean():.2f}")
    print(f"  Median: ${df_result['REVENUE_SCORE_PROXY'].median():.2f}")
    print(f"  Min: ${df_result['REVENUE_SCORE_PROXY'].min():.2f}")
    print(f"  Max: ${df_result['REVENUE_SCORE_PROXY'].max():.2f}")
    print(f"  Std Dev: ${df_result['REVENUE_SCORE_PROXY'].std():.2f}")
    
    return df_result


def revenue_proxy_feature_pipeline(data_path: Optional[str] = None,
                                  config_path: Optional[str] = None,
                                  output_path: Optional[str] = None,
                                  save_results: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete pipeline for revenue proxy feature engineering.
    
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
    print("Starting Revenue Proxy Feature Engineering Pipeline...")
    
    # Load configuration
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../config/config_revenue_proxy.yaml')
    
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
                'output_path': 'data/processed_data/revenue_proxy_features.csv'
            },
            'features': {
                'weights': {
                    'transaction_activity': 0.4,
                    'dex_defi_activity': 0.35,
                    'bridge_activity': 0.25
                },
                'handle_missing_values': True,
                'fill_method': 'zero'
            }
        }
    
    # Determine data path
    if data_path is None:
        data_path = config.get('data', {}).get('input_path', 'data/raw_data/new_raw_data_polygon.csv')
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
    feature_config = config.get('features', {})
    weights = feature_config.get('weights', None)
    handle_missing = feature_config.get('handle_missing_values', True)
    fill_method = feature_config.get('fill_method', 'zero')
    
    # Apply revenue proxy calculation
    df_processed = calculate_revenue_score_proxy(
        df=df,
        weights=weights,
        handle_missing=handle_missing,
        fill_method=fill_method
    )
    
    # Prepare output path
    if output_path is None:
        output_path = config.get('data', {}).get('output_path', 'data/processed_data/revenue_proxy_features.csv')
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
    
    # Prepare processing info
    processing_info = {
        'original_shape': original_shape,
        'processed_shape': df_processed.shape,
        'new_features_added': [
            'REVENUE_SCORE_PROXY',
            'REVENUE_PROXY_TRANSACTION_COMPONENT',
            'REVENUE_PROXY_DEX_DEFI_COMPONENT', 
            'REVENUE_PROXY_BRIDGE_COMPONENT'
        ],
        'weights_used': weights,
        'missing_value_handling': {
            'enabled': handle_missing,
            'method': fill_method
        },
        'input_file': data_path,
        'output_file': output_path if save_results else None
    }
    
    print("\nRevenue Proxy Feature Engineering Pipeline completed!")
    print(f"Added {len(processing_info['new_features_added'])} new features")
    
    return df_processed, processing_info


if __name__ == "__main__":
    """Test the revenue proxy feature engineering functionality."""
    print("Revenue Proxy Feature Engineering - Test Run")
    print("=" * 50)
    
    try:
        df_result, info = revenue_proxy_feature_pipeline()
        print("✅ Pipeline completed successfully!")
        print(f"📊 Processed {info['processed_shape'][0]:,} wallets")
        print(f"🎯 Added {len(info['new_features_added'])} new features")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
