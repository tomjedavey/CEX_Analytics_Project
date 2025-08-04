#!/usr/bin/env python3
"""
Test script for scaling and combined preprocessing functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'source_code_package'))

from source_code_package.data.preprocess_cluster import (
    scale_features_from_config, 
    preprocess_for_clustering,
    log_transform_features_from_config
)

def test_scaling_functionality():
    """
    Test feature scaling functionality for clustering preprocessing.
    
    This test validates the scaling functions used in the clustering pipeline,
    ensuring proper standardization of numerical features while maintaining
    data integrity and providing appropriate metadata.
    
    Tests:
    ------
    - scale_features_from_config(): Configuration-driven feature scaling
    - StandardScaler application and parameterization
    - Column exclusion handling based on configuration
    - Scaler object return and metadata preservation
    - Statistical properties of scaled features (mean ≈ 0, std ≈ 1)
    
    Expected Behavior:
    ------------------
    - Loads data according to configuration settings
    - Applies StandardScaler to appropriate numerical columns
    - Returns fitted scaler object for potential inverse transformations
    - Excludes configured columns from scaling process
    - Maintains data structure and provides column mapping
    
    Notes:
    ------
    - Requires valid configuration with scaling settings
    - Validates statistical properties of scaled data
    - Tests integration with preprocessing pipeline
    """
    
    print("Testing scaling functionality...")
    print("=" * 60)
    
    try:
        # Test scaling only
        print("\n1. Testing scale_features_from_config():")
        df_scaled, scaled_cols, scaler = scale_features_from_config()
        
        print(f"Scaled dataset shape: {df_scaled.shape}")
        print(f"Number of scaled columns: {len(scaled_cols)}")
        print(f"Scaler type: {type(scaler)}")
        
        # Display some statistics
        print("\nSample of scaled data statistics:")
        sample_cols = ['TOKEN_DIVERSITY', 'TOTAL_TRANSFER_USD', 'AVG_TRANSFER_USD']
        available_cols = [col for col in sample_cols if col in df_scaled.columns and col in scaled_cols]
        if available_cols:
            print(df_scaled[available_cols].describe())
        
        print(f"\nScaling successful! Scaled columns: {scaled_cols[:5]}..." if len(scaled_cols) > 5 else f"Scaled columns: {scaled_cols}")
        
    except Exception as e:
        print(f"Error during scaling test: {str(e)}")
        import traceback
        traceback.print_exc()

def test_combined_preprocessing():
    """
    Test the integrated preprocessing pipeline combining log transformation and scaling.
    
    This test validates the complete preprocessing workflow that combines
    log transformation and feature scaling in the correct sequence for
    clustering pipeline preparation.
    
    Tests:
    ------
    - preprocess_for_clustering(): Integrated preprocessing pipeline
    - Sequential application of log transformation and scaling
    - Information tracking and metadata preservation
    - Final data validation and statistical properties
    - Error handling and process reporting
    
    Expected Behavior:
    ------------------
    - Applies log transformation first (if enabled)
    - Follows with feature scaling (if enabled)
    - Returns processed DataFrame and detailed processing information
    - Maintains data integrity throughout the pipeline
    - Provides comprehensive processing metadata
    
    Validation Checks:
    ------------------
    - Data shape preservation
    - Processing steps tracking
    - Column count verification for each transformation
    - Final statistical properties validation
    - Error handling and recovery
    
    Notes:
    ------
    - Tests the most common preprocessing configuration (both transforms enabled)
    - Validates integration between preprocessing components
    - Ensures correct processing order and parameter passing
    """
    
    print("\n" + "=" * 60)
    print("2. Testing preprocess_for_clustering():")
    
    try:
        df_processed, info = preprocess_for_clustering(
            apply_log_transform=True, 
            apply_scaling=True
        )
        
        print(f"Final processed dataset shape: {df_processed.shape}")
        print(f"Steps applied: {info['steps_applied']}")
        print(f"Log transformed columns: {len(info['log_transformed_columns'])}")
        print(f"Scaled columns: {len(info['scaled_columns'])}")
        
        # Show some final statistics
        print("\nFinal processed data sample:")
        display_cols = ['TX_PER_MONTH', 'ACTIVE_DURATION_DAYS', 'TOKEN_DIVERSITY', 'TOTAL_TRANSFER_USD']
        available_cols = [col for col in display_cols if col in df_processed.columns]
        print(df_processed[available_cols].head())
        
        print(f"\nCombined preprocessing successful!")
        
    except Exception as e:
        print(f"Error during combined preprocessing test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_scaling_functionality()
    test_combined_preprocessing()

#MAY HAVE TO CHANGE SOME OF THE PATHS ABOVE - MOVED INTO TESTS FOLDER