#!/usr/bin/env python3
"""
Test script for log transformation functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'source_code_package'))

from source_code_package.data.preprocess_cluster import log_transform_features_from_config, log_transform_features

def test_log_transformation():
    """Test the log transformation functions"""
    
    print("Testing log transformation functionality...")
    print("=" * 50)
    
    try:
        # Test the config-based function
        print("\n1. Testing log_transform_features_from_config():")
        df_transformed, transformed_cols = log_transform_features_from_config()
        
        print(f"\nOriginal dataset shape: {df_transformed.shape}")
        print(f"Number of transformed columns: {len(transformed_cols)}")
        
        # Display first few rows of specific columns
        print("\nSample of transformed data:")
        sample_cols = ['WALLET', 'TX_PER_MONTH', 'ACTIVE_DURATION_DAYS', 'TOKEN_DIVERSITY', 'TOTAL_TRANSFER_USD']
        available_cols = [col for col in sample_cols if col in df_transformed.columns]
        print(df_transformed[available_cols].head())
        
        # Verify excluded columns weren't transformed
        excluded_cols = ['TX_PER_MONTH', 'ACTIVE_DURATION_DAYS']
        for col in excluded_cols:
            if col in df_transformed.columns:
                original_data = df_transformed[col].head()
                print(f"\n{col} (should NOT be log-transformed):")
                print(original_data.values)
        
        print(f"\nTransformation successful!")
        print(f"Columns that were log-transformed: {transformed_cols[:5]}..." if len(transformed_cols) > 5 else f"Columns that were log-transformed: {transformed_cols}")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_log_transformation()


# run the following in terminal to execute the test:
#cd /Users/tomdavey/Documents/GitHub/MLProject1 && python test_log_transform.py
#or 
#cd /Users/tomdavey/Documents/GitHub/MLProject1 && ls -la test_log_transform.py
#MAY HAVE TO CHANGE THE PATH ABOVE TO WHERE THE FILE IS LOCATED ON YOUR MACHINE - added to tests folder after building the terminal command