#!/usr/bin/env python3
"""
Test script to demonstrate the new exclude_no_columns functionality.

This script shows how to run the UMAP + HDBSCAN clustering pipeline
with the option to exclude no columns from preprocessing, effectively
using ALL numerical columns in the dataset.

Usage:
    python test_exclude_no_columns.py
"""

import os
import sys
import yaml
import pandas as pd

# Add the source code package to path
sys.path.append('/Users/tomdavey/Documents/GitHub/MLProject1/source_code_package')

from models.clustering_functionality.UMAP_dim_reduction import validate_feature_consistency, umap_with_preprocessing
from models.clustering_functionality.HBDSCAN_cluster import run_flexible_hdbscan_pipeline, run_umap_hdbscan_pipeline

def test_exclude_no_columns_functionality():
    """Test the new exclude_no_columns functionality."""
    
    print("=" * 80)
    print("TESTING EXCLUDE_NO_COLUMNS FUNCTIONALITY")
    print("=" * 80)
    
    # Define config path
    config_path = '/Users/tomdavey/Documents/GitHub/MLProject1/source_code_package/config/config_cluster.yaml'
    
    # Test 1: Load and show current configuration
    print("\n1. CURRENT CONFIGURATION:")
    print("-" * 40)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    preprocessing_config = config.get('preprocessing', {})
    umap_config = config.get('umap', {})
    
    print(f"Log transformation exclude_no_columns: {preprocessing_config.get('log_transformation', {}).get('exclude_no_columns', 'not set')}")
    print(f"Scaling exclude_no_columns: {preprocessing_config.get('scaling', {}).get('exclude_no_columns', 'not set')}")
    print(f"UMAP include_all_columns: {umap_config.get('include_all_columns', 'not set')}")
    print(f"UMAP include_columns: {umap_config.get('include_columns', [])}")
    
    # Test 2: Show available columns in the dataset
    print("\n2. AVAILABLE COLUMNS IN DATASET:")
    print("-" * 40)
    
    data_path = '/Users/tomdavey/Documents/GitHub/MLProject1/data/raw_data/new_raw_data_polygon.csv'
    df = pd.read_csv(data_path)
    
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    print(f"Total columns: {len(df.columns)}")
    print(f"Numerical columns ({len(numerical_columns)}): {numerical_columns}")
    
    # Test 3: Validate current configuration
    print("\n3. CURRENT CONFIGURATION VALIDATION:")
    print("-" * 40)
    
    validation_results = validate_feature_consistency(config_path)
    
    print(f"Include all columns: {validation_results.get('include_all_columns', False)}")
    print(f"Log exclude no columns: {validation_results.get('log_exclude_no_columns', False)}")
    print(f"Scale exclude no columns: {validation_results.get('scale_exclude_no_columns', False)}")
    
    if validation_results['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    
    print("\n📋 RECOMMENDATIONS:")
    for rec in validation_results['recommendations']:
        print(f"  - {rec}")
    
    print("\n" + "=" * 80)
    print("TO ENABLE EXCLUDE_NO_COLUMNS FUNCTIONALITY:")
    print("=" * 80)
    print("""
To use ALL numerical columns in your clustering pipeline:

1. Set preprocessing options in config_cluster.yaml:
   preprocessing:
     log_transformation:
       exclude_no_columns: true
     scaling:
       exclude_no_columns: true

2. Set UMAP option in config_cluster.yaml:
   umap:
     include_all_columns: true

3. This will:
   - Apply log transformation to ALL numerical columns
   - Apply scaling to ALL numerical columns  
   - Use ALL available columns in UMAP/clustering

Example configuration snippet:
---
preprocessing:
  log_transformation:
    enabled: true
    exclude_no_columns: true  # NEW: Ignores exclude_columns
  scaling:
    enabled: true
    exclude_no_columns: true  # NEW: Ignores exclude_columns

umap:
  enabled: true
  include_all_columns: true   # NEW: Ignores include_columns
---
""")

def demonstrate_with_example_config():
    """Demonstrate with a temporary modified configuration."""
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION WITH EXAMPLE CONFIGURATION")
    print("=" * 80)
    
    # Create a temporary config with exclude_no_columns enabled
    config_path = '/Users/tomdavey/Documents/GitHub/MLProject1/source_code_package/config/config_cluster.yaml'
    temp_config_path = '/Users/tomdavey/Documents/GitHub/MLProject1/temp_config_all_columns.yaml'
    
    # Load original config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for demonstration
    config['preprocessing']['log_transformation']['exclude_no_columns'] = True
    config['preprocessing']['scaling']['exclude_no_columns'] = True
    config['umap']['include_all_columns'] = True
    config['umap']['enabled'] = True  # Make sure UMAP is enabled
    
    # Save temporary config
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created temporary configuration: {temp_config_path}")
    print("Modified settings:")
    print("  - exclude_no_columns: true (for both log and scaling)")
    print("  - include_all_columns: true (for UMAP)")
    
    # Validate the modified configuration
    print("\nValidating modified configuration...")
    validation_results = validate_feature_consistency(temp_config_path)
    
    if validation_results['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    
    print("\n📋 RECOMMENDATIONS:")
    for rec in validation_results['recommendations']:
        print(f"  - {rec}")
    
    print(f"\nTo test the pipeline with these settings, run:")
    print(f"  - UMAP pipeline: python source_code_package/models/clustering_functionality/UMAP_dim_reduction.py")
    print(f"  - Full pipeline: python scripts/clustering/run_hdbscan_clustering.py")
    
    # Clean up
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
        print(f"\nCleaned up temporary config file.")

if __name__ == "__main__":
    test_exclude_no_columns_functionality()
    demonstrate_with_example_config()
