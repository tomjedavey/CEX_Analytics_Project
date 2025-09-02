#!/usr/bin/env python3
"""
Execution script for enhanced feature engineering supporting multiple datasets.

This script serves as a simple execution wrapper around the core functionality
in source_code_package.data.enhanced_feature_engineering module.

Author: Tom Davey
Date: July 2025
"""

import os
import sys

# Add source_code_package to path
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
source_package_path = os.path.join(project_root, 'source_code_package')
sys.path.insert(0, source_package_path)

try:
    from data.enhanced_feature_engineering import process_multiple_datasets_batch
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure source_code_package structure is correct")
    exit(1)


def main():
    """Main function to execute enhanced feature engineering."""
    
    # Define configuration paths
    config_dir = os.path.join(project_root, 'source_code_package', 'config')
    
    config_paths = [
        os.path.join(config_dir, 'config_AS_1_full_dataset.yaml'),
        os.path.join(config_dir, 'config_AS_1_cluster_0.yaml'),
        os.path.join(config_dir, 'config_AS_1_cluster_1.yaml')
    ]
    
    print("AS_1 ENHANCED FEATURE ENGINEERING")
    print("=" * 50)
    print("This script will process multiple datasets for AS_1 analysis:")
    print("1. Full dataset (new_raw_data_polygon.csv)")
    print("2. Cluster 0 dataset")
    print("3. Cluster 1 dataset")
    print("\nEnsure that cluster datasets have been created first using prepare_cluster_datasets.py")
    
    # Check if cluster datasets exist
    cluster_0_path = os.path.join(project_root, 'data/raw_data/cluster_datasets/new_raw_data_polygon_cluster_0.csv')
    cluster_1_path = os.path.join(project_root, 'data/raw_data/cluster_datasets/new_raw_data_polygon_cluster_1.csv')
    
    if not os.path.exists(cluster_0_path) or not os.path.exists(cluster_1_path):
        print("\n⚠️  WARNING: Cluster datasets not found!")
        print("Please run prepare_cluster_datasets.py first to create cluster-specific datasets.")
        print("\nProceeding with full dataset only...")
        config_paths = [config_paths[0]]  # Only process full dataset
    
    # Process datasets using core functionality
    results = process_multiple_datasets_batch(config_paths, verbose=True)
    
    # Check results
    if all(r['success'] for r in results.values()):
        print("\n🎉 All datasets processed successfully!")
        print("Ready for AS_1 model training.")
        return True
    else:
        print("\n⚠️  Some datasets failed to process. Check errors above.")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
