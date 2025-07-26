#!/usr/bin/env python3
"""
Script to execute HDBSCAN clustering on UMAP-reduced data.

This script demonstrates how to use the HDBSCAN clustering functionality
built in the source_code_package. It can be run standalone or integrated
into larger data processing pipelines.

Author: Tom Davey
Date: July 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import yaml
from typing import Optional

# Add source_code_package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../source_code_package'))

# Import our custom clustering functionality
from models.clustering_functionality.HBDSCAN_cluster import (
    hdbscan_clustering_pipeline,
    run_umap_hdbscan_pipeline,
      run_flexible_hdbscan_pipeline,
    load_hdbscan_config,
    validate_hdbscan_config
)


def main():
    """
    Main function to demonstrate HDBSCAN clustering usage.
    
    This function shows three different ways to use the HDBSCAN functionality:
    1. Configuration validation
    2. Clustering with existing UMAP data
    3. Complete UMAP + HDBSCAN pipeline
    """
    print("HDBSCAN Clustering Script")
    print("=" * 50)
    
    # Configuration paths
    config_path = os.path.join(os.path.dirname(__file__), 
                              '../../source_code_package/config/config_cluster.yaml')
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), '../../clustering_output')
    
    # Example 1: Configuration validation
    print("\n1. Validating HDBSCAN configuration...")
    try:
        config = load_hdbscan_config(config_path)
        validation = validate_hdbscan_config(config)
        
        print(f"Configuration status: {'VALID' if validation['valid'] else 'INVALID'}")
        
        if validation['warnings']:
            print("Configuration warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        if validation['recommendations']:
            print("Configuration recommendations:")
            for rec in validation['recommendations']:
                print(f"  - {rec}")
                
        if validation['errors']:
            print("Configuration errors:")
            for error in validation['errors']:
                print(f"  - {error}")
                
    except Exception as e:
        print(f"Error validating configuration: {e}")
        return
    
    # Example 2: Check configuration and existing UMAP data
    print("\n2. Determining clustering approach based on configuration...")
    
    # First check if UMAP is enabled in config
    try:
        with open(config_path, 'r') as file:
            config_check = yaml.safe_load(file)
        umap_enabled_in_config = config_check.get('umap', {}).get('enabled', True)
    except Exception as e:
        print(f"Warning: Could not read config for UMAP setting: {e}")
        umap_enabled_in_config = True  # Default to enabled if can't read config
    
     # Always run the flexible pipeline - no more dependency on saved UMAP data
    print(f"✓ UMAP is {'enabled' if umap_enabled_in_config else 'disabled'} in config")
    print("Running flexible pipeline that will handle UMAP in-memory...")
    print("(UMAP data will not be saved to CSV files - processed in memory only)")
    
    # Example 3: Flexible pipeline (respects UMAP enabled/disabled setting)
    print("\n3. Running flexible HDBSCAN pipeline...")
    print("   (This will check config to decide whether to use UMAP or not)")
    
    try:
        # Run flexible pipeline that respects UMAP configuration
        results = run_flexible_hdbscan_pipeline(
            data_path=None,  # Will use path from config
            config_path=config_path,
            output_dir=os.path.join(output_dir, "flexible_pipeline")
        )
        
        print("Flexible pipeline completed!")
        if results['umap_enabled']:
            print("  - UMAP dimensionality reduction was applied")
            print(f"  - Original features: {results['pipeline_info']['n_original_features']}")
            print(f"  - Reduced features: {results['pipeline_info']['n_reduced_features']}")
        else:
            print("  - UMAP was disabled - clustering performed on preprocessed features")
            print(f"  - Features used: {results['pipeline_info']['n_reduced_features']}")
        
        print(f"  - Clusters found: {results['pipeline_info']['n_clusters_found']}")
        print(f"  - Total data points: {results['pipeline_info']['total_data_points']}")
        print(f"  - Noise points: {results['pipeline_info']['noise_points']}")
        
    except Exception as e:
        print(f"Error running flexible pipeline: {e}")
        print("This might be due to missing data files or configuration issues.")
    
    # Optional: Only run complete pipeline if specifically requested or for comparison
    # Note: This always uses UMAP regardless of config setting
    run_complete_pipeline = False  # Set to True if you want to compare with UMAP-enabled results
    
    if run_complete_pipeline:
        print("\n4. Running complete UMAP + HDBSCAN pipeline (for comparison)...")
        print("   (This ignores config UMAP setting and always applies UMAP)")
        
        try:
            # Run complete pipeline
            results = run_umap_hdbscan_pipeline(
                data_path=None,  # Will use path from config
                config_path=config_path,
                output_dir=os.path.join(output_dir, "complete_pipeline")
            )
            
            print("Complete pipeline finished!")
            print(f"  - Original features: {results['pipeline_info']['n_original_features']}")
            print(f"  - Reduced features: {results['pipeline_info']['n_reduced_features']}")
            print(f"  - Clusters found: {results['pipeline_info']['n_clusters_found']}")
            print(f"  - Total data points: {results['pipeline_info']['total_data_points']}")
            print(f"  - Noise points: {results['pipeline_info']['noise_points']}")
            
        except Exception as e:
            print(f"Error running complete pipeline: {e}")
            print("This might be due to missing data files or configuration issues.")
    else:
        print("\n4. Complete UMAP + HDBSCAN pipeline skipped")
        print("   (Set run_complete_pipeline = True if you want to compare with UMAP results)")
    
    print(f"\nResults saved to: {output_dir}")
    print("Script completed!")


if __name__ == "__main__":
    main()
