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
from typing import Optional

# Add source_code_package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../source_code_package'))

# Import our custom clustering functionality
from models.clustering_functionality.HBDSCAN_cluster import (
    hdbscan_clustering_pipeline,
    run_umap_hdbscan_pipeline,
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
    
    # Example 2: Check if UMAP-reduced data exists
    print("\n2. Checking for existing UMAP-reduced data...")
    umap_data_path = os.path.join(os.path.dirname(__file__), 
                                 '../../data/processed_data/umap_reduced_data.csv')
    
    if os.path.exists(umap_data_path):
        print(f"Found UMAP data at: {umap_data_path}")
        print("Running HDBSCAN clustering on existing UMAP data...")
        
        try:
            # Load UMAP data
            umap_df = pd.read_csv(umap_data_path)
            print(f"Loaded UMAP data shape: {umap_df.shape}")
            
            # Convert to numpy array (assuming all columns are numeric)
            umap_data = umap_df.select_dtypes(include=[np.number]).values
            
            # Run HDBSCAN clustering
            results = hdbscan_clustering_pipeline(
                umap_data=umap_data,
                config_path=config_path,
                evaluate_quality=True,
                create_visualizations=True,
                save_results=True,
                output_dir=os.path.join(output_dir, "hdbscan_only")
            )
            
            print(f"Clustering completed!")
            print(f"  - Found {results['cluster_info']['n_clusters']} clusters")
            print(f"  - Noise points: {results['cluster_info']['n_noise_points']}")
            print(f"  - Noise percentage: {results['cluster_info']['noise_percentage']:.1f}%")
            
            if 'evaluation_metrics' in results:
                metrics = results['evaluation_metrics']
                if 'silhouette_score' in metrics:
                    print(f"  - Silhouette Score: {metrics['silhouette_score']:.3f}")
                    
        except Exception as e:
            print(f"Error running HDBSCAN on existing data: {e}")
    
    else:
        print("No existing UMAP data found. Will run complete pipeline.")
        
        # Example 3: Complete UMAP + HDBSCAN pipeline
        print("\n3. Running complete UMAP + HDBSCAN pipeline...")
        
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
    
    print(f"\nResults saved to: {output_dir}")
    print("Script completed!")


if __name__ == "__main__":
    main()
