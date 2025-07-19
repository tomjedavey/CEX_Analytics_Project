#!/usr/bin/env python3
"""
Example script demonstrating how to run HDBSCAN clustering without UMAP dimensionality reduction.

This script shows how to configure and run HDBSCAN directly on preprocessed features
by disabling UMAP in the configuration file.

Author: Tom Davey
Date: July 2025
"""

import os
import sys
import yaml

# Add source_code_package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../source_code_package'))

from models.clustering_functionality.HBDSCAN_cluster import run_flexible_hdbscan_pipeline


def main():
    """
    Demonstrates running HDBSCAN without UMAP dimensionality reduction.
    """
    print("HDBSCAN without UMAP - Example Script")
    print("=" * 50)
    
    # Configuration paths
    config_path = os.path.join(os.path.dirname(__file__), 
                              '../../source_code_package/config/config_cluster.yaml')
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), '../../clustering_output/no_umap_example')
    
    print("\nThis example demonstrates running HDBSCAN without UMAP dimensionality reduction.")
    print("The script will automatically check the configuration and guide you through setup.")
    print("\n" + "=" * 50)
    
    # Check current UMAP configuration
    try:
        import yaml
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        umap_enabled = config.get('umap', {}).get('enabled', True)
        
        if umap_enabled:
            print("\n⚠️  UMAP is currently ENABLED in the configuration.")
            print("This script is designed to demonstrate HDBSCAN WITHOUT UMAP.")
            print("\nTo disable UMAP, you need to edit config_cluster.yaml and set:")
            print("umap:")
            print("  enabled: false")
            print("\nWould you like to continue anyway? (UMAP will still be used)")
            print("Or manually edit the config file and run the script again.")
            print("\nContinuing with current configuration...")
        else:
            print("\n✓ UMAP is DISABLED in the configuration.")
            print("This script will run HDBSCAN directly on preprocessed features.")
            
    except Exception as e:
        print(f"\n⚠️  Could not check configuration: {e}")
        print("Proceeding with default behavior...")
    
    print("\n" + "=" * 50)
    
    try:
        # Run the flexible pipeline
        print("\nRunning flexible HDBSCAN pipeline...")
        results = run_flexible_hdbscan_pipeline(
            data_path=None,  # Will use path from config
            config_path=config_path,
            output_dir=output_dir
        )
        
        # Display results
        print("\nResults:")
        print("-" * 30)
        if results['umap_enabled']:
            print("✓ UMAP dimensionality reduction was applied")
            print(f"  Original features: {results['pipeline_info']['n_original_features']}")
            print(f"  Reduced to: {results['pipeline_info']['n_reduced_features']} dimensions")
        else:
            print("✓ UMAP was disabled - clustering performed on preprocessed features")
            print(f"  Features used: {results['pipeline_info']['n_reduced_features']}")
        
        print(f"✓ Clusters found: {results['pipeline_info']['n_clusters_found']}")
        print(f"✓ Total data points: {results['pipeline_info']['total_data_points']}")
        print(f"✓ Noise points: {results['pipeline_info']['noise_points']}")
        print(f"✓ Noise percentage: {(results['pipeline_info']['noise_points'] / results['pipeline_info']['total_data_points'] * 100):.1f}%")
        
        # Show evaluation metrics if available
        if 'evaluation_metrics' in results.get('hdbscan_results', {}):
            metrics = results['hdbscan_results']['evaluation_metrics']
            print("\nClustering Quality Metrics:")
            print("-" * 30)
            if 'silhouette_score' in metrics:
                print(f"✓ Silhouette Score: {metrics['silhouette_score']:.3f}")
            if 'calinski_harabasz_score' in metrics:
                print(f"✓ Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
            if 'davies_bouldin_score' in metrics:
                print(f"✓ Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")
        
        print(f"\n✓ Results saved to: {output_dir}")
        print("\nScript completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure your data file exists at the path specified in config")
        print("2. Check that all required Python packages are installed")
        print("3. Verify the config file format is correct")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
