#!/usr/bin/env python3
"""
Example script demonstrating how to run HDBSCAN clustering without UMAP dimensionality reduction.

This script shows how to force direct HDBSCAN clustering on preprocessed features
using the simplified clustering interface.

Author: Tom Davey
Date: July 2025
"""

import os
import sys

# Add source_code_package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../source_code_package'))

from models.clustering_functionality.simplified_clustering import run_clustering_pipeline


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
    print("The script uses force_umap=False to override any config setting.")
    print("\n" + "=" * 50)
    
    try:
        # Run the pipeline with UMAP forced OFF
        print("\nRunning HDBSCAN pipeline without UMAP...")
        results = run_clustering_pipeline(
            config_path=config_path,
            data_path=None,  # Will use path from config
            output_dir=output_dir,
            force_umap=False  # Force direct HDBSCAN, ignore config UMAP setting
        )
        
        if not results['success']:
            print(f"\n❌ Pipeline failed: {results['error_message']}")
            return 1
        
        # Display results
        print("\n✅ Pipeline completed successfully!")
        print("\nResults:")
        print("-" * 30)
        
        if results['umap_enabled']:
            print("⚠️  Warning: UMAP was still used despite force_umap=False")
            print("    This might indicate a configuration issue.")
        else:
            print("✓ UMAP was disabled - clustering performed on preprocessed features")
        
        pipeline_info = results['pipeline_info']
        print(f"✓ Features used: {pipeline_info['n_reduced_features']}")
        print(f"✓ Clusters found: {pipeline_info['n_clusters_found']}")
        print(f"✓ Total data points: {pipeline_info['total_data_points']:,}")
        print(f"✓ Noise points: {pipeline_info['noise_points']:,}")
        
        if pipeline_info['total_data_points'] > 0:
            noise_percentage = (pipeline_info['noise_points'] / pipeline_info['total_data_points']) * 100
            print(f"✓ Noise percentage: {noise_percentage:.1f}%")
        
        # Show evaluation metrics if available
        hdbscan_results = results.get('hdbscan_results', {})
        if 'evaluation_metrics' in hdbscan_results:
            metrics = hdbscan_results['evaluation_metrics']
            print("\nClustering Quality Metrics:")
            print("-" * 30)
            if 'silhouette_score' in metrics:
                print(f"✓ Silhouette Score: {metrics['silhouette_score']:.3f}")
            if 'calinski_harabasz_score' in metrics:
                print(f"✓ Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
            if 'davies_bouldin_score' in metrics:
                print(f"✓ Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")
        
        print(f"\n✓ Results saved to: {output_dir}")
        print("\n" + "=" * 50)
        print("💡 Key Points:")
        print("   • This script used force_umap=False to ensure direct HDBSCAN")
        print("   • The config UMAP setting was ignored")
        print("   • Clustering was performed on preprocessed features")
        print("   • This approach is useful when you want to work with")
        print("     the original feature space without dimensionality reduction")
        
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure your data file exists at the path specified in config")
        print("2. Check that all required Python packages are installed")
        print("3. Verify the config file format is correct")
        print("4. Try running: python scripts/clustering/run_simple_clustering.py --validate-only")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
