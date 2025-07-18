#!/usr/bin/env python3
"""
Parameter tuning script for HDBSCAN clustering.

This script tests different parameter combinations to find optimal settings for HDBSCAN clustering.
It should be run from the project root directory:

Usage:
    cd /path/to/MLProject1
    python3 source_code_package/models/clustering_functionality/tune_hdbscan_parameters.py

The script will:
1. Load UMAP-reduced data from data/processed_data/umap_reduced_data.csv
2. Test various parameter combinations
3. Evaluate clustering quality using silhouette scores
4. Provide recommendations for optimal parameters

Author: Tom Davey
Date: July 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add source_code_package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from models.clustering_functionality.HBDSCAN_cluster import apply_hdbscan_clustering, evaluate_clustering_quality

def test_hdbscan_parameters(data, parameter_combinations):
    """Test different HDBSCAN parameter combinations."""
    
    import hdbscan
    results = []
    
    for i, params in enumerate(parameter_combinations):
        print(f"\nTesting combination {i+1}/{len(parameter_combinations)}: {params}")
        
        try:
            # Apply HDBSCAN directly with test parameters
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=params['min_cluster_size'],
                min_samples=params['min_samples'],
                metric=params.get('metric', 'euclidean'),
                alpha=1.0,
                algorithm='best',
                leaf_size=40,
                cluster_selection_method='eom',
                allow_single_cluster=False,
                prediction_data=True,
                core_dist_n_jobs=-1,
                gen_min_span_tree=False,
                approx_min_span_tree=True,
                match_reference_implementation=False
            )
            
            labels = clusterer.fit_predict(data)
            
            # Calculate basic stats
            n_clusters = len(np.unique(labels[labels >= 0]))
            n_noise = np.sum(labels == -1)
            noise_pct = (n_noise / len(labels)) * 100
            
            # Quick evaluation (skip expensive metrics for many clusters)
            if n_clusters <= 50 and n_clusters >= 2:
                try:
                    evaluation = evaluate_clustering_quality(data, labels, clusterer)
                    silhouette = evaluation.get('silhouette_score', 'N/A')
                except:
                    silhouette = 'Error'
            else:
                silhouette = 'Skipped'
            
            result = {
                'min_cluster_size': params['min_cluster_size'],
                'min_samples': params['min_samples'],
                'metric': params.get('metric', 'euclidean'),
                'n_clusters': n_clusters,
                'noise_percentage': noise_pct,
                'silhouette_score': silhouette
            }
            
            results.append(result)
            
            print(f"  Results: {n_clusters} clusters, {noise_pct:.1f}% noise, silhouette: {silhouette}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results.append({
                'min_cluster_size': params['min_cluster_size'],
                'min_samples': params['min_samples'],
                'metric': params.get('metric', 'euclidean'),
                'n_clusters': 'Error',
                'noise_percentage': 'Error',
                'silhouette_score': 'Error'
            })
    
    return results

def main():
    """Main function to run parameter tuning."""
    
    print("HDBSCAN Parameter Tuning for Large Dataset")
    print("=" * 50)
    
    # Load your UMAP data - try multiple possible paths
    possible_data_paths = [
        # From clustering_functionality folder
        os.path.join(os.path.dirname(__file__), '../../../data/processed_data/umap_reduced_data.csv'),
        # From project root
        os.path.join(os.getcwd(), 'data/processed_data/umap_reduced_data.csv'),
        # Relative to script location
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../data/processed_data/umap_reduced_data.csv')
    ]
    
    umap_data_path = None
    for path in possible_data_paths:
        if os.path.exists(path):
            umap_data_path = path
            break
    
    if umap_data_path is None:
        print("Error: UMAP data not found. Tried the following paths:")
        for path in possible_data_paths:
            print(f"  - {path}")
        print("\nPlease ensure you're running from the project root directory and that umap_reduced_data.csv exists.")
        return
    
    print(f"Loading UMAP data from: {umap_data_path}")
    umap_df = pd.read_csv(umap_data_path)
    umap_data = umap_df.select_dtypes(include=[np.number]).values
    
    print(f"Data shape: {umap_data.shape}")
    
    # Test parameters optimized for your dataset size (~20K points)
    parameter_combinations = [
        # Conservative approach - fewer, larger clusters
        {'min_cluster_size': 200, 'min_samples': 50},
        {'min_cluster_size': 150, 'min_samples': 40},
        {'min_cluster_size': 100, 'min_samples': 30},
        
        # Moderate approach
        {'min_cluster_size': 100, 'min_samples': 25},
        {'min_cluster_size': 75, 'min_samples': 20},
        {'min_cluster_size': 50, 'min_samples': 15},
        
        # More sensitive approach
        {'min_cluster_size': 50, 'min_samples': 10},
        {'min_cluster_size': 30, 'min_samples': 8},
        
        # Test different metrics with best size parameters
        {'min_cluster_size': 100, 'min_samples': 25, 'metric': 'manhattan'},
        {'min_cluster_size': 100, 'min_samples': 25, 'metric': 'cosine'},
    ]
    
    print(f"\nTesting {len(parameter_combinations)} parameter combinations...")
    
    results = test_hdbscan_parameters(umap_data, parameter_combinations)
    
    # Display results
    print("\n" + "=" * 80)
    print("PARAMETER TUNING RESULTS")
    print("=" * 80)
    print(f"{'MinClusterSize':<15} {'MinSamples':<12} {'Metric':<12} {'Clusters':<10} {'Noise%':<8} {'Silhouette':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['min_cluster_size']:<15} {result['min_samples']:<12} {result['metric']:<12} "
              f"{result['n_clusters']:<10} {result['noise_percentage']:<8} {result['silhouette_score']:<12}")
    
    # Find best results
    valid_results = [r for r in results if isinstance(r['n_clusters'], int) and 
                     isinstance(r['silhouette_score'], (int, float)) and 
                     r['silhouette_score'] != 'Skipped']
    
    if valid_results:
        print("\n" + "=" * 50)
        print("RECOMMENDATIONS")
        print("=" * 50)
        
        # Best by silhouette score
        best_silhouette = max(valid_results, key=lambda x: x['silhouette_score'] if isinstance(x['silhouette_score'], (int, float)) else -1)
        print(f"Best Silhouette Score: {best_silhouette['silhouette_score']:.3f}")
        print(f"  Parameters: min_cluster_size={best_silhouette['min_cluster_size']}, min_samples={best_silhouette['min_samples']}")
        print(f"  Results: {best_silhouette['n_clusters']} clusters, {best_silhouette['noise_percentage']:.1f}% noise")
        
        # Best balance (reasonable number of clusters, low noise)
        balanced_results = [r for r in valid_results if 5 <= r['n_clusters'] <= 20 and r['noise_percentage'] < 15]
        if balanced_results:
            best_balanced = max(balanced_results, key=lambda x: x['silhouette_score'] if isinstance(x['silhouette_score'], (int, float)) else -1)
            print(f"\nBest Balanced Result:")
            print(f"  Parameters: min_cluster_size={best_balanced['min_cluster_size']}, min_samples={best_balanced['min_samples']}")
            print(f"  Results: {best_balanced['n_clusters']} clusters, {best_balanced['noise_percentage']:.1f}% noise")
            print(f"  Silhouette Score: {best_balanced['silhouette_score']:.3f}")
        
        # Recommendation for config file
        recommended = best_balanced if balanced_results else best_silhouette
        print(f"\n" + "=" * 50)
        print("RECOMMENDED CONFIG UPDATE:")
        print("=" * 50)
        print("hdbscan:")
        print(f"  min_cluster_size: {recommended['min_cluster_size']}")
        print(f"  min_samples: {recommended['min_samples']}")
        print(f"  metric: \"{recommended['metric']}\"")
        print("  # ... other parameters unchanged")

if __name__ == "__main__":
    main()
