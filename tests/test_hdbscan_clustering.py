#!/usr/bin/env python3
"""
Test script for HDBSCAN clustering on UMAP-reduced data.

This script validates the HDBSCAN clustering pipeline, ensuring that clustering is performed correctly on UMAP-reduced features, and that cluster labels and quality metrics are produced as expected.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the source_code_package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../source_code_package'))

from source_code_package.models.clustering_functionality.UMAP_dim_reduction import (
    umap_with_preprocessing
)
from source_code_package.models.clustering_functionality.HBDSCAN_cluster import (
    hdbscan_clustering_pipeline
)

def test_hdbscan_clustering():
    """Test HDBSCAN clustering on UMAP-reduced data."""
    print("=== Testing HDBSCAN Clustering on UMAP-Reduced Data ===")
    # Step 1: Run UMAP preprocessing to get reduced data
    reduced_data, umap_model, preprocessed_data, preprocessing_info = umap_with_preprocessing(
        apply_log_transform=True,
        apply_scaling=True
    )
    assert isinstance(reduced_data, np.ndarray), "Reduced data should be a numpy array"
    assert reduced_data.shape[0] > 0 and reduced_data.shape[1] > 0, "Reduced data should not be empty"
    print(f"Reduced data shape: {reduced_data.shape}")
    # Step 2: Run HDBSCAN clustering
    results = hdbscan_clustering_pipeline(
        umap_data=reduced_data,
        evaluate_quality=True,
        create_visualizations=False,
        save_results=False
    )
    assert 'cluster_labels' in results, "Results should contain cluster_labels"
    cluster_labels = results['cluster_labels']
    assert isinstance(cluster_labels, np.ndarray), "Cluster labels should be a numpy array"
    assert cluster_labels.shape[0] == reduced_data.shape[0], "Cluster labels should match number of data points"
    print(f"Number of clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
    # Step 3: Check quality metrics
    metrics = results.get('evaluation_metrics', {})
    for metric in ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']:
        assert metric in metrics, f"Missing quality metric: {metric}"
        print(f"{metric}: {metrics[metric]}")
    print("✅ test_hdbscan_clustering passed!")


def main():
    """Run all tests."""
    try:
        test_hdbscan_clustering()
        print("\n=== Summary ===")
        print("✅ HDBSCAN clustering test passed!")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
