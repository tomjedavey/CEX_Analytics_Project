#!/usr/bin/env python3
"""
Test script for verifying that HDBSCAN clustering results are saved correctly as datasets
using the cluster_datasets.py functionality.

This script ensures that the clustering pipeline output is saved to disk in the expected format,
and that the saved files contain the correct data and structure.
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile
import shutil

# Add the source_code_package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../source_code_package'))

from source_code_package.data.cluster_datasets import prepare_cluster_datasets

def test_prepare_cluster_datasets():
    """Test that clustering results are saved as datasets correctly using prepare_cluster_datasets."""
    print("=== Testing prepare_cluster_datasets ===")
    # Step 1: Create temporary original data and cluster label files
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a small dummy dataset
        original_data = pd.DataFrame({
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10),
            'feature3': np.random.randint(0, 100, 10)
        })
        cluster_labels = pd.DataFrame({
            'cluster_label': [0, 1, 0, 1, -1, 0, 1, 0, 1, -1]
        })
        original_data_path = os.path.join(temp_dir, 'original_data.csv')
        cluster_labels_path = os.path.join(temp_dir, 'cluster_labels.csv')
        original_data.to_csv(original_data_path, index=False)
        cluster_labels.to_csv(cluster_labels_path, index=False)

        # Step 2: Run prepare_cluster_datasets
        output_dir = os.path.join(temp_dir, 'output')
        result = prepare_cluster_datasets(
            original_data_path=original_data_path,
            clustering_results_path=cluster_labels_path,
            output_directory=output_dir,
            noise_strategy='exclude'
        )
        assert result['success'], f"prepare_cluster_datasets failed: {result.get('error', '')}"

        # Step 3: Check that cluster CSVs and summary files are created
        files = os.listdir(output_dir)
        print(f"Files saved: {files}")
        assert any(f.endswith('.csv') for f in files), "No cluster CSV files saved!"
        assert 'cluster_datasets_summary.json' in files, "Summary JSON not saved!"
        assert 'cluster_datasets_summary.txt' in files, "Summary TXT not saved!"

        # Step 4: Check that at least one cluster file is non-empty and has expected columns
        for f in files:
            if f.endswith('.csv'):
                df = pd.read_csv(os.path.join(output_dir, f))
                assert not df.empty, f"Saved file {f} is empty!"
                assert set(original_data.columns).issubset(df.columns), f"Saved file {f} missing expected columns!"

        print("✅ test_prepare_cluster_datasets passed!")
    finally:
        shutil.rmtree(temp_dir)

def main():
    """Run all tests."""
    try:
        test_prepare_cluster_datasets()
        print("\n=== Summary ===")
        print("✅ Cluster dataset saving test passed!")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
