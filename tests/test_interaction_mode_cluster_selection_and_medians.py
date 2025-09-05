#!/usr/bin/env python3
"""
Test script for verifying cluster selection and feature median production for interaction mode clustering results.

This script ensures that:
- The cluster selection method works for each clustering result from different datasets
- Feature medians are correctly produced from the selected clusters

It uses the following source files:
- interaction_mode_median_production_source.py
- interaction_mode_median_production_exection.py
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile
import shutil

# Add the source_code_package to the path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '../source_code_package'))

# Import the relevant functions (update import paths as needed)

# Use the actual source file and functions
from source_code_package.features.interaction_mode_median_production_source import (
    select_strongest_cluster_for_feature,
    calculate_median_feature_values_for_clusters
)

def test_cluster_selection_and_feature_medians():
    """Test cluster selection and feature median calculation using the real pipeline logic."""
    print("=== Testing cluster selection and feature median calculation ===")
    temp_dir = tempfile.mkdtemp()
    try:
        # Step 1: Create a dummy clustering result directory structure
        results_dir = os.path.join(temp_dir, "interaction_mode_results")
        os.makedirs(results_dir, exist_ok=True)

        # Create two fake clustering result folders: main_clustering and cluster_0_clustering
        for dataset_name in ["main", "cluster_0"]:
            folder = os.path.join(results_dir, f"{dataset_name}_clustering" if dataset_name != "main" else "main_clustering")
            os.makedirs(folder, exist_ok=True)
            # Create dummy cluster labels
            n = 60
            cluster_labels = pd.DataFrame({"cluster_label": np.random.choice([0, 1, 2, -1], n)})
            cluster_labels.to_csv(os.path.join(folder, "cluster_labels.csv"), index=False)
            # Create dummy data file for 'main' dataset
            if dataset_name == "main":
                df = pd.DataFrame({
                    "DEX_EVENTS": np.random.randint(0, 10, n),
                    "CEX_EVENTS": np.random.randint(0, 10, n),
                    "DEFI_EVENTS": np.random.randint(0, 10, n),
                    "BRIDGE_EVENTS": np.random.randint(0, 10, n)
                })
                raw_data_path = os.path.join(temp_dir, "new_raw_data_polygon.csv")
                df.to_csv(raw_data_path, index=False)
                # Patch the expected path for the test
                os.makedirs(os.path.dirname("data/raw_data/"), exist_ok=True)
                df.to_csv("data/raw_data/new_raw_data_polygon.csv", index=False)
            else:
                # For cluster datasets, create a dummy file in the expected location
                df = pd.DataFrame({
                    "DEX_EVENTS": np.random.randint(0, 10, n),
                    "CEX_EVENTS": np.random.randint(0, 10, n),
                    "DEFI_EVENTS": np.random.randint(0, 10, n),
                    "BRIDGE_EVENTS": np.random.randint(0, 10, n)
                })
                cluster_data_path = f"data/processed_data/cluster_datasets/new_raw_data_polygon_cluster_0.csv"
                os.makedirs(os.path.dirname(cluster_data_path), exist_ok=True)
                df.to_csv(cluster_data_path, index=False)

        # Step 2: Run the real median calculation pipeline
        output_path = os.path.join(results_dir, "test_output.yaml")
        results = calculate_median_feature_values_for_clusters(
            results_dir=results_dir,
            min_activity_threshold=0.1,
            min_cluster_size=10,
            output_path=output_path
        )

        # Step 3: Check that results contain expected structure and medians
        assert 'datasets' in results, "Results missing 'datasets' key"
        for dataset_name in ["main", "cluster_0"]:
            assert dataset_name in results['datasets'], f"Missing dataset {dataset_name} in results"
            feature_selections = results['datasets'][dataset_name]['feature_selections']
            for feature in ["DEX_EVENTS", "CEX_EVENTS", "DEFI_EVENTS", "BRIDGE_EVENTS"]:
                assert feature in feature_selections, f"Missing feature {feature} in selections for {dataset_name}"
                sel = feature_selections[feature]
                assert 'selected_cluster' in sel, f"No selected_cluster for {feature}"
                assert 'median_nonzero_value' in sel, f"No median_nonzero_value for {feature}"
                print(f"{dataset_name}: {feature} -> Cluster {sel['selected_cluster']}, Median Nonzero: {sel['median_nonzero_value']}")

        print("✅ test_cluster_selection_and_feature_medians passed!")
    finally:
        shutil.rmtree(temp_dir)

def main():
    """Run all tests."""
    try:
        test_cluster_selection_and_feature_medians()
        print("\n=== Summary ===")
        print("✅ Cluster selection and feature median production test passed!")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
