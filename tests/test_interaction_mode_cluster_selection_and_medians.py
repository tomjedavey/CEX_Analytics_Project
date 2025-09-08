 #!/usr/bin/env python3
"""
Test script for verifying cluster selection and feature median production for interaction mode clustering results.

This script ensures that:
- The cluster selection method works for each clustering result from different datasets
- Feature medians are correctly produced from the selected clusters

It uses the following source files:
- interaction_mode_median_production_source.py
- interaction_mode_median_production_exection.py

Key here - a lot of the file directories etc are hardcoded in the source files (unlike other functionality) in order to dynamically work with whatever datasets present.
"""

import numpy as np
import pandas as pd
from unittest.mock import patch

                                    "CEX_EVENTS": np.random.randint(0, 10, n),
    return pd.DataFrame({
        "DEX_EVENTS": np.random.randint(0, 10, n),
        "CEX_EVENTS": np.random.randint(0, 10, n),
        "DEFI_EVENTS": np.random.randint(0, 10, n),
        "BRIDGE_EVENTS": np.random.randint(0, 10, n)
    })

                                    "DEFI_EVENTS": np.random.randint(0, 10, n),
    return pd.DataFrame({"cluster_label": np.random.choice([0, 1, 2, -1], n)})

@patch("pandas.read_csv")
@patch("os.path.exists")
def test_cluster_selection_and_feature_medians(mock_exists, mock_read_csv):
    # Always return True for file existence
    mock_exists.return_value = True

    # Prepare dummy data for each read_csv call in order
    # 1. main data, 2. main labels, 3. cluster_0 data, 4. cluster_0 labels
    dummy_main = make_dummy_data()
    dummy_main_labels = make_dummy_labels()
    dummy_cluster = make_dummy_data()
    dummy_cluster_labels = make_dummy_labels()
    mock_read_csv.side_effect = [
        dummy_main, dummy_main_labels, dummy_cluster, dummy_cluster_labels
    ]

    from source_code_package.features.interaction_mode_median_production_source import (
        calculate_median_feature_values_for_clusters
    )
    results = calculate_median_feature_values_for_clusters(
        results_dir="unused",  # Path is ignored due to mocking
        min_activity_threshold=0.1,
        min_cluster_size=10,
        output_path="unused"
    )

    # Assert results structure
    assert "datasets" in results
    assert "main" in results["datasets"]
    assert "cluster_0" in results["datasets"]
    for dataset_name in ["main", "cluster_0"]:
        feature_selections = results["datasets"][dataset_name]["feature_selections"]
        for feature in ["DEX_EVENTS", "CEX_EVENTS", "DEFI_EVENTS", "BRIDGE_EVENTS"]:
            assert feature in feature_selections
            sel = feature_selections[feature]
            assert "selected_cluster" in sel
            assert "median_nonzero_value" in sel
            print(f"{dataset_name}: {feature} -> Cluster {sel['selected_cluster']}, Median Nonzero: {sel['median_nonzero_value']}")

    print("✅ test_cluster_selection_and_feature_medians passed!")

if __name__ == "__main__":
    test_cluster_selection_and_feature_medians()
                                    "BRIDGE_EVENTS": np.random.randint(0, 10, n)
                                })

                            def make_dummy_labels(n=60):
                                return pd.DataFrame({"cluster_label": np.random.choice([0, 1, 2, -1], n)})

                            @patch("pandas.read_csv")
                            @patch("os.path.exists")
                            def test_cluster_selection_and_feature_medians(mock_exists, mock_read_csv):
                                # Always return True for file existence
                                mock_exists.return_value = True

                                # Prepare dummy data for each read_csv call in order
                                # 1. main data, 2. main labels, 3. cluster_0 data, 4. cluster_0 labels
                                dummy_main = make_dummy_data()
                                dummy_main_labels = make_dummy_labels()
                                dummy_cluster = make_dummy_data()
                                dummy_cluster_labels = make_dummy_labels()
                                mock_read_csv.side_effect = [
                                    dummy_main, dummy_main_labels, dummy_cluster, dummy_cluster_labels
                                ]

                                from source_code_package.features.interaction_mode_median_production_source import (
                                    calculate_median_feature_values_for_clusters
                                )
                                results = calculate_median_feature_values_for_clusters(
                                    results_dir="unused",  # Path is ignored due to mocking
                                    min_activity_threshold=0.1,
                                    min_cluster_size=10,
                                    output_path="unused"
                                )

                                # Assert results structure
                                assert "datasets" in results
                                assert "main" in results["datasets"]
                                assert "cluster_0" in results["datasets"]
                                for dataset_name in ["main", "cluster_0"]:
                                    feature_selections = results["datasets"][dataset_name]["feature_selections"]
                                    for feature in ["DEX_EVENTS", "CEX_EVENTS", "DEFI_EVENTS", "BRIDGE_EVENTS"]:
                                        assert feature in feature_selections
                                        sel = feature_selections[feature]
                                        assert "selected_cluster" in sel
                                        assert "median_nonzero_value" in sel
                                        print(f"{dataset_name}: {feature} -> Cluster {sel['selected_cluster']}, Median Nonzero: {sel['median_nonzero_value']}")

                                print("✅ test_cluster_selection_and_feature_medians passed!")

                            if __name__ == "__main__":
                                test_cluster_selection_and_feature_medians()
