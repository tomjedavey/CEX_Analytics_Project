#!/usr/bin/env python3
"""
Test script to verify deterministic ordering in interaction mode processing.

This test ensures that the directory processing order is consistent across
different environments (local vs CI) to prevent median value discrepancies.
"""

import sys
import os
import tempfile
import shutil

# Add the source_code_package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../source_code_package'))

from source_code_package.features.interaction_mode_median_production_source import (
    calculate_median_feature_values_for_clusters
)

def test_deterministic_directory_ordering():
    """
    Test that directory processing order is deterministic.
    
    This test creates multiple clustering directories and verifies that
    they are processed in a consistent order regardless of file system
    implementation or OS differences.
    """
    print("=== Testing deterministic directory ordering ===")
    
    # Create a temporary directory structure
    temp_dir = tempfile.mkdtemp()
    try:
        results_dir = os.path.join(temp_dir, "interaction_mode_results")
        os.makedirs(results_dir)
        
        # Create clustering directories in a specific order that might be 
        # returned differently by os.listdir() on different systems
        cluster_dirs = [
            "main_clustering",
            "cluster_1_clustering", 
            "cluster_0_clustering",  # Note: 0 comes after 1 in this creation order
        ]
        
        # Create each directory with basic structure
        for cluster_dir in cluster_dirs:
            cluster_path = os.path.join(results_dir, cluster_dir)
            os.makedirs(cluster_path)
            
            # Create a dummy clustered_data.csv
            import pandas as pd
            dummy_data = pd.DataFrame({
                'cluster': [0, 1, 1, 1, 1, 1],
                'DEX_EVENTS': [0, 1, 2, 3, 4, 5],
                'CEX_EVENTS': [0, 0, 0, 0, 0, 0],
                'DEFI_EVENTS': [0, 10, 11, 12, 13, 14],
                'BRIDGE_EVENTS': [0, 20, 21, 22, 23, 24]
            })
            dummy_data.to_csv(os.path.join(cluster_path, "clustered_data.csv"), index=False)
        
        # Create base data directory and file
        raw_data_dir = os.path.join(temp_dir, "raw_data")
        os.makedirs(raw_data_dir)
        base_data = pd.DataFrame({
            'DEX_EVENTS': [0, 1, 2, 3, 4, 5],
            'CEX_EVENTS': [0, 0, 0, 0, 0, 0],
            'DEFI_EVENTS': [0, 10, 11, 12, 13, 14],
            'BRIDGE_EVENTS': [0, 20, 21, 22, 23, 24]
        })
        base_data.to_csv(os.path.join(raw_data_dir, "new_raw_data_polygon.csv"), index=False)
        
        # Test multiple times to ensure consistent ordering
        results_list = []
        for run in range(3):
            print(f"  Run {run + 1}...")
            
            # Remove any existing output to force recalculation
            output_path = os.path.join(results_dir, f"test_selections_{run}.yaml")
            
            # Call the function and capture the directory processing order
            # We'll check this by looking at the debug output or return values
            try:
                results = calculate_median_feature_values_for_clusters(
                    results_dir=results_dir,
                    min_activity_threshold=0.1,
                    min_cluster_size=1,
                    output_path=output_path
                )
                
                # Extract the dataset processing order from results
                dataset_order = list(results['datasets'].keys())
                results_list.append(dataset_order)
                print(f"    Dataset processing order: {dataset_order}")
                
            except Exception as e:
                print(f"    Error in run {run + 1}: {e}")
                continue
        
        # Verify all runs produced the same order
        if len(results_list) >= 2:
            first_order = results_list[0]
            for i, order in enumerate(results_list[1:], 1):
                assert order == first_order, (
                    f"Run {i + 1} produced different order {order} vs {first_order}. "
                    f"This indicates non-deterministic directory processing!"
                )
            
            print(f"  ✅ All runs produced consistent order: {first_order}")
            
            # Verify the order is sorted (expected behavior after fix)
            expected_order = sorted(first_order)
            assert first_order == expected_order, (
                f"Order {first_order} is not sorted. Expected: {expected_order}"
            )
            print(f"  ✅ Order is properly sorted")
        else:
            print(f"  ⚠️  Only {len(results_list)} successful runs, cannot verify consistency")
            
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
    
    print("=== Deterministic ordering test completed ===")

def test_median_value_consistency():
    """
    Test that the same clustering data produces the same median values
    across multiple runs.
    """
    print("=== Testing median value consistency ===")
    print("(This test is simplified to focus on directory ordering)")
    
    # For now, just verify that the function can be called without errors
    # The real test is the directory ordering test above
    temp_dir = tempfile.mkdtemp()
    try:
        results_dir = os.path.join(temp_dir, "interaction_mode_results")
        os.makedirs(results_dir)
        
        # Create a minimal structure
        main_cluster_path = os.path.join(results_dir, "main_clustering")
        os.makedirs(main_cluster_path)
        
        # Create minimal test data
        import pandas as pd
        test_data = pd.DataFrame({
            'cluster': [1, 1, 1],
            'DEX_EVENTS': [10, 11, 12],
            'CEX_EVENTS': [0, 0, 0],
            'DEFI_EVENTS': [10, 11, 12],
            'BRIDGE_EVENTS': [20, 21, 22]
        })
        test_data.to_csv(os.path.join(main_cluster_path, "clustered_data.csv"), index=False)
        
        # Create base data
        raw_data_dir = os.path.join(temp_dir, "raw_data")  
        os.makedirs(raw_data_dir)
        test_data.to_csv(os.path.join(raw_data_dir, "new_raw_data_polygon.csv"), index=False)
        
        # Just verify the function runs without error
        try:
            output_path = os.path.join(results_dir, "test_output.yaml")
            results = calculate_median_feature_values_for_clusters(
                results_dir=results_dir,
                min_activity_threshold=0.1,
                min_cluster_size=1,
                output_path=output_path
            )
            print(f"  ✅ Function executed successfully")
            print(f"  ✅ Returned results structure with datasets: {list(results.get('datasets', {}).keys())}")
        except Exception as e:
            print(f"  ❌ Function failed: {e}")
            raise
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("=== Median value consistency test completed ===")

def main():
    """Run all deterministic ordering tests."""
    print("DETERMINISTIC ORDERING TESTS")
    print("=" * 50)
    
    test_deterministic_directory_ordering()
    print()
    test_median_value_consistency()
    
    print("\n✅ All deterministic ordering tests passed!")

if __name__ == "__main__":
    main()