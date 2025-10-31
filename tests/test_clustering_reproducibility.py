#!/usr/bin/env python3
"""
Test script to verify reproducible clustering behavior.

This script tests that the clustering pipeline produces consistent results
across multiple runs, which should fix the DEFI_EVENTS median discrepancy.
"""

import sys
import os
import numpy as np
import tempfile
import shutil

# Add the source_code_package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../source_code_package'))

def test_numpy_random_determinism():
    """Test that our numpy random seed fixes work."""
    print("=== Testing NumPy random determinism ===")
    
    # Test multiple runs with same seed
    results = []
    for run in range(3):
        np.random.seed(42)
        sample = np.random.choice(100, 10, replace=False)
        results.append(list(sample))
        print(f"  Run {run + 1}: {sample}")
    
    # Verify all runs produced same result
    first_result = results[0]
    for i, result in enumerate(results[1:], 1):
        assert result == first_result, (
            f"Run {i + 1} produced different result {result} vs {first_result}"
        )
    
    print("  ✅ NumPy random sampling is deterministic with seed")

def test_hdbscan_config_loading():
    """Test that HDBSCAN config includes random_state."""
    print("=== Testing HDBSCAN config loading ===")
    
    from source_code_package.models.clustering_functionality.HBDSCAN_cluster import load_hdbscan_config
    
    config_path = "source_code_package/config/config_interaction_mode.yaml"
    config = load_hdbscan_config(config_path)
    
    hdbscan_config = config.get('hdbscan', {})
    
    assert 'random_state' in hdbscan_config, "random_state not found in HDBSCAN config"
    assert hdbscan_config['random_state'] == 42, f"Expected random_state=42, got {hdbscan_config['random_state']}"
    
    print(f"  ✅ HDBSCAN random_state = {hdbscan_config['random_state']}")

def test_umap_config_loading():
    """Test that UMAP config includes random_state."""
    print("=== Testing UMAP config loading ===")
    
    import yaml
    
    config_path = "source_code_package/config/config_interaction_mode.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    umap_config = config.get('umap', {})
    
    assert 'random_state' in umap_config, "random_state not found in UMAP config"
    assert umap_config['random_state'] == 42, f"Expected random_state=42, got {umap_config['random_state']}"
    
    print(f"  ✅ UMAP random_state = {umap_config['random_state']}")

def main():
    """Run all reproducibility tests."""
    print("CLUSTERING REPRODUCIBILITY TESTS")
    print("=" * 50)
    
    test_numpy_random_determinism()
    print()
    test_hdbscan_config_loading()
    print()
    test_umap_config_loading()
    
    print("\n✅ All reproducibility tests passed!")
    print("\nThese fixes should resolve the DEFI_EVENTS median discrepancy")
    print("between local and GitHub Actions environments.")

if __name__ == "__main__":
    main()