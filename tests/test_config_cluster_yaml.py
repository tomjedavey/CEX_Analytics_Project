#!/usr/bin/env python3
"""
Test script for validating config_cluster.yaml configuration for UMAP, HDBSCAN, and preprocessing.

This script loads the configuration and runs validation functions to ensure all sections are present and consistent.
"""

import sys
import os

# Add the source_code_package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../source_code_package'))

from source_code_package.models.clustering_functionality.UMAP_dim_reduction import load_umap_config, validate_feature_consistency
from source_code_package.models.clustering_functionality.HBDSCAN_cluster import load_hdbscan_config, validate_hdbscan_config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../source_code_package/config/config_cluster.yaml')

def test_umap_config_loading():
    """Test loading UMAP config from YAML."""
    config = load_umap_config(CONFIG_PATH)
    assert isinstance(config, dict), "UMAP config should be a dictionary"
    assert 'n_neighbors' in config or 'include_columns' in config, "UMAP config missing expected keys"
    print("UMAP config loaded and contains expected keys.")

def test_umap_feature_consistency():
    """Test validation of UMAP and preprocessing feature consistency."""
    results = validate_feature_consistency(CONFIG_PATH)
    assert isinstance(results, dict), "Validation results should be a dictionary"
    print("UMAP feature consistency validation results:", results)
    assert 'warnings' in results and 'recommendations' in results, "Validation results missing keys"

def test_hdbscan_config_loading():
    """Test loading HDBSCAN config from YAML."""
    config = load_hdbscan_config(CONFIG_PATH)
    assert isinstance(config, dict), "HDBSCAN config should be a dictionary"
    assert 'hdbscan' in config, "HDBSCAN config missing 'hdbscan' section"
    print("HDBSCAN config loaded and contains 'hdbscan' section.")

def test_hdbscan_config_validation():
    """Test validation of HDBSCAN config parameters."""
    config = load_hdbscan_config(CONFIG_PATH)
    results = validate_hdbscan_config(config)
    assert isinstance(results, dict), "Validation results should be a dictionary"
    print("HDBSCAN config validation results:", results)
    assert 'valid' in results, "Validation results missing 'valid' key"

def main():
    test_umap_config_loading()
    test_umap_feature_consistency()
    test_hdbscan_config_loading()
    test_hdbscan_config_validation()
    print("\n✅ All config_cluster.yaml configuration tests passed!")

if __name__ == "__main__":
    main()
