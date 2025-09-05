#!/usr/bin/env python3
"""
Test script for validating config_interaction_mode.yaml configuration for interaction mode feature engineering and clustering.

This script loads the configuration and runs validation functions to ensure all sections are present and consistent, similar to the config_cluster.yaml test.
"""

import sys
import os
# Add the source_code_package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../source_code_package'))
from source_code_package.features.interaction_mode_median_production_source import load_config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../source_code_package/config/config_interaction_mode.yaml')

def test_interaction_mode_config_loading():
    """Test loading interaction mode config from YAML."""
    config = load_config(CONFIG_PATH)
    assert isinstance(config, dict), "Interaction mode config should be a dictionary"
    assert 'data' in config, "Interaction mode config missing 'data' section"
    assert 'preprocessing' in config, "Interaction mode config missing 'preprocessing' section"
    assert 'umap' in config, "Interaction mode config missing 'umap' section"
    print("Interaction mode config loaded and contains expected keys.")

def main():
    test_interaction_mode_config_loading()
    print("\n✅ config_interaction_mode.yaml configuration test passed!")

if __name__ == "__main__":
    main()
