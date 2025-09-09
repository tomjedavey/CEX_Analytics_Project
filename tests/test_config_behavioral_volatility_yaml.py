#!/usr/bin/env python3
"""
Test script for validating config_behavioral_volatility.yaml configuration.

Checks structure, required keys, and prints config for manual inspection.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../source_code_package'))
from source_code_package.features.behavioural_volatility_features import load_config

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../source_code_package/config/config_behavioral_volatility.yaml')

def test_behavioral_volatility_config_loading():
    import yaml
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    assert isinstance(config, dict), "Config should be a dictionary"
    assert 'data' in config, "Missing 'data' section"
    assert 'features' in config, "Missing 'features' section"
    print("Behavioral Volatility config loaded:", config)

if __name__ == "__main__":
    test_behavioral_volatility_config_loading()
    print("\n✅ config_behavioral_volatility.yaml configuration test passed!")
