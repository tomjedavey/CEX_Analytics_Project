#!/usr/bin/env python3
"""
Test script for validating config_cross_domain_engagement.yaml configuration.

Checks structure, required keys, and prints config for manual inspection.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../source_code_package'))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../source_code_package/config/config_cross_domain_engagement.yaml')

def test_cross_domain_engagement_config_loading():
    import yaml
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    assert isinstance(config, dict), "Config should be a dictionary"
    assert 'data' in config, "Missing 'data' section"
    assert 'features' in config, "Missing 'features' section"
    print("Cross Domain Engagement config loaded:", config)

if __name__ == "__main__":
    test_cross_domain_engagement_config_loading()
    print("\n✅ config_cross_domain_engagement.yaml configuration test passed!")
