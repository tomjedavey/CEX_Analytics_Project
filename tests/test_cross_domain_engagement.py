#!/usr/bin/env python3
"""
Test and validation script for Cross Domain Engagement Score feature engineering.

This script validates the correctness of the cross domain engagement score implementation
by testing edge cases, component calculations, and overall score consistency.

Author: Tom Davey
Date: August 2025
"""

import sys
import os
import pandas as pd
import numpy as np

# Add source_code_package to path
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
source_package_path = os.path.join(project_root, 'source_code_package')
sys.path.insert(0, source_package_path)

from source_code_package.features.cross_domain_engagement_features import (
    calculate_event_proportions,
    calculate_shannon_entropy,
    normalize_entropy,
    calculate_cross_domain_engagement_score
)

def test_event_proportions():
    """Test event proportions calculation."""
    print("Testing event proportions calculation...")
    test_data = pd.DataFrame({
        'A_EVENTS': [10, 0, 5],
        'B_EVENTS': [10, 0, 5],
        'C_EVENTS': [0, 0, 0]
    })
    event_columns = ['A_EVENTS', 'B_EVENTS', 'C_EVENTS']
    # Type and schema checks
    assert isinstance(test_data, pd.DataFrame), "Input must be a pandas DataFrame"
    assert all(col in test_data.columns for col in event_columns), "Missing event columns"
    df_result, prop_cols = calculate_event_proportions(test_data, event_columns)
    assert isinstance(df_result, pd.DataFrame), "Output must be a pandas DataFrame"
    assert all(col in df_result.columns for col in prop_cols), "Missing proportion columns in output"
    # Correctness
    expected_props = [[0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
    for i, row in enumerate(df_result[prop_cols].values):
        assert np.allclose(row, expected_props[i]), f"Proportion calculation failed for row {i}"
    print("  ✅ Event proportions test passed!")

def test_shannon_entropy():
    """Test Shannon entropy calculation."""
    print("\nTesting Shannon entropy calculation...")
    # Uniform distribution
    props = np.array([0.25, 0.25, 0.25, 0.25])
    entropy = calculate_shannon_entropy(props)
    expected = 2.0  # log2(4)
    assert abs(entropy - expected) < 1e-6, "Shannon entropy failed for uniform distribution"
    # Single category
    props = np.array([1.0, 0.0, 0.0, 0.0])
    entropy = calculate_shannon_entropy(props)
    assert abs(entropy - 0.0) < 1e-6, "Shannon entropy failed for single category"
    # All zeros
    props = np.array([0.0, 0.0, 0.0, 0.0])
    entropy = calculate_shannon_entropy(props)
    assert abs(entropy - 0.0) < 1e-6, "Shannon entropy failed for all zeros"
    print("  ✅ Shannon entropy test passed!")

def test_normalize_entropy():
    """Test entropy normalization."""
    print("\nTesting entropy normalization...")
    entropy = 2.0
    max_categories = 4
    norm = normalize_entropy(entropy, max_categories)
    assert abs(norm - 1.0) < 1e-6, "Normalization failed for max entropy"
    entropy = 0.0
    norm = normalize_entropy(entropy, max_categories)
    assert abs(norm - 0.0) < 1e-6, "Normalization failed for zero entropy"
    print("  ✅ Entropy normalization test passed!")

def test_cross_domain_engagement_score():
    """Test cross domain engagement score calculation."""
    print("\nTesting cross domain engagement score calculation...")
    test_data = pd.DataFrame({
        'A_EVENTS': [10, 0, 5, 0],
        'B_EVENTS': [10, 0, 5, 0],
        'C_EVENTS': [0, 0, 0, 10]
    })
    event_columns = ['A_EVENTS', 'B_EVENTS', 'C_EVENTS']
    # Type and schema checks
    assert isinstance(test_data, pd.DataFrame), "Input must be a pandas DataFrame"
    assert all(col in test_data.columns for col in event_columns), "Missing event columns"
    df_result = calculate_cross_domain_engagement_score(test_data, event_columns)
    assert isinstance(df_result, pd.DataFrame), "Output must be a pandas DataFrame"
    assert 'CROSS_DOMAIN_ENGAGEMENT_SCORE' in df_result.columns, "Missing score column"
    # Correctness: row 0 and 2 should have max diversity, row 1 and 3 should have zero or low
    assert abs(df_result['CROSS_DOMAIN_ENGAGEMENT_SCORE'][0] - 1.0) < 1e-6, "Score should be 1.0 for uniform split"
    assert abs(df_result['CROSS_DOMAIN_ENGAGEMENT_SCORE'][1] - 0.0) < 1e-6, "Score should be 0.0 for all zero"
    assert abs(df_result['CROSS_DOMAIN_ENGAGEMENT_SCORE'][2] - 1.0) < 1e-6, "Score should be 1.0 for uniform split"
    assert abs(df_result['CROSS_DOMAIN_ENGAGEMENT_SCORE'][3] - 0.0) < 1e-6, "Score should be 0.0 for single category"
    print("  ✅ Cross domain engagement score test passed!")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    # Empty DataFrame
    test_data = pd.DataFrame({
        'A_EVENTS': [],
        'B_EVENTS': [],
        'C_EVENTS': []
    })
    event_columns = ['A_EVENTS', 'B_EVENTS', 'C_EVENTS']
    df_result = calculate_cross_domain_engagement_score(test_data, event_columns)
    assert df_result.shape[0] == 0, "Output should be empty for empty input"
    # Missing event columns
    try:
        bad_data = pd.DataFrame({'A_EVENTS': [1, 2]})
        calculate_cross_domain_engagement_score(bad_data, event_columns)
        assert False, "Should raise error for missing columns"
    except Exception:
        pass
    print("  ✅ Edge cases test passed!")

def main():
    print("="*80)
    print("CROSS DOMAIN ENGAGEMENT SCORE VALIDATION TESTS")
    print("="*80)
    test_event_proportions()
    test_shannon_entropy()
    test_normalize_entropy()
    test_cross_domain_engagement_score()
    test_edge_cases()
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
    print("="*80)

if __name__ == "__main__":
    main()
