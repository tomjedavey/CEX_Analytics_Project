#!/usr/bin/env python3
"""
Test and validation script for Revenue Proxy Score feature engineering.

This script validates the correctness of the revenue proxy score implementation
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

from source_code_package.features.revenue_proxy_features import (
    calculate_revenue_score_proxy,
    validate_required_columns,
    handle_missing_values
)

def test_required_columns():
    """Test required columns validation."""
    print("Testing required columns validation...")
    test_data = pd.DataFrame({
        'AVG_TRANSFER_USD': [100, 50],
        'TX_PER_MONTH': [10, 5],
        'DEX_EVENTS': [2, 1],
        'DEFI_EVENTS': [3, 2],
        'BRIDGE_TOTAL_VOLUME_USD': [1000, 500]
    })
    required = ['AVG_TRANSFER_USD', 'TX_PER_MONTH', 'DEX_EVENTS', 'DEFI_EVENTS', 'BRIDGE_TOTAL_VOLUME_USD']
    valid, missing = validate_required_columns(test_data, required)
    assert valid, f"Should be valid, missing: {missing}"
    # Remove a column
    test_data2 = test_data.drop(columns=['DEX_EVENTS'])
    valid, missing = validate_required_columns(test_data2, required)
    assert not valid and 'DEX_EVENTS' in missing, "Should detect missing column"
    print("  ✅ Required columns test passed!")

def test_handle_missing_values():
    """Test missing value handling."""
    print("\nTesting missing value handling...")
    test_data = pd.DataFrame({
        'AVG_TRANSFER_USD': [100, np.nan],
        'TX_PER_MONTH': [10, 5],
        'DEX_EVENTS': [2, 1],
        'DEFI_EVENTS': [3, np.nan],
        'BRIDGE_TOTAL_VOLUME_USD': [1000, 500]
    })
    cols = ['AVG_TRANSFER_USD', 'DEFI_EVENTS']
    filled = handle_missing_values(test_data, cols, fill_method='zero')
    assert filled['AVG_TRANSFER_USD'][1] == 0, "Should fill with zero"
    assert filled['DEFI_EVENTS'][1] == 0, "Should fill with zero"
    filled_mean = handle_missing_values(test_data, cols, fill_method='mean')
    assert np.isclose(filled_mean['AVG_TRANSFER_USD'][1], 100), "Should fill with mean (100)"
    print("  ✅ Missing value handling test passed!")

def test_revenue_score_proxy():
    """Test revenue score proxy calculation."""
    print("\nTesting revenue score proxy calculation...")
    test_data = pd.DataFrame({
        'AVG_TRANSFER_USD': [100, 50, 0],
        'TX_PER_MONTH': [10, 5, 0],
        'DEX_EVENTS': [2, 1, 0],
        'DEFI_EVENTS': [3, 2, 0],
        'BRIDGE_TOTAL_VOLUME_USD': [1000, 500, 0]
    })
    # Type and schema checks
    assert isinstance(test_data, pd.DataFrame), "Input must be a pandas DataFrame"
    required = ['AVG_TRANSFER_USD', 'TX_PER_MONTH', 'DEX_EVENTS', 'DEFI_EVENTS', 'BRIDGE_TOTAL_VOLUME_USD']
    assert all(col in test_data.columns for col in required), "Missing required columns"
    df_result = calculate_revenue_score_proxy(test_data)
    assert isinstance(df_result, pd.DataFrame), "Output must be a pandas DataFrame"
    assert 'REVENUE_SCORE_PROXY' in df_result.columns, "Missing score column"
    # Correctness: check calculation for row 0
    expected = 0.4 * 100 * 10 + 0.35 * (2 + 3) * 100 + 0.25 * 1000
    assert np.isclose(df_result['REVENUE_SCORE_PROXY'][0], expected), "Incorrect score calculation for row 0"
    print("  ✅ Revenue score proxy calculation test passed!")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    # All zeros
    test_data = pd.DataFrame({
        'AVG_TRANSFER_USD': [0],
        'TX_PER_MONTH': [0],
        'DEX_EVENTS': [0],
        'DEFI_EVENTS': [0],
        'BRIDGE_TOTAL_VOLUME_USD': [0]
    })
    df_result = calculate_revenue_score_proxy(test_data)
    assert df_result['REVENUE_SCORE_PROXY'][0] == 0, "Score should be zero for all zero input"
    # Missing column
    try:
        bad_data = test_data.drop(columns=['DEX_EVENTS'])
        calculate_revenue_score_proxy(bad_data)
        assert False, "Should raise error for missing columns"
    except ValueError:
        pass
    print("  ✅ Edge cases test passed!")

def main():
    print("="*80)
    print("REVENUE PROXY SCORE VALIDATION TESTS")
    print("="*80)
    test_required_columns()
    test_handle_missing_values()
    test_revenue_score_proxy()
    test_edge_cases()
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
    print("="*80)

if __name__ == "__main__":
    main()
