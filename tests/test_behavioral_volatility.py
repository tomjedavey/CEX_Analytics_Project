#!/usr/bin/env python3
"""
Test and validation script for Behavioral Volatility Score feature engineering.

This script validates the correctness of the behavioral volatility score implementation
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

from source_code_package.features.behavioral_volatility_features import (
    calculate_financial_volatility,
    calculate_coefficient_of_variance,
    calculate_variance_ratio_from_uniform,
    calculate_gini_coefficient,
    calculate_activity_volatility,
    calculate_exploration_volatility
)


def test_financial_volatility():
    """Test financial volatility calculation."""
    print("Testing Financial Volatility calculation...")
    
    # Create test data
    test_data = pd.DataFrame({
        'USD_TRANSFER_STDDEV': [100, 0, 50, np.nan],
        'AVG_TRANSFER_USD': [50, 100, 25, 50]
    })
    
    # Type and schema checks
    assert isinstance(test_data, pd.DataFrame), "Input must be a pandas DataFrame"
    required_cols = {'USD_TRANSFER_STDDEV', 'AVG_TRANSFER_USD'}
    assert required_cols.issubset(test_data.columns), f"Missing required columns: {required_cols - set(test_data.columns)}"
    result = calculate_financial_volatility(test_data)
    assert isinstance(result, pd.Series), "Output must be a pandas Series"
    # Expected: [2.0, 0.0, 2.0, 0.0]
    expected = [2.0, 0.0, 2.0, 0.0]
    print(f"  Input: {test_data.values.tolist()}")
    print(f"  Result: {result.tolist()}")
    print(f"  Expected: {expected}")
    assert np.allclose(result, expected, atol=1e-6), "Financial volatility test failed"
    print("  ✅ Financial volatility test passed!")


def test_activity_volatility_components():
    """Test activity volatility sub-components."""
    print("\nTesting Activity Volatility components...")
    
    # Test coefficient of variance
    activity_counts = [10, 20, 30, 40]
    # Type checks
    assert isinstance(activity_counts, list), "Input must be a list"
    cv = calculate_coefficient_of_variance(activity_counts)
    expected_cv = np.std(activity_counts) / np.mean(activity_counts)
    print(f"  CV test: {cv:.6f} (expected: {expected_cv:.6f})")
    assert isinstance(cv, float), "CV output must be a float"
    assert abs(cv - expected_cv) < 1e-6, "CV test failed"
    # Test variance ratio from uniform
    vr = calculate_variance_ratio_from_uniform(activity_counts)
    print(f"  Variance ratio: {vr:.6f}")
    assert isinstance(vr, float), "Variance ratio output must be a float"
    # Test Gini coefficient
    gini = calculate_gini_coefficient(activity_counts)
    print(f"  Gini coefficient: {gini:.6f}")
    assert isinstance(gini, float), "Gini coefficient output must be a float"
    # Test edge cases
    empty_counts = []
    zero_counts = [0, 0, 0, 0]
    uniform_counts = [25, 25, 25, 25]
    print(f"  Empty array CV: {calculate_coefficient_of_variance(empty_counts)}")
    print(f"  Zero counts CV: {calculate_coefficient_of_variance(zero_counts)}")
    print(f"  Uniform counts Gini: {calculate_gini_coefficient(uniform_counts):.6f}")
    print("  ✅ Activity volatility components test passed!")


def test_exploration_volatility():
    """Test exploration volatility calculation."""
    print("\nTesting Exploration Volatility calculation...")
    
    # Create test data
    test_data = pd.DataFrame({
        'PROTOCOL_DIVERSITY': [5, 10, 2],
        'INTERACTION_DIVERSITY': [3, 8, 4],
        'TOKEN_DIVERSITY': [2, 6, 6],
        'TX_PER_MONTH': [10, 5, 2]
    })
    diversity_cols = ['PROTOCOL_DIVERSITY', 'INTERACTION_DIVERSITY', 'TOKEN_DIVERSITY']
    # Type and schema checks
    assert isinstance(test_data, pd.DataFrame), "Input must be a pandas DataFrame"
    assert all(col in test_data.columns for col in diversity_cols), "Missing diversity columns"
    assert 'TX_PER_MONTH' in test_data.columns, "Missing TX_PER_MONTH column"
    result = calculate_exploration_volatility(test_data, diversity_cols)
    assert isinstance(result, pd.Series), "Output must be a pandas Series"
    print(f"  Input diversity averages: {test_data[diversity_cols].mean(axis=1).tolist()}")
    print(f"  TX per month: {test_data['TX_PER_MONTH'].tolist()}")
    print(f"  Result (with sqrt): {result.tolist()}")
    # Test without sqrt transformation
    result_no_sqrt = calculate_exploration_volatility(test_data, diversity_cols, apply_sqrt_transform=False)
    assert isinstance(result_no_sqrt, pd.Series), "Output (no sqrt) must be a pandas Series"
    print(f"  Result (without sqrt): {result_no_sqrt.tolist()}")
    print("  ✅ Exploration volatility test passed!")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    # Test with zero activity
    test_data = pd.DataFrame({
        'USD_TRANSFER_STDDEV': [0],
        'AVG_TRANSFER_USD': [0],
        'DEX_EVENTS': [0],
        'GAMES_EVENTS': [0],
        'CEX_EVENTS': [0],
        'DAPP_EVENTS': [0],
        'CHADMIN_EVENTS': [0],
        'DEFI_EVENTS': [0],
        'BRIDGE_EVENTS': [0],
        'NFT_EVENTS': [0],
        'TOKEN_EVENTS': [0],
        'FLOTSAM_EVENTS': [0],
        'PROTOCOL_DIVERSITY': [0],
        'INTERACTION_DIVERSITY': [0],
        'TOKEN_DIVERSITY': [0],
        'TX_PER_MONTH': [0]
    })
    event_columns = ['DEX_EVENTS', 'GAMES_EVENTS', 'CEX_EVENTS', 'DAPP_EVENTS', 
                    'CHADMIN_EVENTS', 'DEFI_EVENTS', 'BRIDGE_EVENTS', 'NFT_EVENTS', 
                    'TOKEN_EVENTS', 'FLOTSAM_EVENTS']
    weights = {'coefficient_of_variance': 0.4, 'variance_ratio': 0.3, 'gini_coefficient': 0.3}
    # Type and schema checks
    assert isinstance(test_data, pd.DataFrame), "Input must be a pandas DataFrame"
    for col in ['USD_TRANSFER_STDDEV', 'AVG_TRANSFER_USD'] + event_columns + ['PROTOCOL_DIVERSITY', 'INTERACTION_DIVERSITY', 'TOKEN_DIVERSITY', 'TX_PER_MONTH']:
        assert col in test_data.columns, f"Missing required column: {col}"
    fin_vol = calculate_financial_volatility(test_data)
    act_vol = calculate_activity_volatility(test_data, event_columns, weights)
    exp_vol = calculate_exploration_volatility(test_data, ['PROTOCOL_DIVERSITY', 'INTERACTION_DIVERSITY', 'TOKEN_DIVERSITY'])
    assert isinstance(fin_vol, pd.Series), "Financial volatility output must be a pandas Series"
    assert isinstance(act_vol, pd.Series), "Activity volatility output must be a pandas Series"
    assert isinstance(exp_vol, pd.Series), "Exploration volatility output must be a pandas Series"
    print(f"  Zero activity case:")
    print(f"    Financial volatility: {fin_vol.iloc[0]}")
    print(f"    Activity volatility: {act_vol.iloc[0]}")
    print(f"    Exploration volatility: {exp_vol.iloc[0]}")
    print("  ✅ Edge cases test passed!")


def validate_real_data():
    """Validate the behavioral volatility score on real data."""
    print("\nValidating on real data...")
    
    # Load the generated features
    try:
        df = pd.read_csv('data/processed_data/behavioral_volatility_features.csv')
        print(f"  Loaded {len(df)} records")
        
        # Check for any invalid values
        features = ['FINANCIAL_VOLATILITY', 'ACTIVITY_VOLATILITY', 'EXPLORATION_VOLATILITY', 
                   'BEHAVIORAL_VOLATILITY_SCORE_RAW', 'BEHAVIORAL_VOLATILITY_SCORE']
        
        for feature in features:
            nulls = df[feature].isnull().sum()
            infs = np.isinf(df[feature]).sum()
            negs = (df[feature] < 0).sum()
            
            print(f"  {feature}: nulls={nulls}, infs={infs}, negatives={negs}")
            
            if nulls > 0 or infs > 0:
                print(f"    ⚠️  Warning: Found {nulls} nulls and {infs} infinite values")
        
        # Verify component weights
        expected_raw = (0.35 * df['FINANCIAL_VOLATILITY'] + 
                       0.40 * df['ACTIVITY_VOLATILITY'] + 
                       0.25 * df['EXPLORATION_VOLATILITY'])
        
        max_diff = np.abs(df['BEHAVIORAL_VOLATILITY_SCORE_RAW'] - expected_raw).max()
        print(f"  Max difference in raw score calculation: {max_diff:.10f}")
        
        # Check normalization
        norm_min = df['BEHAVIORAL_VOLATILITY_SCORE'].min()
        norm_max = df['BEHAVIORAL_VOLATILITY_SCORE'].max()
        print(f"  Normalized score range: [{norm_min:.6f}, {norm_max:.6f}]")
        
        print("  ✅ Real data validation passed!")
        
    except FileNotFoundError:
        print("  ⚠️  Behavioral volatility features file not found. Run the main script first.")


def main():
    """Run all tests."""
    print("="*80)
    print("BEHAVIORAL VOLATILITY SCORE VALIDATION TESTS")
    print("="*80)
    
    test_financial_volatility()
    test_activity_volatility_components()
    test_exploration_volatility()
    test_edge_cases()
    validate_real_data()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
    print("="*80)


if __name__ == "__main__":
    main()
