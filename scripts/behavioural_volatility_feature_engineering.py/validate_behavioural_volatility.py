#!/usr/bin/env python3
"""
Behavioural Volatility Feature Validation and Analysis Script

This script validates the calculated BEHAVIOURAL_VOLATILITY_SCORE feature and provides
basic analysis to help understand the behavioural volatility scores in the context
of user or wallet activity analytics.

Author: Tom Davey
Date: August 2025
"""

import pandas as pd
import numpy as np
import os
import sys

# Adjust the path as needed to import project modules if required
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../source_code_package'))


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """Load the behavioural volatility features and validate the data structure."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Behavioural volatility features file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded data with {len(df)} records and {len(df.columns)} features")
    
    # Validate required columns exist
    required_cols = [
        'BEHAVIOURAL_VOLATILITY_SCORE',
        # Add any other required columns here, e.g. 'WALLET_ADDRESS', 'TOTAL_EVENTS', etc.
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    print("✅ All required columns present")
    
    # Check for NaNs or invalid values
    if df['BEHAVIOURAL_VOLATILITY_SCORE'].isnull().any():
        raise ValueError("Null values found in BEHAVIOURAL_VOLATILITY_SCORE column")
    if not np.isfinite(df['BEHAVIOURAL_VOLATILITY_SCORE']).all():
        raise ValueError("Non-finite values found in BEHAVIOURAL_VOLATILITY_SCORE column")
    print("✅ No missing or invalid values in BEHAVIOURAL_VOLATILITY_SCORE")
    return df


def analyze_score_distribution(df: pd.DataFrame) -> dict:
    """Analyze the distribution of behavioural volatility scores."""
    print("\nAnalyzing BEHAVIOURAL_VOLATILITY_SCORE distribution...")
    score = df['BEHAVIOURAL_VOLATILITY_SCORE']
    stats = {
        'min': score.min(),
        'max': score.max(),
        'mean': score.mean(),
        'std': score.std(),
        'median': score.median(),
        'nunique': score.nunique(),
        'missing': score.isnull().sum(),
        'zero_count': (score == 0).sum(),
        'negative_count': (score < 0).sum(),
    }
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats

def validate_behavioural_volatility_formula(df: pd.DataFrame, tolerance: float = 1e-6) -> bool:
    """
    Placeholder for formula validation. If the formula for BEHAVIOURAL_VOLATILITY_SCORE is known,
    recalculate and compare to stored values. Otherwise, just return True.
    """
    print("\nValidating behavioural volatility score calculations...")
    # Example: If you know the formula, implement it here and compare
    # For now, just print a message and return True
    print("(No formula validation implemented - add logic if formula is available)")
    return True

def main():
    file_path = os.path.join(os.path.dirname(__file__), '../../data/processed_data/behavioural_volatility_features.csv')
    try:
        df = load_and_validate_data(file_path)
        formula_valid = validate_behavioural_volatility_formula(df)
        stats = analyze_score_distribution(df)
        if formula_valid:
            print("\n✅ Behavioural volatility feature validation completed successfully.")
        else:
            print("\n❌ Formula validation failed.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
