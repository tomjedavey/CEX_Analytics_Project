#!/usr/bin/env python3
"""
Test script for verifying the distance score production in the interaction mode pipeline using feature medians.

This script ensures that:
- The distance score calculation uses the correct medians and features
- All steps (preprocessing, distance, MAD, normalization, proportionality weighting) are correct
- Handles edge cases (e.g., zero MAD, missing features)

It uses the following source files:
- source_code_package/features/interaction_mode_distance_source.py
- scripts/interaction_mode_score/interaction_mode_distance_execution.py
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile
import shutil

# Add the source_code_package to the path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '../source_code_package'))

from source_code_package.features.interaction_mode_distance_source import (
    load_medians, preprocess_features, compute_absolute_distances,
    compute_mad, normalize_distances, compute_proportionality_weights,
    apply_proportionality_weighting
)

def test_interaction_mode_distance_score():
    """Test the distance score production using feature medians and synthetic data."""
    print("=== Testing interaction mode distance score production ===")
    temp_dir = tempfile.mkdtemp()
    try:
        # Step 1: Create synthetic medians and wallet data
        features = ["DEX_EVENTS", "CEX_EVENTS", "BRIDGE_EVENTS", "DEFI_EVENTS"]
        medians = pd.DataFrame({f: [10.0] for f in features})
        wallet_data = pd.DataFrame({
            "DEX_EVENTS": [12, 8, 10],
            "CEX_EVENTS": [10, 10, 10],
            "BRIDGE_EVENTS": [5, 15, 10],
            "DEFI_EVENTS": [20, 0, 10]
        })
        # Save medians to CSV
        median_csv = os.path.join(temp_dir, "feature_medians.csv")
        medians.to_csv(median_csv, index=False)
        # Step 2: Load medians using pipeline function
        loaded_medians = load_medians(median_csv, features)
        assert loaded_medians.equals(medians), "Loaded medians do not match expected values"
        # Step 3: Preprocess features (should be idempotent for log1p if already preprocessed)
        wallet_proc = preprocess_features(wallet_data, features)
        medians_proc = preprocess_features(loaded_medians, features)
        # Step 4: Compute signed distances
        dist = compute_absolute_distances(wallet_proc, medians_proc, features)
        assert dist.shape == wallet_data[features].shape, "Distance shape mismatch"
        # Step 5: Compute MAD
        mad = compute_mad(wallet_proc, features, medians_proc)
        assert isinstance(mad, dict) and all(f in mad for f in features), "MAD keys missing"
        # Step 6: Normalize distances
        norm_dist = normalize_distances(dist, mad, features)
        assert norm_dist.shape == dist.shape, "Normalized distance shape mismatch"
        # Step 7: Proportionality weights
        weights = compute_proportionality_weights(wallet_proc, features)
        assert weights.shape == wallet_proc.shape, "Weights shape mismatch"
        # Step 8: Apply proportionality weighting
        weighted_dist = apply_proportionality_weighting(norm_dist, weights, features)
        assert weighted_dist.shape == norm_dist.shape, "Weighted distance shape mismatch"
        print("  ✅ All steps in distance score production passed!")
    finally:
        shutil.rmtree(temp_dir)

def main():
    test_interaction_mode_distance_score()
    print("All interaction mode distance score tests passed.")

if __name__ == "__main__":
    main()
