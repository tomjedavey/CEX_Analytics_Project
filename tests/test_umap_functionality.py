#!/usr/bin/env python3
"""
Test script to verify the updated UMAP functionality with include_columns configuration.
"""

import sys
import os

# Add the source_code_package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../source_code_package'))

from source_code_package.models.clustering_functionality.UMAP_dim_reduction import (
    validate_feature_consistency,
    load_umap_config,
    run_umap_pipeline_example
)

def test_config_loading():
    """Test that the configuration is loaded correctly."""
    print("=== Testing Configuration Loading ===")
    
    config = load_umap_config()
    print(f"UMAP config keys: {list(config.keys())}")
    
    include_columns = config.get('include_columns', [])
    print(f"Include columns: {include_columns}")
    print(f"Number of columns to include: {len(include_columns)}")
    
    # Rationale: Use assert to validate that include_columns is a list and not empty, so the test runner can detect failures.
    assert isinstance(include_columns, list), "include_columns should be a list"
    assert len(include_columns) > 0, "include_columns should not be empty"
    
    print("✅ test_config_loading passed!")

def test_validation():
    """Test the feature consistency validation."""
    print("\n=== Testing Feature Consistency Validation ===")
    
    validation_results = validate_feature_consistency()
    
    print(f"Include columns: {validation_results['include_columns']}")
    print(f"Log transform excluded: {validation_results['log_transform_excluded']}")
    print(f"Scaling excluded: {validation_results['scaling_excluded']}")
    
    if validation_results['warnings']:
        print("⚠️  Warnings found:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    else:
        print("✅ No warnings found")
    
    print("📋 Recommendations:")
    for rec in validation_results['recommendations']:
        print(f"  - {rec}")
    
    # Rationale: Use assert to validate the structure and keys of validation_results, so the test runner can detect failures.
    assert isinstance(validation_results, dict), "validation_results should be a dict"
    expected_keys = ['include_columns', 'include_all_columns', 'log_transform_excluded', 'log_exclude_no_columns', 'scaling_excluded', 'scale_exclude_no_columns', 'warnings', 'recommendations']
    for key in expected_keys:
        assert key in validation_results, f"Missing key in validation_results: {key}"
    
    print("✅ test_validation passed!")

def main():
    """Run all tests."""
    try:
        # Test 1: Configuration loading
        test_config_loading()
        
        # Test 2: Validation
        test_validation()
        
        print("\n=== Summary ===")
        print("✅ All UMAP configuration and validation tests passed!")
        
        print("\n=== NOTE ===")
        print("To test the full pipeline, ensure you have:")
        print("1. The required data file: data/raw_data/new_raw_data_polygon.csv")
        print("2. All required Python packages installed (umap-learn, scikit-learn, etc.)")
        print("3. Run: python -c 'from scripts.test_umap_functionality import run_full_pipeline_test; run_full_pipeline_test()'")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def run_full_pipeline_test():
    """Run the full pipeline test - only call this if you have the data and dependencies."""
    print("=== Running Full Pipeline Test ===")
    try:
        result = run_umap_pipeline_example()
        if result[0] is not None:
            print("✅ Full pipeline completed successfully!")
            print(f"Reduced data shape: {result[0].shape}")
        else:
            print("❌ Pipeline failed")
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")

if __name__ == "__main__":
    main()
