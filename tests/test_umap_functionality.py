#!/usr/bin/env python3
"""
Test script to verify the updated UMAP functionality with include_columns configuration.
"""

import sys
import os

# Add the source_code_package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../source_code_package'))

from models.clustering_functionality.UMAP_dim_reduction import (
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
    
    return include_columns

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
    
    return validation_results

def main():
    """Run all tests."""
    try:
        # Test 1: Configuration loading
        include_columns = test_config_loading()
        
        # Test 2: Validation
        validation_results = test_validation()
        
        print("\n=== Summary ===")
        if include_columns:
            print(f"✅ Successfully loaded {len(include_columns)} include_columns from config")
        else:
            print("❌ No include_columns found in config")
        
        if not validation_results['warnings']:
            print("✅ Configuration is consistent")
        else:
            print("⚠️  Configuration has potential issues")
        
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

#**MAY HAVE TO CHANGE SOME OF THE PATHS - MOVED INTO TEST FOLDER FROM SCRIPTS**