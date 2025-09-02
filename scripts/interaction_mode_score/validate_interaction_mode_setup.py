#!/usr/bin/env python3
"""
Test script to validate the interaction mode clustering setup.

This script performs basic validation of the new interaction mode clustering
configuration and ensures all components are properly set up.
"""

import os
import sys
import yaml

# Add source_code_package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../source_code_package'))

def test_config_loading():
    """Test loading the interaction mode configuration."""
    print("Testing configuration loading...")
    
    config_path = os.path.join(
        os.path.dirname(__file__), 
        '../../source_code_package/config/config_interaction_mode.yaml'
    )
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded successfully from: {config_path}")
        
        # Validate key sections
        required_sections = ['data', 'preprocessing', 'umap', 'hdbscan', 'interaction_mode']
        for section in required_sections:
            if section in config:
                print(f"   ✅ {section} section found")
            else:
                print(f"   ❌ {section} section missing")
                return False
                
        return True
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

def test_dataset_files():
    """Test that all required dataset files exist."""
    print("\nTesting dataset files...")
    
    raw_data_dir = os.path.join(os.path.dirname(__file__), '../../data/raw_data')
    processed_results_dir = os.path.join(os.path.dirname(__file__), '../../data/processed_data/cluster_datasets')

    # Always check for the main dataset
    datasets = {'main': os.path.join(raw_data_dir, 'new_raw_data_polygon.csv')}

    # Dynamically find all cluster datasets
    for fname in os.listdir(processed_results_dir):
        if fname.startswith('new_raw_data_polygon_cluster_') and fname.endswith('.csv'):
            cluster_name = fname.replace('new_raw_data_polygon_', '').replace('.csv', '')
            datasets[cluster_name] = os.path.join(processed_results_dir, fname)

    all_exist = True
    for name, path in datasets.items():
        if os.path.exists(path):
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"   ✅ {name}: {os.path.basename(path)} ({file_size:.1f} MB)")
        else:
            print(f"   ❌ {name}: {path} - File not found")
            all_exist = False

    return all_exist

def test_imports():
    """Test that required modules can be imported."""
    print("\nTesting module imports...")
    
    try:
        from source_code_package.models.clustering_functionality.simplified_clustering import (
            run_clustering_pipeline,
            load_hdbscan_config,
            validate_hdbscan_config
        )
        print("   ✅ Clustering functionality imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False

def test_config_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    
    try:
        from source_code_package.models.clustering_functionality.simplified_clustering import validate_hdbscan_config
        
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '../../source_code_package/config/config_interaction_mode.yaml'
        )
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        validation_result = validate_hdbscan_config(config)
        
        if validation_result.get('valid', True):
            print("   ✅ Configuration validation passed")
            
            warnings = validation_result.get('warnings', [])
            if warnings:
                print("   ⚠️  Warnings:")
                for warning in warnings:
                    print(f"      • {warning}")
                    
            return True
        else:
            print("   ❌ Configuration validation failed")
            errors = validation_result.get('errors', [])
            for error in errors:
                print(f"      • {error}")
            return False
            
    except Exception as e:
        print(f"   ❌ Validation error: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🔍 Interaction Mode Clustering Setup Validation")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Dataset Files", test_dataset_files),
        ("Module Imports", test_imports),
        ("Configuration Validation", test_config_validation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The interaction mode clustering setup is ready.")
        print("\nNext steps:")
        print("1. Run the clustering pipeline: python scripts/clustering/run_interaction_mode_clustering.py --validate-only")
        print("2. Process all datasets: python scripts/clustering/run_interaction_mode_clustering.py")
        print("3. Process specific datasets: python scripts/clustering/run_interaction_mode_clustering.py --datasets main cluster_0")
        return 0
    else:
        print("❌ Some tests failed. Please address the issues before running the pipeline.")
        return 1

if __name__ == "__main__":
    exit(main())
