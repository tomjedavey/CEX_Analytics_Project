#!/usr/bin/env python3
"""
Enhanced AS_1 Linear Regression Model Testing Script

This script provides an execution interface for testing trained AS_1 linear regression
models using the enhanced testing module with comprehensive evaluation capabilities.

Author: Tom Davey
Date: July 2025
"""

import os
import sys
from pathlib import Path

# Add source code package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../source_code_package/'))

# Import enhanced testing functions
from models.AS_1_functionality.enhanced_test_model_source import (
    test_model_from_config,
    test_multiple_models_batch,
    load_as1_testing_config
)


def main():
    """Main execution function for AS_1 model testing."""
    print("AS_1 Enhanced Model Testing")
    print("=" * 40)
    
    # Configuration file path
    config_path = "/Users/tomdavey/Documents/GitHub/MLProject1/source_code_package/config/config_AS_1_full_dataset.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    try:
        print(f"Testing AS_1 model with configuration: {Path(config_path).name}")
        print()
        
        # Test the model
        result = test_model_from_config(config_path, verbose=True)
        
        if result['success']:
            print("\n✅ Model testing completed successfully!")
            metrics = result['test_results']['performance_metrics']
            print(f"Final Test Metrics:")
            print(f"  - R²: {metrics['r2']:.4f}")
            print(f"  - RMSE: {metrics['rmse']:.4f}")
            print(f"  - MAE: {metrics['mae']:.4f}")
            print(f"  - Test samples: {result['test_results']['test_samples']}")
        else:
            print(f"\n❌ Model testing failed: {result['error']}")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")


if __name__ == "__main__":
    main()
