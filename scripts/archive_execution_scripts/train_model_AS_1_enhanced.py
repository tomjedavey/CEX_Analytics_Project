#!/usr/bin/env python3
"""
Enhanced AS_1 Model Training Execution Script

This script serves as an execution wrapper around the enhanced AS_1 training functionality
in source_code_package.models.AS_1_functionality.enhanced_train_model_source module.

Author: Tom Davey
Date: July 2025
"""

import os
import sys
from pathlib import Path

# Add source code package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../source_code_package/'))

# Import enhanced training functions
from models.AS_1_functionality.enhanced_train_model_source import (
    train_model_from_config,
    train_multiple_models_batch,
    load_as1_training_config
)


def main():
    """Main function to execute enhanced AS_1 model training."""
    print("AS_1 ENHANCED MODEL TRAINING")
    print("=" * 50)
    
    # Define configuration paths
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'source_code_package', 'config')
    
    config_paths = [
        os.path.join(config_dir, 'config_AS_1_full_dataset.yaml'),
        os.path.join(config_dir, 'config_AS_1_cluster_0.yaml'),
        os.path.join(config_dir, 'config_AS_1_cluster_1.yaml')
    ]
    
    # Check if processed data files exist
    processed_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/processed_data/AS_1_feature_data_full_dataset.csv'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/processed_data/AS_1_feature_data_cluster_0.csv'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data/processed_data/AS_1_feature_data_cluster_1.csv')
    ]
    
    missing_files = [path for path in processed_paths if not os.path.exists(path)]
    
    if missing_files:
        print("\n⚠️  WARNING: Missing processed data files!")
        print("Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run feature engineering first using feature_engineering_AS_1_enhanced.py")
        
        # Ask user if they want to proceed with available datasets
        proceed = input("\nProceed with available datasets? (y/n): ").lower().strip()
        if proceed != 'y':
            print("Training aborted.")
            return False
        
        # Filter to only existing datasets
        available_configs = []
        for i, path in enumerate(processed_paths):
            if os.path.exists(path):
                available_configs.append(config_paths[i])
        config_paths = available_configs
        
        if not config_paths:
            print("No processed datasets available. Please run feature engineering first.")
            return False
    
    print(f"\nTraining models for {len(config_paths)} datasets...")
    
    # Train models using core functionality
    results = train_multiple_models_batch(config_paths, verbose=True)
    
    # Check results and provide summary
    successful = sum(1 for r in results.values() if r['success'])
    
    print(f"\n{'='*50}")
    print(f"TRAINING SUMMARY")
    print(f"Successful: {successful}/{len(config_paths)}")
    
    if successful > 0:
        print("\nTrained Models:")
        for config_path, result in results.items():
            if result['success']:
                dataset_name = os.path.basename(config_path).replace('config_AS_1_', '').replace('.yaml', '')
                metrics = result['metrics']
                print(f"  📊 {dataset_name}:")
                print(f"     - Model saved: {result['model_path']}")
                print(f"     - Test R²: {metrics['test_r2']:.4f}")
                print(f"     - Test RMSE: {metrics['test_rmse']:.4f}")
                print(f"     - Training samples: {metrics['training_samples']}")
                print(f"     - Test samples: {metrics['test_samples']}")
        
        print(f"\n🎉 Training completed successfully!")
        print("Next steps:")
        print("1. Run model testing: test_model_AS_1_enhanced.py")
        print("2. Run comparative analysis: comparative_analysis_AS_1.py")
        return True
    else:
        print("\n⚠️  All training attempts failed. Check errors above.")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
