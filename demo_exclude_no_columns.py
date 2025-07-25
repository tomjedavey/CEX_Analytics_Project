#!/usr/bin/env python3
"""
Demonstration of the new exclude_no_columns functionality.

This script shows how the new configuration options work to include
ALL numerical columns in the UMAP + clustering pipeline.
"""

import os
import sys
import yaml
import pandas as pd
import tempfile

# Add the source code package to path
sys.path.append('/Users/tomdavey/Documents/GitHub/MLProject1/source_code_package')

def create_demo_config():
    """Create a demo configuration with all-columns options enabled."""
    
    # Load original config
    config_path = '/Users/tomdavey/Documents/GitHub/MLProject1/source_code_package/config/config_cluster.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify to enable all columns
    config['preprocessing']['log_transformation']['exclude_no_columns'] = True
    config['preprocessing']['scaling']['exclude_no_columns'] = True
    config['umap']['include_all_columns'] = True
    
    # Create temporary config file
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config, temp_config, default_flow_style=False, sort_keys=False)
    temp_config.close()
    
    return temp_config.name

def demonstrate_functionality():
    """Demonstrate the new functionality."""
    
    print("=" * 80)
    print("EXCLUDE_NO_COLUMNS FUNCTIONALITY DEMONSTRATION")
    print("=" * 80)
    
    # Show available columns in dataset
    print("\n1. AVAILABLE COLUMNS IN DATASET:")
    print("-" * 50)
    
    data_path = '/Users/tomdavey/Documents/GitHub/MLProject1/data/raw_data/new_raw_data_polygon.csv'
    df = pd.read_csv(data_path)
    
    all_columns = list(df.columns)
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    print(f"Total columns: {len(all_columns)}")
    print(f"All columns: {all_columns}")
    print(f"\nNumerical columns ({len(numerical_columns)}): {numerical_columns}")
    
    # Show original configuration
    print("\n2. ORIGINAL CONFIGURATION:")
    print("-" * 50)
    
    config_path = '/Users/tomdavey/Documents/GitHub/MLProject1/source_code_package/config/config_cluster.yaml'
    with open(config_path, 'r') as f:
        original_config = yaml.safe_load(f)
    
    print("Original settings:")
    preprocessing = original_config.get('preprocessing', {})
    umap_config = original_config.get('umap', {})
    
    print(f"  Log transformation exclude_columns: {preprocessing.get('log_transformation', {}).get('exclude_columns', [])}")
    print(f"  Scaling exclude_columns: {preprocessing.get('scaling', {}).get('exclude_columns', [])}")
    print(f"  UMAP include_columns: {umap_config.get('include_columns', [])}")
    
    print(f"\nWith original config, UMAP uses {len(umap_config.get('include_columns', []))} specific columns")
    
    # Show new configuration options
    print("\n3. NEW CONFIGURATION OPTIONS ADDED:")
    print("-" * 50)
    
    print("New options added to config_cluster.yaml:")
    print("""
preprocessing:
  log_transformation:
    exclude_no_columns: false  # NEW: When true, processes ALL numerical columns
  scaling:
    exclude_no_columns: false  # NEW: When true, processes ALL numerical columns

umap:
  include_all_columns: false   # NEW: When true, uses ALL available columns
""")
    
    # Demonstrate with modified config
    print("\n4. DEMONSTRATION WITH ALL-COLUMNS ENABLED:")
    print("-" * 50)
    
    # Create demo config
    demo_config_path = create_demo_config()
    
    try:
        # Load demo config
        with open(demo_config_path, 'r') as f:
            demo_config = yaml.safe_load(f)
        
        print("Demo configuration:")
        preprocessing = demo_config.get('preprocessing', {})
        umap_config = demo_config.get('umap', {})
        
        print(f"  Log transformation exclude_no_columns: {preprocessing.get('log_transformation', {}).get('exclude_no_columns', False)}")
        print(f"  Scaling exclude_no_columns: {preprocessing.get('scaling', {}).get('exclude_no_columns', False)}")
        print(f"  UMAP include_all_columns: {umap_config.get('include_all_columns', False)}")
        
        print(f"\nWith new config, all {len(numerical_columns)} numerical columns would be used:")
        for i, col in enumerate(numerical_columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Test validation function
        print("\n5. CONFIGURATION VALIDATION:")
        print("-" * 50)
        
        try:
            from models.clustering_functionality.UMAP_dim_reduction import validate_feature_consistency
            
            validation_results = validate_feature_consistency(demo_config_path)
            
            print("Validation results:")
            print(f"  Include all columns: {validation_results.get('include_all_columns', False)}")
            print(f"  Log exclude no columns: {validation_results.get('log_exclude_no_columns', False)}")
            print(f"  Scale exclude no columns: {validation_results.get('scale_exclude_no_columns', False)}")
            
            if validation_results.get('warnings'):
                print("\n⚠️  Warnings:")
                for warning in validation_results['warnings']:
                    print(f"    - {warning}")
            else:
                print("\n✅ No configuration warnings")
                
            print("\n📋 Recommendations:")
            for rec in validation_results.get('recommendations', []):
                print(f"    - {rec}")
                
        except ImportError as e:
            print(f"Could not import validation function: {e}")
        
    finally:
        # Clean up temp file
        os.unlink(demo_config_path)
    
    # Usage instructions
    print("\n6. HOW TO USE THIS FUNCTIONALITY:")
    print("-" * 50)
    
    print("""
To enable processing of ALL numerical columns:

1. Edit your config_cluster.yaml file:
   
   preprocessing:
     log_transformation:
       exclude_no_columns: true    # Apply log transformation to ALL numerical columns
     scaling:
       exclude_no_columns: true    # Apply scaling to ALL numerical columns
   
   umap:
     include_all_columns: true     # Use ALL available columns in UMAP

2. Run your clustering pipeline as usual:
   - The preprocessing will now include ALL numerical columns
   - UMAP will use ALL preprocessed columns
   - HDBSCAN will cluster using the full feature space

3. Benefits:
   - Maximum information retention
   - No manual column selection needed
   - Automatic adaptation to new datasets
   - Simplified configuration management

4. Considerations:
   - Higher dimensionality may require UMAP for effective clustering
   - Processing time may increase with more features
   - Results may differ from column-specific configurations
""")

if __name__ == "__main__":
    demonstrate_functionality()
