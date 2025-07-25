#!/usr/bin/env python3
"""
Quick Start Guide: Using the exclude_no_columns functionality

This script shows you exactly how to enable and use the new
exclude_no_columns functionality in your clustering pipeline.
"""

import yaml

def show_configuration_example():
    """Show the exact configuration needed."""
    
    print("=" * 80)
    print("QUICK START: EXCLUDE_NO_COLUMNS FUNCTIONALITY")
    print("=" * 80)
    
    print("\n1. EDIT YOUR CONFIG FILE")
    print("-" * 40)
    print("Add these lines to your config_cluster.yaml:")
    
    config_example = """
preprocessing:
  log_transformation:
    enabled: true
    exclude_no_columns: true    # NEW: Process ALL numerical columns
    method: "log1p"
  
  scaling:
    enabled: true
    exclude_no_columns: true    # NEW: Process ALL numerical columns
    method: "standard"

umap:
  enabled: true
  include_all_columns: true     # NEW: Use ALL available columns
  n_neighbors: 15
  n_components: 2
  # ... keep your other UMAP parameters
"""
    
    print(config_example)
    
    print("\n2. WHAT THIS DOES")
    print("-" * 40)
    print("✅ Log transformation: Applied to ALL 21 numerical columns (instead of excluding some)")
    print("✅ Scaling: Applied to ALL 21 numerical columns (instead of excluding some)")
    print("✅ UMAP: Uses ALL available columns (instead of only 4 specific columns)")
    print("✅ HDBSCAN: Clusters using the full feature space")
    
    print("\n3. RUN YOUR PIPELINE AS USUAL")
    print("-" * 40)
    print("No code changes needed! Just run your existing scripts:")
    print("  python scripts/clustering/run_hdbscan_clustering.py")
    print("  python source_code_package/models/clustering_functionality/UMAP_dim_reduction.py")
    
    print("\n4. EXPECTED OUTPUT")
    print("-" * 40)
    print("You should see:")
    print('  "exclude_no_columns is enabled - applying log transformation to ALL numerical columns"')
    print('  "exclude_no_columns is enabled - applying scaling to ALL numerical columns"')
    print('  "include_all_columns is enabled - using ALL numerical columns from preprocessed data"')
    print('  "Using all 21 numerical columns"')
    
    print("\n5. VERIFY IT'S WORKING")
    print("-" * 40)
    print("Check the preprocessing output - you should see 21 columns being processed instead of 4.")
    
    print("\n6. BENEFITS")
    print("-" * 40)
    print("• Maximum information retention")
    print("• No manual column selection needed")
    print("• Automatic adaptation to new datasets")
    print("• Simplified configuration management")
    
    print("\n" + "=" * 80)
    print("That's it! Your pipeline now uses ALL available features.")
    print("=" * 80)

def create_sample_config():
    """Create a sample configuration file."""
    
    sample_config = {
        'data': {
            'raw_data_path': 'data/raw_data/new_raw_data_polygon.csv'
        },
        'preprocessing': {
            'log_transformation': {
                'enabled': True,
                'exclude_no_columns': True,  # NEW OPTION
                'method': 'log1p'
            },
            'scaling': {
                'enabled': True,
                'exclude_no_columns': True,  # NEW OPTION
                'method': 'standard'
            }
        },
        'umap': {
            'enabled': True,
            'include_all_columns': True,  # NEW OPTION
            'n_neighbors': 15,
            'n_components': 2,
            'metric': 'cosine',
            'min_dist': 0.05,
            'random_state': 42
        },
        'hdbscan': {
            'min_cluster_size': 50,
            'min_samples': 15,
            'metric': 'euclidean'
        }
    }
    
    # Save sample config
    with open('sample_all_columns_config.yaml', 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n📄 Sample configuration saved to: sample_all_columns_config.yaml")
    print("You can use this as a template for your own configuration.")

if __name__ == "__main__":
    show_configuration_example()
    create_sample_config()
