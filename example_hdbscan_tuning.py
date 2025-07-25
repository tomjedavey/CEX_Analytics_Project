#!/usr/bin/env python3
"""
Example script demonstrating HDBSCAN parameter optimization functionality.

This script shows how to use the tune_hdbscan_parameters module to optimize
HDBSCAN clustering parameters using UMAP dimensionality reduced data.
"""

import os
import sys

# Add the parent directory to path to import from source_code_package
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from source_code_package.models.clustering_functionality.tune_hdbscan_parameters import (
    optimize_hdbscan_parameters, HDBSCANParameterOptimizer
)


def run_quick_optimization():
    """Run a quick parameter optimization example."""
    print("HDBSCAN Parameter Optimization Example")
    print("=" * 50)
    
    # Configuration
    config_path = "source_code_package/config/config_cluster.yaml"
    output_dir = "tuning_results_example"
    
    print("Running quick grid search optimization...")
    print("This will test a small parameter space for demonstration.")
    
    try:
        # Run optimization
        results = optimize_hdbscan_parameters(
            config_path=config_path,
            method="grid_search",
            search_type="quick",
            max_workers=1,  # Use single worker for this example
            save_results=True,
            output_dir=output_dir,
            update_config=False
        )
        
        if results:
            print(f"\n✓ Optimization completed successfully!")
            print(f"✓ Tested {len(results)} parameter combinations")
            print(f"✓ Best composite score: {results[0]['composite_score']:.2f}")
            
            print("\nBest UMAP parameters:")
            for key, value in results[0]['umap_params'].items():
                print(f"  {key}: {value}")
            
            print("\nBest HDBSCAN parameters:")
            for key, value in results[0]['hdbscan_params'].items():
                print(f"  {key}: {value}")
            
            print(f"\nBest result details:")
            print(f"  - Clusters found: {results[0]['n_clusters']}")
            print(f"  - Noise percentage: {results[0]['noise_percentage']:.1f}%")
            print(f"  - Silhouette score: {results[0]['silhouette_score']:.3f}")
            print(f"  - Calinski-Harabasz score: {results[0]['calinski_harabasz_score']:.2f}")
            print(f"  - Davies-Bouldin score: {results[0]['davies_bouldin_score']:.3f}")
            
            print(f"\n✓ Results saved to: {output_dir}/")
            
        else:
            print("❌ No successful optimization results")
            
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        print("\nThis might happen if:")
        print("- Required dependencies are not installed")
        print("- Data file is not found")
        print("- Configuration file is not properly formatted")


def run_advanced_optimization():
    """Run more advanced optimization examples."""
    print("\nAdvanced Optimization Examples")
    print("=" * 30)
    
    # Initialize optimizer
    optimizer = HDBSCANParameterOptimizer(
        config_path="source_code_package/config/config_cluster.yaml"
    )
    
    try:
        print("Loading and preprocessing data...")
        data = optimizer.load_and_preprocess_data()
        print(f"✓ Data loaded: {data.shape[0]} samples, {data.shape[1]} features")
        
        # Example: Random search
        print("\nRunning random search (10 trials)...")
        results = optimizer.random_search_optimization(
            n_trials=10,
            max_workers=1,
            save_results=True,
            output_dir="tuning_results_random"
        )
        
        if results:
            print(f"✓ Random search completed: best score {results[0]['composite_score']:.2f}")
        
    except Exception as e:
        print(f"❌ Error in advanced optimization: {e}")


def show_parameter_spaces():
    """Demonstrate different parameter space configurations."""
    print("\nParameter Space Examples")
    print("=" * 30)
    
    from source_code_package.models.clustering_functionality.tune_hdbscan_parameters import ParameterSpace
    
    search_types = ["quick", "default", "comprehensive", "fine_tune"]
    
    for search_type in search_types:
        print(f"\n{search_type.upper()} search space:")
        
        umap_space = ParameterSpace.get_umap_parameter_space(search_type)
        hdbscan_space = ParameterSpace.get_hdbscan_parameter_space(search_type)
        
        # Calculate total combinations
        umap_combinations = 1
        for param_values in umap_space.values():
            umap_combinations *= len(param_values)
        
        hdbscan_combinations = 1
        for param_values in hdbscan_space.values():
            hdbscan_combinations *= len(param_values)
        
        total_combinations = umap_combinations * hdbscan_combinations
        
        print(f"  UMAP parameters: {len(umap_space)} types")
        print(f"  HDBSCAN parameters: {len(hdbscan_space)} types")
        print(f"  Total combinations: {total_combinations}")


def main():
    """Main execution function."""
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("HDBSCAN Parameter Tuning Demonstration")
    print("=" * 60)
    
    # Show parameter spaces
    show_parameter_spaces()
    
    # Run quick optimization
    run_quick_optimization()
    
    # Optionally run advanced examples (uncomment to test)
    # run_advanced_optimization()
    
    print("\n" + "=" * 60)
    print("Demonstration completed!")
    print("\nFor production use, consider:")
    print("- Using 'comprehensive' or 'default' search types")
    print("- Enabling parallel processing with max_workers > 1")
    print("- Using random_search or bayesian optimization for large spaces")
    print("- Setting update_config=True to automatically update configuration")


if __name__ == "__main__":
    main()
