#!/usr/bin/env python3
"""
Interaction Mode Score HDBSCAN Clustering Pipeline

This script runs HDBSCAN clustering on three different datasets to produce
interaction mode scores. It utilizes the existing HDBSCAN clustering 
functionality with a specialized configuration for interaction mode detection.

The script processes:
1. new_raw_data_polygon.csv (main dataset)
2. new_raw_data_polygon_cluster_0.csv (cluster 0 subset)  
3. new_raw_data_polygon_cluster_1.csv (cluster 1 subset)

Each dataset is clustered separately using the same HDBSCAN configuration,
producing separate clustering results that can later be used to compute
interaction mode scores.

Author: Tom Davey
Date: August 2025
"""

import os
import sys
import yaml
import argparse
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from datetime import datetime

# Add source_code_package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../source_code_package'))

# Import the simplified clustering pipeline
try:
    from models.clustering_functionality.simplified_clustering import (
        run_clustering_pipeline,
        load_hdbscan_config,
        validate_hdbscan_config
    )
except ImportError:
    print("❌ Error: Could not import simplified clustering functionality.")
    print("Please ensure the simplified_clustering.py module is available.")
    sys.exit(1)


def get_default_paths() -> Dict[str, str]:
    """Get default paths for configuration and output directory."""
    script_dir = os.path.dirname(__file__)
    return {
        'config_path': os.path.join(script_dir, '../../source_code_package/config/config_interaction_mode.yaml'),
        'output_dir': os.path.join(script_dir, '../../data/raw_data/interaction_mode_results')
    }


def load_interaction_mode_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the interaction mode configuration."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"✅ Loaded configuration from: {config_path}")
        return config
    except Exception as e:
        raise ValueError(f"Error loading configuration from {config_path}: {e}")


def get_dataset_paths() -> Dict[str, str]:
    """Get the full paths to all three datasets."""
    script_dir = os.path.dirname(__file__)
    base_data_dir = os.path.join(script_dir, '../../data/raw_data')
    cluster_data_dir = os.path.join(base_data_dir, 'cluster_datasets')
    
    datasets = {
        'main': os.path.join(base_data_dir, 'new_raw_data_polygon.csv'),
        'cluster_0': os.path.join(cluster_data_dir, 'new_raw_data_polygon_cluster_0.csv'),
        'cluster_1': os.path.join(cluster_data_dir, 'new_raw_data_polygon_cluster_1.csv')
    }
    
    # Verify all datasets exist
    for name, path in datasets.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
    
    return datasets


def validate_dataset(dataset_path: str, dataset_name: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate a dataset before processing.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset file
    dataset_name : str
        Name/identifier for the dataset
        
    Returns:
    --------
    Tuple[bool, Dict[str, Any]]
        Validation result and dataset info
    """
    try:
        df = pd.read_csv(dataset_path)
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        print(f"  📊 {dataset_name}: {info['rows']:,} rows, {info['columns']} columns")
        print(f"     Memory usage: {info['memory_usage_mb']:.1f} MB")
        
        return True, info
    except Exception as e:
        print(f"  ❌ Error validating {dataset_name}: {e}")
        return False, {}


def create_dataset_config(config: Dict[str, Any], dataset_path: str, output_dir: str) -> str:
    """
    Create a temporary configuration file for a specific dataset.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Base configuration
    dataset_path : str
        Path to the dataset to be processed
    output_dir : str
        Output directory for this dataset's results
        
    Returns:
    --------
    str
        Path to the temporary configuration file
    """
    # Create a copy of the base config
    dataset_config = config.copy()
    
    # Update the data path for this specific dataset
    dataset_config['data']['raw_data_path'] = dataset_path
    
    # Update output path if needed
    if 'output' in dataset_config:
        dataset_config['output']['preprocessing_info_path'] = os.path.join(
            output_dir, 'preprocessing_info.yaml'
        )
    
    # Create temporary config file
    temp_config_path = os.path.join(output_dir, 'temp_config.yaml')
    os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    return temp_config_path


def validate_configuration(config_path: str) -> bool:
    """
    Validate HDBSCAN configuration and display results clearly.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
        
    Returns:
    --------
    bool
        True if configuration is valid
    """
    print("📋 Validating configuration...")
    
    try:
        config = load_hdbscan_config(config_path)
        validation = validate_hdbscan_config(config)
        
        print(f"Configuration status: {'✅ VALID' if validation['valid'] else '❌ INVALID'}")
        
        if validation['warnings']:
            print("⚠️  Configuration warnings:")
            for warning in validation['warnings']:
                print(f"    • {warning}")
        
        if validation['recommendations']:
            print("💡 Configuration recommendations:")
            for rec in validation['recommendations']:
                print(f"    • {rec}")
                
        if validation['errors']:
            print("❌ Configuration errors:")
            for error in validation['errors']:
                print(f"    • {error}")
            return False
                
        return validation['valid']
        
    except Exception as e:
        print(f"❌ Error validating configuration: {e}")
        return False


def run_clustering_for_dataset(config: Dict[str, Any], dataset_name: str, dataset_path: str, 
                              output_dir: str, force_umap: Optional[bool] = None) -> Dict[str, Any]:
    """
    Run HDBSCAN clustering for a single dataset.
    
    Parameters:
    -----------
    config : Dict[str, Any]
        Base configuration
    dataset_name : str
        Name/identifier for the dataset
    dataset_path : str
        Path to the dataset file
    output_dir : str
        Output directory for results
    force_umap : bool, optional
        Force UMAP usage (None = use config setting)
        
    Returns:
    --------
    Dict[str, Any]
        Clustering results
    """
    print(f"\n🎯 Processing {dataset_name}...")
    print(f"   Dataset: {os.path.basename(dataset_path)}")
    print(f"   Output: {output_dir}")
    
    # Create dataset-specific configuration
    temp_config_path = create_dataset_config(config, dataset_path, output_dir)
    
    try:
        # Run the clustering pipeline
        results = run_clustering_pipeline(
            config_path=temp_config_path,
            data_path=dataset_path,
            output_dir=output_dir,
            force_umap=force_umap
        )
        
        if results.get('success', True):
            print(f"   ✅ Clustering completed successfully")
            
            # Display key results
            pipeline_info = results.get('pipeline_info', {})
            print(f"   📈 Clusters found: {pipeline_info.get('n_clusters_found', 'N/A')}")
            print(f"   📊 Data points: {pipeline_info.get('total_data_points', 'N/A'):,}")
            print(f"   🔍 Noise points: {pipeline_info.get('noise_points', 'N/A'):,}")
            
            # Calculate noise percentage
            total_points = pipeline_info.get('total_data_points', 0)
            noise_points = pipeline_info.get('noise_points', 0)
            if total_points > 0:
                noise_pct = (noise_points / total_points) * 100
                print(f"   📉 Noise percentage: {noise_pct:.1f}%")
            
        else:
            print(f"   ❌ Clustering failed: {results.get('error_message', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        error_msg = f"Error processing {dataset_name}: {str(e)}"
        print(f"   ❌ {error_msg}")
        return {'success': False, 'error_message': error_msg}
    
    finally:
        # Clean up temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def validate_all_datasets(dataset_paths: Dict[str, str]) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate all datasets before processing.
    
    Parameters:
    -----------
    dataset_paths : Dict[str, str]
        Dictionary mapping dataset names to file paths
        
    Returns:
    --------
    Tuple[bool, Dict[str, Dict[str, Any]]]
        Validation success status and detailed results
    """
    print("🔍 Validating datasets...")
    validation_results = {}
    all_valid = True
    
    for name, path in dataset_paths.items():
        is_valid, info = validate_dataset(path, name)
        validation_results[name] = {'valid': is_valid, 'info': info}
        if not is_valid:
            print(f"❌ Invalid dataset: {name}")
            all_valid = False
    
    return all_valid, validation_results


def save_pipeline_summary(results: Dict[str, Any], validation_results: Dict[str, Any], 
                         config: Dict[str, Any], output_dir: str, config_path: str) -> None:
    """Save a comprehensive summary of the pipeline execution."""
    summary_path = os.path.join(output_dir, "interaction_mode_pipeline_summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    # Add timestamp to track when the results were last updated
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(summary_path, 'w') as f:
        f.write("Interaction Mode Score Clustering Pipeline Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Last Updated: {current_time}\n\n")
        
        # Dataset validation summary
        f.write("Dataset Validation Results:\n")
        f.write("-" * 30 + "\n")
        for name, validation in validation_results.items():
            if validation['valid']:
                info = validation['info']
                f.write(f"{name}:\n")
                f.write(f"  ✅ Valid dataset\n")
                f.write(f"  📊 Rows: {info['rows']:,}\n")
                f.write(f"  📋 Columns: {info['columns']}\n")
                f.write(f"  💾 Memory: {info['memory_usage_mb']:.1f} MB\n\n")
            else:
                f.write(f"{name}: ❌ Invalid dataset\n\n")
        
        # Clustering results summary
        f.write("Clustering Results Summary:\n")
        f.write("-" * 30 + "\n")
        for name, result in results.items():
            f.write(f"{name}:\n")
            if result.get('success', True):
                pipeline_info = result.get('pipeline_info', {})
                f.write(f"  ✅ Clustering successful\n")
                f.write(f"  🎯 Clusters: {pipeline_info.get('n_clusters_found', 'N/A')}\n")
                f.write(f"  📊 Data points: {pipeline_info.get('total_data_points', 'N/A'):,}\n")
                f.write(f"  🔍 Noise points: {pipeline_info.get('noise_points', 'N/A'):,}\n")
                
                # Calculate noise percentage
                total_points = pipeline_info.get('total_data_points', 0)
                noise_points = pipeline_info.get('noise_points', 0)
                if total_points > 0:
                    noise_pct = (noise_points / total_points) * 100
                    f.write(f"  📉 Noise percentage: {noise_pct:.1f}%\n")
                
                # UMAP usage
                if result.get('umap_enabled', False):
                    f.write(f"  🔄 UMAP: Enabled\n")
                else:
                    f.write(f"  🔄 UMAP: Disabled (direct HDBSCAN)\n")
            else:
                f.write(f"  ❌ Clustering failed: {result.get('error_message', 'Unknown error')}\n")
            f.write("\n")
        
        # Configuration summary
        f.write("Configuration Summary:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Config file: {config_path}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"UMAP enabled: {config.get('umap', {}).get('enabled', 'N/A')}\n")
        
        hdbscan_config = config.get('hdbscan', {})
        f.write(f"Min cluster size: {hdbscan_config.get('min_cluster_size', 'N/A')}\n")
        f.write(f"Min samples: {hdbscan_config.get('min_samples', 'N/A')}\n")
    
    print(f"📄 Pipeline summary saved to: {summary_path}")


def display_pipeline_summary(results: Dict[str, Any], successful_runs: int, total_datasets: int) -> None:
    """Display a nicely formatted summary of the full pipeline results."""
    print(f"\n� Pipeline Summary")
    print("=" * 40)
    print(f"Datasets processed: {total_datasets}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {total_datasets - successful_runs}")
    
    if successful_runs == total_datasets:
        print("✅ All datasets processed successfully!")
    elif successful_runs > 0:
        print(f"⚠️  {successful_runs}/{total_datasets} datasets processed successfully")
    else:
        print("❌ No datasets processed successfully")


def main():
    """Main function for the interaction mode clustering script."""
    parser = argparse.ArgumentParser(
        description="Interaction Mode Score HDBSCAN Clustering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_interaction_mode_clustering.py                    # Process all datasets (updates data/raw_data/interaction_mode_results/)
    python run_interaction_mode_clustering.py --force-umap       # Force UMAP for all datasets  
    python run_interaction_mode_clustering.py --no-umap          # Force direct HDBSCAN for all
    python run_interaction_mode_clustering.py --datasets main cluster_0  # Process specific datasets
    python run_interaction_mode_clustering.py --validate-only    # Only validate configuration

Note: Each run updates the same output directory, overwriting previous results with new clustering data.
        """
    )
    
    parser.add_argument('--output-dir', '-o',
                       default=None,
                       help='Base output directory for all results (default: data/raw_data/interaction_mode_results)')
    
    parser.add_argument('--config-path', '-c',
                       help='Path to interaction mode configuration file (default: auto-detect)')
    
    parser.add_argument('--force-umap', action='store_true',
                       help='Force UMAP + HDBSCAN pipeline for all datasets (ignores config UMAP setting)')
    
    parser.add_argument('--no-umap', action='store_true',
                       help='Force direct HDBSCAN pipeline for all datasets (no UMAP)')
    
    parser.add_argument('--datasets', nargs='+', 
                       choices=['main', 'cluster_0', 'cluster_1'],
                       help='Specific datasets to process (default: all)')
    
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate configuration and datasets, do not run clustering')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress detailed progress output')
    
    args = parser.parse_args()
    
    # Validate mutually exclusive arguments
    if args.force_umap and args.no_umap:
        print("❌ Error: --force-umap and --no-umap cannot be used together.")
        return 1
    
    # Determine UMAP setting
    force_umap = None
    if args.force_umap:
        force_umap = True
    elif args.no_umap:
        force_umap = False
    
    # Header
    if not args.quiet:
        print("Interaction Mode Score HDBSCAN Clustering Pipeline")
        print("=" * 60)
    
    try:
        # Get default paths
        defaults = get_default_paths()
        
        # Determine configuration and output paths
        config_path = args.config_path or defaults['config_path']
        output_dir = args.output_dir or defaults['output_dir']
        
        if not args.quiet:
            print(f"📁 Configuration: {os.path.relpath(config_path)}")
            print(f"📂 Output directory: {os.path.relpath(output_dir)}")
            print()
        
        # Step 1: Load and validate configuration
        if not args.quiet:
            print("Step 1: Configuration Validation")
            print("-" * 40)
        
        config = load_interaction_mode_config(config_path)
        
        if not validate_configuration(config_path):
            print("❌ Configuration validation failed. Please fix configuration errors.")
            return 1
        
        if not args.quiet:
            print()
        
        # Step 2: Get and validate datasets
        if not args.quiet:
            print("Step 2: Dataset Discovery and Validation")
            print("-" * 40)
        
        all_dataset_paths = get_dataset_paths()
        
        # Filter datasets if specified
        if args.datasets:
            dataset_paths = {k: v for k, v in all_dataset_paths.items() 
                           if k in args.datasets}
            print(f"📋 Processing selected datasets: {', '.join(args.datasets)}")
        else:
            dataset_paths = all_dataset_paths
            print(f"📋 Processing all datasets: {', '.join(dataset_paths.keys())}")
        
        print(f"\n📋 Will process {len(dataset_paths)} datasets:")
        for name, path in dataset_paths.items():
            print(f"   • {name}: {os.path.basename(path)}")
        print()
        
        # Validate datasets
        datasets_valid, validation_results = validate_all_datasets(dataset_paths)
        if not datasets_valid:
            print("❌ Dataset validation failed. Please check your datasets.")
            return 1
        
        if args.validate_only:
            print("\n✅ Validation complete. Exiting without running clustering.")
            return 0
        
        if not args.quiet:
            print()
        
        # Step 3: Setup output directory
        if not args.quiet:
            print("Step 3: Output Directory Setup")
            print("-" * 40)
        
        if os.path.exists(output_dir):
            print(f"📁 Updating existing results directory: {output_dir}")
        else:
            print(f"📁 Creating new results directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        if not args.quiet:
            print()
        
        # Step 4: Run clustering pipeline
        if not args.quiet:
            print("Step 4: Clustering Pipeline Execution")
            print("-" * 40)
        
        # Determine UMAP setting message
        if force_umap is True:
            umap_setting = "Forced UMAP usage"
        elif force_umap is False:
            umap_setting = "Forced direct HDBSCAN"
        else:
            umap_setting = "Auto-detect from config"
        
        print(f"🔧 UMAP setting: {umap_setting}")
        
        # Process each dataset
        print(f"\n🔄 Processing datasets...")
        all_results = {}
        successful_runs = 0
        
        for dataset_name, dataset_path in dataset_paths.items():
            # Create output directory for this dataset
            dataset_output_dir = os.path.join(output_dir, f"{dataset_name}_clustering")
            
            # Run clustering for this dataset
            results = run_clustering_for_dataset(
                config=config,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                output_dir=dataset_output_dir,
                force_umap=force_umap
            )
            
            all_results[dataset_name] = results
            if results.get('success', True):
                successful_runs += 1
        
        # Step 5: Results summary and cleanup
        if not args.quiet:
            print("\nStep 5: Results Summary")
            print("-" * 40)
        
        display_pipeline_summary(all_results, successful_runs, len(dataset_paths))
        
        # Save overall summary
        save_pipeline_summary(all_results, validation_results, config, output_dir, config_path)
        print(f"🔄 Results directory updated: {output_dir}")
        
        # Final status
        if successful_runs == len(dataset_paths):
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"📁 Results saved to: {output_dir}")
            return 0
        else:
            print(f"\n❌ Pipeline completed with errors!")
            print(f"   Successful: {successful_runs}/{len(dataset_paths)}")
            return 1
            
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
