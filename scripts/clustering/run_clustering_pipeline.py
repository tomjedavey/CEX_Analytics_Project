#!/usr/bin/env python3
"""
Simplified HDBSCAN Clustering Script

This script provides a streamlined, user-friendly interface for running HDBSCAN 
clustering with optional UMAP dimensionality reduction. It replaces the complex
dual-interface approach with a simple, configuration-driven design.

Author: Tom Davey
Date: July 2025
"""

import os
import sys
import argparse
from typing import Optional

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


def display_results_summary(results: dict) -> None:
    """Display a nicely formatted summary of pipeline results."""
    if not results['success']:
        print(f"❌ Pipeline failed: {results['error_message']}")
        return
    
    print("✅ Pipeline completed successfully!")
    print()
    print("📊 Results Summary:")
    print(f"    • UMAP enabled: {'Yes' if results['umap_enabled'] else 'No'}")
    print(f"    • Clusters found: {results['pipeline_info']['n_clusters_found']}")
    print(f"    • Total data points: {results['pipeline_info']['total_data_points']:,}")
    print(f"    • Noise points: {results['pipeline_info']['noise_points']:,}")
    
    # Calculate noise percentage
    total_points = results['pipeline_info']['total_data_points']
    noise_points = results['pipeline_info']['noise_points']
    if total_points > 0:
        noise_percentage = (noise_points / total_points) * 100
        print(f"    • Noise percentage: {noise_percentage:.1f}%")
    
    # Display feature reduction info
    original_features = results['pipeline_info']['n_original_features']
    reduced_features = results['pipeline_info']['n_reduced_features']
    print(f"    • Features: {original_features} → {reduced_features}")
    
    # Display quality metrics if available
    hdbscan_results = results.get('hdbscan_results', {})
    if 'evaluation_metrics' in hdbscan_results:
        metrics = hdbscan_results['evaluation_metrics']
        print()
        print("📏 Quality Metrics:")
        if 'silhouette_score' in metrics:
            print(f"    • Silhouette Score: {metrics['silhouette_score']:.3f}")
        if 'calinski_harabasz_score' in metrics:
            print(f"    • Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
        if 'davies_bouldin_score' in metrics:
            print(f"    • Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")


def main():
    """Main function for simplified clustering script."""
    parser = argparse.ArgumentParser(
        description="Simplified HDBSCAN Clustering Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_simple_clustering.py                             # Auto-detect UMAP from config
    python run_simple_clustering.py --force-umap                # Force UMAP usage
    python run_simple_clustering.py --no-umap                   # Force direct HDBSCAN
    python run_simple_clustering.py --validate-only             # Only validate config
    python run_simple_clustering.py -o my_results               # Custom output directory
        """
    )
    
    parser.add_argument('--output-dir', '-o', 
                       default='clustering_results',
                       help='Output directory for results (default: clustering_results)')
    
    parser.add_argument('--config-path', '-c',
                       help='Path to configuration file (default: auto-detect)')
    
    parser.add_argument('--data-path', '-d',
                       help='Path to data file (default: use config setting)')
    
    parser.add_argument('--force-umap', action='store_true',
                       help='Force UMAP + HDBSCAN pipeline (ignores config UMAP setting)')
    
    parser.add_argument('--no-umap', action='store_true',
                       help='Force direct HDBSCAN pipeline (no UMAP)')
    
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate configuration, do not run pipeline')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress detailed progress output')
    
    args = parser.parse_args()
    
    # Validate mutually exclusive arguments
    if args.force_umap and args.no_umap:
        print("❌ Error: --force-umap and --no-umap cannot be used together.")
        return 1
    
    # Header
    if not args.quiet:
        print("Simplified HDBSCAN Clustering Script")
        print("=" * 50)
    
    # Determine configuration path
    if args.config_path:
        config_path = args.config_path
    else:
        config_path = os.path.join(os.path.dirname(__file__), 
                                  '../../source_code_package/config/config_cluster.yaml')
    
    # Determine output directory (convert to absolute path)
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(os.path.dirname(__file__), '../../', args.output_dir)
    else:
        output_dir = args.output_dir
    
    if not args.quiet:
        print(f"📁 Configuration: {os.path.relpath(config_path)}")
        print(f"📂 Output directory: {os.path.relpath(output_dir)}")
        print()
    
    # Step 1: Validate configuration
    if not args.quiet:
        print("Step 1: Configuration Validation")
        print("-" * 40)
    
    if not validate_configuration(config_path):
        print("❌ Configuration validation failed. Please fix configuration errors.")
        return 1
    
    if args.validate_only:
        print("✅ Configuration validation complete. Exiting (--validate-only specified).")
        return 0
    
    if not args.quiet:
        print()
    
    # Step 2: Determine UMAP setting
    force_umap = None
    if args.force_umap:
        force_umap = True
        umap_setting = "Forced UMAP usage"
    elif args.no_umap:
        force_umap = False
        umap_setting = "Forced direct HDBSCAN"
    else:
        umap_setting = "Auto-detect from config"
    
    if not args.quiet:
        print("Step 2: Pipeline Configuration")
        print("-" * 40)
        print(f"🔧 UMAP setting: {umap_setting}")
        print()
    
    # Step 3: Run pipeline
    if not args.quiet:
        print("Step 3: Pipeline Execution")
        print("-" * 40)
    
    try:
        results = run_clustering_pipeline(
            config_path=config_path,
            data_path=args.data_path,
            output_dir=output_dir,
            force_umap=force_umap
        )
        
        if not args.quiet:
            print()
            print("Step 4: Results Summary")
            print("-" * 40)
        
        display_results_summary(results)
        
        if results['success']:
            if not args.quiet:
                print()
                print("📂 Output files saved to:")
                print(f"    {os.path.relpath(output_dir)}")
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if not args.quiet:
            import traceback
            print()
            print("🔍 Full error details:")
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
