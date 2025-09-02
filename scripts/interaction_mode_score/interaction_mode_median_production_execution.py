#!/usr/bin/env python3
"""
Interaction Mode Cluster Selection + Median Production Script

This script addresses the sparse data problem by requiring minimum activity 
thresholds for cluster selection, ensuring selected clusters represent meaningful 
activity levels rather than just large groups of inactive wallets.
"""

import argparse

import sys
import os
# Dynamically add the absolute path to source_code_package to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
source_code_path = os.path.join(project_root, 'source_code_package')
if source_code_path not in sys.path:
    sys.path.insert(0, source_code_path)


import importlib.util
features_path = os.path.join(source_code_path, 'features', 'interaction_mode_median_production_source.py')
spec = importlib.util.spec_from_file_location('interaction_mode_median_production_source', features_path)
interaction_mode_median_production_source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interaction_mode_median_production_source)
calculate_median_feature_values_for_clusters = interaction_mode_median_production_source.calculate_median_feature_values_for_clusters

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Interaction Mode Cluster Selection with Activity Requirements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard usage with 10% activity threshold
  python3 %(prog)s --threshold 0.1
  
  # Strict selection requiring 25% activity
  python3 %(prog)s --threshold 0.25 --min-size 100
  
  # Lenient selection for sparse features
  python3 %(prog)s --threshold 0.05 --min-size 20
  
  # Custom paths
  python3 %(prog)s --results-dir /path/to/results --output /path/to/output.yaml
        """
    )
    
    parser.add_argument(
        '--results-dir', 
        type=str,
        default='data/processed_data/interaction_mode_results',
        help='Directory containing interaction mode clustering results (should be data/processed_data/interaction_mode_results)'
    )

    parser.add_argument(
        '--output', 
        type=str,
        default='data/processed_data/interaction_mode_results/interaction_mode_cluster_selections.yaml',
        help='Output file path for cluster selections (should be in data/processed_data/interaction_mode_results)'
    )
    
    parser.add_argument(
        '--threshold', 
        type=float,
        default=0.3,
        help='Minimum activity threshold (0.0-1.0). Default: 0.3 (30%% of wallets must be active)'
    )
    
    parser.add_argument(
        '--min-size', 
        type=int,
        default=50,
        help='Minimum cluster size to consider. Default: 50'
    )
    
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Overwrite existing output files'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("Activity threshold must be between 0.0 and 1.0")
    
    if args.min_size < 1:
        parser.error("Minimum cluster size must be at least 1")
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        print("Make sure you've run the interaction mode clustering pipeline first.")
        return 1
    
    # Check if output file exists (unless force)
    if os.path.exists(args.output) and not args.force:
        response = input(f"Output file {args.output} exists. Overwrite? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled.")
            return 0
    
    print(f"🚀 Starting Cluster Selection")
    print(f"📁 Results Directory: {args.results_dir}")
    print(f"🎯 Activity Threshold: {args.threshold*100:.1f}%")
    print(f"👥 Minimum Cluster Size: {args.min_size}")
    print(f"💾 Output: {args.output}")
    print()
    
    try:
        # Run the cluster selection
        results = calculate_median_feature_values_for_clusters(
            results_dir=args.results_dir,
            min_activity_threshold=args.threshold,
            min_cluster_size=args.min_size,
            output_path=args.output
        )
        print(f"\n✅ Cluster selection completed successfully!")
        # Provide interpretation guidance
        print(f"\n💡 INTERPRETATION:")
        print(f"   The algorithm now requires clusters to have at least {args.threshold*100:.1f}% active wallets.")
        print(f"   This ensures selected clusters represent meaningful activity levels.")
        print(f"   If no clusters meet the threshold, the algorithm falls back to the most active cluster.")
        # Show compliance summary
        summary = results.get('summary', {})
        if 'activity_threshold_met' in summary:
            compliance_rate = summary['activity_threshold_met'] / max(summary['successful_selections'], 1) * 100
            print(f"   Threshold Compliance: {compliance_rate:.1f}% of selections met the activity requirement")
        return 0
        
    except Exception as e:
        print(f"❌ Error during cluster selection: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
