#!/usr/bin/env python3
"""
AS_1 Comparative Analysis Execution Script

This script serves as a execution wrapper around the AS_1 comparative analysis functionality
in source_code_package.models.AS_1_functionality.comparative_analysis_source module.

Author: Tom Davey
Date: July 2025
"""

import os
import sys

# Add source_code_package to path
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
source_package_path = os.path.join(project_root, 'source_code_package')
sys.path.insert(0, source_package_path)

try:
    from models.AS_1_functionality.comparative_analysis_source import compare_as1_models
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure source_code_package structure is correct")
    exit(1)


def main():
    """Main function to execute AS_1 comparative analysis."""
    
    # Define configuration paths
    config_dir = os.path.join(project_root, 'source_code_package', 'config')
    
    config_paths = [
        os.path.join(config_dir, 'config_AS_1_full_dataset.yaml'),
        os.path.join(config_dir, 'config_AS_1_cluster_0.yaml'),
        os.path.join(config_dir, 'config_AS_1_cluster_1.yaml')
    ]
    
    print("AS_1 COMPARATIVE ANALYSIS")
    print("=" * 50)
    print("This script will compare AS_1 linear regression models across:")
    print("1. Full dataset model")
    print("2. Cluster 0 dataset model")
    print("3. Cluster 1 dataset model")
    print("\nEnsure that both training and testing have been completed for all models.")
    
    # Check if required result files exist
    required_files = []
    
    # Check for training metrics
    for config_path in config_paths:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        metrics_path = config.get('output', {}).get('metrics_path', 'data/scores/AS_1_metrics.json')
        if not os.path.isabs(metrics_path):
            metrics_path = os.path.join(project_root, metrics_path)
        required_files.append(metrics_path)
        
        # Test metrics path
        test_metrics_path = metrics_path.replace('metrics', 'test_metrics')
        required_files.append(test_metrics_path)
        
        # Test results CSV
        test_results_path = config.get('output', {}).get('test_results_path', 'data/scores/AS_1_test_results.csv')
        if not os.path.isabs(test_results_path):
            test_results_path = os.path.join(project_root, test_results_path)
        required_files.append(test_results_path)
    
    missing_files = [path for path in required_files if not os.path.exists(path)]
    
    if missing_files:
        print("\n⚠️  WARNING: Missing result files!")
        print("Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure that both training and testing have been completed.")
        print("Run the following scripts in order:")
        print("1. train_model_AS_1_enhanced.py")
        print("2. test_model_AS_1_enhanced.py")
        
        proceed = input("\nProceed with available data? (y/n): ").lower().strip()
        if proceed != 'y':
            print("Analysis aborted.")
            return False
    
    print(f"\nRunning comparative analysis...")
    
    try:
        # Run comprehensive comparative analysis
        results = compare_as1_models(config_paths, verbose=True)
        
        print(f"\n{'='*60}")
        print("COMPARATIVE ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        # Performance summary
        if 'performance' in results:
            perf_results = results['performance']
            
            if 'best_performers' in perf_results:
                best = perf_results['best_performers']
                print(f"\n🏆 BEST PERFORMERS:")
                print(f"   Highest R²: {best['highest_test_r2']['dataset']} ({best['highest_test_r2']['value']:.4f})")
                print(f"   Lowest RMSE: {best['lowest_test_rmse']['dataset']} ({best['lowest_test_rmse']['value']:.4f})")
                print(f"   Lowest Overfitting: {best['lowest_overfitting']['dataset']} (gap: {best['lowest_overfitting']['value']:.4f})")
            
            if 'summary_table' in perf_results:
                print(f"\n📊 PERFORMANCE COMPARISON:")
                import pandas as pd
                summary_df = pd.DataFrame(perf_results['summary_table'])
                if not summary_df.empty:
                    print(summary_df[['Dataset', 'Test_R2', 'Test_RMSE', 'Test_MAE', 'Overfitting_Gap_R2']].to_string(index=False))
        
        # Business insights
        if 'insights' in results:
            insights = results['insights']
            
            print(f"\n💡 KEY FINDINGS:")
            for finding in insights.get('key_findings', []):
                print(f"   • {finding}")
            
            if 'model_selection_guidance' in insights:
                guidance = insights['model_selection_guidance']
                print(f"\n🎯 RECOMMENDED MODEL: {guidance['recommended_model']}")
                print(f"   Reasoning: {guidance['reasoning']}")
                
                if guidance.get('alternative_considerations'):
                    print(f"   Alternative considerations:")
                    for consideration in guidance['alternative_considerations']:
                        print(f"   • {consideration}")
            
            print(f"\n📋 RECOMMENDATIONS:")
            for rec in insights.get('recommendations', []):
                print(f"   • {rec}")
        
        print(f"\n📄 Detailed report saved to: {results.get('report_path', 'data/reports/')}")
        
        print(f"\n🎉 Comparative analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Comparative analysis failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
