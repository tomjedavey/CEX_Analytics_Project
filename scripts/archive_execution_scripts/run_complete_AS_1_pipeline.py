#!/usr/bin/env python3
"""
AS_1 Complete Pipeline Execution Script

This script runs the complete AS_1 linear regression pipeline for cluster analysis:
1. Model Training for all datasets
2. Model Testing for all datasets  
3. Comparative Analysis across datasets
4. Final reporting and recommendations

Author: Tom Davey
Date: July 2025
"""

import os
import sys
import subprocess
from datetime import datetime

# Add source_code_package to path
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
source_package_path = os.path.join(project_root, 'source_code_package')
sys.path.insert(0, source_package_path)


def run_script(script_path, script_name):
    """Run a Python script and capture its output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("CHECKING PREREQUISITES")
    print("=" * 30)
    
    # Check if processed data exists
    processed_files = [
        'data/processed_data/AS_1_feature_data_full_dataset.csv',
        'data/processed_data/AS_1_feature_data_cluster_0.csv',
        'data/processed_data/AS_1_feature_data_cluster_1.csv'
    ]
    
    missing_files = []
    for file_path in processed_files:
        full_path = os.path.join(project_root, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
        else:
            print(f"✅ Found: {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing processed data files:")
        for file in missing_files:
            print(f"   - {file}")
        print(f"\nPlease run feature engineering first:")
        print(f"   python scripts/AS_1/feature_engineering_AS_1_enhanced.py")
        return False
    
    print(f"✅ All prerequisites met")
    return True


def main():
    """Main function to run the complete AS_1 pipeline."""
    
    start_time = datetime.now()
    
    print("AS_1 COMPLETE PIPELINE EXECUTION")
    print("=" * 60)
    print("This script will run the complete AS_1 analysis pipeline:")
    print("1. Training linear regression models for all datasets")
    print("2. Testing models with comprehensive evaluation")
    print("3. Comparative analysis across datasets")
    print("4. Business insights and recommendations")
    print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Aborting pipeline.")
        return False
    
    # Define script paths
    scripts_dir = os.path.join(project_root, 'scripts', 'AS_1')
    
    pipeline_scripts = [
        (os.path.join(scripts_dir, 'train_model_AS_1_enhanced.py'), "Model Training"),
        (os.path.join(scripts_dir, 'test_model_AS_1_enhanced.py'), "Model Testing"),
        (os.path.join(scripts_dir, 'comparative_analysis_AS_1.py'), "Comparative Analysis")
    ]
    
    # Track results
    results = {}
    
    # Run pipeline
    for script_path, script_name in pipeline_scripts:
        if not os.path.exists(script_path):
            print(f"❌ Script not found: {script_path}")
            results[script_name] = False
            continue
        
        success = run_script(script_path, script_name)
        results[script_name] = success
        
        if not success:
            print(f"\n⚠️  {script_name} failed. Do you want to continue with the next step?")
            continue_choice = input("Continue pipeline? (y/n): ").lower().strip()
            if continue_choice != 'y':
                print("Pipeline aborted by user.")
                break
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("PIPELINE EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    
    successful_steps = sum(1 for success in results.values() if success)
    total_steps = len(results)
    
    print(f"\nSteps completed: {successful_steps}/{total_steps}")
    
    for step_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {step_name}: {status}")
    
    if successful_steps == total_steps:
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("• Review the comparative analysis report")
        print("• Check model performance metrics")
        print("• Consider implementing the recommended model")
        print("• Monitor model performance in production")
        
        # Show output locations
        print(f"\nOutput Locations:")
        print(f"• Models: linear_regression_model_*.pkl")
        print(f"• Test Results: data/scores/AS_1_test_results_*.csv")
        print(f"• Metrics: data/scores/AS_1_*metrics*.json")
        print(f"• Analysis Report: data/reports/AS_1_comparative_analysis_*.json")
        print(f"• Logs: data/logs/AS_1_*.log")
        
        return True
    else:
        print(f"\n⚠️  Pipeline completed with {total_steps - successful_steps} failed steps.")
        print("Please review the error messages above and resolve issues.")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
