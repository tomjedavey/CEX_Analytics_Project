#!/usr/bin/env python3
"""
Execution script for Behavioral Volatility Score Feature Engineering.

This script serves as a simple execution wrapper around the core functionality
in source_code_package.features.behavioral_volatility_features module.

The script calculates BEHAVIORAL_VOLATILITY_SCORE based on three main components:
1. Financial Volatility (35%): USD_TRANSFER_STDDEV / AVG_TRANSFER_USD
2. Activity Volatility (40%): Composite of CV, variance ratio, and Gini coefficient
3. Exploration Volatility (25%): Exploration intensity based on diversity metrics

The behavioral volatility score measures the inconsistency and unpredictability 
of wallet behavior across financial, activity, and exploration dimensions.

Author: Tom Davey
Date: August 2025
"""

import os
import sys
import argparse

# Add source_code_package to path
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
source_package_path = os.path.join(project_root, 'source_code_package')
sys.path.insert(0, source_package_path)

try:
    from source_code_package.features.behavioral_volatility_features import behavioral_volatility_pipeline
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure source_code_package structure is correct")
    exit(1)


def main():
    """Main function to execute behavioral volatility score feature engineering."""
    
    parser = argparse.ArgumentParser(
        description="Calculate BEHAVIORAL_VOLATILITY_SCORE for crypto wallet behavioral pattern analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/behavioral_volatility_feature_engineering.py
  python scripts/behavioral_volatility_feature_engineering.py --input data/raw_data/custom_data.csv
  python scripts/behavioral_volatility_feature_engineering.py --output data/processed_data/custom_output.csv
  python scripts/behavioral_volatility_feature_engineering.py --config source_code_package/config/custom_config.yaml

Component Details:
  Financial Volatility (35%):
    - Measures transfer amount volatility: USD_TRANSFER_STDDEV / AVG_TRANSFER_USD
    - Higher values indicate inconsistent transfer patterns
  
  Activity Volatility (40%):
    - Coefficient of Variance (40%): CV of activity counts across event types
    - Variance Ratio (30%): Variance ratio from uniform distribution
    - Gini Coefficient (30%): Inequality measure of activity distribution
  
  Exploration Volatility (25%):
    - Exploration intensity: Average diversity / TX_PER_MONTH
    - Square root transformation applied to intensity value
    - Measures exploration relative to activity rate
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input CSV file (default: from config file)'
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str,
        help='Path to output CSV file (default: from config file)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file (default: source_code_package/config/config_behavioral_volatility.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()
    
    if args.verbose:
        print("Starting Behavioral Volatility Score Feature Engineering...")
        print(f"Input file: {args.input or 'from config'}")
        print(f"Output file: {args.output or 'from config'}")
        print(f"Config file: {args.config or 'default'}")
        print("-" * 60)
    
    try:
        # Execute the pipeline
        df_result, config = behavioral_volatility_pipeline(
            input_path=args.input,
            output_path=args.output,
            config_path=args.config
        )
        
        if args.verbose:
            print("\n" + "="*60)
            print("EXECUTION SUMMARY")
            print("="*60)
            print(f"Successfully processed {len(df_result)} wallet records")
            print(f"Generated behavioral volatility features:")
            print(f"  - FINANCIAL_VOLATILITY")
            print(f"  - ACTIVITY_VOLATILITY") 
            print(f"  - EXPLORATION_VOLATILITY")
            print(f"  - BEHAVIORAL_VOLATILITY_SCORE_RAW")
            print(f"  - BEHAVIORAL_VOLATILITY_SCORE")
            
            # Component weights used
            weights = config['features']['component_weights']
            print(f"\nComponent weights used:")
            print(f"  - Financial Volatility: {weights['financial_volatility']:.1%}")
            print(f"  - Activity Volatility: {weights['activity_volatility']:.1%}")
            print(f"  - Exploration Volatility: {weights['exploration_volatility']:.1%}")
            
            normalize = config['features']['normalize_score']
            method = config['features']['normalization_method'] if normalize else "none"
            print(f"\nNormalization: {method}")
        
        print("\n✅ Behavioral Volatility Score feature engineering completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
