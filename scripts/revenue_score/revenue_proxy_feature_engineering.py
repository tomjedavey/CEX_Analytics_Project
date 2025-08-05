#!/usr/bin/env python3
"""
Execution script for Revenue Proxy Feature Engineering.

This script serves as a simple execution wrapper around the core functionality
in source_code_package.features.revenue_proxy_features module.

The script calculates REVENUE_SCORE_PROXY for cryptocurrency wallets using the formula:
REVENUE_SCORE_PROXY = 0.4 * AVG_TRANSFER_USD * TX_PER_MONTH + 
                     0.35 * (DEX_EVENTS + DEFI_EVENTS) * AVG_TRANSFER_USD + 
                     0.25 * BRIDGE_TOTAL_VOLUME_USD

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
    from features.revenue_proxy_features import revenue_proxy_feature_pipeline
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure source_code_package structure is correct")
    exit(1)


def main():
    """Main function to execute revenue proxy feature engineering."""
    
    parser = argparse.ArgumentParser(
        description="Calculate REVENUE_SCORE_PROXY feature for crypto wallet analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/revenue_proxy_feature_engineering.py
  python scripts/revenue_proxy_feature_engineering.py --input data/raw_data/custom_data.csv
  python scripts/revenue_proxy_feature_engineering.py --output data/processed_data/custom_output.csv
  python scripts/revenue_proxy_feature_engineering.py --config source_code_package/config/custom_config.yaml
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input CSV file (default: from config)'
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str,
        help='Path to output CSV file (default: from config)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file (default: config_revenue_proxy.yaml)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving results to file (useful for testing)'
    )
    
    args = parser.parse_args()
    
    print("REVENUE PROXY FEATURE ENGINEERING")
    print("=" * 50)
    print("Calculating REVENUE_SCORE_PROXY for crypto wallet analysis")
    print("\nFormula Components:")
    print("• Transaction Activity (40%): AVG_TRANSFER_USD × TX_PER_MONTH") 
    print("• DEX/DeFi Activity (35%): (DEX_EVENTS + DEFI_EVENTS) × AVG_TRANSFER_USD")
    print("• Bridge Activity (25%): BRIDGE_TOTAL_VOLUME_USD")
    print("")
    
    # Prepare configuration path
    if args.config:
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(project_root, config_path)
    else:
        config_path = os.path.join(project_root, 'source_code_package', 'config', 'config_revenue_proxy.yaml')
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"⚠️  Configuration file not found: {config_path}")
        print("Proceeding with default configuration...")
        config_path = None
    
    try:
        # Execute the revenue proxy feature engineering pipeline
        df_result, processing_info = revenue_proxy_feature_pipeline(
            data_path=args.input,
            config_path=config_path,
            output_path=args.output,
            save_results=not args.no_save
        )
        
        print("\n🎉 Revenue Proxy Feature Engineering completed successfully!")
        print(f"\nProcessing Summary:")
        print(f"• Input file: {processing_info['input_file']}")
        print(f"• Original data: {processing_info['original_shape'][0]:,} wallets, {processing_info['original_shape'][1]} features")
        print(f"• Processed data: {processing_info['processed_shape'][0]:,} wallets, {processing_info['processed_shape'][1]} features")
        print(f"• New features added: {len(processing_info['new_features_added'])}")
        
        if processing_info['output_file']:
            print(f"• Output saved to: {processing_info['output_file']}")
        
        # Display some statistics about the revenue proxy scores
        revenue_scores = df_result['REVENUE_SCORE_PROXY']
        print(f"\nRevenue Proxy Score Statistics:")
        print(f"• Mean: ${revenue_scores.mean():.2f}")
        print(f"• Median: ${revenue_scores.median():.2f}")  
        print(f"• Standard Deviation: ${revenue_scores.std():.2f}")
        print(f"• Min: ${revenue_scores.min():.2f}")
        print(f"• Max: ${revenue_scores.max():.2f}")
        
        # Show top 5 wallets by revenue proxy score
        print(f"\nTop 5 Wallets by Revenue Proxy Score:")
        top_wallets = df_result.nlargest(5, 'REVENUE_SCORE_PROXY')[['WALLET', 'REVENUE_SCORE_PROXY']]
        for idx, (_, row) in enumerate(top_wallets.iterrows(), 1):
            print(f"{idx}. {row['WALLET']}: ${row['REVENUE_SCORE_PROXY']:.2f}")
        
        print(f"\n✅ Ready for further analysis and modeling!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during revenue proxy feature engineering:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
