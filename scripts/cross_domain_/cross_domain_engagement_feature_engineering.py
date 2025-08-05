#!/usr/bin/env python3
"""
Execution script for Cross Domain Engagement Score Feature Engineering using Shannon Entropy.

This script serves as a simple execution wrapper around the core functionality
in source_code_package.features.cross_domain_engagement_features module.

The script calculates CROSS_DOMAIN_ENGAGEMENT_SCORE using Shannon entropy based on 
event count proportions:

H(X) = -Σ(i=1 to n) pi × log₂(pi)

Where pi is the proportion of each event type, and the result is normalized 
to [0,1] scale by dividing by the maximum possible entropy.

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
    from features.cross_domain_engagement_features import cross_domain_engagement_pipeline
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure source_code_package structure is correct")
    exit(1)


def main():
    """Main function to execute cross domain engagement score feature engineering."""
    
    parser = argparse.ArgumentParser(
        description="Calculate CROSS_DOMAIN_ENGAGEMENT_SCORE using Shannon entropy for crypto wallet analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cross_domain_engagement_feature_engineering.py
  python scripts/cross_domain_engagement_feature_engineering.py --input data/raw_data/custom_data.csv
  python scripts/cross_domain_engagement_feature_engineering.py --output data/processed_data/custom_output.csv
  python scripts/cross_domain_engagement_feature_engineering.py --config source_code_package/config/custom_config.yaml

Shannon Entropy Formula:
  H(X) = -Σ(pi × log₂(pi))
  
  Where:
  - pi = proportion of event type i
  - Result normalized to [0,1] by dividing by log₂(number_of_categories)
  
Cross Domain Engagement Interpretation:
  0.0 - 0.2: Specialist (focuses on 1-2 activity types)
  0.2 - 0.4: Somewhat focused
  0.4 - 0.6: Moderate cross-domain engagement  
  0.6 - 0.8: High cross-domain engagement
  0.8 - 1.0: Generalist (very diverse engagement portfolio)
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
        help='Path to configuration YAML file (default: config_cross_domain_engagement.yaml)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving results to file (useful for testing)'
    )
    
    args = parser.parse_args()
    
    print("CROSS DOMAIN ENGAGEMENT SCORE FEATURE ENGINEERING")
    print("=" * 60)
    print("Calculating CROSS_DOMAIN_ENGAGEMENT_SCORE using Shannon entropy")
    print("\nMethod: Event Count Proportions → Shannon Entropy → Normalized Score")
    print("Formula: H(X) = -Σ(pi × log₂(pi)) / log₂(max_categories)")
    print("\nEvent Types Analyzed (from config):")
    print("• DEX_EVENTS, GAMES_EVENTS, CEX_EVENTS, DAPP_EVENTS")
    print("• CHADMIN_EVENTS, DEFI_EVENTS, BRIDGE_EVENTS") 
    print("• NFT_EVENTS, TOKEN_EVENTS, FLOTSAM_EVENTS")
    print("")
    
    # Prepare configuration path
    if args.config:
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(project_root, config_path)
    else:
        config_path = os.path.join(project_root, 'source_code_package', 'config', 'config_cross_domain_engagement.yaml')
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"⚠️  Configuration file not found: {config_path}")
        print("Proceeding with default configuration...")
        config_path = None
    
    try:
        # Execute the cross domain engagement score feature engineering pipeline
        df_result, processing_info = cross_domain_engagement_pipeline(
            data_path=args.input,
            config_path=config_path,
            output_path=args.output,
            save_results=not args.no_save
        )
        
        print("\n🎉 Cross Domain Engagement Score Feature Engineering completed successfully!")
        print(f"\nProcessing Summary:")
        print(f"• Input file: {processing_info['input_file']}")
        print(f"• Original data: {processing_info['original_shape'][0]:,} wallets, {processing_info['original_shape'][1]} features")
        print(f"• Processed data: {processing_info['processed_shape'][0]:,} wallets, {processing_info['processed_shape'][1]} features")
        print(f"• Event types found: {len(processing_info['event_columns_found'])}")
        print(f"• New features added: {len(processing_info['new_features_added'])}")
        
        if processing_info['output_file']:
            print(f"• Output saved to: {processing_info['output_file']}")
        
        # Display statistics about the engagement scores
        stats = processing_info['statistics']
        print(f"\nCross Domain Engagement Score Statistics:")
        print(f"• Mean: {stats['mean_diversity_score']:.4f}")
        print(f"• Median: {stats['median_diversity_score']:.4f}")  
        print(f"• Standard Deviation: {stats['std_diversity_score']:.4f}")
        print(f"• Min: {stats['min_diversity_score']:.4f}")
        print(f"• Max: {stats['max_diversity_score']:.4f}")
        
        # Show distribution by engagement categories
        engagement_scores = df_result['CROSS_DOMAIN_ENGAGEMENT_SCORE']
        
        print(f"\nCross Domain Engagement Distribution:")
        specialist = (engagement_scores <= 0.2).sum()
        focused = ((engagement_scores > 0.2) & (engagement_scores <= 0.4)).sum()
        moderate = ((engagement_scores > 0.4) & (engagement_scores <= 0.6)).sum()
        diverse = ((engagement_scores > 0.6) & (engagement_scores <= 0.8)).sum()
        generalist = (engagement_scores > 0.8).sum()
        
        total = len(engagement_scores)
        print(f"• Specialists (0.0-0.2): {specialist:,} wallets ({specialist/total*100:.1f}%)")
        print(f"• Focused (0.2-0.4): {focused:,} wallets ({focused/total*100:.1f}%)")
        print(f"• Moderate (0.4-0.6): {moderate:,} wallets ({moderate/total*100:.1f}%)")
        print(f"• Diverse (0.6-0.8): {diverse:,} wallets ({diverse/total*100:.1f}%)")
        print(f"• Generalists (0.8-1.0): {generalist:,} wallets ({generalist/total*100:.1f}%)")
        
        # Show top 5 most engaged wallets
        print(f"\nTop 5 Most Cross-Domain Engaged Wallets:")
        top_engaged = df_result.nlargest(5, 'CROSS_DOMAIN_ENGAGEMENT_SCORE')[['WALLET', 'CROSS_DOMAIN_ENGAGEMENT_SCORE', 'TOTAL_EVENTS']]
        for idx, (_, row) in enumerate(top_engaged.iterrows(), 1):
            print(f"{idx}. {row['WALLET']}: Engagement = {row['CROSS_DOMAIN_ENGAGEMENT_SCORE']:.4f}, Total Events = {row['TOTAL_EVENTS']}")
        
        print(f"\n✅ Ready for further analysis and modeling!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during cross domain engagement score feature engineering:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
