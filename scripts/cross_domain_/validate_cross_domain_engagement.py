#!/usr/bin/env python3
"""
Cross Domain Engagement Score Feature Validation and Analysis Script

This script validates the calculated CROSS_DOMAIN_ENGAGEMENT_SCORE feature using Shannon entropy
and provides comprehensive analysis to understand cross-domain engagement patterns in 
cryptocurrency wallet behavior.

Author: Tom Davey  
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """Load the cross domain engagement features and validate the data structure."""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cross domain engagement features file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded data with {len(df)} wallets and {len(df.columns)} features")
    
    # Validate required columns exist
    required_cols = [
        'CROSS_DOMAIN_ENGAGEMENT_SCORE', 
        'TOTAL_EVENTS'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for proportion columns
    proportion_cols = [col for col in df.columns if col.endswith('_PROPORTION')]
    print(f"Found {len(proportion_cols)} proportion columns")
    
    print("✅ All required columns present")
    return df


def validate_shannon_entropy_calculations(df: pd.DataFrame, tolerance: float = 1e-6) -> bool:
    """Validate that the Shannon entropy calculations are mathematically correct."""
    
    print("\nValidating Shannon entropy calculations...")
    
    # Get proportion columns
    proportion_cols = [col for col in df.columns if col.endswith('_PROPORTION')]
    
    if len(proportion_cols) == 0:
        print("❌ No proportion columns found")
        return False
    
    print(f"Validating entropy for {len(proportion_cols)} event types")
    
    errors = []
    max_error = 0
    
    # Validate a sample of wallets (first 100 for performance)
    sample_size = min(100, len(df))
    sample_df = df.head(sample_size)
    
    for idx, row in sample_df.iterrows():
        # Get proportions for this wallet
        proportions = row[proportion_cols].values.astype(float)
        
        # Check if proportions sum to 1 (for wallets with activity)
        if row['TOTAL_EVENTS'] > 0:
            prop_sum = np.sum(proportions)
            if abs(prop_sum - 1.0) > tolerance:
                errors.append(f"Wallet {idx}: proportions sum to {prop_sum:.6f}, not 1.0")
        
        # Recalculate entropy
        filtered_props = proportions[proportions > 0]
        if len(filtered_props) > 0:
            # Calculate entropy manually
            manual_entropy = -np.sum(filtered_props * np.log2(filtered_props))
            
            # Normalize by max entropy for this wallet
            max_entropy = np.log2(len(filtered_props))
            if max_entropy > 0:
                manual_normalized = manual_entropy / max_entropy
            else:
                manual_normalized = 0.0
                
            # Compare with stored value
            stored_entropy = row['CROSS_DOMAIN_ENGAGEMENT_SCORE']
            error = abs(stored_entropy - manual_normalized)
            max_error = max(max_error, error)
            
            if error > tolerance:
                errors.append(f"Wallet {idx}: entropy difference {error:.2e}")
    
    print(f"Validation Results:")
    print(f"  Sample size: {sample_size} wallets")
    print(f"  Max entropy calculation error: {max_error:.2e}")
    print(f"  Errors found: {len(errors)}")
    
    if len(errors) > 0:
        print("  First few errors:")
        for error in errors[:5]:
            print(f"    {error}")
    
    is_valid = len(errors) == 0 and max_error < tolerance
    
    if is_valid:
        print("✅ Shannon entropy calculations are correct")
    else:
        print("❌ Shannon entropy calculation errors detected")
    
    return is_valid


def analyze_engagement_distribution(df: pd.DataFrame) -> dict:
    """Analyze the distribution of cross domain engagement scores."""
    
    print("\nAnalyzing cross domain engagement score distribution...")
    
    engagement_scores = df['CROSS_DOMAIN_ENGAGEMENT_SCORE']
    
    # Basic statistics
    stats = {
        'count': len(engagement_scores),
        'mean': engagement_scores.mean(),
        'median': engagement_scores.median(),
        'std': engagement_scores.std(),
        'min': engagement_scores.min(),
        'max': engagement_scores.max(),
        'q25': engagement_scores.quantile(0.25),
        'q75': engagement_scores.quantile(0.75),
        'q95': engagement_scores.quantile(0.95),
        'q99': engagement_scores.quantile(0.99)
    }
    
    print(f"Cross Domain Engagement Score Statistics:")
    print(f"  Count: {stats['count']:,}")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  Std Dev: {stats['std']:.4f}")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    print(f"  25th percentile: {stats['q25']:.4f}")
    print(f"  75th percentile: {stats['q75']:.4f}")
    print(f"  95th percentile: {stats['q95']:.4f}")
    print(f"  99th percentile: {stats['q99']:.4f}")
    
    # Distribution characteristics
    zero_engagement = (engagement_scores == 0).sum()
    perfect_engagement = (engagement_scores == 1.0).sum()
    
    print(f"\nDistribution Characteristics:")
    print(f"  Wallets with zero engagement: {zero_engagement:,} ({zero_engagement/len(df)*100:.1f}%)")
    print(f"  Wallets with perfect engagement: {perfect_engagement:,} ({perfect_engagement/len(df)*100:.1f}%)")
    
    return stats


def analyze_engagement_patterns(df: pd.DataFrame) -> dict:
    """Analyze engagement patterns by cross domain engagement score."""
    
    print("\nAnalyzing engagement patterns by cross domain engagement score...")
    
    # Define engagement categories
    engagement_scores = df['CROSS_DOMAIN_ENGAGEMENT_SCORE']
    
    categories = {
        'Specialists': df[engagement_scores <= 0.2],
        'Focused': df[(engagement_scores > 0.2) & (engagement_scores <= 0.4)],
        'Moderate': df[(engagement_scores > 0.4) & (engagement_scores <= 0.6)],
        'Diverse': df[(engagement_scores > 0.6) & (engagement_scores <= 0.8)],
        'Generalists': df[engagement_scores > 0.8]
    }
    
    # Use hardcoded event columns (should match config)
    event_cols = [
        'DEX_EVENTS', 'GAMES_EVENTS', 'CEX_EVENTS', 'DAPP_EVENTS',
        'CHADMIN_EVENTS', 'DEFI_EVENTS', 'BRIDGE_EVENTS', 
        'NFT_EVENTS', 'TOKEN_EVENTS', 'FLOTSAM_EVENTS'
    ]
    
    print(f"Engagement Pattern Analysis by Cross Domain Engagement Category:")
    print(f"{'Category':<12} {'Count':<8} {'Avg Events':<12} {'Most Common Activity':<25}")
    print("-" * 70)
    
    pattern_analysis = {}
    
    for cat_name, cat_df in categories.items():
        if len(cat_df) > 0:
            avg_total_events = cat_df['TOTAL_EVENTS'].mean()
            
            # Find most common activity type
            event_means = cat_df[event_cols].mean()
            most_common_activity = event_means.idxmax()
            most_common_value = event_means.max()
            
            pattern_analysis[cat_name] = {
                'count': len(cat_df),
                'avg_total_events': avg_total_events,
                'most_common_activity': most_common_activity,
                'most_common_value': most_common_value
            }
            
            print(f"{cat_name:<12} {len(cat_df):<8,} {avg_total_events:<12.1f} {most_common_activity:<25}")
    
    return pattern_analysis


def analyze_entropy_vs_activity_relationship(df: pd.DataFrame) -> dict:
    """Analyze the relationship between total activity and cross domain engagement."""
    
    print("\nAnalyzing relationship between total activity and cross domain engagement...")
    
    # Calculate correlation
    correlation = df['CROSS_DOMAIN_ENGAGEMENT_SCORE'].corr(df['TOTAL_EVENTS'])
    
    print(f"Correlation between engagement score and total events: {correlation:.4f}")
    
    # Analyze by activity level
    total_events = df['TOTAL_EVENTS']
    engagement = df['CROSS_DOMAIN_ENGAGEMENT_SCORE']
    
    # Create activity level bins
    activity_bins = {
        'Very Low (0-1 events)': df[total_events <= 1],
        'Low (2-10 events)': df[(total_events > 1) & (total_events <= 10)],
        'Medium (11-50 events)': df[(total_events > 10) & (total_events <= 50)],
        'High (51-200 events)': df[(total_events > 50) & (total_events <= 200)],
        'Very High (200+ events)': df[total_events > 200]
    }
    
    print(f"\nCross Domain Engagement by Activity Level:")
    print(f"{'Activity Level':<25} {'Count':<8} {'Avg Engagement':<15} {'Engagement Range':<20}")
    print("-" * 75)
    
    activity_analysis = {}
    
    for level_name, level_df in activity_bins.items():
        if len(level_df) > 0:
            avg_engagement = level_df['CROSS_DOMAIN_ENGAGEMENT_SCORE'].mean()
            min_engagement = level_df['CROSS_DOMAIN_ENGAGEMENT_SCORE'].min()
            max_engagement = level_df['CROSS_DOMAIN_ENGAGEMENT_SCORE'].max()
            
            activity_analysis[level_name] = {
                'count': len(level_df),
                'avg_engagement': avg_engagement,
                'min_engagement': min_engagement,
                'max_engagement': max_engagement
            }
            
            range_str = f"{min_engagement:.3f} - {max_engagement:.3f}"
            print(f"{level_name:<25} {len(level_df):<8,} {avg_engagement:<15.4f} {range_str:<20}")
    
    return {'correlation': correlation, 'by_activity_level': activity_analysis}


def main():
    """Main analysis function."""
    
    print("CROSS DOMAIN ENGAGEMENT SCORE FEATURE VALIDATION & ANALYSIS")
    print("=" * 70)
    
    # Load data
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    data_file = os.path.join(project_root, 'data', 'processed_data', 'cross_domain_engagement_features.csv')
    
    try:
        df = load_and_validate_data(data_file)
        
        # Validate calculations
        calc_valid = validate_shannon_entropy_calculations(df)
        
        # Analyze distribution
        distribution_stats = analyze_engagement_distribution(df)
        
        # Analyze engagement patterns
        pattern_analysis = analyze_engagement_patterns(df)
        
        # Analyze entropy vs activity relationship
        relationship_analysis = analyze_entropy_vs_activity_relationship(df)
        
        print(f"\n" + "="*70)
        print("SUMMARY & BUSINESS INSIGHTS")
        print("="*70)
        
        if calc_valid:
            print("✅ Shannon entropy calculations are mathematically correct")
        else:
            print("❌ Shannon entropy calculations have errors")
        
        print(f"\n📊 CROSS DOMAIN ENGAGEMENT INSIGHTS:")
        print(f"• {distribution_stats['count']:,} wallets analyzed")
        print(f"• Mean engagement score: {distribution_stats['mean']:.4f}")
        print(f"• Median engagement score: {distribution_stats['median']:.4f}")
        
        # Key insights from the analysis
        specialists_pct = (df['CROSS_DOMAIN_ENGAGEMENT_SCORE'] <= 0.2).sum() / len(df) * 100
        generalists_pct = (df['CROSS_DOMAIN_ENGAGEMENT_SCORE'] > 0.8).sum() / len(df) * 100
        
        print(f"\n🎯 WALLET ENGAGEMENT PATTERNS:")
        print(f"• Specialists (low engagement): {specialists_pct:.1f}% of wallets")
        print(f"• Generalists (high engagement): {generalists_pct:.1f}% of wallets")
        print(f"• Correlation with activity level: {relationship_analysis['correlation']:.4f}")
        
        print(f"\n💡 BUSINESS APPLICATIONS:")
        print(f"• Specialists: Target with focused features for specific domains")
        print(f"• Generalists: Offer comprehensive cross-domain platforms")
        print(f"• Medium engagement users: Prime candidates for domain expansion")
        print(f"• Zero engagement: New users or inactive accounts needing activation")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
