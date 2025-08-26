"""
Enhanced Interaction Mode Features - Version 2
Addresses the sparse data problem by requiring minimum activity thresholds.
"""

import os
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Tuple, Any, Optional
import logging

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = "source_code_package/config/config_interaction_mode.yaml"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_target_features(config: Dict[str, Any]) -> List[str]:
    """Extract target features from config."""
    return [
        "DEX_EVENTS",
        "CEX_EVENTS", 
        "DEFI_EVENTS",
        "BRIDGE_EVENTS"
    ]

def load_dataset_and_clusters(dataset_name: str, base_data_path: str, clusters_path: str) -> pd.DataFrame:
    """Load and merge dataset with cluster labels."""
    # The cluster files only contain cluster labels, we need the original data with features
    # Load the appropriate base dataset based on dataset name
    if dataset_name == 'main':
        df = pd.read_csv(base_data_path)  # This should be the main dataset with all features
    elif dataset_name == 'cluster_0':
        df = pd.read_csv('data/raw_data/cluster_datasets/new_raw_data_polygon_cluster_0.csv')
    elif dataset_name == 'cluster_1':
        df = pd.read_csv('data/raw_data/cluster_datasets/new_raw_data_polygon_cluster_1.csv')
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    # Load cluster labels
    clusters_df = pd.read_csv(clusters_path)
    
    # Add cluster labels to the dataset (index-based merge since they should align)
    if len(df) != len(clusters_df):
        raise ValueError(f"Dataset length {len(df)} doesn't match cluster labels length {len(clusters_df)}")
    
    # Use the correct column name from the cluster file
    df['cluster'] = clusters_df['cluster_label'].values
    
    return df

def calculate_cluster_density(df: pd.DataFrame, cluster_id: int) -> float:
    """Calculate cluster density (placeholder - returns 1.0 for now)."""
    return 1.0

#**NEED TO CHANGE THIS TO THE ACTUAL VALUE UTILISING THE PROBABILITIES PRODUCED FROM THE CLUSTERING RESULTS AT ONE POINT**

def calculate_enhanced_selection_score(
    cluster_size: int,
    density: float,
    median_nonzero_value: float,
    non_zero_proportion: float,
    min_activity_threshold: float = 0.3  # NEW: Minimum 30% activity required
) -> float:
    """
    Enhanced selection score with activity threshold requirement.
    Args:
        cluster_size: Number of wallets in cluster
        density: Cluster density (currently 1.0)
        median_nonzero_value: Median of non-zero feature values in cluster
        non_zero_proportion: Proportion of wallets with non-zero values
        min_activity_threshold: Minimum activity rate required for consideration
    Returns:
        Selection score (0 if below activity threshold)
    """
    # Reject clusters with insufficient activity
    if non_zero_proportion < min_activity_threshold:
        return 0.0
    # New scoring: score = (non-zero proportion) * (median non-zero value)
    score = non_zero_proportion * median_nonzero_value
    return score

def select_strongest_cluster_for_feature(
    df: pd.DataFrame, 
    feature: str,
    min_activity_threshold: float = 0.3,
    min_cluster_size: int = 50,  # NEW: Minimum cluster size
    prefer_high_activity: bool = True  # NEW: Preference for active clusters
) -> Tuple[int, Dict[str, Any]]:
    """
    Select the cluster with the strongest representation of a feature.
    
    Enhanced version that requires minimum activity levels.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature {feature} not found in dataset")
    
    # Remove noise points (cluster = -1)
    valid_df = df[df['cluster'] != -1].copy()
    
    best_cluster = -1
    best_score = -1
    best_stats = {}
    
    candidate_clusters = []
    
    for cluster_id in valid_df['cluster'].unique():
        cluster_data = valid_df[valid_df['cluster'] == cluster_id]
        # Skip small clusters
        if len(cluster_data) < min_cluster_size:
            continue
        feature_values = cluster_data[feature].values
        non_zero_values = feature_values[feature_values > 0]
        non_zero_count = len(non_zero_values)
        non_zero_proportion = non_zero_count / len(feature_values)
        median_nonzero_value = float(np.median(non_zero_values)) if non_zero_count > 0 else 0.0
        mean_val = np.mean(feature_values)
        median_val = np.median(feature_values)
        std_val = np.std(feature_values)
        variance = np.var(feature_values)
        density = calculate_cluster_density(cluster_data, cluster_id)
        score = calculate_enhanced_selection_score(
            cluster_size=len(cluster_data),
            density=density,
            median_nonzero_value=median_nonzero_value,
            non_zero_proportion=non_zero_proportion,
            min_activity_threshold=min_activity_threshold
        )
        stats = {
            'cluster_id': cluster_id,
            'cluster_size': len(cluster_data),
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'variance': variance,
            'non_zero_proportion': non_zero_proportion,
            'activity_count': non_zero_count,
            'median_nonzero_value': median_nonzero_value,
            'score': score
        }
        candidate_clusters.append(stats)
        if score > best_score:
            best_score = score
            best_cluster = cluster_id
            best_stats = stats
    
    # If no clusters meet activity threshold, fall back to highest activity rate
    if best_cluster == -1 and candidate_clusters:
        logging.warning(f"No clusters for {feature} meet activity threshold {min_activity_threshold}")
        logging.warning("Falling back to cluster with highest activity rate...")
        
        # Sort by activity rate, then by mean value, then by cluster size
        fallback_clusters = sorted(candidate_clusters, 
                                 key=lambda x: (x['non_zero_proportion'], x['mean'], x['cluster_size']), 
                                 reverse=True)
        
        if fallback_clusters:
            best_stats = fallback_clusters[0]
            best_cluster = best_stats['cluster_id']
            best_score = best_stats['score']
    
    if best_cluster == -1:
        raise ValueError(f"No valid clusters found for feature {feature}")
    
    return best_cluster, best_stats

def calculate_median_feature_values_for_clusters_v2(
    results_dir: str = "data/raw_data/interaction_mode_results",
    min_activity_threshold: float = 0.3,  # Require 30% activity minimum
    min_cluster_size: int = 50,
    output_path: str = "data/processed_data/interaction_mode_cluster_selections_v2.yaml"
) -> Dict[str, Any]:
    """
    Enhanced version: Calculate median feature values with activity requirements.
    
    Args:
        results_dir: Directory containing interaction mode clustering results
        min_activity_threshold: Minimum proportion of active wallets required (0.0-1.0)
        min_cluster_size: Minimum cluster size to consider
        output_path: Where to save results
    """
    
    # Load configuration
    config = load_config()
    target_features = get_target_features(config)
    
    # Initialize results structure
    results = {
        'target_features': target_features,
        'selection_parameters': {
            'min_activity_threshold': min_activity_threshold,
            'min_cluster_size': min_cluster_size,
            'algorithm_version': 'v2_activity_required'
        },
        'datasets': {},
        'summary': {}
    }
    
    dataset_names = ['main', 'cluster_0', 'cluster_1']
    total_datasets = 0
    
    for dataset_name in dataset_names:
        print(f"\n🔍 Processing {dataset_name} dataset...")
        # Construct file paths - updated for actual directory structure
        if dataset_name == 'main':
            base_data_path = "data/raw_data/new_raw_data_polygon.csv"
            clusters_path = os.path.join(results_dir, "main_clustering/cluster_labels.csv")
        else:
            base_data_path = None  # Will be handled in load_dataset_and_clusters
            clusters_path = os.path.join(results_dir, f"{dataset_name}_clustering/cluster_labels.csv")

        if not os.path.exists(clusters_path):
            print(f"⚠️  Skipping {dataset_name} - cluster labels not found: {clusters_path}")
            continue

        if dataset_name == 'main' and not os.path.exists(base_data_path):
            print(f"⚠️  Skipping {dataset_name} - base data not found: {base_data_path}")
            continue

        try:
            # Load data with corrected logic
            df = load_dataset_and_clusters(dataset_name, base_data_path, clusters_path)

            # Dataset statistics
            total_points = len(df)
            noise_points = len(df[df['cluster'] == -1])
            valid_clusters = len(df[df['cluster'] != -1]['cluster'].unique())

            print(f"   📊 {total_points:,} wallets, {valid_clusters} clusters, {noise_points:,} noise points")

            # Initialize dataset results
            dataset_results = {
                'total_points': total_points,
                'valid_clusters': valid_clusters,
                'noise_points': noise_points,
                'feature_selections': {}
            }

            # For outputting medians to CSV

            feature_medians = []  # For backward compatibility, but will use dict for wide format

            # Process each target feature
            for feature in target_features:
                print(f"   🎯 Analyzing {feature}...")

                try:
                    selected_cluster, stats = select_strongest_cluster_for_feature(
                        df, feature,
                        min_activity_threshold=min_activity_threshold,
                        min_cluster_size=min_cluster_size
                    )

                    # Extract median non-zero value from selected cluster
                    cluster_data = df[df['cluster'] == selected_cluster]
                    feature_values = cluster_data[feature].values
                    non_zero_values = feature_values[feature_values > 0]
                    median_nonzero_value = float(np.median(non_zero_values)) if len(non_zero_values) > 0 else 0.0
                    # Enhanced feature statistics
                    feature_stats = {
                        'mean': float(np.mean(feature_values)),
                        'median': float(np.median(feature_values)),
                        'median_nonzero': median_nonzero_value,
                        'std': float(np.std(feature_values)),
                        'variance': float(np.var(feature_values)),
                        'min': float(np.min(feature_values)),
                        'max': float(np.max(feature_values)),
                        'non_zero_proportion': float(stats['non_zero_proportion']),
                        'activity_count': int(stats['activity_count'])
                    }
                    dataset_results['feature_selections'][feature] = {
                        'selected_cluster': selected_cluster,
                        'median_nonzero_value': median_nonzero_value,
                        'cluster_size': stats['cluster_size'],
                        'selection_score': float(stats['score']),
                        'feature_stats': feature_stats,
                        'meets_activity_threshold': stats['non_zero_proportion'] >= min_activity_threshold
                    }
                    # Add to medians list for CSV output (wide format)
                    feature_medians.append((feature, median_nonzero_value))
                    print(f"      ✅ Cluster {selected_cluster}: median_nonzero={median_nonzero_value:.1f}, "
                          f"activity={stats['non_zero_proportion']*100:.1f}%, "
                          f"size={stats['cluster_size']:,}")

                except Exception as e:
                    print(f"      ❌ Error processing {feature}: {e}")
                    continue


            # Output feature medians to CSV for this dataset (wide format)
            if feature_medians:
                csv_name = f"{dataset_name}_clustering_feature_medians.csv" if dataset_name != 'main' else "main_clustering_feature_medians.csv"
                csv_path = os.path.join(results_dir, csv_name)
                import pandas as pd
                # Convert list of tuples to dict for wide format
                feature_medians_dict = {feature: median for feature, median in feature_medians}
                pd.DataFrame([feature_medians_dict]).to_csv(csv_path, index=False)
                print(f"      💾 Feature medians saved to: {csv_path} (wide format)")

            results['datasets'][dataset_name] = dataset_results
            total_datasets += 1

        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")
            continue
    
    # Generate summary statistics
    results['summary'] = generate_summary_statistics_v2(results, target_features)
    results['summary']['total_datasets_processed'] = total_datasets
    
    # Save results
    save_cluster_selection_results(results, output_path)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Print summary
    print_cluster_selection_summary_v2(results)
    
    return results

def generate_summary_statistics_v2(results: Dict[str, Any], target_features: List[str]) -> Dict[str, Any]:
    """Generate enhanced summary statistics."""
    summary = {
        'features_analyzed': len(target_features),
        'successful_selections': 0,
        'activity_threshold_met': 0,
        'fallback_selections': 0,
        'median_value_ranges': {}
    }
    
    for feature in target_features:
        medians = []
        activity_rates = []
        threshold_met = 0
        for dataset_name, dataset_data in results['datasets'].items():
            if feature in dataset_data['feature_selections']:
                selection = dataset_data['feature_selections'][feature]
                # Use the new key for median of non-zero values
                medians.append(selection['median_nonzero_value'])
                activity_rates.append(selection['feature_stats']['non_zero_proportion'])
                if selection.get('meets_activity_threshold', False):
                    threshold_met += 1
        if medians:
            summary['median_value_ranges'][feature] = {
                'min': float(np.min(medians)),
                'max': float(np.max(medians)),
                'mean': float(np.mean(medians)),
                'datasets_with_selections': len(medians),
                'avg_activity_rate': float(np.mean(activity_rates)),
                'threshold_compliance': f"{threshold_met}/{len(medians)}"
            }
            summary['successful_selections'] += len(medians)
            summary['activity_threshold_met'] += threshold_met
    
    return summary

def save_cluster_selection_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results with proper numpy type conversion."""
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):  # Handle numpy boolean
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert numpy types to native Python types
    converted_results = convert_numpy_types(results)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as YAML
    with open(output_path, 'w') as f:
        yaml.dump(converted_results, f, default_flow_style=False, indent=2)
    
    # Also save as JSON for readability
    json_path = output_path.replace('.yaml', '.json')
    import json
    with open(json_path, 'w') as f:
        json.dump(converted_results, f, indent=2)

def print_cluster_selection_summary_v2(results: Dict[str, Any]) -> None:
    """Print enhanced summary of cluster selection results."""
    print(f"\n{'='*80}")
    print("ENHANCED CLUSTER SELECTION SUMMARY (V2)")
    print(f"{'='*80}")
    print(f"🎯 Algorithm Version: {results['selection_parameters']['algorithm_version']}")
    print(f"📋 Activity Threshold: {results['selection_parameters']['min_activity_threshold']*100:.1f}%")
    print(f"👥 Minimum Cluster Size: {results['selection_parameters']['min_cluster_size']}")
    print()
    print(f"📊 Processing Results:")
    print(f"   • Datasets Processed: {results['summary']['total_datasets_processed']}")
    print(f"   • Successful Selections: {results['summary']['successful_selections']}")
    print(f"   • Activity Threshold Met: {results['summary']['activity_threshold_met']}")
    print(f"   • Fallback Selections: {results['summary'].get('fallback_selections', 0)}")
    print()
    print("🔍 Feature Analysis (per dataset):")
    for dataset_name, dataset_data in results['datasets'].items():
        print(f"\n  Dataset: {dataset_name}")
        for feature, selection in dataset_data['feature_selections'].items():
            print(f"    Feature: {feature}")
            print(f"      Selected Cluster: {selection['selected_cluster']}")
            print(f"      Median Non-Zero Value: {selection['median_nonzero_value']:.1f}")
            print(f"      Cluster Size: {selection['cluster_size']}")
            print(f"      Activity Rate: {selection['feature_stats']['non_zero_proportion']*100:.1f}%")
            print(f"      Meets Activity Threshold: {selection['meets_activity_threshold']}")
            print(f"      Selection Score: {selection['selection_score']:.3f}")

if __name__ == "__main__":
    # Test the enhanced version
    results = calculate_median_feature_values_for_clusters_v2(
        min_activity_threshold=0.1,  # Require 10% minimum activity
        min_cluster_size=50
    )
