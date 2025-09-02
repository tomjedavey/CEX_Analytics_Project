#!/usr/bin/env python3
"""
Example: Using Interaction Mode Clustering Results

This script demonstrates how to load and analyze the results from the 
interaction mode clustering pipeline. This serves as a foundation for 
developing interaction mode scoring algorithms.

Author: Tom Davey
Date: August 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple

def load_clustering_results(results_dir: str = "data/processed_data/interaction_mode_results") -> Dict[str, pd.DataFrame]:
    """
    Load clustering results from all three datasets.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing the clustering results
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary containing cluster labels for each dataset
    """
    datasets = ['main', 'cluster_0', 'cluster_1']
    results = {}
    
    for dataset in datasets:
        labels_path = os.path.join(results_dir, f"{dataset}_clustering", "cluster_labels.csv")
        
        if os.path.exists(labels_path):
            df = pd.read_csv(labels_path)
            results[dataset] = df
            print(f"✅ Loaded {len(df)} cluster labels for {dataset}")
        else:
            print(f"❌ Could not find results for {dataset} at {labels_path}")
    
    return results

def analyze_cluster_distributions(results: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Analyze the distribution of clusters across datasets.
    
    Parameters:
    -----------
    results : Dict[str, pd.DataFrame]
        Clustering results from load_clustering_results
        
    Returns:
    --------
    Dict[str, Dict]
        Analysis results for each dataset
    """
    analysis = {}
    
    for dataset_name, df in results.items():
        cluster_counts = df['cluster_label'].value_counts().sort_index()
        
        analysis[dataset_name] = {
            'total_points': len(df),
            'num_clusters': len(cluster_counts),
            'cluster_sizes': cluster_counts.to_dict(),
            'largest_cluster': cluster_counts.max(),
            'smallest_cluster': cluster_counts.min(),
            'noise_points': len(df[df['cluster_label'] == -1]) if -1 in df['cluster_label'].values else 0
        }
        
        print(f"\n📊 {dataset_name.upper()} Dataset Analysis:")
        print(f"   Total points: {analysis[dataset_name]['total_points']:,}")
        print(f"   Number of clusters: {analysis[dataset_name]['num_clusters']}")
        print(f"   Cluster sizes: {analysis[dataset_name]['cluster_sizes']}")
        print(f"   Noise points: {analysis[dataset_name]['noise_points']}")
    
    return analysis

def compare_clustering_patterns(results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Compare clustering patterns across datasets to identify interaction modes.
    
    This is a placeholder function that demonstrates the type of analysis
    that could be used for interaction mode score development.
    
    Parameters:
    -----------
    results : Dict[str, pd.DataFrame]
        Clustering results from load_clustering_results
        
    Returns:
    --------
    Dict[str, Any]
        Comparison metrics and patterns
    """
    print("\n🔍 Comparing Clustering Patterns:")
    
    comparison = {
        'dataset_cluster_counts': {},
        'cluster_consistency': {},
        'potential_interaction_modes': []
    }
    
    # Extract cluster counts for each dataset
    for dataset_name, df in results.items():
        cluster_counts = df['cluster_label'].value_counts().sort_index()
        comparison['dataset_cluster_counts'][dataset_name] = len(cluster_counts)
        print(f"   {dataset_name}: {len(cluster_counts)} clusters")
    
    # Identify potential interaction mode patterns
    main_clusters = comparison['dataset_cluster_counts'].get('main', 0)
    cluster_0_clusters = comparison['dataset_cluster_counts'].get('cluster_0', 0)
    cluster_1_clusters = comparison['dataset_cluster_counts'].get('cluster_1', 0)
    
    if cluster_1_clusters > cluster_0_clusters:
        comparison['potential_interaction_modes'].append(
            "Cluster 1 subset shows higher internal diversity (more sub-clusters)"
        )
    
    if cluster_0_clusters == main_clusters:
        comparison['potential_interaction_modes'].append(
            "Cluster 0 subset maintains similar structure to main dataset"
        )
    
    # This is where more sophisticated interaction mode analysis would go
    print(f"   Potential interaction mode indicators:")
    for mode in comparison['potential_interaction_modes']:
        print(f"      • {mode}")
    
    return comparison

def create_summary_visualization(results: Dict[str, pd.DataFrame], 
                               analysis: Dict[str, Dict],
                               output_path: str = None) -> None:
    """
    Create a summary visualization of clustering results across datasets.
    
    Parameters:
    -----------
    results : Dict[str, pd.DataFrame]
        Clustering results
    analysis : Dict[str, Dict]
        Analysis results from analyze_cluster_distributions
    output_path : str, optional
        Path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Interaction Mode Clustering Results Summary', fontsize=16, fontweight='bold')
    
    # Dataset sizes comparison
    dataset_names = list(analysis.keys())
    dataset_sizes = [analysis[name]['total_points'] for name in dataset_names]
    
    axes[0, 0].bar(dataset_names, dataset_sizes, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_title('Dataset Sizes')
    axes[0, 0].set_ylabel('Number of Data Points')
    for i, v in enumerate(dataset_sizes):
        axes[0, 0].text(i, v + 100, f'{v:,}', ha='center', va='bottom')
    
    # Number of clusters per dataset
    cluster_counts = [analysis[name]['num_clusters'] for name in dataset_names]
    
    axes[0, 1].bar(dataset_names, cluster_counts, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 1].set_title('Number of Clusters per Dataset')
    axes[0, 1].set_ylabel('Number of Clusters')
    for i, v in enumerate(cluster_counts):
        axes[0, 1].text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    # Cluster size distribution for main dataset
    if 'main' in results:
        main_cluster_sizes = results['main']['cluster_label'].value_counts().sort_index()
        axes[1, 0].pie(main_cluster_sizes.values, 
                      labels=[f'Cluster {i}' for i in main_cluster_sizes.index],
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Main Dataset Cluster Distribution')
    
    # Noise points comparison
    noise_counts = [analysis[name]['noise_points'] for name in dataset_names]
    
    axes[1, 1].bar(dataset_names, noise_counts, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1, 1].set_title('Noise Points per Dataset')
    axes[1, 1].set_ylabel('Number of Noise Points')
    for i, v in enumerate(noise_counts):
        axes[1, 1].text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Summary visualization saved to: {output_path}")
    
    plt.show()

def compute_interaction_mode_prototype_scores(results: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Prototype function for computing interaction mode scores.
    
    This is a placeholder that demonstrates the structure for future 
    interaction mode score computation algorithms.
    
    Parameters:
    -----------
    results : Dict[str, pd.DataFrame]
        Clustering results from all datasets
        
    Returns:
    --------
    Dict[str, float]
        Prototype interaction mode scores
    """
    print("\n🧮 Computing Prototype Interaction Mode Scores:")
    
    scores = {}
    
    # Example score 1: Cluster Diversity Score
    # Measures how much clustering structure varies between datasets
    if 'main' in results and 'cluster_0' in results and 'cluster_1' in results:
        main_clusters = len(results['main']['cluster_label'].unique())
        c0_clusters = len(results['cluster_0']['cluster_label'].unique())
        c1_clusters = len(results['cluster_1']['cluster_label'].unique())
        
        # Higher diversity = higher score
        diversity_score = (c0_clusters + c1_clusters) / (2 * main_clusters)
        scores['cluster_diversity'] = diversity_score
        print(f"   Cluster Diversity Score: {diversity_score:.3f}")
    
    # Example score 2: Cluster Stability Score
    # Measures consistency of clustering across subsets
    if 'cluster_0' in results and 'cluster_1' in results:
        c0_clusters = len(results['cluster_0']['cluster_label'].unique())
        c1_clusters = len(results['cluster_1']['cluster_label'].unique())
        
        # Lower difference = higher stability
        stability_score = 1.0 / (1.0 + abs(c0_clusters - c1_clusters))
        scores['cluster_stability'] = stability_score
        print(f"   Cluster Stability Score: {stability_score:.3f}")
    
    # Example score 3: Interaction Complexity Score
    # Measures overall interaction pattern complexity
    total_unique_patterns = 0
    for dataset_name, df in results.items():
        unique_clusters = len(df['cluster_label'].unique())
        total_unique_patterns += unique_clusters
    
    complexity_score = total_unique_patterns / len(results)
    scores['interaction_complexity'] = complexity_score
    print(f"   Interaction Complexity Score: {complexity_score:.3f}")
    
    print(f"\n📋 Prototype scores computed. These are examples for future development.")
    
    return scores

def main():
    """Main demonstration function."""
    print("🚀 Interaction Mode Clustering Results Analysis")
    print("=" * 60)
    
    # Check if results exist
    results_dir = "data/processed_data/interaction_mode_results"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        print("   Please run the interaction mode clustering pipeline first:")
        print("   python3 scripts/clustering/run_interaction_mode_clustering.py")
        return 1
    
    try:
        # Load clustering results
        print("📂 Loading clustering results...")
        results = load_clustering_results(results_dir)
        
        if not results:
            print("❌ No clustering results found.")
            return 1
        
        # Analyze cluster distributions
        analysis = analyze_cluster_distributions(results)
        
        # Compare clustering patterns
        comparison = compare_clustering_patterns(results)
        
        # Create summary visualization
        print("\n📊 Creating summary visualization...")
        viz_path = os.path.join(results_dir, "interaction_mode_analysis_summary.png")
        create_summary_visualization(results, analysis, viz_path)
        
        # Compute prototype interaction mode scores
        scores = compute_interaction_mode_prototype_scores(results)
        
        # Summary
        print("\n✅ Analysis Complete!")
        print("=" * 40)
        print("This analysis provides the foundation for developing")
        print("interaction mode scoring algorithms. Key next steps:")
        print("1. Develop domain-specific scoring functions")
        print("2. Validate scores against known interaction patterns")
        print("3. Implement real-time score computation")
        print("4. Create interaction mode classification system")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
