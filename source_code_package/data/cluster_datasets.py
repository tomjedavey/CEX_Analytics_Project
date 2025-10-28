#!/usr/bin/env python3
"""
Core functionality for preparing cluster-specific datasets for later analytic algorithm + score production.

This module provides functions to:
1. Load and merge original data with clustering results
2. Analyze cluster composition and statistics
3. Handle noise points with different strategies
4. Create cluster-specific datasets
5. Generate summary reports

Author: Tom Davey
Date: July 2025
"""


import os
import pandas as pd
import numpy as np
import json
from typing import Dict, Optional



def load_data_and_clusters(original_data_path: str, clustering_results_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the original dataset and clustering results.
    """
    print("Loading original dataset and clustering results...")
    original_data = pd.read_csv(original_data_path)
    print(f"Original dataset shape: {original_data.shape}")
    cluster_labels = pd.read_csv(clustering_results_path)
    print(f"Clustering results shape: {cluster_labels.shape}")
    if len(original_data) != len(cluster_labels):
        raise ValueError(f"Data size mismatch: original ({len(original_data)}) vs clusters ({len(cluster_labels)})")
    return original_data, cluster_labels
    

def merge_data_with_clusters(original_data: pd.DataFrame, cluster_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Merge original data with cluster labels.
    """
    print("Merging data with cluster labels...")
    merged_data = original_data.copy()
    merged_data['cluster_label'] = cluster_labels['cluster_label'].values
    return merged_data
    

def analyze_clusters(merged_data: pd.DataFrame) -> Dict:
    """
    Analyze cluster composition and statistics.
    """
    print("Analyzing cluster composition...")
    cluster_stats = {}
    cluster_counts = merged_data['cluster_label'].value_counts().sort_index()
    print(f"Cluster distribution:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count:,} records ({count/len(merged_data)*100:.1f}%)")
        cluster_stats[f'cluster_{cluster_id}'] = {
            'count': int(count),
            'percentage': round(count/len(merged_data)*100, 2)
        }
    noise_count = (merged_data['cluster_label'] == -1).sum()
    if noise_count > 0:
        print(f"  Noise points (cluster -1): {noise_count:,} records ({noise_count/len(merged_data)*100:.1f}%)")
        cluster_stats['noise_points'] = {
            'count': int(noise_count),
            'percentage': round(noise_count/len(merged_data)*100, 2)
        }
    cluster_stats['total_records'] = len(merged_data)
    return cluster_stats
    

def handle_noise_points(merged_data: pd.DataFrame, strategy: str = 'exclude') -> pd.DataFrame:
    """
    Handle noise points according to specified strategy.
    """
    noise_count = (merged_data['cluster_label'] == -1).sum()
    if noise_count == 0:
        print("No noise points found.")
        return merged_data
    print(f"Handling {noise_count} noise points using strategy: {strategy}")
    if strategy == 'exclude':
        filtered_data = merged_data[merged_data['cluster_label'] != -1].copy()
        print(f"  Excluded {noise_count} noise points")
        return filtered_data
    elif strategy == 'assign_to_largest':
        cluster_counts = merged_data[merged_data['cluster_label'] != -1]['cluster_label'].value_counts()
        largest_cluster = cluster_counts.index[0]
        modified_data = merged_data.copy()
        modified_data.loc[modified_data['cluster_label'] == -1, 'cluster_label'] = largest_cluster
        print(f"  Assigned {noise_count} noise points to cluster {largest_cluster}")
        return modified_data
    elif strategy == 'separate':
        print(f"  Keeping {noise_count} noise points as separate group")
        return merged_data
    else:
        raise ValueError(f"Unknown noise handling strategy: {strategy}")
    

def create_cluster_datasets(processed_data: pd.DataFrame, output_directory: str) -> Dict:
    """
    Create separate CSV files for each cluster.
    """
    print("Creating cluster-specific datasets...")
    os.makedirs(output_directory, exist_ok=True)
    created_files = {}
    unique_clusters = sorted(processed_data['cluster_label'].unique())
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            cluster_data = processed_data[processed_data['cluster_label'] == cluster_id].copy()
            filename = f'new_raw_data_polygon_noise_points.csv'
            description = "noise points"
        else:
            cluster_data = processed_data[processed_data['cluster_label'] == cluster_id].copy()
            filename = f'new_raw_data_polygon_cluster_{cluster_id}.csv'
            description = f"cluster {cluster_id}"
        cluster_data_for_export = cluster_data.drop('cluster_label', axis=1)
        file_path = os.path.join(output_directory, filename)
        cluster_data_for_export.to_csv(file_path, index=False)
        created_files[f'cluster_{cluster_id}'] = {
            'file_path': file_path,
            'filename': filename,
            'record_count': len(cluster_data_for_export),
            'description': description
        }
        print(f"  Created {filename} with {len(cluster_data_for_export):,} records ({description})")
    return created_files
    

def _get_noise_strategy_rationale(strategy: str) -> str:
    rationales = {
        'exclude': 'Noise points excluded to focus on clear cluster patterns',
        'assign_to_largest': 'Noise points assigned to largest cluster for complete dataset coverage',
        'separate': 'Noise points kept separate for specialized analysis'
    }
    return rationales.get(strategy, 'Custom noise handling strategy')

def generate_summary_report(
    created_files: Dict,
    noise_strategy: str,
    output_directory: str,
    original_data_path: str,
    clustering_results_path: str,
    cluster_stats: Dict
) -> tuple[str, str]:
    """
    Generate a comprehensive summary report of cluster dataset creation process.
    """
    print("Generating summary report...")
    summary = {
        'generation_timestamp': pd.Timestamp.now().isoformat(),
        'source_data': {
            'original_file': os.path.basename(original_data_path),
            'clustering_results': os.path.basename(clustering_results_path),
            'total_records': cluster_stats['total_records'] if cluster_stats else 0
        },
        'cluster_statistics': cluster_stats,
        'created_datasets': created_files,
        'noise_handling': {
            'strategy': noise_strategy,
            'rationale': _get_noise_strategy_rationale(noise_strategy)
        },
        'next_steps': [
            'Run feature engineering on each cluster dataset',
            'Train separate AS_1 models for each cluster',
            'Compare cluster-specific model performance',
            'Analyze feature importance differences between clusters'
        ]
    }
    summary_path = os.path.join(output_directory, 'cluster_datasets_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    text_summary_path = os.path.join(output_directory, 'cluster_datasets_summary.txt')
    with open(text_summary_path, 'w') as f:
        f.write("CLUSTER DATASETS CREATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {summary['generation_timestamp']}\n")
        f.write(f"Total records processed: {cluster_stats['total_records']:,}\n\n")
        f.write("CLUSTER DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        for key, stats in cluster_stats.items():
            if key.startswith('cluster_'):
                cluster_num = key.split('_')[1]
                f.write(f"Cluster {cluster_num}: {stats['count']:,} records ({stats['percentage']:.1f}%)\n")
        if 'noise_points' in cluster_stats:
            f.write(f"Noise points: {cluster_stats['noise_points']['count']:,} records "
                    f"({cluster_stats['noise_points']['percentage']:.1f}%)\n")
        f.write("\nCREATED FILES:\n")
        f.write("-" * 30 + "\n")
        for key, file_info in created_files.items():
            f.write(f"{file_info['filename']}: {file_info['record_count']:,} records\n")
        f.write("\nNEXT STEPS:\n")
        f.write("-" * 30 + "\n")
        for step in summary['next_steps']:
            f.write(f"- {step}\n")
    print(f"Summary reports saved:")
    print(f"  JSON: {summary_path}")
    print(f"  Text: {text_summary_path}")
    return summary_path, text_summary_path
    

def prepare_cluster_datasets(
    original_data_path: str,
    clustering_results_path: str,
    output_directory: str,
    noise_strategy: str = 'exclude'
) -> Dict:
    """
    Prepare cluster datasets in one call (functional style).
    """
    try:
        # Step 1: Load data
        original_data, cluster_labels = load_data_and_clusters(original_data_path, clustering_results_path)
        # Step 2: Merge data with clusters
        merged_data = merge_data_with_clusters(original_data, cluster_labels)
        # Step 3: Analyze clusters
        cluster_stats = analyze_clusters(merged_data)
        # Step 4: Handle noise points
        processed_data = handle_noise_points(merged_data, strategy=noise_strategy)
        # Step 5: Create cluster datasets
        created_files = create_cluster_datasets(processed_data, output_directory)
        # Step 6: Generate summary report
        summary_json, summary_txt = generate_summary_report(
            created_files,
            noise_strategy,
            output_directory,
            original_data_path,
            clustering_results_path,
            cluster_stats
        )
        return {
            'success': True,
            'cluster_stats': cluster_stats,
            'created_files': created_files,
            'summary_files': {
                'json': summary_json,
                'text': summary_txt
            },
            'output_directory': output_directory
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'output_directory': output_directory
        }
