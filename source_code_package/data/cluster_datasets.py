#!/usr/bin/env python3
"""
Core functionality for preparing cluster-specific datasets for AS_1 analysis.

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
from typing import Dict, Tuple, Optional
import json


class ClusterDatasetManager:
    """
    Manager class for handling cluster-specific dataset creation and management.
    """
    
    def __init__(self, 
                 original_data_path: str,
                 clustering_results_path: str,
                 output_directory: str):
        """
        Initialize the ClusterDatasetManager.
        
        Parameters:
        -----------
        original_data_path : str
            Path to the original dataset
        clustering_results_path : str
            Path to the clustering results CSV
        output_directory : str
            Directory to save cluster datasets
        """
        self.original_data_path = original_data_path
        self.clustering_results_path = clustering_results_path
        self.output_directory = output_directory
        self.original_data = None
        self.cluster_labels = None
        self.merged_data = None
        self.cluster_stats = None
        
    def load_data_and_clusters(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the original dataset and clustering results.
        
        Returns:
        --------
        tuple
            (original_data, cluster_labels)
        """
        print("Loading original dataset and clustering results...")
        
        # Load original dataset
        self.original_data = pd.read_csv(self.original_data_path)
        print(f"Original dataset shape: {self.original_data.shape}")
        
        # Load clustering results
        self.cluster_labels = pd.read_csv(self.clustering_results_path)
        print(f"Clustering results shape: {self.cluster_labels.shape}")
        
        # Verify data alignment
        if len(self.original_data) != len(self.cluster_labels):
            raise ValueError(f"Data size mismatch: original ({len(self.original_data)}) vs clusters ({len(self.cluster_labels)})")
        
        return self.original_data, self.cluster_labels
    
    def merge_data_with_clusters(self) -> pd.DataFrame:
        """
        Merge original data with cluster labels.
        
        Returns:
        --------
        pd.DataFrame
            Merged dataset with cluster labels
        """
        if self.original_data is None or self.cluster_labels is None:
            self.load_data_and_clusters()
            
        print("Merging data with cluster labels...")
        
        # Add cluster labels to original data
        self.merged_data = self.original_data.copy()
        self.merged_data['cluster_label'] = self.cluster_labels['cluster_label'].values
        
        return self.merged_data
    
    def analyze_clusters(self) -> Dict:
        """
        Analyze cluster composition and statistics.
        
        Returns:
        --------
        dict
            Cluster analysis results
        """
        if self.merged_data is None:
            self.merge_data_with_clusters()
            
        print("Analyzing cluster composition...")
        
        cluster_stats = {}
        
        # Count records in each cluster
        cluster_counts = self.merged_data['cluster_label'].value_counts().sort_index()
        print(f"Cluster distribution:")
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id}: {count:,} records ({count/len(self.merged_data)*100:.1f}%)")
            cluster_stats[f'cluster_{cluster_id}'] = {
                'count': int(count),
                'percentage': round(count/len(self.merged_data)*100, 2)
            }
        
        # Handle noise points (cluster -1)
        noise_count = (self.merged_data['cluster_label'] == -1).sum()
        if noise_count > 0:
            print(f"  Noise points (cluster -1): {noise_count:,} records ({noise_count/len(self.merged_data)*100:.1f}%)")
            cluster_stats['noise_points'] = {
                'count': int(noise_count),
                'percentage': round(noise_count/len(self.merged_data)*100, 2)
            }
        
        cluster_stats['total_records'] = len(self.merged_data)
        self.cluster_stats = cluster_stats
        
        return cluster_stats
    
    def handle_noise_points(self, strategy: str = 'exclude') -> pd.DataFrame:
        """
        Handle noise points according to specified strategy.
        
        Parameters:
        -----------
        strategy : str
            Strategy for handling noise points:
            - 'exclude': Remove noise points
            - 'assign_to_largest': Assign to largest cluster
            - 'separate': Keep as separate cluster
        
        Returns:
        --------
        pd.DataFrame
            Data with noise points handled
        """
        if self.merged_data is None:
            self.merge_data_with_clusters()
            
        noise_count = (self.merged_data['cluster_label'] == -1).sum()
        
        if noise_count == 0:
            print("No noise points found.")
            return self.merged_data
        
        print(f"Handling {noise_count} noise points using strategy: {strategy}")
        
        if strategy == 'exclude':
            # Remove noise points
            filtered_data = self.merged_data[self.merged_data['cluster_label'] != -1].copy()
            print(f"  Excluded {noise_count} noise points")
            return filtered_data
        
        elif strategy == 'assign_to_largest':
            # Find largest cluster
            cluster_counts = self.merged_data[self.merged_data['cluster_label'] != -1]['cluster_label'].value_counts()
            largest_cluster = cluster_counts.index[0]
            
            # Assign noise points to largest cluster
            modified_data = self.merged_data.copy()
            modified_data.loc[modified_data['cluster_label'] == -1, 'cluster_label'] = largest_cluster
            print(f"  Assigned {noise_count} noise points to cluster {largest_cluster}")
            return modified_data
        
        elif strategy == 'separate':
            # Keep noise points as separate analysis group
            print(f"  Keeping {noise_count} noise points as separate group")
            return self.merged_data
        
        else:
            raise ValueError(f"Unknown noise handling strategy: {strategy}")
    
    def create_cluster_datasets(self, processed_data: pd.DataFrame) -> Dict:
        """
        Create separate CSV files for each cluster.
        
        Parameters:
        -----------
        processed_data : pd.DataFrame
            Data with cluster labels (after noise handling)
        
        Returns:
        --------
        dict
            Information about created datasets
        """
        print("Creating cluster-specific datasets...")
        
        # Ensure output directory exists
        os.makedirs(self.output_directory, exist_ok=True)
        
        created_files = {}
        
        # Get unique clusters (excluding noise if handled)
        unique_clusters = sorted(processed_data['cluster_label'].unique())
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                # Handle noise points separately
                cluster_data = processed_data[processed_data['cluster_label'] == cluster_id].copy()
                filename = f'new_raw_data_polygon_noise_points.csv'
                description = "noise points"
            else:
                # Regular clusters
                cluster_data = processed_data[processed_data['cluster_label'] == cluster_id].copy()
                filename = f'new_raw_data_polygon_cluster_{cluster_id}.csv'
                description = f"cluster {cluster_id}"
            
            # Remove cluster_label column for AS_1 processing (to match original data format)
            cluster_data_for_export = cluster_data.drop('cluster_label', axis=1)
            
            # Save to file
            file_path = os.path.join(self.output_directory, filename)
            cluster_data_for_export.to_csv(file_path, index=False)
            
            created_files[f'cluster_{cluster_id}'] = {
                'file_path': file_path,
                'filename': filename,
                'record_count': len(cluster_data_for_export),
                'description': description
            }
            
            print(f"  Created {filename} with {len(cluster_data_for_export):,} records ({description})")
        
        return created_files
    
    def generate_summary_report(self, created_files: Dict, noise_strategy: str = 'exclude'):
        """
        Generate a summary report of the cluster dataset creation process.
        
        Parameters:
        -----------
        created_files : dict
            Information about created files
        noise_strategy : str
            Strategy used for noise handling
        """
        print("Generating summary report...")
        
        # Create comprehensive summary
        summary = {
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'source_data': {
                'original_file': os.path.basename(self.original_data_path),
                'clustering_results': os.path.basename(self.clustering_results_path),
                'total_records': self.cluster_stats['total_records'] if self.cluster_stats else 0
            },
            'cluster_statistics': self.cluster_stats,
            'created_datasets': created_files,
            'noise_handling': {
                'strategy': noise_strategy,
                'rationale': self._get_noise_strategy_rationale(noise_strategy)
            },
            'next_steps': [
                'Run feature engineering on each cluster dataset',
                'Train separate AS_1 models for each cluster',
                'Compare cluster-specific model performance',
                'Analyze feature importance differences between clusters'
            ]
        }
        
        # Save summary as JSON
        summary_path = os.path.join(self.output_directory, 'cluster_datasets_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save summary as readable text
        text_summary_path = os.path.join(self.output_directory, 'cluster_datasets_summary.txt')
        with open(text_summary_path, 'w') as f:
            f.write("CLUSTER DATASETS CREATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {summary['generation_timestamp']}\n")
            f.write(f"Total records processed: {self.cluster_stats['total_records']:,}\n\n")
            
            f.write("CLUSTER DISTRIBUTION:\n")
            f.write("-" * 30 + "\n")
            for key, stats in self.cluster_stats.items():
                if key.startswith('cluster_'):
                    cluster_num = key.split('_')[1]
                    f.write(f"Cluster {cluster_num}: {stats['count']:,} records ({stats['percentage']:.1f}%)\n")
            
            if 'noise_points' in self.cluster_stats:
                f.write(f"Noise points: {self.cluster_stats['noise_points']['count']:,} records "
                       f"({self.cluster_stats['noise_points']['percentage']:.1f}%)\n")
            
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
    
    def _get_noise_strategy_rationale(self, strategy: str) -> str:
        """Get rationale for noise handling strategy."""
        rationales = {
            'exclude': 'Noise points excluded to focus on clear cluster patterns',
            'assign_to_largest': 'Noise points assigned to largest cluster for complete dataset coverage',
            'separate': 'Noise points kept separate for specialized analysis'
        }
        return rationales.get(strategy, 'Custom noise handling strategy')
    
    def process_full_pipeline(self, noise_strategy: str = 'exclude') -> Dict:
        """
        Execute the complete cluster dataset preparation pipeline.
        
        Parameters:
        -----------
        noise_strategy : str
            Strategy for handling noise points
            
        Returns:
        --------
        dict
            Results of the pipeline execution
        """
        try:
            # Step 1: Load data
            self.load_data_and_clusters()
            
            # Step 2: Merge data with clusters
            self.merge_data_with_clusters()
            
            # Step 3: Analyze clusters
            self.analyze_clusters()
            
            # Step 4: Handle noise points
            processed_data = self.handle_noise_points(strategy=noise_strategy)
            
            # Step 5: Create cluster datasets
            created_files = self.create_cluster_datasets(processed_data)
            
            # Step 6: Generate summary report
            summary_json, summary_txt = self.generate_summary_report(created_files, noise_strategy)
            
            return {
                'success': True,
                'cluster_stats': self.cluster_stats,
                'created_files': created_files,
                'summary_files': {
                    'json': summary_json,
                    'text': summary_txt
                },
                'output_directory': self.output_directory
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'output_directory': self.output_directory
            }


# Convenience functions for backward compatibility and easier usage
def prepare_cluster_datasets(original_data_path: str,
                            clustering_results_path: str,
                            output_directory: str,
                            noise_strategy: str = 'exclude') -> Dict:
    """
    Convenience function to prepare cluster datasets in one call.
    
    Parameters:
    -----------
    original_data_path : str
        Path to the original dataset
    clustering_results_path : str
        Path to the clustering results CSV
    output_directory : str
        Directory to save cluster datasets
    noise_strategy : str
        Strategy for handling noise points
        
    Returns:
    --------
    dict
        Results of the pipeline execution
    """
    manager = ClusterDatasetManager(
        original_data_path=original_data_path,
        clustering_results_path=clustering_results_path,
        output_directory=output_directory
    )
    
    return manager.process_full_pipeline(noise_strategy=noise_strategy)
