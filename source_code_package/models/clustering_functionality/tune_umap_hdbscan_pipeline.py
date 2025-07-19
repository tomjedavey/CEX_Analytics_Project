#!/usr/bin/env python3
"""
Comprehensive parameter tuning script for UMAP + HDBSCAN clustering pipeline.

This script optimizes both UMAP dimensionality reduction and HDBSCAN clustering parameters
together to achieve the best possible clustering results. It uses a grid search approach
with comprehensive evaluation metrics.

Usage:
    cd /path/to/MLProject1
    python3 source_code_package/models/clustering_functionality/tune_umap_hdbscan_pipeline.py

The script will:
1. Load preprocessed data from the preprocessing pipeline
2. Test various UMAP parameter combinations
3. For each UMAP result, test HDBSCAN parameter combinations
4. Evaluate the complete pipeline using multiple quality metrics
5. Provide recommendations for optimal parameter combinations
6. Generate a comprehensive report with visualizations

Author: Tom Davey
Date: July 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import time
import warnings
from typing import Dict, Any, List, Tuple, Optional
from itertools import product
import json
from datetime import datetime

# Suppress warnings for cleaner output during parameter testing
warnings.filterwarnings('ignore')

# Add source_code_package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from models.clustering_functionality.UMAP_dim_reduction import apply_umap_reduction, evaluate_umap_quality
from models.clustering_functionality.HBDSCAN_cluster import apply_hdbscan_clustering, evaluate_clustering_quality
from data.preprocess_cluster import preprocess_for_clustering


class UMAPHDBSCANTuner:
    """
    Comprehensive parameter tuner for UMAP + HDBSCAN clustering pipeline.
    """
    
    def __init__(self, data: pd.DataFrame, config_path: Optional[str] = None):
        """
        Initialize the tuner with data and configuration.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Preprocessed input data for the pipeline
        config_path : str, optional
            Path to configuration file
        """
        self.data = data
        self.config_path = config_path
        self.results = []
        self.best_result = None
        
        print(f"Initialized UMAP-HDBSCAN tuner with data shape: {data.shape}")
        
    def define_parameter_space(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Define the parameter space for both UMAP and HDBSCAN.
        
        Returns:
        --------
        Tuple[List[Dict], List[Dict]]
            UMAP parameter combinations and HDBSCAN parameter combinations
        """
        
        # UMAP parameter combinations
        # Optimized for clustering quality rather than just dimensionality reduction
        umap_params = [
            # Conservative approach - preserve more global structure
            {'n_components': 15, 'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'euclidean'},
            {'n_components': 12, 'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'euclidean'},
            {'n_components': 10, 'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'euclidean'},
            
            # Moderate approach - balance local and global structure
            {'n_components': 15, 'n_neighbors': 30, 'min_dist': 0.05, 'metric': 'euclidean'},
            {'n_components': 12, 'n_neighbors': 30, 'min_dist': 0.05, 'metric': 'euclidean'},
            {'n_components': 10, 'n_neighbors': 30, 'min_dist': 0.05, 'metric': 'euclidean'},
            
            # Focus on local structure - better for tight clusters
            {'n_components': 15, 'n_neighbors': 50, 'min_dist': 0.01, 'metric': 'euclidean'},
            {'n_components': 12, 'n_neighbors': 50, 'min_dist': 0.01, 'metric': 'euclidean'},
            {'n_components': 10, 'n_neighbors': 50, 'min_dist': 0.01, 'metric': 'euclidean'},
            
            # Different metrics for comparison
            {'n_components': 12, 'n_neighbors': 30, 'min_dist': 0.05, 'metric': 'manhattan'},
            {'n_components': 12, 'n_neighbors': 30, 'min_dist': 0.05, 'metric': 'cosine'},
            
            # Higher dimensional embeddings for complex data
            {'n_components': 20, 'n_neighbors': 30, 'min_dist': 0.05, 'metric': 'euclidean'},
            {'n_components': 25, 'n_neighbors': 30, 'min_dist': 0.05, 'metric': 'euclidean'},
        ]
        
        # HDBSCAN parameter combinations
        # Optimized for dataset size (~20K points) and clustering quality
        hdbscan_params = [
            # Conservative - larger, more stable clusters
            {'min_cluster_size': 200, 'min_samples': 50, 'metric': 'euclidean'},
            {'min_cluster_size': 150, 'min_samples': 40, 'metric': 'euclidean'},
            {'min_cluster_size': 100, 'min_samples': 30, 'metric': 'euclidean'},
            
            # Moderate approach
            {'min_cluster_size': 100, 'min_samples': 25, 'metric': 'euclidean'},
            {'min_cluster_size': 75, 'min_samples': 20, 'metric': 'euclidean'},
            {'min_cluster_size': 50, 'min_samples': 15, 'metric': 'euclidean'},
            
            # More sensitive - smaller clusters
            {'min_cluster_size': 50, 'min_samples': 10, 'metric': 'euclidean'},
            {'min_cluster_size': 30, 'min_samples': 8, 'metric': 'euclidean'},
            {'min_cluster_size': 25, 'min_samples': 5, 'metric': 'euclidean'},
            
            # Different metrics with best size parameters
            {'min_cluster_size': 75, 'min_samples': 20, 'metric': 'manhattan'},
            {'min_cluster_size': 75, 'min_samples': 20, 'metric': 'cosine'},
        ]
        
        return umap_params, hdbscan_params
    
    def evaluate_pipeline(self, umap_data: np.ndarray, cluster_labels: np.ndarray, 
                         hdbscan_model, umap_params: Dict, hdbscan_params: Dict) -> Dict[str, Any]:
        """
        Comprehensive evaluation of the complete UMAP + HDBSCAN pipeline.
        
        Parameters:
        -----------
        umap_data : np.ndarray
            UMAP-reduced data
        cluster_labels : np.ndarray
            HDBSCAN cluster labels
        hdbscan_model : hdbscan.HDBSCAN
            Fitted HDBSCAN model
        umap_params : Dict
            UMAP parameters used
        hdbscan_params : Dict
            HDBSCAN parameters used
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive evaluation metrics
        """
        
        n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
        n_noise = np.sum(cluster_labels == -1)
        noise_percentage = (n_noise / len(cluster_labels)) * 100
        
        metrics = {
            'umap_params': umap_params,
            'hdbscan_params': hdbscan_params,
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'noise_percentage': noise_percentage,
            'n_data_points': len(cluster_labels),
            'umap_dimensions': umap_data.shape[1],
            'original_dimensions': self.data.select_dtypes(include=[np.number]).shape[1]
        }
        
        # Clustering quality metrics (only if we have valid clusters)
        if n_clusters >= 2 and n_noise < len(cluster_labels) * 0.95:  # At least 5% clustered
            try:
                # Remove noise points for silhouette calculation
                clustered_mask = cluster_labels >= 0
                if np.sum(clustered_mask) > 1:
                    clustered_data = umap_data[clustered_mask]
                    clustered_labels = cluster_labels[clustered_mask]
                    
                    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                    
                    # Silhouette score (higher is better)
                    metrics['silhouette_score'] = silhouette_score(clustered_data, clustered_labels)
                    
                    # Calinski-Harabasz index (higher is better)
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(clustered_data, clustered_labels)
                    
                    # Davies-Bouldin index (lower is better)
                    metrics['davies_bouldin_score'] = davies_bouldin_score(clustered_data, clustered_labels)
                    
                    # HDBSCAN-specific metrics
                    if hasattr(hdbscan_model, 'cluster_persistence_'):
                        cluster_persistence = hdbscan_model.cluster_persistence_
                        metrics['mean_cluster_persistence'] = np.mean(cluster_persistence) if len(cluster_persistence) > 0 else 0
                        metrics['min_cluster_persistence'] = np.min(cluster_persistence) if len(cluster_persistence) > 0 else 0
                    
                    # Stability score from HDBSCAN
                    if hasattr(hdbscan_model, 'probabilities_'):
                        stable_points = hdbscan_model.probabilities_ > 0.5
                        metrics['stability_ratio'] = np.sum(stable_points) / len(stable_points)
                    
            except Exception as e:
                print(f"Warning: Could not calculate some clustering metrics: {e}")
                metrics['silhouette_score'] = None
                metrics['calinski_harabasz_score'] = None
                metrics['davies_bouldin_score'] = None
        
        # Composite score for ranking
        # Higher is better
        composite_score = 0
        
        if metrics.get('silhouette_score') is not None:
            # Silhouette score (0 to 1, higher better)
            composite_score += metrics['silhouette_score'] * 100
            
        # Penalty for too much noise
        if noise_percentage < 5:
            composite_score += 20  # Bonus for low noise
        elif noise_percentage > 50:
            composite_score -= 50  # Heavy penalty for high noise
        
        # Bonus for reasonable number of clusters
        if 2 <= n_clusters <= 20:
            composite_score += 10
        elif n_clusters > 50:
            composite_score -= 30
        
        # Penalty for too few clustered points
        clustered_percentage = 100 - noise_percentage
        if clustered_percentage > 80:
            composite_score += 15
        elif clustered_percentage < 50:
            composite_score -= 25
        
        metrics['composite_score'] = composite_score
        
        return metrics
    
    def run_comprehensive_tuning(self, max_combinations: int = 50, 
                                time_limit_per_combination: int = 300) -> List[Dict[str, Any]]:
        """
        Run comprehensive parameter tuning for the complete pipeline.
        
        Parameters:
        -----------
        max_combinations : int, default 50
            Maximum number of UMAP-HDBSCAN combinations to test
        time_limit_per_combination : int, default 300
            Maximum time in seconds per combination
            
        Returns:
        --------
        List[Dict[str, Any]]
            Results for all tested combinations
        """
        
        print("Starting comprehensive UMAP + HDBSCAN parameter tuning...")
        print("=" * 60)
        
        umap_params_list, hdbscan_params_list = self.define_parameter_space()
        
        # Create all combinations but limit to max_combinations
        all_combinations = list(product(umap_params_list, hdbscan_params_list))
        
        if len(all_combinations) > max_combinations:
            print(f"Limiting to {max_combinations} combinations out of {len(all_combinations)} possible")
            # Sample combinations to get a good spread
            step = len(all_combinations) // max_combinations
            combinations_to_test = all_combinations[::step][:max_combinations]
        else:
            combinations_to_test = all_combinations
        
        print(f"Testing {len(combinations_to_test)} UMAP-HDBSCAN combinations...")
        print(f"Estimated total time: {len(combinations_to_test) * 2:.0f}-{len(combinations_to_test) * 5:.0f} minutes\\n")
        
        start_time = time.time()
        
        for i, (umap_params, hdbscan_params) in enumerate(combinations_to_test):
            combination_start = time.time()
            
            print(f"\\nCombination {i+1}/{len(combinations_to_test)}")
            print(f"UMAP: {umap_params}")
            print(f"HDBSCAN: {hdbscan_params}")
            
            try:
                # Step 1: Apply UMAP
                umap_start = time.time()
                reduced_data, umap_model = apply_umap_reduction(
                    data=self.data,
                    config_path=self.config_path,
                    **umap_params
                )
                umap_time = time.time() - umap_start
                print(f"  UMAP completed in {umap_time:.2f}s")
                
                # Step 2: Apply HDBSCAN
                hdbscan_start = time.time()
                
                # Create a temporary config for HDBSCAN
                temp_config = {
                    'hdbscan': hdbscan_params.copy()
                }
                
                # Create temporary config file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump(temp_config, f)
                    temp_config_path = f.name
                
                try:
                    cluster_labels, hdbscan_model, cluster_info = apply_hdbscan_clustering(
                        data=reduced_data,
                        config_path=temp_config_path,
                        validate_config=False
                    )
                finally:
                    # Clean up temporary file
                    os.unlink(temp_config_path)
                
                hdbscan_time = time.time() - hdbscan_start
                print(f"  HDBSCAN completed in {hdbscan_time:.2f}s")
                
                # Step 3: Evaluate pipeline
                eval_start = time.time()
                metrics = self.evaluate_pipeline(
                    reduced_data, cluster_labels, hdbscan_model, 
                    umap_params, hdbscan_params
                )
                eval_time = time.time() - eval_start
                
                metrics['umap_time'] = umap_time
                metrics['hdbscan_time'] = hdbscan_time
                metrics['evaluation_time'] = eval_time
                metrics['total_time'] = time.time() - combination_start
                
                self.results.append(metrics)
                
                print(f"  Results: {metrics['n_clusters']} clusters, "
                      f"{metrics['noise_percentage']:.1f}% noise, "
                      f"composite score: {metrics['composite_score']:.1f}")
                
                if metrics.get('silhouette_score') is not None:
                    print(f"  Silhouette: {metrics['silhouette_score']:.3f}")
                
                # Check if this is the best result so far
                if (self.best_result is None or 
                    metrics['composite_score'] > self.best_result['composite_score']):
                    self.best_result = metrics
                    print(f"  *** NEW BEST RESULT! ***")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                error_result = {
                    'umap_params': umap_params,
                    'hdbscan_params': hdbscan_params,
                    'error': str(e),
                    'composite_score': -1000  # Very low score for errors
                }
                self.results.append(error_result)
            
            # Check time limit
            elapsed_time = time.time() - start_time
            if elapsed_time > time_limit_per_combination * len(combinations_to_test):
                print(f"\\nTime limit reached. Stopping after {i+1} combinations.")
                break
        
        total_time = time.time() - start_time
        print(f"\\nTuning completed in {total_time/60:.1f} minutes")
        
        return self.results
    
    def generate_report(self, output_dir: str = "tuning_results") -> str:
        """
        Generate a comprehensive report of tuning results.
        
        Parameters:
        -----------
        output_dir : str, default "tuning_results"
            Directory to save the report and visualizations
            
        Returns:
        --------
        str
            Path to the generated report file
        """
        
        if not self.results:
            print("No results to report. Run tuning first.")
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Filter valid results
        valid_results = [r for r in self.results if 'error' not in r]
        
        if not valid_results:
            print("No valid results to report.")
            return None
        
        # Sort by composite score
        valid_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"umap_hdbscan_tuning_report_{timestamp}.md")
        
        with open(report_path, 'w') as f:
            f.write("# UMAP + HDBSCAN Parameter Tuning Report\\n\\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Data shape: {self.data.shape}\\n")
            f.write(f"Total combinations tested: {len(self.results)}\\n")
            f.write(f"Valid results: {len(valid_results)}\\n\\n")
            
            # Best results
            f.write("## Top 5 Results\\n\\n")
            for i, result in enumerate(valid_results[:5]):
                f.write(f"### Rank {i+1}\\n")
                f.write(f"**Composite Score:** {result['composite_score']:.2f}\\n\\n")
                f.write(f"**UMAP Parameters:**\\n")
                for k, v in result['umap_params'].items():
                    f.write(f"- {k}: {v}\\n")
                f.write(f"\\n**HDBSCAN Parameters:**\\n")
                for k, v in result['hdbscan_params'].items():
                    f.write(f"- {k}: {v}\\n")
                f.write(f"\\n**Results:**\\n")
                f.write(f"- Clusters: {result['n_clusters']}\\n")
                f.write(f"- Noise percentage: {result['noise_percentage']:.1f}%\\n")
                if result.get('silhouette_score'):
                    f.write(f"- Silhouette score: {result['silhouette_score']:.3f}\\n")
                f.write(f"- Total time: {result.get('total_time', 0):.1f}s\\n\\n")
            
            # Configuration recommendations
            if self.best_result:
                f.write("## Recommended Configuration\\n\\n")
                f.write("```yaml\\n")
                f.write("umap:\\n")
                for k, v in self.best_result['umap_params'].items():
                    f.write(f"  {k}: {v}\\n")
                f.write("\\nhdbscan:\\n")
                for k, v in self.best_result['hdbscan_params'].items():
                    f.write(f"  {k}: {v}\\n")
                f.write("```\\n\\n")
        
        # Create visualizations
        self._create_visualizations(valid_results, output_dir)
        
        # Save detailed results as JSON
        results_path = os.path.join(output_dir, f"detailed_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(valid_results, f, indent=2, default=str)
        
        print(f"Report generated: {report_path}")
        print(f"Detailed results saved: {results_path}")
        
        return report_path
    
    def _create_visualizations(self, results: List[Dict], output_dir: str):
        """Create visualization plots for the tuning results."""
        
        # Convert results to DataFrame for easier plotting
        df_data = []
        for r in results:
            row = {}
            row.update(r['umap_params'])
            row.update(r['hdbscan_params'])
            for k, v in r.items():
                if k not in ['umap_params', 'hdbscan_params']:
                    row[k] = v
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('UMAP + HDBSCAN Parameter Tuning Results', fontsize=16)
        
        # Plot 1: Composite score vs number of clusters
        axes[0, 0].scatter(df['n_clusters'], df['composite_score'], alpha=0.7)
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Composite Score')
        axes[0, 0].set_title('Composite Score vs Number of Clusters')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Noise percentage vs silhouette score
        silhouette_data = df[df['silhouette_score'].notna()]
        if not silhouette_data.empty:
            axes[0, 1].scatter(silhouette_data['noise_percentage'], 
                             silhouette_data['silhouette_score'], alpha=0.7)
            axes[0, 1].set_xlabel('Noise Percentage')
            axes[0, 1].set_ylabel('Silhouette Score')
            axes[0, 1].set_title('Noise vs Silhouette Score')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: UMAP components vs performance
        axes[1, 0].scatter(df['n_components'], df['composite_score'], alpha=0.7)
        axes[1, 0].set_xlabel('UMAP Components')
        axes[1, 0].set_ylabel('Composite Score')
        axes[1, 0].set_title('UMAP Dimensions vs Performance')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: HDBSCAN min_cluster_size vs clusters found
        axes[1, 1].scatter(df['min_cluster_size'], df['n_clusters'], alpha=0.7)
        axes[1, 1].set_xlabel('HDBSCAN Min Cluster Size')
        axes[1, 1].set_ylabel('Number of Clusters Found')
        axes[1, 1].set_title('Min Cluster Size vs Clusters Found')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tuning_results_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")


def main():
    """Main function to run the comprehensive parameter tuning."""
    
    print("UMAP + HDBSCAN Comprehensive Parameter Tuning")
    print("=" * 50)
    
    # Load data - try multiple possible paths
    possible_data_paths = [
        # From clustering_functionality folder
        os.path.join(os.path.dirname(__file__), '../../../data/processed_data/AS_1_feature_data.csv'),
        # From project root
        os.path.join(os.getcwd(), 'data/processed_data/AS_1_feature_data.csv'),
    ]
    
    data_path = None
    for path in possible_data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print("Error: Feature data not found. Tried the following paths:")
        for path in possible_data_paths:
            print(f"  - {path}")
        print("\\nPlease ensure you're running from the project root directory.")
        return
    
    print(f"Loading feature data from: {data_path}")
    
    # Load and preprocess data
    try:
        raw_data = pd.read_csv(data_path)
        print(f"Raw data shape: {raw_data.shape}")
        
        # Preprocess data for clustering using the file path
        processed_data, preprocessing_info = preprocess_for_clustering(
            data_path=data_path, 
            apply_log_transform=True, 
            apply_scaling=True
        )
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Preprocessing steps applied: {preprocessing_info['steps_applied']}")
        
    except Exception as e:
        print(f"Error loading/preprocessing data: {e}")
        return
    
    # Initialize tuner
    tuner = UMAPHDBSCANTuner(processed_data)
    
    # Run tuning
    print("\\nStarting parameter tuning...")
    results = tuner.run_comprehensive_tuning(
        max_combinations=6,  # Start with fewer combinations for testing
        time_limit_per_combination=120  # 2 minutes per combination
    )
    
    # Generate report
    print("\\nGenerating comprehensive report...")
    report_path = tuner.generate_report()
    
    # Print summary
    if tuner.best_result:
        print("\\n" + "=" * 60)
        print("BEST CONFIGURATION FOUND:")
        print("=" * 60)
        print("UMAP Parameters:")
        for k, v in tuner.best_result['umap_params'].items():
            print(f"  {k}: {v}")
        print("\\nHDBSCAN Parameters:")
        for k, v in tuner.best_result['hdbscan_params'].items():
            print(f"  {k}: {v}")
        print(f"\\nResults:")
        print(f"  Clusters: {tuner.best_result['n_clusters']}")
        print(f"  Noise: {tuner.best_result['noise_percentage']:.1f}%")
        print(f"  Composite Score: {tuner.best_result['composite_score']:.2f}")
        if tuner.best_result.get('silhouette_score'):
            print(f"  Silhouette Score: {tuner.best_result['silhouette_score']:.3f}")
    
    print(f"\\nDetailed report available at: {report_path}")


if __name__ == "__main__":
    main()
