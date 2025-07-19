#File to produce the functionality of clustering using HBDSCAN algorithm - to be executed in the scripts folder in logic integrating both UMAP and HBDSCAN on processed data.
#HBDSCAN clusters the output of the UMAP dimensionality reduction, making use of configurations specified in the config_cluster.yaml file.

import pandas as pd
import numpy as np
import yaml
import os
from typing import Optional, Tuple, Dict, Any, List
import hdbscan
import warnings
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import sys

# Configure warnings to reduce noise during clustering
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero encountered.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*n_init.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*default value of n_init.*')

# Add the parent directory to path to import from source_code_package
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


# ================================
# CONFIGURATION LOADING
# ================================

def load_hdbscan_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load HDBSCAN configuration from YAML file.
    
    This function reads the clustering configuration file and extracts HDBSCAN-specific
    parameters that control the clustering algorithm's behavior.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to the config YAML file. If None, uses default config_cluster.yaml.
        
    Returns:
    --------
    dict
        Dictionary containing HDBSCAN configuration parameters including:
        - min_cluster_size: Minimum number of samples in a cluster
        - min_samples: Number of samples in neighborhood for core point
        - metric: Distance metric to use
        - cluster_selection_method: Method for selecting clusters
        - Additional algorithm parameters
        
    Example:
    --------
    >>> config = load_hdbscan_config()
    >>> min_cluster_size = config['min_cluster_size']
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../../config/config_cluster.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('hdbscan', {})


def validate_hdbscan_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate HDBSCAN configuration parameters and provide warnings/recommendations.
    
    This function checks the validity of HDBSCAN parameters and provides guidance
    on parameter selection based on best practices.
    
    Parameters:
    -----------
    config : dict
        HDBSCAN configuration dictionary
        
    Returns:
    --------
    dict
        Dictionary containing validation results, warnings, and recommendations
        
    Example:
    --------
    >>> config = load_hdbscan_config()
    >>> validation = validate_hdbscan_config(config)
    >>> print(validation['warnings'])
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'recommendations': [],
        'errors': []
    }
    
    # Check min_cluster_size
    min_cluster_size = config.get('min_cluster_size', 15)
    if min_cluster_size < 5:
        validation_results['warnings'].append(
            f"min_cluster_size ({min_cluster_size}) is very small. Consider values >= 5 for stable clusters."
        )
    elif min_cluster_size > 100:
        validation_results['warnings'].append(
            f"min_cluster_size ({min_cluster_size}) is very large. This may result in very few clusters."
        )
    
    # Check min_samples
    min_samples = config.get('min_samples', 5)
    if min_samples > min_cluster_size:
        validation_results['errors'].append(
            f"min_samples ({min_samples}) should not be larger than min_cluster_size ({min_cluster_size})"
        )
        validation_results['valid'] = False
    
    # Check metric
    valid_metrics = ['euclidean', 'manhattan', 'cosine', 'minkowski']
    metric = config.get('metric', 'euclidean')
    if metric not in valid_metrics:
        validation_results['warnings'].append(
            f"Metric '{metric}' may not be supported. Consider using: {valid_metrics}"
        )
    
    # Check cluster_selection_method
    valid_methods = ['eom', 'leaf']
    method = config.get('cluster_selection_method', 'eom')
    if method not in valid_methods:
        validation_results['errors'].append(
            f"cluster_selection_method '{method}' is invalid. Use one of: {valid_methods}"
        )
        validation_results['valid'] = False
    
    # Provide recommendations
    if min_cluster_size < 15:
        validation_results['recommendations'].append(
            "Consider increasing min_cluster_size to 15-30 for more stable clusters in typical datasets."
        )
    
    if min_samples == 1:
        validation_results['recommendations'].append(
            "Consider setting min_samples > 1 to reduce noise sensitivity."
        )
    
    return validation_results


# ================================
# CORE CLUSTERING FUNCTIONALITY
# ================================

def apply_hdbscan_clustering(data: np.ndarray, config_path: Optional[str] = None, 
                           validate_config: bool = True) -> Tuple[np.ndarray, hdbscan.HDBSCAN, Dict[str, Any]]:
    """
    Apply HDBSCAN clustering to the input data (typically UMAP-reduced data).
    
    This is the core clustering function that takes dimensionality-reduced data
    and applies HDBSCAN clustering algorithm to identify clusters.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data for clustering (typically UMAP-reduced data)
        Shape: (n_samples, n_features)
    config_path : str, optional
        Path to configuration file. If None, uses default config.
    validate_config : bool, default=True
        Whether to validate configuration parameters before clustering
        
    Returns:
    --------
    tuple
        - cluster_labels : np.ndarray
            Cluster labels for each data point (-1 indicates noise)
        - clusterer : hdbscan.HDBSCAN
            Fitted HDBSCAN clusterer object
        - cluster_info : dict
            Dictionary containing clustering information and statistics
            
    Example:
    --------
    >>> umap_data = np.random.rand(100, 10)  # Example UMAP output
    >>> labels, clusterer, info = apply_hdbscan_clustering(umap_data)
    >>> print(f"Found {info['n_clusters']} clusters")
    """
    # Load configuration
    config = load_hdbscan_config(config_path)
    
    # Validate configuration if requested
    if validate_config:
        validation = validate_hdbscan_config(config)
        if not validation['valid']:
            raise ValueError(f"Invalid HDBSCAN configuration: {validation['errors']}")
        
        # Print warnings and recommendations
        for warning in validation['warnings']:
            print(f"WARNING: {warning}")
        for rec in validation['recommendations']:
            print(f"RECOMMENDATION: {rec}")
    
    # Prepare HDBSCAN parameters
    hdbscan_params = {
        'min_cluster_size': config.get('min_cluster_size', 15),
        'min_samples': config.get('min_samples', 5),
        'cluster_selection_epsilon': config.get('cluster_selection_epsilon', 0.0),
        'metric': config.get('metric', 'euclidean'),
        'alpha': config.get('alpha', 1.0),
        'algorithm': config.get('algorithm', 'best'),
        'leaf_size': config.get('leaf_size', 40),
        'cluster_selection_method': config.get('cluster_selection_method', 'eom'),
        'allow_single_cluster': config.get('allow_single_cluster', False),
        'max_cluster_size': config.get('max_cluster_size', 0),
        'prediction_data': config.get('prediction_data', True),
        'core_dist_n_jobs': config.get('core_dist_n_jobs', -1),
        'gen_min_span_tree': config.get('gen_min_span_tree', False),
        'approx_min_span_tree': config.get('approx_min_span_tree', True),
        'match_reference_implementation': config.get('match_reference_implementation', False)
    }
    
    # Remove parameters that are 0 or None where appropriate
    if hdbscan_params['max_cluster_size'] == 0:
        del hdbscan_params['max_cluster_size']
    if hdbscan_params['cluster_selection_epsilon'] == 0.0:
        del hdbscan_params['cluster_selection_epsilon']
    
    print(f"Applying HDBSCAN clustering with parameters: {hdbscan_params}")
    
    # Initialize HDBSCAN
    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    cluster_labels = clusterer.fit_predict(data)
    
    # Calculate cluster statistics
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
    n_noise = np.sum(cluster_labels == -1)
    n_points = len(cluster_labels)
    
    # Calculate cluster sizes
    cluster_sizes = {}
    for label in unique_labels:
        if label >= 0:  # Exclude noise
            cluster_sizes[label] = np.sum(cluster_labels == label)
    
    # Create cluster information dictionary
    cluster_info = {
        'n_clusters': n_clusters,
        'n_noise_points': n_noise,
        'n_total_points': n_points,
        'noise_percentage': (n_noise / n_points) * 100,
        'cluster_sizes': cluster_sizes,
        'cluster_labels': cluster_labels,
        'has_prediction_data': hasattr(clusterer, 'prediction_data_') and clusterer.prediction_data_ is not None,
        'probabilities': clusterer.probabilities_ if hasattr(clusterer, 'probabilities_') else None,
        'config_used': config
    }
    
    print(f"Clustering completed: {n_clusters} clusters, {n_noise} noise points ({cluster_info['noise_percentage']:.1f}%)")
    
    return cluster_labels, clusterer, cluster_info


# ================================
# CLUSTERING EVALUATION METRICS
# ================================

def evaluate_clustering_quality(data: np.ndarray, cluster_labels: np.ndarray, 
                               clusterer: hdbscan.HDBSCAN = None) -> Dict[str, Any]:
    """
    Evaluate the quality of HDBSCAN clustering results using multiple metrics.
    
    This function computes various clustering quality metrics to assess how well
    the algorithm has performed on the given data.
    
    Parameters:
    -----------
    data : np.ndarray
        The data that was clustered
    cluster_labels : np.ndarray
        Cluster labels assigned by HDBSCAN
    clusterer : hdbscan.HDBSCAN, optional
        The fitted HDBSCAN clusterer object
        
    Returns:
    --------
    dict
        Dictionary containing various clustering quality metrics:
        - silhouette_score: Average silhouette score
        - calinski_harabasz_score: Variance ratio criterion
        - davies_bouldin_score: Davies-Bouldin index
        - cluster_validity_metrics: Additional HDBSCAN-specific metrics
        
    Example:
    --------
    >>> evaluation = evaluate_clustering_quality(data, cluster_labels, clusterer)
    >>> print(f"Silhouette Score: {evaluation['silhouette_score']:.3f}")
    """
    evaluation_results = {}
    
    # Filter out noise points for some metrics
    non_noise_mask = cluster_labels >= 0
    non_noise_data = data[non_noise_mask]
    non_noise_labels = cluster_labels[non_noise_mask]
    
    # Check if we have enough clusters and points for evaluation
    n_clusters = len(np.unique(non_noise_labels))
    n_points = len(non_noise_labels)
    
    if n_clusters < 2:
        evaluation_results['warning'] = "Less than 2 clusters found. Most metrics cannot be computed."
        evaluation_results['n_clusters'] = n_clusters
        evaluation_results['n_points'] = n_points
        return evaluation_results
    
    if n_points < 10:
        evaluation_results['warning'] = "Too few points for reliable evaluation metrics."
        evaluation_results['n_clusters'] = n_clusters
        evaluation_results['n_points'] = n_points
        return evaluation_results
    
    try:
        # Suppress overflow warnings during metric computation
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow encountered.*')
            
            # Silhouette Score (higher is better, range: -1 to 1)
            silhouette_avg = silhouette_score(non_noise_data, non_noise_labels)
            evaluation_results['silhouette_score'] = silhouette_avg
            
            # Calinski-Harabasz Score (higher is better)
            ch_score = calinski_harabasz_score(non_noise_data, non_noise_labels)
            evaluation_results['calinski_harabasz_score'] = ch_score
            
            # Davies-Bouldin Score (lower is better)
            db_score = davies_bouldin_score(non_noise_data, non_noise_labels)
            evaluation_results['davies_bouldin_score'] = db_score
        
    except Exception as e:
        evaluation_results['error'] = f"Error computing basic metrics: {str(e)}"
    
    # HDBSCAN-specific metrics
    if clusterer is not None:
        try:
            # Cluster persistence (from HDBSCAN tree structure)
            if hasattr(clusterer, 'cluster_persistence_'):
                evaluation_results['cluster_persistence'] = clusterer.cluster_persistence_
            
            # Cluster probabilities statistics
            if hasattr(clusterer, 'probabilities_'):
                probs = clusterer.probabilities_[non_noise_mask]
                evaluation_results['probability_stats'] = {
                    'mean_probability': np.mean(probs),
                    'std_probability': np.std(probs),
                    'min_probability': np.min(probs),
                    'max_probability': np.max(probs)
                }
            
            # Outlier scores
            if hasattr(clusterer, 'outlier_scores_'):
                evaluation_results['outlier_score_stats'] = {
                    'mean_outlier_score': np.mean(clusterer.outlier_scores_),
                    'std_outlier_score': np.std(clusterer.outlier_scores_),
                    'max_outlier_score': np.max(clusterer.outlier_scores_)
                }
                
        except Exception as e:
            evaluation_results['hdbscan_metrics_error'] = f"Error computing HDBSCAN metrics: {str(e)}"
    
    # Cluster balance metrics
    unique_labels, counts = np.unique(non_noise_labels, return_counts=True)
    if len(counts) > 1:
        # Coefficient of variation for cluster sizes
        cv_cluster_sizes = np.std(counts) / np.mean(counts)
        evaluation_results['cluster_size_cv'] = cv_cluster_sizes
        
        # Entropy of cluster size distribution (higher = more balanced)
        normalized_counts = counts / np.sum(counts)
        size_entropy = entropy(normalized_counts)
        evaluation_results['cluster_size_entropy'] = size_entropy
    
    evaluation_results['n_clusters'] = n_clusters
    evaluation_results['n_points'] = n_points
    evaluation_results['noise_points'] = np.sum(cluster_labels == -1)
    
    return evaluation_results


# ================================
# CLUSTER PREDICTION FUNCTIONALITY
# ================================

def predict_cluster_membership(clusterer: hdbscan.HDBSCAN, new_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict cluster membership for new data points using a fitted HDBSCAN clusterer.
    
    This function allows you to assign new data points to existing clusters
    identified by HDBSCAN, which is useful for streaming data or new observations.
    
    Parameters:
    -----------
    clusterer : hdbscan.HDBSCAN
        A fitted HDBSCAN clusterer with prediction_data=True
    new_data : np.ndarray
        New data points to assign to clusters
        Shape: (n_new_samples, n_features)
        
    Returns:
    --------
    tuple
        - predicted_labels : np.ndarray
            Predicted cluster labels for new data points
        - predicted_probabilities : np.ndarray
            Prediction probabilities for cluster assignments
            
    Example:
    --------
    >>> new_points = np.random.rand(10, 10)  # 10 new points
    >>> labels, probs = predict_cluster_membership(clusterer, new_points)
    >>> print(f"Predicted labels: {labels}")
    """
    if not hasattr(clusterer, 'prediction_data_') or clusterer.prediction_data_ is None:
        raise ValueError("Clusterer was not fitted with prediction_data=True. Cannot predict new points.")
    
    # Predict cluster membership
    predicted_labels, predicted_probabilities = hdbscan.approximate_predict(clusterer, new_data)
    
    return predicted_labels, predicted_probabilities


# ================================
# VISUALIZATION FUNCTIONS
# ================================

def plot_clustering_results(data: np.ndarray, cluster_labels: np.ndarray, 
                          title: str = "HDBSCAN Clustering Results",
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None) -> None:
    """
    Visualize HDBSCAN clustering results with scatter plots and cluster statistics.
    
    This function creates comprehensive visualizations of the clustering results
    including cluster scatter plots and summary statistics.
    
    Parameters:
    -----------
    data : np.ndarray
        The clustered data (typically 2D UMAP reduction for visualization)
    cluster_labels : np.ndarray
        Cluster labels from HDBSCAN
    title : str, default="HDBSCAN Clustering Results"
        Title for the plot
    figsize : tuple, default=(12, 8)
        Figure size for matplotlib
    save_path : str, optional
        Path to save the plot. If None, plot is displayed but not saved.
        
    Example:
    --------
    >>> plot_clustering_results(umap_2d_data, cluster_labels, 
    ...                        title="Customer Segmentation Results",
    ...                        save_path="clustering_results.png")
    """
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Determine if we can make 2D plots
    if data.shape[1] >= 2:
        # Main scatter plot
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points
                mask = cluster_labels == label
                ax1.scatter(data[mask, 0], data[mask, 1], c='black', marker='x', 
                          s=50, alpha=0.6, label='Noise')
            else:
                mask = cluster_labels == label
                ax1.scatter(data[mask, 0], data[mask, 1], c=[color], 
                          s=50, alpha=0.7, label=f'Cluster {label}')
        
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        ax1.set_title('Cluster Visualization')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Density plot
        ax2.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        ax2.set_title('Cluster Density')
        
    else:
        ax1.text(0.5, 0.5, 'Data dimensionality too low\nfor 2D visualization', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Cluster Visualization')
        
        ax2.text(0.5, 0.5, 'Data dimensionality too low\nfor 2D visualization', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Cluster Density')
    
    # Cluster size distribution
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    cluster_labels_for_hist = unique_labels[unique_labels >= 0]  # Exclude noise
    cluster_counts = counts[unique_labels >= 0]
    
    if len(cluster_labels_for_hist) > 0:
        ax3.bar(range(len(cluster_labels_for_hist)), cluster_counts, 
                color='skyblue', alpha=0.7)
        ax3.set_xlabel('Cluster Index')
        ax3.set_ylabel('Number of Points')
        ax3.set_title('Cluster Size Distribution')
        
        # Only show x-tick labels if we have a reasonable number of clusters
        if len(cluster_labels_for_hist) <= 20:
            ax3.set_xticks(range(len(cluster_labels_for_hist)))
            ax3.set_xticklabels(cluster_labels_for_hist)
        else:
            # For many clusters, show only some tick marks
            n_ticks = min(10, len(cluster_labels_for_hist))
            tick_indices = np.linspace(0, len(cluster_labels_for_hist)-1, n_ticks, dtype=int)
            ax3.set_xticks(tick_indices)
            ax3.set_xticklabels([str(cluster_labels_for_hist[i]) for i in tick_indices], rotation=45)
    
    # Summary statistics
    n_clusters = len(cluster_labels_for_hist)
    n_noise = np.sum(cluster_labels == -1)
    n_total = len(cluster_labels)
    
    # Calculate cluster size statistics safely
    min_size = np.min(cluster_counts) if len(cluster_counts) > 0 else 0
    max_size = np.max(cluster_counts) if len(cluster_counts) > 0 else 0
    mean_size = np.mean(cluster_counts) if len(cluster_counts) > 0 else 0
    
    stats_text = f"""
    Total Points: {n_total}
    Clusters Found: {n_clusters}
    Noise Points: {n_noise}
    Noise %: {(n_noise/n_total)*100:.1f}%
    
    Cluster Sizes:
    Min: {min_size}
    Max: {max_size}
    Mean: {mean_size:.1f}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    ax4.set_title('Clustering Statistics')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


# ================================
# PIPELINE INTEGRATION FUNCTIONS
# ================================

def hdbscan_clustering_pipeline(umap_data: np.ndarray, config_path: Optional[str] = None,
                               evaluate_quality: bool = True, 
                               create_visualizations: bool = True,
                               save_results: bool = True,
                               output_dir: str = "clustering_results") -> Dict[str, Any]:
    """
    Complete HDBSCAN clustering pipeline for UMAP-reduced data.
    
    This is a comprehensive pipeline function that performs clustering, evaluation,
    and visualization in a single call. It's designed to be used from script files.
    
    Parameters:
    -----------
    umap_data : np.ndarray
        UMAP-reduced data for clustering
    config_path : str, optional
        Path to configuration file
    evaluate_quality : bool, default=True
        Whether to compute clustering quality metrics
    create_visualizations : bool, default=True
        Whether to create and save visualizations
    save_results : bool, default=True
        Whether to save clustering results to files
    output_dir : str, default="clustering_results"
        Directory to save results
        
    Returns:
    --------
    dict
        Complete results dictionary containing:
        - cluster_labels: Cluster assignments
        - clusterer: Fitted HDBSCAN object
        - cluster_info: Clustering statistics
        - evaluation_metrics: Quality metrics (if requested)
        - file_paths: Paths to saved files (if requested)
        
    Example:
    --------
    >>> # Assuming you have UMAP-reduced data
    >>> results = hdbscan_clustering_pipeline(umap_data, 
    ...                                     config_path="my_config.yaml",
    ...                                     output_dir="results")
    >>> print(f"Found {results['cluster_info']['n_clusters']} clusters")
    """
    print("Starting HDBSCAN clustering pipeline...")
    
    # Create output directory if needed
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: Apply HDBSCAN clustering
    print("Step 1: Applying HDBSCAN clustering...")
    cluster_labels, clusterer, cluster_info = apply_hdbscan_clustering(
        umap_data, config_path=config_path, validate_config=True
    )
    
    # Initialize results dictionary
    results = {
        'cluster_labels': cluster_labels,
        'clusterer': clusterer,
        'cluster_info': cluster_info,
        'file_paths': {}
    }
    
    # Step 2: Evaluate clustering quality
    if evaluate_quality:
        print("Step 2: Evaluating clustering quality...")
        evaluation_metrics = evaluate_clustering_quality(umap_data, cluster_labels, clusterer)
        results['evaluation_metrics'] = evaluation_metrics
        
        # Print key metrics
        if 'silhouette_score' in evaluation_metrics:
            print(f"  Silhouette Score: {evaluation_metrics['silhouette_score']:.3f}")
        if 'calinski_harabasz_score' in evaluation_metrics:
            print(f"  Calinski-Harabasz Score: {evaluation_metrics['calinski_harabasz_score']:.2f}")
        if 'davies_bouldin_score' in evaluation_metrics:
            print(f"  Davies-Bouldin Score: {evaluation_metrics['davies_bouldin_score']:.3f}")
    
    # Step 3: Create visualizations
    if create_visualizations:
        print("Step 3: Creating visualizations...")
        try:
            if umap_data.shape[1] >= 2:
                # For visualization, use first 2 dimensions if more than 2D
                print(f"  Data shape for visualization: {umap_data.shape}")
                viz_data = umap_data[:, :2] if umap_data.shape[1] > 2 else umap_data
                print(f"  Visualization data shape: {viz_data.shape}")
                
                if save_results:
                    viz_path = os.path.join(output_dir, "hdbscan_clustering_results.png")
                    plot_clustering_results(viz_data, cluster_labels, 
                                          title="HDBSCAN Clustering Results",
                                          save_path=viz_path)
                    results['file_paths']['visualization'] = viz_path
                else:
                    plot_clustering_results(viz_data, cluster_labels, 
                                          title="HDBSCAN Clustering Results")
            else:
                print("  Warning: Data has less than 2 dimensions, skipping visualization")
        except Exception as e:
            print(f"  Warning: Error creating visualization: {e}")
            print("  Continuing without visualization...")
    
    # Step 4: Save results
    if save_results:
        print("Step 4: Saving results...")
        
        # Save cluster labels
        labels_path = os.path.join(output_dir, "cluster_labels.csv")
        pd.DataFrame({'cluster_label': cluster_labels}).to_csv(labels_path, index=False)
        results['file_paths']['cluster_labels'] = labels_path
        
        # Save clustered data with labels
        clustered_data_path = os.path.join(output_dir, "clustered_data.csv")
        clustered_df = pd.DataFrame(umap_data, columns=[f'dim_{i}' for i in range(umap_data.shape[1])])
        clustered_df['cluster_label'] = cluster_labels
        clustered_df.to_csv(clustered_data_path, index=False)
        results['file_paths']['clustered_data'] = clustered_data_path
        
        # Save cluster summary
        summary_path = os.path.join(output_dir, "clustering_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("HDBSCAN Clustering Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Number of clusters: {cluster_info['n_clusters']}\n")
            f.write(f"Number of noise points: {cluster_info['n_noise_points']}\n")
            f.write(f"Total points: {cluster_info['n_total_points']}\n")
            f.write(f"Noise percentage: {cluster_info['noise_percentage']:.1f}%\n\n")
            
            f.write("Cluster sizes:\n")
            for cluster_id, size in cluster_info['cluster_sizes'].items():
                f.write(f"  Cluster {cluster_id}: {size} points\n")
            
            if evaluate_quality and 'evaluation_metrics' in results:
                f.write("\nQuality Metrics:\n")
                metrics = results['evaluation_metrics']
                if 'silhouette_score' in metrics:
                    f.write(f"  Silhouette Score: {metrics['silhouette_score']:.3f}\n")
                if 'calinski_harabasz_score' in metrics:
                    f.write(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}\n")
                if 'davies_bouldin_score' in metrics:
                    f.write(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}\n")
        
        results['file_paths']['summary'] = summary_path
        
        print(f"Results saved to: {output_dir}")
    
    print("HDBSCAN clustering pipeline completed!")
    return results


# ================================
# INTEGRATION WITH UMAP
# ================================

def run_umap_hdbscan_pipeline(data_path: Optional[str] = None, 
                            config_path: Optional[str] = None,
                            output_dir: str = "umap_hdbscan_results") -> Dict[str, Any]:
    """
    Complete pipeline integrating UMAP dimensionality reduction with HDBSCAN clustering.
    
    This function provides a complete end-to-end pipeline that:
    1. Loads and preprocesses data
    2. Applies UMAP dimensionality reduction
    3. Performs HDBSCAN clustering
    4. Evaluates and visualizes results
    
    Parameters:
    -----------
    data_path : str, optional
        Path to data file. If None, uses path from config file.
    config_path : str, optional
        Path to configuration file
    output_dir : str, default="umap_hdbscan_results"
        Directory to save all results
        
    Returns:
    --------
    dict
        Complete pipeline results including UMAP and HDBSCAN results
        
    Example:
    --------
    >>> # Run complete pipeline
    >>> results = run_umap_hdbscan_pipeline(
    ...     data_path="data/processed_data.csv",
    ...     config_path="config/config_cluster.yaml",
    ...     output_dir="final_results"
    ... )
    >>> print(f"Pipeline completed with {results['hdbscan_results']['cluster_info']['n_clusters']} clusters")
    """
    print("Starting complete UMAP + HDBSCAN pipeline...")
    
    # Import UMAP functionality
    try:
        from .UMAP_dim_reduction import umap_with_preprocessing
    except ImportError:
        try:
            from UMAP_dim_reduction import umap_with_preprocessing
        except ImportError:
            print("ERROR: Could not import UMAP functionality. Ensure UMAP_dim_reduction.py is available.")
            raise
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: UMAP Dimensionality Reduction
    print("Step 1: Applying UMAP dimensionality reduction...")
    umap_results = umap_with_preprocessing(
        data_path=data_path,
        config_path=config_path,
        save_results=True,
        output_dir=os.path.join(output_dir, "umap_results")
    )
    
    # Step 2: HDBSCAN Clustering
    print("Step 2: Applying HDBSCAN clustering...")
    hdbscan_results = hdbscan_clustering_pipeline(
        umap_data=umap_results['reduced_data'],
        config_path=config_path,
        evaluate_quality=True,
        create_visualizations=True,
        save_results=True,
        output_dir=os.path.join(output_dir, "hdbscan_results")
    )
    
    # Step 3: Create combined results
    print("Step 3: Creating combined results...")
    combined_results = {
        'umap_results': umap_results,
        'hdbscan_results': hdbscan_results,
        'pipeline_info': {
            'n_original_features': umap_results.get('original_data_shape', [None, None])[1],
            'n_reduced_features': umap_results['reduced_data'].shape[1],
            'n_clusters_found': hdbscan_results['cluster_info']['n_clusters'],
            'total_data_points': len(hdbscan_results['cluster_labels']),
            'noise_points': hdbscan_results['cluster_info']['n_noise_points']
        }
    }
    
    # Save combined summary
    combined_summary_path = os.path.join(output_dir, "pipeline_summary.txt")
    with open(combined_summary_path, 'w') as f:
        f.write("UMAP + HDBSCAN Pipeline Summary\n")
        f.write("=" * 40 + "\n\n")
        
        # UMAP section
        f.write("UMAP Dimensionality Reduction:\n")
        f.write(f"  Original dimensions: {combined_results['pipeline_info']['n_original_features']}\n")
        f.write(f"  Reduced dimensions: {combined_results['pipeline_info']['n_reduced_features']}\n")
        f.write(f"  Total data points: {combined_results['pipeline_info']['total_data_points']}\n\n")
        
        # HDBSCAN section
        f.write("HDBSCAN Clustering:\n")
        f.write(f"  Clusters found: {combined_results['pipeline_info']['n_clusters_found']}\n")
        f.write(f"  Noise points: {combined_results['pipeline_info']['noise_points']}\n")
        f.write(f"  Noise percentage: {(combined_results['pipeline_info']['noise_points'] / combined_results['pipeline_info']['total_data_points']) * 100:.1f}%\n\n")
        
        # Quality metrics
        if 'evaluation_metrics' in hdbscan_results:
            f.write("Clustering Quality Metrics:\n")
            metrics = hdbscan_results['evaluation_metrics']
            if 'silhouette_score' in metrics:
                f.write(f"  Silhouette Score: {metrics['silhouette_score']:.3f}\n")
            if 'calinski_harabasz_score' in metrics:
                f.write(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}\n")
            if 'davies_bouldin_score' in metrics:
                f.write(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}\n")
    
    combined_results['file_paths'] = {'pipeline_summary': combined_summary_path}
    
    print(f"Complete pipeline results saved to: {output_dir}")
    print("Pipeline completed successfully!")
    
    return combined_results


def run_flexible_hdbscan_pipeline(data_path: Optional[str] = None, 
                                config_path: Optional[str] = None,
                                output_dir: str = "flexible_hdbscan_results") -> Dict[str, Any]:
    """
    Flexible pipeline that can run HDBSCAN with or without UMAP dimensionality reduction.
    
    This function checks the configuration to determine whether to apply UMAP 
    dimensionality reduction before HDBSCAN clustering. If UMAP is disabled in the 
    config, it will run HDBSCAN directly on the preprocessed features.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to data file. If None, uses path from config file.
    config_path : str, optional
        Path to configuration file
    output_dir : str, default="flexible_hdbscan_results"
        Directory to save all results
        
    Returns:
    --------
    dict
        Complete pipeline results. Structure varies based on whether UMAP was used:
        - If UMAP enabled: includes both UMAP and HDBSCAN results
        - If UMAP disabled: includes preprocessing and HDBSCAN results only
        
    Example:
    --------
    >>> # Run pipeline with UMAP (if enabled in config)
    >>> results = run_flexible_hdbscan_pipeline(
    ...     data_path="data/processed_data.csv",
    ...     config_path="config/config_cluster.yaml",
    ...     output_dir="results"
    ... )
    >>> if results['umap_enabled']:
    ...     print("Used UMAP dimensionality reduction")
    ... else:
    ...     print("Ran HDBSCAN directly on preprocessed features")
    """
    print("Starting flexible HDBSCAN pipeline...")
    
    # Load configuration
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../../config/config_cluster.yaml')
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        raise ValueError(f"Error loading configuration from {config_path}: {e}")
    
    # Check if UMAP is enabled
    umap_enabled = config.get('umap', {}).get('enabled', True)  # Default to True for backward compatibility
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if umap_enabled:
        print("UMAP is enabled - running complete UMAP + HDBSCAN pipeline...")
        results = run_umap_hdbscan_pipeline(
            data_path=data_path,
            config_path=config_path,
            output_dir=output_dir
        )
        results['umap_enabled'] = True
        return results
    else:
        print("UMAP is disabled - running HDBSCAN directly on preprocessed features...")
        
        # Import preprocessing functionality
        try:
            from ...data.preprocess_cluster import preprocess_for_clustering
        except ImportError:
            try:
                from data.preprocess_cluster import preprocess_for_clustering
            except ImportError:
                print("ERROR: Could not import preprocessing functionality. Ensure preprocess_cluster.py is available.")
                raise
        
        # Step 1: Load and preprocess data (without UMAP)
        print("Step 1: Loading and preprocessing data...")
        
        # Determine data path
        if data_path is None:
            data_path = config.get('data', {}).get('raw_data_path', 'data/raw_data/new_raw_data_polygon.csv')
            # Convert relative path to absolute
            if not os.path.isabs(data_path):
                data_path = os.path.join(os.path.dirname(config_path), '../../', data_path)
        
        # Preprocess data
        preprocessed_data, preprocessing_info = preprocess_for_clustering(
            data_path=data_path,
            config_path=config_path
        )
        
        print(f"Preprocessed data shape: {preprocessed_data.shape}")
        
        # Apply column selection (use UMAP include_columns even though UMAP is disabled)
        include_columns = config.get('umap', {}).get('include_columns', None)
        if include_columns:
            print(f"Selecting specified columns: {include_columns}")
            # Filter to only include specified columns that exist in the data
            available_columns = [col for col in include_columns if col in preprocessed_data.columns]
            if available_columns:
                preprocessed_data = preprocessed_data[available_columns]
                print(f"After column selection, data shape: {preprocessed_data.shape}")
            else:
                print("Warning: None of the specified include_columns were found in the data")
        else:
            print("No column selection specified - using all numeric columns")
            # Select only numeric columns to avoid string columns like 'WALLET'
            preprocessed_data = preprocessed_data.select_dtypes(include=[np.number])
            print(f"After selecting numeric columns, data shape: {preprocessed_data.shape}")
        
        # Step 2: Apply HDBSCAN clustering directly
        print("Step 2: Applying HDBSCAN clustering...")
        hdbscan_results = hdbscan_clustering_pipeline(
            umap_data=preprocessed_data,  # Using preprocessed data instead of UMAP data
            config_path=config_path,
            evaluate_quality=True,
            create_visualizations=True,
            save_results=True,
            output_dir=os.path.join(output_dir, "hdbscan_results")
        )
        
        # Combine results
        complete_results = {
            'umap_enabled': False,
            'preprocessing_results': {
                'preprocessed_data': preprocessed_data,
                'preprocessing_info': preprocessing_info
            },
            'hdbscan_results': hdbscan_results,
            'pipeline_info': {
                'n_original_features': preprocessed_data.shape[1],
                'n_reduced_features': preprocessed_data.shape[1],  # Same as original since no UMAP
                'n_clusters_found': hdbscan_results['cluster_info']['n_clusters'],
                'total_data_points': preprocessed_data.shape[0],
                'noise_points': hdbscan_results['cluster_info']['n_noise_points'],
                'umap_applied': False
            }
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, "pipeline_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Flexible HDBSCAN Pipeline Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"UMAP Dimensionality Reduction: DISABLED\n")
            f.write(f"Original Features: {complete_results['pipeline_info']['n_original_features']}\n")
            f.write(f"Features Used for Clustering: {complete_results['pipeline_info']['n_reduced_features']}\n")
            f.write(f"Clusters Found: {complete_results['pipeline_info']['n_clusters_found']}\n")
            f.write(f"Total Data Points: {complete_results['pipeline_info']['total_data_points']}\n")
            f.write(f"Noise Points: {complete_results['pipeline_info']['noise_points']}\n")
            f.write(f"Noise Percentage: {(complete_results['pipeline_info']['noise_points'] / complete_results['pipeline_info']['total_data_points'] * 100):.1f}%\n")
        
        print(f"Pipeline completed! Results saved to: {output_dir}")
        return complete_results


if __name__ == "__main__":
    """
    Example usage of the HDBSCAN clustering functionality.
    This section demonstrates how to use the functions for testing purposes.
    """
    print("HDBSCAN Clustering Functionality - Test Run")
    print("=" * 50)
    
    # Example 1: Load and validate configuration
    print("\n1. Loading and validating configuration...")
    try:
        config = load_hdbscan_config()
        print(f"Loaded configuration with {len(config)} parameters")
        
        validation = validate_hdbscan_config(config)
        print(f"Configuration validation: {'PASSED' if validation['valid'] else 'FAILED'}")
        
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
                
    except Exception as e:
        print(f"Error loading configuration: {e}")
    
    # Example 2: Test with synthetic data
    print("\n2. Testing with synthetic data...")
    try:
        # Create synthetic UMAP-like data
        np.random.seed(42)
        synthetic_data = np.random.rand(100, 10)  # 100 points, 10 dimensions
        
        # Apply clustering
        labels, clusterer, info = apply_hdbscan_clustering(synthetic_data)
        print(f"Synthetic data clustering: {info['n_clusters']} clusters, {info['n_noise_points']} noise points")
        
        # Evaluate quality
        evaluation = evaluate_clustering_quality(synthetic_data, labels, clusterer)
        if 'silhouette_score' in evaluation:
            print(f"Silhouette score: {evaluation['silhouette_score']:.3f}")
            
    except Exception as e:
        print(f"Error with synthetic data test: {e}")
    
    print("\nTest run completed!")
