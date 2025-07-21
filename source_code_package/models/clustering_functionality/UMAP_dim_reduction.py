#This file includes the necessary functionality for UMAP dimensionality reduction before HDBSCAN clustering.
#This file utilises the features and configurations specified in the config_cluster.yaml file.

import pandas as pd
import numpy as np
import yaml
import os
from typing import Optional, Tuple, Dict, Any
import umap
import warnings
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist
import sys

# Configure warnings to reduce noise during quality evaluation
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero encountered.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*n_init.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*default value of n_init.*')

# Add the parent directory to path to import from source_code_package
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from data.preprocess_cluster import preprocess_for_clustering


# ================================
# CONFIGURATION & VALIDATION
# ================================

def load_umap_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load UMAP configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to the config YAML file. If None, will use default config_cluster.yaml.
    
    Returns:
    --------
    dict
        UMAP configuration dictionary including include_columns
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../../config/config_cluster.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Return the full UMAP configuration including include_columns
    return config.get('umap', {})


def validate_feature_consistency(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate consistency between preprocessing configuration and UMAP include_columns.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to the config YAML file. If None, will use default config_cluster.yaml.
    
    Returns:
    --------
    dict
        Dictionary containing validation results and recommendations
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../../config/config_cluster.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract relevant configurations
    umap_config = config.get('umap', {})
    preprocessing_config = config.get('preprocessing', {})
    
    include_columns = umap_config.get('include_columns', [])
    log_exclude_columns = preprocessing_config.get('log_transformation', {}).get('exclude_columns', [])
    scale_exclude_columns = preprocessing_config.get('scaling', {}).get('exclude_columns', [])
    
    validation_results = {
        'include_columns': include_columns,
        'log_transform_excluded': log_exclude_columns,
        'scaling_excluded': scale_exclude_columns,
        'warnings': [],
        'recommendations': []
    }
    
    # Check if any include_columns are excluded from log transformation
    log_excluded_but_included = set(include_columns) & set(log_exclude_columns)
    if log_excluded_but_included:
        validation_results['warnings'].append(
            f"Columns {list(log_excluded_but_included)} are included in UMAP but excluded from log transformation"
        )
    
    # Check if any include_columns are excluded from scaling
    scale_excluded_but_included = set(include_columns) & set(scale_exclude_columns)
    if scale_excluded_but_included:
        validation_results['warnings'].append(
            f"Columns {list(scale_excluded_but_included)} are included in UMAP but excluded from scaling"
        )
    
    # Add recommendations
    if validation_results['warnings']:
        validation_results['recommendations'].append(
            "Consider reviewing preprocessing exclusions to ensure consistency with UMAP feature selection"
        )
    else:
        validation_results['recommendations'].append(
            "Configuration appears consistent between preprocessing and UMAP feature selection"
        )
    
    return validation_results


# ================================
# UTILITY FUNCTIONS FOR QUALITY EVALUATION
# ================================

def _safe_correlation(x: np.ndarray, y: np.ndarray, method: str = 'spearman') -> Tuple[float, float]:
    """
    Safely calculate correlation between two arrays with proper error handling.
    
    Parameters:
    -----------
    x : np.ndarray
        First array
    y : np.ndarray
        Second array
    method : str, default 'spearman'
        Correlation method ('spearman' or 'pearson')
    
    Returns:
    --------
    tuple
        Correlation coefficient and p-value, or (0.0, 1.0) if calculation fails
    """
    try:
        # Remove NaN and infinite values
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 3:  # Need at least 3 points for correlation
            return 0.0, 1.0
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Check for constant arrays
        if np.std(x_clean) == 0 or np.std(y_clean) == 0:
            return 0.0, 1.0
        
        # Calculate correlation
        if method == 'spearman':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                correlation, p_value = spearmanr(x_clean, y_clean)
        else:  # pearson
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                correlation, p_value = pearsonr(x_clean, y_clean)
        
        # Handle NaN results
        if np.isnan(correlation) or np.isnan(p_value):
            return 0.0, 1.0
        
        return correlation, p_value
        
    except Exception:
        return 0.0, 1.0


def _safe_distance_calculation(data: np.ndarray, max_samples: int = 500) -> np.ndarray:
    """
    Safely calculate pairwise distances with proper error handling.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data array
    max_samples : int, default 500
        Maximum number of samples to use for distance calculation
    
    Returns:
    --------
    np.ndarray
        Pairwise distances, or empty array if calculation fails
    """
    try:
        # Limit sample size for performance
        if data.shape[0] > max_samples:
            sample_indices = np.random.choice(data.shape[0], max_samples, replace=False)
            data_sample = data[sample_indices]
        else:
            data_sample = data
        
        # Check for valid data
        if data_sample.shape[0] < 2:
            return np.array([])
        
        # Remove rows with NaN or infinite values
        mask = np.all(np.isfinite(data_sample), axis=1)
        if np.sum(mask) < 2:
            return np.array([])
        
        data_clean = data_sample[mask]
        
        # Calculate distances
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            distances = pdist(data_clean)
        
        # Remove invalid distances
        distances = distances[np.isfinite(distances)]
        
        return distances
        
    except Exception:
        return np.array([])


def _safe_clustering_metrics(data: np.ndarray, labels: np.ndarray, metric_name: str) -> Optional[float]:
    """
    Safely calculate clustering metrics with proper error handling.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    labels : np.ndarray
        Cluster labels
    metric_name : str
        Name of the metric to calculate
    
    Returns:
    --------
    float or None
        Metric value or None if calculation fails
    """
    try:
        # Check for valid inputs
        if len(np.unique(labels)) < 2:
            return None
        
        if data.shape[0] != len(labels):
            return None
        
        # Remove samples with NaN or infinite values
        mask = np.all(np.isfinite(data), axis=1)
        if np.sum(mask) < 2:
            return None
        
        data_clean = data[mask]
        labels_clean = labels[mask]
        
        # Check if we still have multiple clusters
        if len(np.unique(labels_clean)) < 2:
            return None
        
        # Calculate metric
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if metric_name == 'silhouette':
                return silhouette_score(data_clean, labels_clean)
            elif metric_name == 'calinski_harabasz':
                return calinski_harabasz_score(data_clean, labels_clean)
            elif metric_name == 'davies_bouldin':
                return davies_bouldin_score(data_clean, labels_clean)
            else:
                return None
                
    except Exception:
        return None


# ================================
# CORE UMAP FUNCTIONALITY
# ================================

def apply_umap_reduction(data: pd.DataFrame, config_path: Optional[str] = None, 
                        **kwargs) -> Tuple[np.ndarray, umap.UMAP]:
    """
    Apply UMAP dimensionality reduction to the input data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data for dimensionality reduction
    config_path : str, optional
        Path to the config YAML file. If None, will use default config_cluster.yaml.
    **kwargs : dict
        Additional parameters to override config settings
    
    Returns:
    --------
    tuple
        - Reduced dimensional data as numpy array
        - Fitted UMAP model
    
    Notes:
    ------
    - Uses columns specified in config include_columns, or all numerical columns if not specified
    - NaN values are handled by dropping rows with NaN
    """
    
    # Load UMAP configuration
    umap_config = load_umap_config(config_path)
    
    # Override config with any provided kwargs
    umap_params = {**umap_config, **kwargs}
    
    # Remove include_columns from umap_params since it's not a UMAP parameter
    include_columns = umap_params.pop('include_columns', None)
    
    # Remove enabled from umap_params since it's not a UMAP parameter
    umap_params.pop('enabled', None)
    
    # Define valid UMAP parameters to filter out any config parameters that aren't UMAP parameters
    valid_umap_params = {
        'n_neighbors', 'n_components', 'metric', 'min_dist', 'spread', 'learning_rate',
        'n_epochs', 'init', 'random_state', 'verbose', 'low_memory', 'metric_kwds',
        'output_metric', 'output_metric_kwds', 'negative_sample_rate', 'transform_queue_size',
        'angular_rp_forest', 'set_op_mix_ratio', 'local_connectivity', 'repulsion_strength',
        'n_jobs', 'transform_seed'
    }
    
    # Filter umap_params to only include valid UMAP parameters
    umap_params = {k: v for k, v in umap_params.items() if k in valid_umap_params}
    
    # Select columns for UMAP
    if include_columns:
        # Use specified columns from config
        print(f"Using specified columns from config: {include_columns}")
        
        # Check if data has already been filtered (e.g., from preprocessing pipeline)
        if len(data.columns) == len(include_columns) and all(col in data.columns for col in include_columns):
            print("Data appears to already be filtered to include_columns. Using as-is.")
            numerical_data = data.select_dtypes(include=[np.number])
        else:
            # Validate that all specified columns exist in the data
            missing_columns = [col for col in include_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"The following columns specified in include_columns are missing from the data: {missing_columns}")
            
            # Select only the specified columns
            numerical_data = data[include_columns]
            
            # Validate that all selected columns are numerical
            non_numerical = numerical_data.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numerical:
                print(f"Warning: The following specified columns are not numerical and will be excluded: {non_numerical}")
                numerical_data = numerical_data.select_dtypes(include=[np.number])
    else:
        # Fallback to all numerical columns if include_columns not specified
        print("No include_columns specified in config. Using all numerical columns.")
        numerical_data = data.select_dtypes(include=[np.number])
    
    # Handle NaN values
    if numerical_data.isnull().any().any():
        print(f"Warning: Found NaN values. Dropping {numerical_data.isnull().any(axis=1).sum()} rows with NaN.")
        numerical_data = numerical_data.dropna()
    
    print(f"Applying UMAP with parameters: {umap_params}")
    print(f"Input data shape: {numerical_data.shape}")
    print(f"Selected columns: {list(numerical_data.columns)}")
    
    # Initialize UMAP with configuration parameters
    umap_model = umap.UMAP(**umap_params)
    
    # Fit and transform the data
    reduced_data = umap_model.fit_transform(numerical_data)
    
    print(f"Output data shape: {reduced_data.shape}")
    #print(f"Explained variance (approximation): {umap_model.explained_variance_ratio_[:2] if hasattr(umap_model, 'explained_variance_ratio_') else 'Not available'}")
    #ABOVE COMMENTED OUT AS UMAP DOES NOT PROVIDE EXPLAINED VARIANCE LIKE PCA
    
    return reduced_data, umap_model


def umap_with_preprocessing(data_path: Optional[str] = None, config_path: Optional[str] = None,
                           apply_log_transform: bool = True, apply_scaling: bool = True,
                           **umap_kwargs) -> Tuple[np.ndarray, umap.UMAP, pd.DataFrame, Dict]:
    """
    Complete pipeline: preprocessing + UMAP dimensionality reduction.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the data file. If None, will use the path from config file.
    config_path : str, optional
        Path to the config YAML file. If None, will use default config_cluster.yaml.
    apply_log_transform : bool, default True
        Whether to apply log transformation during preprocessing.
    apply_scaling : bool, default True
        Whether to apply scaling during preprocessing.
    **umap_kwargs : dict
        Additional UMAP parameters to override config settings
    
    Returns:
    --------
    tuple
        - Reduced dimensional data as numpy array
        - Fitted UMAP model
        - Preprocessed DataFrame
        - Preprocessing information dictionary
    """
    
    print("Starting complete preprocessing + UMAP pipeline...")
    
    # Step 0: Validate feature consistency
    print("\n=== CONFIGURATION VALIDATION ===")
    validation_results = validate_feature_consistency(config_path)
    
    print(f"UMAP include_columns: {validation_results['include_columns']}")
    print(f"Log transform excluded: {validation_results['log_transform_excluded']}")
    print(f"Scaling excluded: {validation_results['scaling_excluded']}")
    
    if validation_results['warnings']:
        print("⚠️  WARNINGS:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    
    print("📋 RECOMMENDATIONS:")
    for rec in validation_results['recommendations']:
        print(f"  - {rec}")
    
    # Step 1: Preprocess the data
    print("\n=== PREPROCESSING STAGE ===")
    preprocessed_data, preprocessing_info = preprocess_for_clustering(
        data_path=data_path,
        config_path=config_path,
        apply_log_transform=apply_log_transform,
        apply_scaling=apply_scaling
    )
    
    # Step 1.5: Filter preprocessed data to only include UMAP columns
    print("\n=== FILTERING PREPROCESSED DATA FOR UMAP ===")
    umap_config = load_umap_config(config_path)
    include_columns = umap_config.get('include_columns', None)
    
    if include_columns:
        print(f"Filtering preprocessed data to include only UMAP columns: {include_columns}")
        
        # Check which columns are actually available in preprocessed data
        available_columns = [col for col in include_columns if col in preprocessed_data.columns]
        missing_columns = [col for col in include_columns if col not in preprocessed_data.columns]
        
        if missing_columns:
            print(f"Warning: These UMAP columns are missing from preprocessed data: {missing_columns}")
        
        print(f"Using {len(available_columns)} out of {len(include_columns)} specified UMAP columns")
        preprocessed_data = preprocessed_data[available_columns]
        
    else:
        print("No include_columns specified, using all numerical columns from preprocessed data")
        preprocessed_data = preprocessed_data.select_dtypes(include=[np.number])
    
    print(f"Preprocessed data shape after filtering: {preprocessed_data.shape}")
    
    # Step 2: Apply UMAP
    print("\n=== UMAP DIMENSIONALITY REDUCTION STAGE ===")
    reduced_data, umap_model = apply_umap_reduction(
        data=preprocessed_data,
        config_path=config_path,
        **umap_kwargs
    )
    
    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"Original data shape: {preprocessed_data.shape}")
    print(f"Reduced data shape: {reduced_data.shape}")
    print(f"Preprocessing steps applied: {preprocessing_info['steps_applied']}")
    
    return reduced_data, umap_model, preprocessed_data, preprocessing_info


# ================================
# QUALITY EVALUATION FUNCTIONS
# ================================

def _evaluate_umap_quality_internal(original_data: pd.DataFrame, reduced_data: np.ndarray, 
                                   labels: Optional[np.ndarray] = None, 
                                   k_neighbors: int = 15,
                                   evaluate_clustering: bool = True,
                                   n_clusters_range: range = range(2, 11),
                                   verbose: bool = True) -> Dict[str, Any]:
    """
    Internal function that performs the actual quality evaluation.
    """
    
    if verbose:
        print("=== COMPREHENSIVE UMAP QUALITY EVALUATION ===")
    
    metrics = {}
    original_numerical = original_data.select_dtypes(include=[np.number])
    original_dims = original_numerical.shape[1]
    reduced_dims = reduced_data.shape[1]
    
    # ====== SECTION 1: BASIC METRICS ======
    if verbose:
        print("1. Calculating basic quality metrics...")
    
    basic_metrics = {}
    
    # Calculate dimension reduction ratio
    basic_metrics['dimension_reduction_ratio'] = reduced_dims / original_dims
    basic_metrics['original_dimensions'] = original_dims
    basic_metrics['reduced_dimensions'] = reduced_dims
    basic_metrics['data_points'] = reduced_data.shape[0]
    
    # Calculate silhouette score if labels are provided
    if labels is not None and len(np.unique(labels)) > 1:
        try:
            silhouette_avg = _safe_clustering_metrics(reduced_data, labels, 'silhouette')
            basic_metrics['silhouette_score'] = silhouette_avg
        except Exception as e:
            if verbose:
                print(f"Could not calculate silhouette score: {e}")
            basic_metrics['silhouette_score'] = None
    
    # Calculate variance preserved (approximate using PCA)
    try:
        # Ensure we have clean numerical data
        clean_data = original_numerical.dropna()
        if clean_data.shape[0] > 0 and clean_data.shape[1] > 0:
            pca = PCA(n_components=min(reduced_dims, clean_data.shape[1]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pca.fit(clean_data)
            basic_metrics['variance_preserved_estimate'] = np.sum(pca.explained_variance_ratio_[:reduced_dims])
        else:
            basic_metrics['variance_preserved_estimate'] = None
    except Exception as e:
        if verbose:
            print(f"Could not estimate variance preserved: {e}")
        basic_metrics['variance_preserved_estimate'] = None
    
    metrics['basic_metrics'] = basic_metrics
    
    # ====== SECTION 2: NEIGHBORHOOD PRESERVATION ======
    if verbose:
        print("2. Evaluating neighborhood preservation...")
    
    try:
        neighborhood_metrics = {}
        
        # Limit sample size for performance on large datasets
        sample_size = min(1000, original_numerical.shape[0])
        if sample_size < original_numerical.shape[0]:
            sample_indices = np.random.choice(original_numerical.shape[0], sample_size, replace=False)
            orig_sample = original_numerical.iloc[sample_indices].values
            red_sample = reduced_data[sample_indices]
        else:
            orig_sample = original_numerical.values
            red_sample = reduced_data
        
        # Find k-nearest neighbors in both spaces
        k_actual = min(k_neighbors, sample_size - 1)
        nbrs_original = NearestNeighbors(n_neighbors=k_actual+1).fit(orig_sample)
        nbrs_reduced = NearestNeighbors(n_neighbors=k_actual+1).fit(red_sample)
        
        # Get distances and indices
        distances_orig, indices_orig = nbrs_original.kneighbors(orig_sample)
        distances_red, indices_red = nbrs_reduced.kneighbors(red_sample)
        
        # Calculate trustworthiness and continuity
        trustworthiness_scores = []
        continuity_scores = []
        
        for i in range(sample_size):
            neighbors_orig = set(indices_orig[i][1:])  # Exclude self
            neighbors_red = set(indices_red[i][1:])    # Exclude self
            
            intersection = neighbors_orig.intersection(neighbors_red)
            score = len(intersection) / k_actual
            trustworthiness_scores.append(score)
            continuity_scores.append(score)
        
        neighborhood_metrics['trustworthiness'] = np.mean(trustworthiness_scores)
        neighborhood_metrics['continuity'] = np.mean(continuity_scores)
        neighborhood_metrics['trustworthiness_std'] = np.std(trustworthiness_scores)
        
        # Distance correlation
        orig_distances_flat = distances_orig[:, 1:].flatten()
        red_distances_flat = distances_red[:, 1:].flatten()
        
        correlation, p_value = _safe_correlation(orig_distances_flat, red_distances_flat, method='spearman')
        neighborhood_metrics['distance_correlation'] = correlation
        neighborhood_metrics['distance_correlation_pvalue'] = p_value
        
        metrics['neighborhood_preservation'] = neighborhood_metrics
        
    except Exception as e:
        if verbose:
            print(f"Warning: Could not calculate neighborhood preservation: {e}")
        metrics['neighborhood_preservation'] = None
    
    # ====== SECTION 3: CLUSTERING QUALITY ASSESSMENT ======
    if evaluate_clustering and verbose:
        print("3. Evaluating clustering quality...")
    
    if evaluate_clustering:
        try:
            cluster_metrics = {
                'optimal_clusters': {},
                'cluster_comparison': {}
            }
            
            # Limit sample size for performance
            sample_size = min(500, original_numerical.shape[0])
            if sample_size < original_numerical.shape[0]:
                sample_indices = np.random.choice(original_numerical.shape[0], sample_size, replace=False)
                orig_sample = original_numerical.iloc[sample_indices].values
                red_sample = reduced_data[sample_indices]
            else:
                orig_sample = original_numerical.values
                red_sample = reduced_data
            
            silhouette_scores_orig = []
            silhouette_scores_red = []
            calinski_scores_orig = []
            calinski_scores_red = []
            davies_bouldin_orig = []
            davies_bouldin_red = []
            
            for n_clusters in n_clusters_range:
                if n_clusters >= sample_size:
                    break
                    
                # Cluster both original and reduced data
                kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans_red = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                
                labels_orig = kmeans_orig.fit_predict(orig_sample)
                labels_red = kmeans_red.fit_predict(red_sample)
                
                # Calculate metrics
                try:
                    sil_orig = _safe_clustering_metrics(orig_sample, labels_orig, 'silhouette')
                    sil_red = _safe_clustering_metrics(red_sample, labels_red, 'silhouette')
                    cal_orig = _safe_clustering_metrics(orig_sample, labels_orig, 'calinski_harabasz')
                    cal_red = _safe_clustering_metrics(red_sample, labels_red, 'calinski_harabasz')
                    db_orig = _safe_clustering_metrics(orig_sample, labels_orig, 'davies_bouldin')
                    db_red = _safe_clustering_metrics(red_sample, labels_red, 'davies_bouldin')
                    
                    # Only append if metrics are valid
                    if sil_orig is not None and sil_red is not None:
                        silhouette_scores_orig.append(sil_orig)
                        silhouette_scores_red.append(sil_red)
                    if cal_orig is not None and cal_red is not None:
                        calinski_scores_orig.append(cal_orig)
                        calinski_scores_red.append(cal_red)
                    if db_orig is not None and db_red is not None:
                        davies_bouldin_orig.append(db_orig)
                        davies_bouldin_red.append(db_red)
                        
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not calculate clustering metrics for {n_clusters} clusters: {e}")
            
            # Find optimal number of clusters
            if silhouette_scores_orig and silhouette_scores_red:
                cluster_metrics['optimal_clusters']['silhouette_orig'] = list(n_clusters_range)[np.argmax(silhouette_scores_orig)]
                cluster_metrics['optimal_clusters']['silhouette_red'] = list(n_clusters_range)[np.argmax(silhouette_scores_red)]
                
                # Store comparison metrics
                cluster_metrics['cluster_comparison']['silhouette_improvement'] = np.mean(silhouette_scores_red) - np.mean(silhouette_scores_orig)
                cluster_metrics['cluster_comparison']['max_silhouette_orig'] = np.max(silhouette_scores_orig)
                cluster_metrics['cluster_comparison']['max_silhouette_red'] = np.max(silhouette_scores_red)
            
            metrics['cluster_quality'] = cluster_metrics
            
        except Exception as e:
            if verbose:
                print(f"Warning: Could not calculate cluster quality: {e}")
            metrics['cluster_quality'] = None
    
    # ====== SECTION 4: MANIFOLD QUALITY ASSESSMENT ======
    if verbose:
        print("4. Evaluating manifold quality...")
    
    try:
        manifold_metrics = {}
        
        # Embedding density and coverage
        embedding_distances = _safe_distance_calculation(reduced_data, max_samples=500)
        
        if len(embedding_distances) > 0:
            manifold_metrics['embedding_density'] = {
                'mean_distance': np.mean(embedding_distances),
                'std_distance': np.std(embedding_distances),
                'min_distance': np.min(embedding_distances),
                'max_distance': np.max(embedding_distances)
            }
        else:
            manifold_metrics['embedding_density'] = None
        
        # Embedding coverage (how much of the embedding space is used)
        embedding_range = np.ptp(reduced_data, axis=0)
        manifold_metrics['embedding_coverage'] = {
            'dimension_ranges': embedding_range.tolist(),
            'total_volume': np.prod(embedding_range),
            'mean_range': np.mean(embedding_range)
        }
        
        # Global distance preservation (sample for performance)
        sample_size = min(200, original_numerical.shape[0])
        if sample_size < original_numerical.shape[0]:
            sample_indices = np.random.choice(original_numerical.shape[0], sample_size, replace=False)
            orig_sample = original_numerical.iloc[sample_indices].values
            red_sample = reduced_data[sample_indices]
        else:
            orig_sample = original_numerical.values
            red_sample = reduced_data
        
        original_distances = _safe_distance_calculation(orig_sample, max_samples=200)
        reduced_distances = _safe_distance_calculation(red_sample, max_samples=200)
        
        if len(original_distances) > 0 and len(reduced_distances) > 0 and len(original_distances) == len(reduced_distances):
            global_correlation, p_val = _safe_correlation(original_distances, reduced_distances, method='pearson')
            manifold_metrics['global_distance_preservation'] = global_correlation
            manifold_metrics['global_distance_preservation_pvalue'] = p_val
        else:
            manifold_metrics['global_distance_preservation'] = None
            manifold_metrics['global_distance_preservation_pvalue'] = None
        
        metrics['manifold_quality'] = manifold_metrics
        
    except Exception as e:
        if verbose:
            print(f"Warning: Could not calculate manifold quality: {e}")
        metrics['manifold_quality'] = None
    
    # ====== SECTION 5: QUALITY SUMMARY ======
    if verbose:
        print("5. Generating quality summary...")
    
    metrics['quality_summary'] = _generate_quality_summary(metrics)
    
    if verbose:
        print("=== QUALITY EVALUATION COMPLETE ===")
        _print_quality_summary(metrics)
    
    return metrics


def evaluate_umap_quality(original_data: pd.DataFrame, reduced_data: np.ndarray, 
                         labels: Optional[np.ndarray] = None, 
                         k_neighbors: int = 15,
                         evaluate_clustering: bool = True,
                         n_clusters_range: range = range(2, 11),
                         verbose: bool = True,
                         suppress_warnings: bool = True) -> Dict[str, Any]:
    """
    Comprehensive evaluation of UMAP dimensionality reduction quality.
    
    Parameters:
    -----------
    original_data : pd.DataFrame
        Original high-dimensional data
    reduced_data : np.ndarray
        UMAP reduced data
    labels : np.ndarray, optional
        Cluster labels for silhouette score calculation
    k_neighbors : int, default 15
        Number of neighbors for neighborhood preservation metrics
    evaluate_clustering : bool, default True
        Whether to evaluate clustering quality metrics
    n_clusters_range : range, default range(2, 11)
        Range of cluster numbers to test for clustering evaluation
    verbose : bool, default True
        Whether to print progress information
    suppress_warnings : bool, default True
        Whether to suppress runtime warnings during evaluation
    
    Returns:
    --------
    dict
        Dictionary containing comprehensive quality metrics
    """
    
    # Set up warning suppression context
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FutureWarning)
            return _evaluate_umap_quality_internal(
                original_data, reduced_data, labels, k_neighbors, 
                evaluate_clustering, n_clusters_range, verbose
            )
    else:
        return _evaluate_umap_quality_internal(
            original_data, reduced_data, labels, k_neighbors, 
            evaluate_clustering, n_clusters_range, verbose
        )


# ================================
# QUALITY SUMMARY & INTERPRETATION
# ================================

def _generate_quality_summary(metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate a human-readable summary of UMAP quality.
    
    Parameters:
    -----------
    metrics : dict
        Comprehensive metrics dictionary
    
    Returns:
    --------
    dict
        Quality summary with interpretations
    """
    
    summary = {}
    
    # Neighborhood preservation interpretation
    if metrics.get('neighborhood_preservation'):
        trustworthiness = metrics['neighborhood_preservation'].get('trustworthiness', 0)
        if trustworthiness > 0.8:
            summary['neighborhood_preservation'] = "Excellent - Local structure well preserved"
        elif trustworthiness > 0.6:
            summary['neighborhood_preservation'] = "Good - Local structure mostly preserved"
        elif trustworthiness > 0.4:
            summary['neighborhood_preservation'] = "Fair - Some local structure preserved"
        else:
            summary['neighborhood_preservation'] = "Poor - Local structure poorly preserved"
    
    # Distance correlation interpretation
    if metrics.get('neighborhood_preservation'):
        dist_corr = metrics['neighborhood_preservation'].get('distance_correlation', 0)
        if dist_corr > 0.7:
            summary['distance_preservation'] = "Excellent - Distances well preserved"
        elif dist_corr > 0.5:
            summary['distance_preservation'] = "Good - Distances reasonably preserved"
        elif dist_corr > 0.3:
            summary['distance_preservation'] = "Fair - Some distance preservation"
        else:
            summary['distance_preservation'] = "Poor - Distances poorly preserved"
    
    # Silhouette score interpretation
    if metrics.get('basic_metrics', {}).get('silhouette_score'):
        sil_score = metrics['basic_metrics']['silhouette_score']
        if sil_score > 0.7:
            summary['cluster_separation'] = "Excellent - Well-separated clusters"
        elif sil_score > 0.5:
            summary['cluster_separation'] = "Good - Moderately separated clusters"
        elif sil_score > 0.3:
            summary['cluster_separation'] = "Fair - Some cluster separation"
        else:
            summary['cluster_separation'] = "Poor - Poorly separated clusters"
    
    # Variance preservation interpretation
    if metrics.get('basic_metrics', {}).get('variance_preserved_estimate'):
        var_preserved = metrics['basic_metrics']['variance_preserved_estimate']
        if var_preserved > 0.8:
            summary['variance_preservation'] = "Excellent - Most variance preserved"
        elif var_preserved > 0.6:
            summary['variance_preservation'] = "Good - Substantial variance preserved"
        elif var_preserved > 0.4:
            summary['variance_preservation'] = "Fair - Some variance preserved"
        else:
            summary['variance_preservation'] = "Poor - Low variance preserved"
    
    # Clustering quality interpretation
    if metrics.get('cluster_quality', {}).get('cluster_comparison', {}).get('silhouette_improvement'):
        sil_improvement = metrics['cluster_quality']['cluster_comparison']['silhouette_improvement']
        if sil_improvement > 0.1:
            summary['clustering_improvement'] = "Excellent - Clustering significantly improved"
        elif sil_improvement > 0.05:
            summary['clustering_improvement'] = "Good - Clustering moderately improved"
        elif sil_improvement > 0:
            summary['clustering_improvement'] = "Fair - Clustering slightly improved"
        else:
            summary['clustering_improvement'] = "Poor - Clustering not improved"
    
    return summary


def _print_quality_summary(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted quality summary to console.
    
    Parameters:
    -----------
    metrics : dict
        Comprehensive metrics dictionary
    """
    
    print("\n=== QUALITY SUMMARY ===")
    
    # Basic metrics
    if metrics.get('basic_metrics'):
        basic = metrics['basic_metrics']
        print(f"📊 Data Shape: {basic.get('data_points', 'N/A')} samples")
        print(f"📉 Dimension Reduction: {basic.get('original_dimensions', 'N/A')} → {basic.get('reduced_dimensions', 'N/A')} (ratio: {basic.get('dimension_reduction_ratio', 'N/A'):.3f})")
        
        if basic.get('variance_preserved_estimate'):
            print(f"📈 Variance Preserved: {basic['variance_preserved_estimate']:.3f} ({basic['variance_preserved_estimate']*100:.1f}%)")
        
        if basic.get('silhouette_score'):
            print(f"🎯 Silhouette Score: {basic['silhouette_score']:.3f}")
    
    # Neighborhood preservation
    if metrics.get('neighborhood_preservation'):
        neighbor = metrics['neighborhood_preservation']
        print(f"🏠 Trustworthiness: {neighbor.get('trustworthiness', 'N/A'):.3f}")
        print(f"🔄 Distance Correlation: {neighbor.get('distance_correlation', 'N/A'):.3f}")
    
    # Clustering quality
    if metrics.get('cluster_quality', {}).get('cluster_comparison'):
        cluster = metrics['cluster_quality']['cluster_comparison']
        print(f"🎭 Clustering Improvement: {cluster.get('silhouette_improvement', 'N/A'):.3f}")
    
    # Manifold quality
    if metrics.get('manifold_quality'):
        manifold = metrics['manifold_quality']
        if manifold.get('global_distance_preservation'):
            print(f"🌐 Global Distance Preservation: {manifold['global_distance_preservation']:.3f}")
    
    # Quality interpretations
    if metrics.get('quality_summary'):
        print("\n=== QUALITY INTERPRETATIONS ===")
        for aspect, interpretation in metrics['quality_summary'].items():
            print(f"• {aspect.replace('_', ' ').title()}: {interpretation}")
    
    print("=" * 50)


# ================================
# FILE I/O & UTILITIES
# ================================

def save_umap_results(reduced_data: np.ndarray, output_path: str, 
                     original_index: Optional[pd.Index] = None) -> None:
    """
    Save UMAP reduced data to CSV file.
    
    Parameters:
    -----------
    reduced_data : np.ndarray
        UMAP reduced dimensional data
    output_path : str
        Path to save the results
    original_index : pd.Index, optional
        Original DataFrame index to preserve row identification
    """
    
    # Create DataFrame with reduced dimensions
    n_components = reduced_data.shape[1]
    column_names = [f'UMAP_{i+1}' for i in range(n_components)]
    
    df_reduced = pd.DataFrame(reduced_data, columns=column_names, index=original_index)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df_reduced.to_csv(output_path, index=True)
    print(f"UMAP results saved to: {output_path}")


# ================================
# MAIN PIPELINE EXECUTION
# ================================

def run_umap_pipeline_example(config_path: Optional[str] = None):
    """
    Example function demonstrating the complete UMAP pipeline.
    This function orchestrates the entire workflow from configuration validation
    through to quality evaluation and result saving.
    """
    
    try:
        # Run the complete pipeline
        reduced_data, umap_model, preprocessed_data, preprocessing_info = umap_with_preprocessing(
            config_path=config_path
        )
        
        # Evaluate quality
        quality_metrics = evaluate_umap_quality(
            original_data=preprocessed_data, 
            reduced_data=reduced_data,
            verbose=True,
            suppress_warnings=True
        )
        
        print(f"\n=== QUALITY METRICS ===")
        for metric, value in quality_metrics.items():
            print(f"{metric}: {value}")
        
        # Save results (optional)
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '../../config/config_cluster.yaml')
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_config = config.get('output', {})
        if output_config.get('save_umap_results', False):
            output_path = output_config.get('umap_output_path', 'data/processed_data/umap_reduced_data.csv')
            save_umap_results(reduced_data, output_path, preprocessed_data.index)
        
        return reduced_data, umap_model, preprocessed_data, preprocessing_info
        
    except ImportError as e:
        print(f"Error: UMAP library not installed. Please install with: pip install umap-learn")
        print(f"Full error: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"Error in UMAP pipeline: {e}")
        return None, None, None, None


# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":
    # Run example pipeline when file is executed directly
    run_umap_pipeline_example()
