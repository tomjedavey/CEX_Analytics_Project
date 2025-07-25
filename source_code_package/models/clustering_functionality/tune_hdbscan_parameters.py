# HDBSCAN Parameter Tuning Functionality
# This module provides comprehensive parameter optimization for HDBSCAN clustering
# using UMAP dimensionality reduced data. It includes grid search, random search,
# and Bayesian optimization approaches with multiple evaluation metrics.

import pandas as pd
import numpy as np
import yaml
import os
import sys
import time
import json
import warnings
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List, Union
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning imports
import hdbscan
import umap
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
from scipy.stats import randint, uniform
from scipy.optimize import minimize

# Configure warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

try:
    from data.preprocess_cluster import preprocess_for_clustering
    from models.clustering_functionality.UMAP_dim_reduction import apply_umap_reduction
    from models.clustering_functionality.HBDSCAN_cluster import (
        apply_hdbscan_clustering, evaluate_clustering_quality, load_hdbscan_config
    )
except ImportError as e:
    print(f"Warning: Could not import some dependencies: {e}")
    print("Ensure all required modules are available in the source_code_package")


# ================================
# PARAMETER SPACE DEFINITIONS
# ================================

class ParameterSpace:
    """Defines parameter spaces for HDBSCAN and UMAP optimization."""
    
    @staticmethod
    def get_hdbscan_parameter_space(search_type: str = "default") -> Dict[str, List]:
        """
        Get HDBSCAN parameter space for optimization.
        
        Parameters:
        -----------
        search_type : str
            Type of search space: "default", "comprehensive", "quick", "fine_tune"
            
        Returns:
        --------
        dict
            Dictionary containing parameter ranges for HDBSCAN
        """
        if search_type == "quick":
            return {
                'min_cluster_size': [50, 100, 150],
                'min_samples': [10, 25, 40],
                'metric': ['euclidean', 'cosine'],
                'cluster_selection_method': ['eom']
            }
        elif search_type == "comprehensive":
            return {
                'min_cluster_size': [25, 50, 75, 100, 150, 200, 300],
                'min_samples': [5, 10, 15, 25, 30, 40, 50],
                'metric': ['euclidean', 'manhattan', 'cosine'],
                'cluster_selection_method': ['eom', 'leaf'],
                'cluster_selection_epsilon': [0.0, 0.01, 0.05]
            }
        elif search_type == "fine_tune":
            return {
                'min_cluster_size': [80, 100, 120, 140, 160, 180, 200],
                'min_samples': [20, 25, 30, 35, 40, 45, 50],
                'metric': ['euclidean'],
                'cluster_selection_method': ['eom']
            }
        else:  # default
            return {
                'min_cluster_size': [50, 75, 100, 150, 200],
                'min_samples': [10, 20, 30, 40],
                'metric': ['euclidean', 'cosine'],
                'cluster_selection_method': ['eom']
            }
    
    @staticmethod
    def get_umap_parameter_space(search_type: str = "default") -> Dict[str, List]:
        """
        Get UMAP parameter space for optimization.
        
        Parameters:
        -----------
        search_type : str
            Type of search space: "default", "comprehensive", "quick", "fine_tune"
            
        Returns:
        --------
        dict
            Dictionary containing parameter ranges for UMAP
        """
        if search_type == "quick":
            return {
                'n_components': [10, 15],
                'n_neighbors': [15, 30],
                'min_dist': [0.1],
                'metric': ['euclidean']
            }
        elif search_type == "comprehensive":
            return {
                'n_components': [5, 10, 12, 15, 20],
                'n_neighbors': [5, 10, 15, 30, 50],
                'min_dist': [0.01, 0.05, 0.1, 0.5],
                'metric': ['euclidean', 'cosine', 'manhattan']
            }
        elif search_type == "fine_tune":
            return {
                'n_components': [10, 12, 15, 18],
                'n_neighbors': [15, 20, 25, 30],
                'min_dist': [0.05, 0.1, 0.15],
                'metric': ['euclidean']
            }
        else:  # default
            return {
                'n_components': [10, 15, 20],
                'n_neighbors': [15, 30, 50],
                'min_dist': [0.01, 0.1],
                'metric': ['euclidean', 'cosine']
            }


# ================================
# SCORING AND EVALUATION
# ================================

class ClusteringEvaluator:
    """Comprehensive clustering evaluation with multiple metrics."""
    
    @staticmethod
    def calculate_composite_score(
        silhouette: float,
        calinski_harabasz: float,
        davies_bouldin: float,
        n_clusters: int,
        noise_percentage: float,
        stability_ratio: float = None,
        weights: Dict[str, float] = None
    ) -> float:
        """
        Calculate composite score for clustering quality.
        
        Parameters:
        -----------
        silhouette : float
            Silhouette score (-1 to 1, higher better)
        calinski_harabasz : float
            Calinski-Harabasz score (higher better)
        davies_bouldin : float
            Davies-Bouldin score (lower better)
        n_clusters : int
            Number of clusters found
        noise_percentage : float
            Percentage of points classified as noise
        stability_ratio : float, optional
            Cluster stability metric
        weights : dict, optional
            Custom weights for different metrics
            
        Returns:
        --------
        float
            Composite score (higher is better)
        """
        if weights is None:
            weights = {
                'silhouette': 40.0,
                'calinski_harabasz': 0.002,  # Scale down the large values
                'davies_bouldin': -10.0,     # Negative because lower is better
                'cluster_penalty': -2.0,     # Penalty for too many/few clusters
                'noise_penalty': -1.0,       # Penalty for high noise
                'stability_bonus': 10.0      # Bonus for stability
            }
        
        score = 0.0
        
        # Silhouette contribution (40% of max score)
        score += weights['silhouette'] * max(0, silhouette)
        
        # Calinski-Harabasz contribution (scaled)
        score += weights['calinski_harabasz'] * min(calinski_harabasz, 50000)
        
        # Davies-Bouldin contribution (negative, so lower DB = higher score)
        score += weights['davies_bouldin'] * davies_bouldin
        
        # Cluster number penalty (prefer 3-10 clusters)
        if n_clusters < 3:
            score += weights['cluster_penalty'] * (3 - n_clusters) * 5
        elif n_clusters > 15:
            score += weights['cluster_penalty'] * (n_clusters - 15)
        
        # Noise penalty (prefer < 5% noise)
        if noise_percentage > 5.0:
            score += weights['noise_penalty'] * (noise_percentage - 5.0)
        
        # Stability bonus
        if stability_ratio is not None:
            score += weights['stability_bonus'] * stability_ratio
        
        return max(0, score)  # Ensure non-negative score
    
    @staticmethod
    def evaluate_single_configuration(
        data: np.ndarray,
        umap_params: Dict[str, Any],
        hdbscan_params: Dict[str, Any],
        config_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single parameter configuration.
        
        Parameters:
        -----------
        data : np.ndarray
            Preprocessed data for clustering
        umap_params : dict
            UMAP parameters to test
        hdbscan_params : dict
            HDBSCAN parameters to test
        config_path : str, optional
            Path to configuration file
            
        Returns:
        --------
        dict
            Evaluation results including metrics and timing
        """
        start_time = time.time()
        
        try:
            # Apply UMAP
            umap_start = time.time()
            
            # Create UMAP reducer with specified parameters
            reducer = umap.UMAP(
                n_components=umap_params.get('n_components', 15),
                n_neighbors=umap_params.get('n_neighbors', 30),
                min_dist=umap_params.get('min_dist', 0.1),
                metric=umap_params.get('metric', 'euclidean'),
                random_state=42,
                verbose=False
            )
            
            umap_data = reducer.fit_transform(data)
            umap_time = time.time() - umap_start
            
            # Apply HDBSCAN
            hdbscan_start = time.time()
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=hdbscan_params.get('min_cluster_size', 50),
                min_samples=hdbscan_params.get('min_samples', 10),
                metric=hdbscan_params.get('metric', 'euclidean'),
                cluster_selection_method=hdbscan_params.get('cluster_selection_method', 'eom'),
                cluster_selection_epsilon=hdbscan_params.get('cluster_selection_epsilon', 0.0),
                prediction_data=True
            )
            
            cluster_labels = clusterer.fit_predict(umap_data)
            hdbscan_time = time.time() - hdbscan_start
            
            # Evaluate clustering quality
            eval_start = time.time()
            evaluation_metrics = evaluate_clustering_quality(umap_data, cluster_labels, clusterer)
            eval_time = time.time() - eval_start
            
            # Calculate statistics
            n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
            n_noise_points = np.sum(cluster_labels == -1)
            noise_percentage = (n_noise_points / len(cluster_labels)) * 100
            
            # Calculate stability ratio if available
            stability_ratio = None
            if hasattr(clusterer, 'cluster_persistence_') and clusterer.cluster_persistence_ is not None:
                if len(clusterer.cluster_persistence_) > 0:
                    stability_ratio = np.mean(clusterer.cluster_persistence_)
            
            # Calculate composite score
            composite_score = ClusteringEvaluator.calculate_composite_score(
                silhouette=evaluation_metrics.get('silhouette_score', 0),
                calinski_harabasz=evaluation_metrics.get('calinski_harabasz_score', 0),
                davies_bouldin=evaluation_metrics.get('davies_bouldin_score', 10),
                n_clusters=n_clusters,
                noise_percentage=noise_percentage,
                stability_ratio=stability_ratio
            )
            
            total_time = time.time() - start_time
            
            return {
                'umap_params': umap_params,
                'hdbscan_params': hdbscan_params,
                'n_clusters': n_clusters,
                'n_noise_points': n_noise_points,
                'noise_percentage': noise_percentage,
                'n_data_points': len(cluster_labels),
                'umap_dimensions': umap_params.get('n_components', 15),
                'original_dimensions': data.shape[1],
                'silhouette_score': evaluation_metrics.get('silhouette_score', 0),
                'calinski_harabasz_score': evaluation_metrics.get('calinski_harabasz_score', 0),
                'davies_bouldin_score': evaluation_metrics.get('davies_bouldin_score', 10),
                'mean_cluster_persistence': stability_ratio,
                'min_cluster_persistence': (
                    np.min(clusterer.cluster_persistence_) 
                    if hasattr(clusterer, 'cluster_persistence_') and clusterer.cluster_persistence_ is not None
                    else None
                ),
                'stability_ratio': stability_ratio,
                'composite_score': composite_score,
                'umap_time': umap_time,
                'hdbscan_time': hdbscan_time,
                'evaluation_time': eval_time,
                'total_time': total_time,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'umap_params': umap_params,
                'hdbscan_params': hdbscan_params,
                'success': False,
                'error': str(e),
                'composite_score': 0,
                'total_time': time.time() - start_time
            }


# ================================
# OPTIMIZATION ALGORITHMS
# ================================

class HDBSCANParameterOptimizer:
    """Main class for HDBSCAN parameter optimization."""
    
    def __init__(self, data_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the optimizer.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to data file
        config_path : str, optional
            Path to configuration file
        """
        self.data_path = data_path
        self.config_path = config_path
        self.data = None
        self.results = []
        
    def load_and_preprocess_data(self) -> np.ndarray:
        """
        Load and preprocess data for clustering.
        
        Returns:
        --------
        np.ndarray
            Preprocessed data ready for clustering
        """
        if self.data is not None:
            return self.data
            
        print("Loading and preprocessing data...")
        
        try:
            # Load config to get data path if not provided
            if self.config_path is None:
                self.config_path = os.path.join(
                    os.path.dirname(__file__), '../../config/config_cluster.yaml'
                )
            
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if self.data_path is None:
                self.data_path = config.get('data', {}).get('raw_data_path')
                if self.data_path and not os.path.isabs(self.data_path):
                    # Convert relative path to absolute
                    self.data_path = os.path.join(
                        os.path.dirname(self.config_path), '../../', self.data_path
                    )
            
            # Preprocess data
            preprocessed_data, _ = preprocess_for_clustering(
                data_path=self.data_path,
                config_path=self.config_path
            )
            
            # Apply column selection based on UMAP config
            umap_config = config.get('umap', {})
            include_columns = umap_config.get('include_columns', None)
            include_all_columns = umap_config.get('include_all_columns', False)
            
            if include_all_columns:
                # Select only numeric columns
                self.data = preprocessed_data.select_dtypes(include=[np.number]).values
            elif include_columns:
                # Filter to specified columns
                available_columns = [col for col in include_columns if col in preprocessed_data.columns]
                if available_columns:
                    self.data = preprocessed_data[available_columns].values
                else:
                    print("Warning: No specified columns found, using all numeric columns")
                    self.data = preprocessed_data.select_dtypes(include=[np.number]).values
            else:
                # Use all numeric columns by default
                self.data = preprocessed_data.select_dtypes(include=[np.number]).values
            
            print(f"Data loaded: {self.data.shape[0]} samples, {self.data.shape[1]} features")
            return self.data
            
        except Exception as e:
            raise ValueError(f"Error loading and preprocessing data: {e}")
    
    def grid_search_optimization(
        self,
        search_type: str = "default",
        max_workers: int = None,
        save_results: bool = True,
        output_dir: str = "tuning_results"
    ) -> List[Dict[str, Any]]:
        """
        Perform grid search optimization of HDBSCAN parameters.
        
        Parameters:
        -----------
        search_type : str
            Type of search: "quick", "default", "comprehensive", "fine_tune"
        max_workers : int, optional
            Number of parallel workers
        save_results : bool
            Whether to save results to file
        output_dir : str
            Directory to save results
            
        Returns:
        --------
        list
            List of evaluation results sorted by composite score
        """
        print(f"Starting grid search optimization (search_type: {search_type})")
        
        # Load data
        data = self.load_and_preprocess_data()
        
        # Get parameter spaces
        umap_space = ParameterSpace.get_umap_parameter_space(search_type)
        hdbscan_space = ParameterSpace.get_hdbscan_parameter_space(search_type)
        
        # Generate all parameter combinations
        umap_combinations = list(ParameterGrid(umap_space))
        hdbscan_combinations = list(ParameterGrid(hdbscan_space))
        
        total_combinations = len(umap_combinations) * len(hdbscan_combinations)
        print(f"Testing {total_combinations} parameter combinations...")
        
        # Create all parameter pairs
        param_pairs = []
        for umap_params in umap_combinations:
            for hdbscan_params in hdbscan_combinations:
                param_pairs.append((data, umap_params, hdbscan_params, self.config_path))
        
        # Run evaluations
        results = []
        if max_workers and max_workers > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_params = {
                    executor.submit(
                        ClusteringEvaluator.evaluate_single_configuration,
                        *params
                    ): params for params in param_pairs
                }
                
                for i, future in enumerate(as_completed(future_to_params)):
                    result = future.result()
                    results.append(result)
                    if (i + 1) % 10 == 0 or (i + 1) == total_combinations:
                        print(f"Completed {i + 1}/{total_combinations} evaluations")
        else:
            # Sequential execution
            for i, params in enumerate(param_pairs):
                result = ClusteringEvaluator.evaluate_single_configuration(*params)
                results.append(result)
                if (i + 1) % 10 == 0 or (i + 1) == total_combinations:
                    print(f"Completed {i + 1}/{total_combinations} evaluations")
        
        # Filter successful results and sort by composite score
        successful_results = [r for r in results if r.get('success', False)]
        successful_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"Completed optimization: {len(successful_results)}/{total_combinations} successful evaluations")
        
        self.results = successful_results
        
        if save_results:
            self._save_results(successful_results, search_type, output_dir)
        
        return successful_results
    
    def random_search_optimization(
        self,
        n_trials: int = 50,
        max_workers: int = None,
        save_results: bool = True,
        output_dir: str = "tuning_results"
    ) -> List[Dict[str, Any]]:
        """
        Perform random search optimization of HDBSCAN parameters.
        
        Parameters:
        -----------
        n_trials : int
            Number of random trials to perform
        max_workers : int, optional
            Number of parallel workers
        save_results : bool
            Whether to save results to file
        output_dir : str
            Directory to save results
            
        Returns:
        --------
        list
            List of evaluation results sorted by composite score
        """
        print(f"Starting random search optimization ({n_trials} trials)")
        
        # Load data
        data = self.load_and_preprocess_data()
        
        # Generate random parameter combinations
        param_pairs = []
        np.random.seed(42)  # For reproducibility
        
        for _ in range(n_trials):
            # Random UMAP parameters
            umap_params = {
                'n_components': np.random.choice([5, 10, 12, 15, 20, 25]),
                'n_neighbors': np.random.choice([5, 10, 15, 20, 30, 50]),
                'min_dist': np.random.choice([0.01, 0.05, 0.1, 0.2, 0.5]),
                'metric': np.random.choice(['euclidean', 'cosine', 'manhattan'])
            }
            
            # Random HDBSCAN parameters
            hdbscan_params = {
                'min_cluster_size': np.random.choice([25, 50, 75, 100, 150, 200, 300]),
                'min_samples': np.random.choice([5, 10, 15, 20, 25, 30, 40, 50]),
                'metric': np.random.choice(['euclidean', 'cosine', 'manhattan']),
                'cluster_selection_method': np.random.choice(['eom', 'leaf']),
                'cluster_selection_epsilon': np.random.choice([0.0, 0.01, 0.05])
            }
            
            param_pairs.append((data, umap_params, hdbscan_params, self.config_path))
        
        # Run evaluations
        results = []
        if max_workers and max_workers > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_params = {
                    executor.submit(
                        ClusteringEvaluator.evaluate_single_configuration,
                        *params
                    ): params for params in param_pairs
                }
                
                for i, future in enumerate(as_completed(future_to_params)):
                    result = future.result()
                    results.append(result)
                    if (i + 1) % 10 == 0 or (i + 1) == n_trials:
                        print(f"Completed {i + 1}/{n_trials} evaluations")
        else:
            # Sequential execution
            for i, params in enumerate(param_pairs):
                result = ClusteringEvaluator.evaluate_single_configuration(*params)
                results.append(result)
                if (i + 1) % 10 == 0 or (i + 1) == n_trials:
                    print(f"Completed {i + 1}/{n_trials} evaluations")
        
        # Filter successful results and sort by composite score
        successful_results = [r for r in results if r.get('success', False)]
        successful_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"Completed random search: {len(successful_results)}/{n_trials} successful evaluations")
        
        self.results = successful_results
        
        if save_results:
            self._save_results(successful_results, "random_search", output_dir)
        
        return successful_results
    
    def bayesian_optimization(
        self,
        n_initial: int = 10,
        n_iterations: int = 40,
        save_results: bool = True,
        output_dir: str = "tuning_results"
    ) -> List[Dict[str, Any]]:
        """
        Perform Bayesian optimization of HDBSCAN parameters.
        
        Parameters:
        -----------
        n_initial : int
            Number of initial random evaluations
        n_iterations : int
            Number of optimization iterations
        save_results : bool
            Whether to save results to file
        output_dir : str
            Directory to save results
            
        Returns:
        --------
        list
            List of evaluation results sorted by composite score
        """
        print(f"Starting Bayesian optimization ({n_initial} initial + {n_iterations} iterations)")
        
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("Bayesian optimization requires scikit-learn with Gaussian processes")
            print("Falling back to random search...")
            return self.random_search_optimization(
                n_trials=n_initial + n_iterations,
                save_results=save_results,
                output_dir=output_dir
            )
        
        # Load data
        data = self.load_and_preprocess_data()
        
        # Parameter bounds for optimization
        param_bounds = [
            (5, 25),     # n_components
            (5, 50),     # n_neighbors
            (0.01, 0.5), # min_dist
            (25, 300),   # min_cluster_size
            (5, 50),     # min_samples
        ]
        
        # Discrete choices for categorical parameters
        metric_choices = ['euclidean', 'cosine']
        cluster_method_choices = ['eom']
        
        # Storage for evaluations
        X_evaluated = []
        y_evaluated = []
        all_results = []
        
        # Initial random evaluations
        print("Performing initial random evaluations...")
        for i in range(n_initial):
            # Random parameters
            params = [
                np.random.randint(param_bounds[0][0], param_bounds[0][1] + 1),  # n_components
                np.random.randint(param_bounds[1][0], param_bounds[1][1] + 1),  # n_neighbors
                np.random.uniform(param_bounds[2][0], param_bounds[2][1]),      # min_dist
                np.random.randint(param_bounds[3][0], param_bounds[3][1] + 1),  # min_cluster_size
                np.random.randint(param_bounds[4][0], param_bounds[4][1] + 1),  # min_samples
            ]
            
            umap_params = {
                'n_components': int(params[0]),
                'n_neighbors': int(params[1]),
                'min_dist': params[2],
                'metric': np.random.choice(metric_choices)
            }
            
            hdbscan_params = {
                'min_cluster_size': int(params[3]),
                'min_samples': int(params[4]),
                'metric': np.random.choice(metric_choices),
                'cluster_selection_method': np.random.choice(cluster_method_choices)
            }
            
            result = ClusteringEvaluator.evaluate_single_configuration(
                data, umap_params, hdbscan_params, self.config_path
            )
            
            if result.get('success', False):
                X_evaluated.append(params)
                y_evaluated.append(result['composite_score'])
                all_results.append(result)
            
            print(f"Initial evaluation {i + 1}/{n_initial} completed")
        
        # Bayesian optimization iterations
        print("Starting Bayesian optimization iterations...")
        
        scaler = StandardScaler()
        gp = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0, nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        
        for iteration in range(n_iterations):
            if len(X_evaluated) < 2:
                # Fall back to random search if not enough points
                params = [
                    np.random.randint(param_bounds[j][0], param_bounds[j][1] + 1) 
                    if j in [0, 1, 3, 4] else np.random.uniform(param_bounds[j][0], param_bounds[j][1])
                    for j in range(5)
                ]
            else:
                # Fit Gaussian process
                X_scaled = scaler.fit_transform(X_evaluated)
                gp.fit(X_scaled, y_evaluated)
                
                # Acquisition function (Expected Improvement)
                def acquisition(x):
                    x_scaled = scaler.transform([x])
                    mu, sigma = gp.predict(x_scaled, return_std=True)
                    best_y = max(y_evaluated)
                    improvement = mu - best_y
                    if sigma > 0:
                        z = improvement / sigma
                        ei = improvement * (0.5 * (1 + np.tanh(z / np.sqrt(2)))) + sigma * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
                        return -ei[0]  # Negative because minimize
                    else:
                        return 0
                
                # Optimize acquisition function
                best_params = None
                best_acquisition = float('inf')
                
                for _ in range(100):  # Random restarts
                    x0 = [
                        np.random.randint(param_bounds[j][0], param_bounds[j][1] + 1) 
                        if j in [0, 1, 3, 4] else np.random.uniform(param_bounds[j][0], param_bounds[j][1])
                        for j in range(5)
                    ]
                    
                    try:
                        result_opt = minimize(
                            acquisition,
                            x0,
                            bounds=param_bounds,
                            method='L-BFGS-B'
                        )
                        
                        if result_opt.fun < best_acquisition:
                            best_acquisition = result_opt.fun
                            best_params = result_opt.x
                    except:
                        continue
                
                params = best_params if best_params is not None else x0
            
            # Evaluate the suggested parameters
            umap_params = {
                'n_components': int(round(params[0])),
                'n_neighbors': int(round(params[1])),
                'min_dist': params[2],
                'metric': np.random.choice(metric_choices)
            }
            
            hdbscan_params = {
                'min_cluster_size': int(round(params[3])),
                'min_samples': int(round(params[4])),
                'metric': np.random.choice(metric_choices),
                'cluster_selection_method': np.random.choice(cluster_method_choices)
            }
            
            result = ClusteringEvaluator.evaluate_single_configuration(
                data, umap_params, hdbscan_params, self.config_path
            )
            
            if result.get('success', False):
                X_evaluated.append(list(params))
                y_evaluated.append(result['composite_score'])
                all_results.append(result)
            
            print(f"Bayesian iteration {iteration + 1}/{n_iterations} completed")
        
        # Sort results by composite score
        all_results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"Completed Bayesian optimization: {len(all_results)} successful evaluations")
        
        self.results = all_results
        
        if save_results:
            self._save_results(all_results, "bayesian_optimization", output_dir)
        
        return all_results
    
    def _save_results(
        self,
        results: List[Dict[str, Any]],
        method_name: str,
        output_dir: str
    ) -> None:
        """
        Save optimization results to files.
        
        Parameters:
        -----------
        results : list
            List of evaluation results
        method_name : str
            Name of optimization method
        output_dir : str
            Directory to save results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        json_file = os.path.join(output_dir, f"detailed_results_{method_name}_{timestamp}.json")
        with open(json_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = []
            for result in results:
                json_result = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        json_result[key] = value.tolist()
                    elif isinstance(value, (np.int64, np.int32)):
                        json_result[key] = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        json_result[key] = float(value)
                    else:
                        json_result[key] = value
                json_results.append(json_result)
            
            json.dump(json_results, f, indent=2)
        
        # Save markdown report
        md_file = os.path.join(output_dir, f"optimization_report_{method_name}_{timestamp}.md")
        self._generate_markdown_report(results, method_name, md_file)
        
        # Create visualization
        plot_file = os.path.join(output_dir, f"optimization_visualization_{method_name}_{timestamp}.png")
        self._create_optimization_plots(results, plot_file)
        
        print(f"Results saved to:")
        print(f"  - {json_file}")
        print(f"  - {md_file}")
        print(f"  - {plot_file}")
    
    def _generate_markdown_report(
        self,
        results: List[Dict[str, Any]],
        method_name: str,
        file_path: str
    ) -> None:
        """Generate markdown report of optimization results."""
        with open(file_path, 'w') as f:
            f.write(f"# HDBSCAN Parameter Optimization Report ({method_name})\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total combinations tested: {len(results)}\n")
            f.write(f"Data shape: ({self.data.shape[0]}, {self.data.shape[1]})\n\n")
            
            # Top 10 results
            f.write("## Top 10 Results\n\n")
            for i, result in enumerate(results[:10]):
                f.write(f"### Rank {i + 1}\n")
                f.write(f"**Composite Score:** {result['composite_score']:.2f}\n\n")
                
                f.write("**UMAP Parameters:**\n")
                for key, value in result['umap_params'].items():
                    f.write(f"- {key}: {value}\n")
                
                f.write("\n**HDBSCAN Parameters:**\n")
                for key, value in result['hdbscan_params'].items():
                    f.write(f"- {key}: {value}\n")
                
                f.write(f"\n**Results:**\n")
                f.write(f"- Clusters: {result['n_clusters']}\n")
                f.write(f"- Noise percentage: {result['noise_percentage']:.1f}%\n")
                f.write(f"- Silhouette score: {result['silhouette_score']:.3f}\n")
                f.write(f"- Total time: {result['total_time']:.1f}s\n\n")
            
            # Best configuration
            best = results[0]
            f.write("## Recommended Configuration\n\n")
            f.write("```yaml\n")
            f.write("umap:\n")
            for key, value in best['umap_params'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\nhdbscan:\n")
            for key, value in best['hdbscan_params'].items():
                f.write(f"  {key}: {value}\n")
            f.write("```\n\n")
            
            # Statistics
            f.write("## Optimization Statistics\n\n")
            scores = [r['composite_score'] for r in results]
            f.write(f"- Best score: {max(scores):.2f}\n")
            f.write(f"- Average score: {np.mean(scores):.2f}\n")
            f.write(f"- Score std: {np.std(scores):.2f}\n")
            
            total_times = [r['total_time'] for r in results]
            f.write(f"- Average evaluation time: {np.mean(total_times):.2f}s\n")
            f.write(f"- Total optimization time: {sum(total_times):.1f}s\n")
    
    def _create_optimization_plots(self, results: List[Dict[str, Any]], file_path: str) -> None:
        """Create visualization plots for optimization results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('HDBSCAN Parameter Optimization Results', fontsize=16)
        
        # Extract data for plotting
        scores = [r['composite_score'] for r in results]
        silhouette_scores = [r['silhouette_score'] for r in results]
        n_clusters_list = [r['n_clusters'] for r in results]
        noise_percentages = [r['noise_percentage'] for r in results]
        min_cluster_sizes = [r['hdbscan_params']['min_cluster_size'] for r in results]
        min_samples_list = [r['hdbscan_params']['min_samples'] for r in results]
        
        # Plot 1: Score distribution
        axes[0, 0].hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Composite Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Composite Scores')
        axes[0, 0].axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.2f}')
        axes[0, 0].legend()
        
        # Plot 2: Silhouette vs Composite Score
        axes[0, 1].scatter(silhouette_scores, scores, alpha=0.6, color='green')
        axes[0, 1].set_xlabel('Silhouette Score')
        axes[0, 1].set_ylabel('Composite Score')
        axes[0, 1].set_title('Silhouette Score vs Composite Score')
        
        # Plot 3: Number of clusters vs Score
        axes[0, 2].scatter(n_clusters_list, scores, alpha=0.6, color='orange')
        axes[0, 2].set_xlabel('Number of Clusters')
        axes[0, 2].set_ylabel('Composite Score')
        axes[0, 2].set_title('Number of Clusters vs Composite Score')
        
        # Plot 4: Noise percentage vs Score
        axes[1, 0].scatter(noise_percentages, scores, alpha=0.6, color='purple')
        axes[1, 0].set_xlabel('Noise Percentage (%)')
        axes[1, 0].set_ylabel('Composite Score')
        axes[1, 0].set_title('Noise Percentage vs Composite Score')
        
        # Plot 5: Parameter space exploration
        scatter = axes[1, 1].scatter(min_cluster_sizes, min_samples_list, c=scores, 
                                   cmap='viridis', alpha=0.7, s=50)
        axes[1, 1].set_xlabel('Min Cluster Size')
        axes[1, 1].set_ylabel('Min Samples')
        axes[1, 1].set_title('Parameter Space (colored by score)')
        plt.colorbar(scatter, ax=axes[1, 1], label='Composite Score')
        
        # Plot 6: Top 10 results
        top_10_scores = scores[:10]
        top_10_labels = [f"Config {i+1}" for i in range(10)]
        axes[1, 2].bar(range(10), top_10_scores, color='lightcoral')
        axes[1, 2].set_xlabel('Configuration Rank')
        axes[1, 2].set_ylabel('Composite Score')
        axes[1, 2].set_title('Top 10 Configurations')
        axes[1, 2].set_xticks(range(10))
        axes[1, 2].set_xticklabels([f"{i+1}" for i in range(10)])
        
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_best_parameters(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get the best parameters from optimization results.
        
        Returns:
        --------
        tuple
            (best_umap_params, best_hdbscan_params)
        """
        if not self.results:
            raise ValueError("No optimization results available. Run optimization first.")
        
        best_result = self.results[0]
        return best_result['umap_params'], best_result['hdbscan_params']
    
    def update_config_file(self, config_path: Optional[str] = None) -> None:
        """
        Update configuration file with best parameters.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file to update
        """
        if not self.results:
            raise ValueError("No optimization results available. Run optimization first.")
        
        if config_path is None:
            config_path = self.config_path
        
        best_umap_params, best_hdbscan_params = self.get_best_parameters()
        
        # Load current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update UMAP parameters
        if 'umap' not in config:
            config['umap'] = {}
        
        for key, value in best_umap_params.items():
            config['umap'][key] = value
        
        # Update HDBSCAN parameters
        if 'hdbscan' not in config:
            config['hdbscan'] = {}
        
        for key, value in best_hdbscan_params.items():
            config['hdbscan'][key] = value
        
        # Save updated config
        backup_path = config_path + '.backup'
        os.rename(config_path, backup_path)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"Configuration updated with best parameters")
        print(f"Backup saved to: {backup_path}")


# ================================
# CONVENIENCE FUNCTIONS
# ================================

def optimize_hdbscan_parameters(
    data_path: Optional[str] = None,
    config_path: Optional[str] = None,
    method: str = "grid_search",
    search_type: str = "default",
    n_trials: int = 50,
    max_workers: int = None,
    save_results: bool = True,
    output_dir: str = "tuning_results",
    update_config: bool = False
) -> List[Dict[str, Any]]:
    """
    Convenient function to optimize HDBSCAN parameters.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to data file
    config_path : str, optional
        Path to configuration file
    method : str
        Optimization method: "grid_search", "random_search", "bayesian"
    search_type : str
        Search space type: "quick", "default", "comprehensive", "fine_tune"
    n_trials : int
        Number of trials for random/bayesian search
    max_workers : int, optional
        Number of parallel workers
    save_results : bool
        Whether to save results
    output_dir : str
        Output directory
    update_config : bool
        Whether to update config file with best parameters
        
    Returns:
    --------
    list
        Optimization results sorted by score
    """
    optimizer = HDBSCANParameterOptimizer(data_path, config_path)
    
    if method == "grid_search":
        results = optimizer.grid_search_optimization(
            search_type=search_type,
            max_workers=max_workers,
            save_results=save_results,
            output_dir=output_dir
        )
    elif method == "random_search":
        results = optimizer.random_search_optimization(
            n_trials=n_trials,
            max_workers=max_workers,
            save_results=save_results,
            output_dir=output_dir
        )
    elif method == "bayesian":
        results = optimizer.bayesian_optimization(
            n_initial=min(10, n_trials // 4),
            n_iterations=n_trials - min(10, n_trials // 4),
            save_results=save_results,
            output_dir=output_dir
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    if update_config and results:
        optimizer.update_config_file()
    
    return results


# ================================
# MAIN EXECUTION
# ================================

def main():
    """
    Main execution function for testing and demonstration.
    """
    print("HDBSCAN Parameter Optimization Module")
    print("=" * 50)
    
    # Example usage
    try:
        # Quick grid search example
        print("Running quick grid search optimization...")
        results = optimize_hdbscan_parameters(
            method="grid_search",
            search_type="quick",
            max_workers=2,
            save_results=True,
            output_dir="tuning_results"
        )
        
        if results:
            print(f"\nOptimization completed! Best composite score: {results[0]['composite_score']:.2f}")
            print("Best UMAP parameters:")
            for key, value in results[0]['umap_params'].items():
                print(f"  {key}: {value}")
            print("Best HDBSCAN parameters:")
            for key, value in results[0]['hdbscan_params'].items():
                print(f"  {key}: {value}")
        else:
            print("No successful optimization results")
            
    except Exception as e:
        print(f"Error during optimization: {e}")
        print("This is expected if dependencies are not available or data is not found")


if __name__ == "__main__":
    main()
