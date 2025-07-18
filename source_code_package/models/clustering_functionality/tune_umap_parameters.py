#!/usr/bin/env python3
"""
Parameter tuning script for UMAP dimensionality reduction.

This script tests different parameter combinations to find optimal settings for UMAP dimensionality reduction.
It should be run from the project root directory:

Usage:
    cd /path/to/MLProject1
    python3 source_code_package/models/clustering_functionality/tune_umap_parameters.py

The script will:
1. Load preprocessed data from the preprocessing pipeline
2. Test various UMAP parameter combinations
3. Evaluate dimensionality reduction quality using comprehensive metrics
4. Provide recommendations for optimal parameters

Author: Tom Davey
Date: July 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import time
from typing import Dict, Any, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output during parameter testing
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add source_code_package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from models.clustering_functionality.UMAP_dim_reduction import (
    apply_umap_reduction, 
    evaluate_umap_quality, 
    umap_with_preprocessing
)
from data.preprocess_cluster import preprocess_for_clustering


def test_umap_parameters(data: pd.DataFrame, parameter_combinations: List[Dict[str, Any]], 
                        config_path: Optional[str] = None, 
                        evaluate_clustering: bool = True,
                        max_evaluation_time: int = 300) -> List[Dict[str, Any]]:
    """
    Test different UMAP parameter combinations and evaluate their quality.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Preprocessed input data for UMAP
    parameter_combinations : List[Dict[str, Any]]
        List of parameter dictionaries to test
    config_path : str, optional
        Path to config file
    evaluate_clustering : bool, default True
        Whether to evaluate clustering quality (can be time-consuming)
    max_evaluation_time : int, default 300
        Maximum time in seconds for quality evaluation per parameter set
    
    Returns:
    --------
    List[Dict[str, Any]]
        Results for each parameter combination tested
    """
    
    results = []
    
    print(f"Testing {len(parameter_combinations)} UMAP parameter combinations...")
    print(f"Input data shape: {data.shape}")
    
    for i, params in enumerate(parameter_combinations):
        print(f"\nTesting combination {i+1}/{len(parameter_combinations)}: {params}")
        
        start_time = time.time()
        
        try:
            # Apply UMAP with test parameters
            reduced_data, umap_model = apply_umap_reduction(
                data=data, 
                config_path=config_path,
                **params
            )
            
            fit_time = time.time() - start_time
            
            # Basic metrics
            n_components = reduced_data.shape[1]
            original_dims = data.select_dtypes(include=[np.number]).shape[1]
            reduction_ratio = n_components / original_dims
            
            print(f"  UMAP completed in {fit_time:.2f}s")
            print(f"  Reduced from {original_dims} to {n_components} dimensions (ratio: {reduction_ratio:.3f})")
            
            # Quality evaluation with timeout
            quality_metrics = None
            evaluation_time = 0
            
            if evaluate_clustering:
                print(f"  Starting quality evaluation (max {max_evaluation_time}s)...")
                eval_start = time.time()
                
                try:
                    # Run quality evaluation with reduced clustering evaluation for speed
                    quality_metrics = evaluate_umap_quality(
                        original_data=data,
                        reduced_data=reduced_data,
                        labels=None,
                        k_neighbors=min(15, data.shape[0] - 1),  # Adjust for small datasets
                        evaluate_clustering=True,
                        n_clusters_range=range(2, min(8, data.shape[0] // 2)),  # Limited range for speed
                        verbose=False,
                        suppress_warnings=True
                    )
                    evaluation_time = time.time() - eval_start
                    
                    if evaluation_time > max_evaluation_time:
                        print(f"  Quality evaluation took {evaluation_time:.2f}s (over limit)")
                    else:
                        print(f"  Quality evaluation completed in {evaluation_time:.2f}s")
                        
                except Exception as e:
                    print(f"  Quality evaluation failed: {str(e)}")
                    quality_metrics = None
                    evaluation_time = time.time() - eval_start
            
            # Extract key metrics for comparison
            result = {
                'parameters': params.copy(),
                'n_components': n_components,
                'original_dimensions': original_dims,
                'reduction_ratio': reduction_ratio,
                'fit_time_seconds': fit_time,
                'evaluation_time_seconds': evaluation_time,
                'data_points': reduced_data.shape[0]
            }
            
            # Add quality metrics if available
            if quality_metrics:
                # Basic metrics
                basic_metrics = quality_metrics.get('basic_metrics', {})
                result['variance_preserved'] = basic_metrics.get('variance_preserved_estimate', None)
                result['silhouette_score'] = basic_metrics.get('silhouette_score', None)
                
                # Neighborhood preservation
                neighborhood = quality_metrics.get('neighborhood_preservation', {})
                result['trustworthiness'] = neighborhood.get('trustworthiness', None)
                result['distance_correlation'] = neighborhood.get('distance_correlation', None)
                
                # Clustering quality
                cluster_quality = quality_metrics.get('cluster_quality', {})
                if cluster_quality and cluster_quality.get('cluster_comparison'):
                    result['clustering_improvement'] = cluster_quality['cluster_comparison'].get('silhouette_improvement', None)
                    result['max_silhouette_reduced'] = cluster_quality['cluster_comparison'].get('max_silhouette_red', None)
                else:
                    result['clustering_improvement'] = None
                    result['max_silhouette_reduced'] = None
                
                # Manifold quality
                manifold = quality_metrics.get('manifold_quality', {})
                result['global_distance_preservation'] = manifold.get('global_distance_preservation', None) if manifold else None
                
                # Quality summary
                quality_summary = quality_metrics.get('quality_summary', {})
                result['quality_summary'] = quality_summary
                
                # Calculate composite score for ranking
                scores = []
                if result['trustworthiness'] is not None:
                    scores.append(result['trustworthiness'])
                if result['distance_correlation'] is not None:
                    scores.append(result['distance_correlation'])
                if result['variance_preserved'] is not None:
                    scores.append(result['variance_preserved'])
                
                result['composite_quality_score'] = np.mean(scores) if scores else None
                
            else:
                # Set quality metrics to None if evaluation failed
                for key in ['variance_preserved', 'silhouette_score', 'trustworthiness', 
                           'distance_correlation', 'clustering_improvement', 'max_silhouette_reduced',
                           'global_distance_preservation', 'quality_summary', 'composite_quality_score']:
                    result[key] = None
            
            results.append(result)
            
            # Print key results
            if result['trustworthiness'] is not None:
                print(f"  Key metrics: Trustworthiness={result['trustworthiness']:.3f}, "
                      f"Distance Corr={result['distance_correlation']:.3f}")
                if result['variance_preserved'] is not None:
                    print(f"  Variance preserved: {result['variance_preserved']:.3f} ({result['variance_preserved']*100:.1f}%)")
            else:
                print(f"  Quality evaluation: Skipped or failed")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            error_result = {
                'parameters': params.copy(),
                'error': str(e),
                'n_components': None,
                'original_dimensions': data.select_dtypes(include=[np.number]).shape[1],
                'reduction_ratio': None,
                'fit_time_seconds': time.time() - start_time,
                'evaluation_time_seconds': 0,
                'data_points': data.shape[0]
            }
            
            # Set all quality metrics to None for failed runs
            for key in ['variance_preserved', 'silhouette_score', 'trustworthiness', 
                       'distance_correlation', 'clustering_improvement', 'max_silhouette_reduced',
                       'global_distance_preservation', 'quality_summary', 'composite_quality_score']:
                error_result[key] = None
            
            results.append(error_result)
    
    return results


def get_default_parameter_combinations() -> List[Dict[str, Any]]:
    """
    Get default parameter combinations to test for UMAP tuning.
    
    Returns:
    --------
    List[Dict[str, Any]]
        List of parameter combinations covering different UMAP use cases
    """
    
    parameter_combinations = [
        # ===== STANDARD COMBINATIONS =====
        # Conservative settings - good for preserving global structure
        {'n_neighbors': 15, 'n_components': 2, 'min_dist': 0.1, 'metric': 'euclidean'},
        {'n_neighbors': 15, 'n_components': 5, 'min_dist': 0.1, 'metric': 'euclidean'},
        {'n_neighbors': 15, 'n_components': 10, 'min_dist': 0.1, 'metric': 'euclidean'},
        
        # ===== VARYING N_NEIGHBORS =====
        # Lower n_neighbors - more focus on local structure
        {'n_neighbors': 5, 'n_components': 2, 'min_dist': 0.0, 'metric': 'euclidean'},
        {'n_neighbors': 10, 'n_components': 2, 'min_dist': 0.0, 'metric': 'euclidean'},
        
        # Higher n_neighbors - more focus on global structure
        {'n_neighbors': 30, 'n_components': 2, 'min_dist': 0.0, 'metric': 'euclidean'},
        {'n_neighbors': 50, 'n_components': 2, 'min_dist': 0.0, 'metric': 'euclidean'},
        
        # ===== VARYING MIN_DIST =====
        # Very tight clusters
        {'n_neighbors': 15, 'n_components': 2, 'min_dist': 0.0, 'metric': 'euclidean'},
        # More spread out
        {'n_neighbors': 15, 'n_components': 2, 'min_dist': 0.25, 'metric': 'euclidean'},
        {'n_neighbors': 15, 'n_components': 2, 'min_dist': 0.5, 'metric': 'euclidean'},
        
        # ===== DIFFERENT METRICS =====
        # Cosine distance - good for high-dimensional data
        {'n_neighbors': 15, 'n_components': 2, 'min_dist': 0.0, 'metric': 'cosine'},
        {'n_neighbors': 15, 'n_components': 5, 'min_dist': 0.0, 'metric': 'cosine'},
        {'n_neighbors': 15, 'n_components': 10, 'min_dist': 0.0, 'metric': 'cosine'},
        
        # Manhattan distance
        {'n_neighbors': 15, 'n_components': 2, 'min_dist': 0.0, 'metric': 'manhattan'},
        {'n_neighbors': 15, 'n_components': 5, 'min_dist': 0.0, 'metric': 'manhattan'},
        
        # Correlation distance - good for finding correlated features
        {'n_neighbors': 15, 'n_components': 2, 'min_dist': 0.0, 'metric': 'correlation'},
        
        # ===== DIMENSION VARIATIONS =====
        # Different output dimensions with optimal settings
        {'n_neighbors': 20, 'n_components': 3, 'min_dist': 0.0, 'metric': 'cosine'},
        {'n_neighbors': 20, 'n_components': 8, 'min_dist': 0.0, 'metric': 'cosine'},
        {'n_neighbors': 20, 'n_components': 15, 'min_dist': 0.0, 'metric': 'cosine'},
        
        # ===== SPECIALIZED COMBINATIONS =====
        # For clustering downstream
        {'n_neighbors': 20, 'n_components': 10, 'min_dist': 0.0, 'metric': 'cosine', 'spread': 1.0},
        {'n_neighbors': 25, 'n_components': 10, 'min_dist': 0.0, 'metric': 'euclidean', 'spread': 0.8},
        
        # For visualization (2D)
        {'n_neighbors': 15, 'n_components': 2, 'min_dist': 0.1, 'metric': 'cosine', 'spread': 1.2},
        {'n_neighbors': 10, 'n_components': 2, 'min_dist': 0.05, 'metric': 'euclidean', 'spread': 1.0},
    ]
    
    return parameter_combinations


def display_results(results: List[Dict[str, Any]]) -> None:
    """
    Display formatted results from parameter testing.
    
    Parameters:
    -----------
    results : List[Dict[str, Any]]
        Results from parameter testing
    """
    
    print("\n" + "=" * 120)
    print("UMAP PARAMETER TUNING RESULTS")
    print("=" * 120)
    
    # Filter out failed results for main display
    successful_results = [r for r in results if 'error' not in r and r['n_components'] is not None]
    failed_results = [r for r in results if 'error' in r or r['n_components'] is None]
    
    if successful_results:
        # Main results table
        print(f"{'N_Neighbors':<12} {'N_Comp':<8} {'Min_Dist':<10} {'Metric':<12} "
              f"{'Fit_Time':<10} {'Trust':<8} {'Dist_Corr':<10} {'Var_Pres':<10} {'Composite':<10}")
        print("-" * 120)
        
        for result in successful_results:
            params = result['parameters']
            n_neighbors = params.get('n_neighbors', 'N/A')
            n_components = params.get('n_components', 'N/A')
            min_dist = params.get('min_dist', 'N/A')
            metric = params.get('metric', 'N/A')
            
            fit_time = f"{result['fit_time_seconds']:.1f}s" if result['fit_time_seconds'] else 'N/A'
            trust = f"{result['trustworthiness']:.3f}" if result['trustworthiness'] is not None else 'N/A'
            dist_corr = f"{result['distance_correlation']:.3f}" if result['distance_correlation'] is not None else 'N/A'
            var_pres = f"{result['variance_preserved']:.3f}" if result['variance_preserved'] is not None else 'N/A'
            composite = f"{result['composite_quality_score']:.3f}" if result['composite_quality_score'] is not None else 'N/A'
            
            print(f"{n_neighbors:<12} {n_components:<8} {min_dist:<10} {metric:<12} "
                  f"{fit_time:<10} {trust:<8} {dist_corr:<10} {var_pres:<10} {composite:<10}")
    
    # Show failed results if any
    if failed_results:
        print(f"\n{len(failed_results)} parameter combinations failed:")
        for result in failed_results:
            params = result['parameters']
            error_msg = result.get('error', 'Unknown error')
            print(f"  {params}: {error_msg}")


def generate_recommendations(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate parameter recommendations based on testing results.
    
    Parameters:
    -----------
    results : List[Dict[str, Any]]
        Results from parameter testing
    
    Returns:
    --------
    Dict[str, Any]
        Recommendations for different use cases
    """
    
    # Filter successful results
    successful_results = [r for r in results if 'error' not in r and r['n_components'] is not None]
    
    if not successful_results:
        return {"error": "No successful parameter combinations found"}
    
    # Results with quality metrics
    quality_results = [r for r in successful_results if r['composite_quality_score'] is not None]
    
    recommendations = {}
    
    if quality_results:
        # Best overall quality
        best_quality = max(quality_results, key=lambda x: x['composite_quality_score'])
        recommendations['best_overall_quality'] = {
            'parameters': best_quality['parameters'],
            'composite_score': best_quality['composite_quality_score'],
            'trustworthiness': best_quality['trustworthiness'],
            'distance_correlation': best_quality['distance_correlation'],
            'variance_preserved': best_quality['variance_preserved'],
            'reason': 'Highest composite quality score'
        }
        
        # Best for clustering (prioritize trustworthiness and higher dimensions)
        clustering_candidates = [r for r in quality_results if 
                               r['parameters'].get('n_components', 0) >= 5 and
                               r['trustworthiness'] is not None]
        
        if clustering_candidates:
            best_clustering = max(clustering_candidates, key=lambda x: x['trustworthiness'])
            recommendations['best_for_clustering'] = {
                'parameters': best_clustering['parameters'],
                'trustworthiness': best_clustering['trustworthiness'],
                'distance_correlation': best_clustering['distance_correlation'],
                'n_components': best_clustering['n_components'],
                'reason': 'High trustworthiness with sufficient dimensions for clustering'
            }
        
        # Best for visualization (2D with good structure preservation)
        viz_candidates = [r for r in quality_results if 
                         r['parameters'].get('n_components', 0) == 2 and
                         r['trustworthiness'] is not None]
        
        if viz_candidates:
            best_viz = max(viz_candidates, key=lambda x: x['trustworthiness'])
            recommendations['best_for_visualization'] = {
                'parameters': best_viz['parameters'],
                'trustworthiness': best_viz['trustworthiness'],
                'distance_correlation': best_viz['distance_correlation'],
                'reason': '2D projection with best local structure preservation'
            }
        
        # Best balanced (good quality with reasonable performance)
        balanced_candidates = [r for r in quality_results if 
                             r['fit_time_seconds'] is not None and
                             r['fit_time_seconds'] < 60 and  # Under 1 minute
                             r['composite_quality_score'] > 0.5]  # Decent quality
        
        if balanced_candidates:
            best_balanced = max(balanced_candidates, key=lambda x: x['composite_quality_score'])
            recommendations['best_balanced'] = {
                'parameters': best_balanced['parameters'],
                'composite_score': best_balanced['composite_quality_score'],
                'fit_time': best_balanced['fit_time_seconds'],
                'reason': 'Good quality with reasonable computation time'
            }
    
    # Fastest completion (regardless of quality)
    fastest = min(successful_results, key=lambda x: x['fit_time_seconds'] if x['fit_time_seconds'] else float('inf'))
    recommendations['fastest'] = {
        'parameters': fastest['parameters'],
        'fit_time': fastest['fit_time_seconds'],
        'n_components': fastest['n_components'],
        'reason': 'Fastest computation time'
    }
    
    return recommendations


def print_recommendations(recommendations: Dict[str, Any]) -> None:
    """
    Print formatted recommendations.
    
    Parameters:
    -----------
    recommendations : Dict[str, Any]
        Recommendations dictionary
    """
    
    if "error" in recommendations:
        print(f"\nError: {recommendations['error']}")
        return
    
    print("\n" + "=" * 80)
    print("PARAMETER RECOMMENDATIONS")
    print("=" * 80)
    
    for use_case, rec in recommendations.items():
        print(f"\n{use_case.replace('_', ' ').title()}:")
        print(f"  Parameters: {rec['parameters']}")
        print(f"  Reason: {rec['reason']}")
        
        # Print relevant metrics
        if 'composite_score' in rec:
            print(f"  Composite Quality Score: {rec['composite_score']:.3f}")
        if 'trustworthiness' in rec:
            print(f"  Trustworthiness: {rec['trustworthiness']:.3f}")
        if 'distance_correlation' in rec:
            print(f"  Distance Correlation: {rec['distance_correlation']:.3f}")
        if 'variance_preserved' in rec:
            print(f"  Variance Preserved: {rec['variance_preserved']:.3f}")
        if 'fit_time' in rec:
            print(f"  Fit Time: {rec['fit_time']:.1f}s")
    
    # Generate config file update suggestion
    if 'best_overall_quality' in recommendations:
        best_params = recommendations['best_overall_quality']['parameters']
        print(f"\n" + "=" * 80)
        print("RECOMMENDED CONFIG UPDATE:")
        print("=" * 80)
        print("umap:")
        for param, value in best_params.items():
            if isinstance(value, str):
                print(f"  {param}: \"{value}\"")
            else:
                print(f"  {param}: {value}")
        print("  # ... other parameters unchanged")


def main():
    """Main function to run UMAP parameter tuning."""
    
    print("UMAP Parameter Tuning")
    print("=" * 50)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '../../config/config_cluster.yaml')
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    
    print(f"Using config: {config_path}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    try:
        preprocessed_data, preprocessing_info = preprocess_for_clustering(
            config_path=config_path,
            apply_log_transform=True,
            apply_scaling=True
        )
        print(f"Preprocessed data shape: {preprocessed_data.shape}")
        print(f"Preprocessing steps: {preprocessing_info['steps_applied']}")
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return
    
    # Get parameter combinations to test
    parameter_combinations = get_default_parameter_combinations()
    
    # Option to test a subset for quick evaluation
    print(f"\nFound {len(parameter_combinations)} parameter combinations to test.")
    
    # Test parameters
    results = test_umap_parameters(
        data=preprocessed_data,
        parameter_combinations=parameter_combinations,
        config_path=config_path,
        evaluate_clustering=True,  # Set to False for faster testing
        max_evaluation_time=300    # 5 minutes max per evaluation
    )
    
    # Display results
    display_results(results)
    
    # Generate and print recommendations
    recommendations = generate_recommendations(results)
    print_recommendations(recommendations)
    
    # Save results to file
    output_dir = os.path.join(os.path.dirname(__file__), '../../../clustering_output')
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, 'umap_parameter_tuning_results.yaml')
    
    try:
        with open(results_path, 'w') as f:
            yaml.dump({
                'parameter_testing_results': results,
                'recommendations': recommendations,
                'test_info': {
                    'data_shape': preprocessed_data.shape,
                    'preprocessing_steps': preprocessing_info['steps_applied'],
                    'timestamp': pd.Timestamp.now().isoformat()
                }
            }, f, default_flow_style=False)
        
        print(f"\nResults saved to: {results_path}")
        
    except Exception as e:
        print(f"Warning: Could not save results to file: {str(e)}")


if __name__ == "__main__":
    main()
