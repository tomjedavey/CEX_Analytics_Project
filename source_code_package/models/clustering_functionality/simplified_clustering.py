#!/usr/bin/env python3
"""
Simplified HDBSCAN Clustering Pipeline

This module provides a streamlined, configuration-driven approach to HDBSCAN clustering
with optional UMAP dimensionality reduction. It replaces the complex object-oriented
wrapper layer with a simple, functional approach while maintaining all benefits.

Author: Tom Davey
Date: July 2025
"""

import pandas as pd
import numpy as np
import yaml
import os
from typing import Optional, Dict, Any
import sys

# Add the parent directory to path to import from source_code_package
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from source_code_package.models.clustering_functionality.HBDSCAN_cluster import (
    load_hdbscan_config,
    validate_hdbscan_config,
    hdbscan_clustering_pipeline
)


def run_clustering_pipeline(config_path: Optional[str] = None,
                          data_path: Optional[str] = None, 
                                                      output_dir: str = "data/processed_data/HDBSCAN_clustering_results_activity",
                          force_umap: Optional[bool] = None,
                          nest_hdbscan_results: bool = True) -> Dict[str, Any]:
    """
    Unified clustering pipeline that automatically determines whether to use UMAP
    based on configuration, with optional override.
    
    This single function replaces both run_flexible_hdbscan_pipeline and 
    run_umap_hdbscan_pipeline by making the choice purely configuration-driven.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to configuration file. If None, uses default.
    data_path : str, optional
        Path to data file. If None, uses path from config.
    output_dir : str, default="clustering_results"
        Directory to save all results.
    force_umap : bool, optional
        If provided, overrides the config UMAP setting.
        True: Force UMAP usage
        False: Force no UMAP (direct HDBSCAN)
        None: Use config setting
        
    Returns:
    --------
    dict
        Standardized results with consistent structure:
        {
            'success': bool,
            'umap_enabled': bool,
            'pipeline_info': {
                'n_original_features': int,
                'n_reduced_features': int, 
                'n_clusters_found': int,
                'total_data_points': int,
                'noise_points': int,
                'umap_applied': bool
            },
            'umap_results': dict,
            'hdbscan_results': dict,
            'file_paths': dict,
            'error_message': str (if success=False)
        }
        
    Example:
    --------
    >>> # Auto-detect from config
    >>> results = run_clustering_pipeline(
    ...     config_path="config/config_cluster.yaml",
    ...     output_dir="results"
    ... )
    >>> 
    >>> # Force UMAP usage
    >>> results = run_clustering_pipeline(
    ...     config_path="config/config_cluster.yaml", 
    ...     force_umap=True
    ... )
    """
    try:
        print("🚀 Starting unified clustering pipeline...")
        
        # Step 1: Load and validate configuration
        print("📋 Loading configuration...")
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 
                '../../config/config_cluster.yaml'
            )
        
        config = load_hdbscan_config(config_path)
        validation = validate_hdbscan_config(config)
        
        if not validation['valid']:
            error_msg = f"Configuration validation failed: {validation['errors']}"
            return _create_error_result(error_msg)
        
        # Step 2: Determine if UMAP should be used
        if force_umap is not None:
            use_umap = force_umap
            print(f"🔀 UMAP usage: {'Enabled' if use_umap else 'Disabled'} (forced)")
        else:
            use_umap = config.get('umap', {}).get('enabled', True)
            print(f"📊 UMAP usage: {'Enabled' if use_umap else 'Disabled'} (from config)")
        
        # Step 3: Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Step 4: Run the appropriate pipeline
        if use_umap:
            print("🔄 Running UMAP + HDBSCAN pipeline...")
            return _run_umap_hdbscan_pipeline(config_path, data_path, output_dir, nest_hdbscan_results)
        else:
            print("🎯 Running direct HDBSCAN pipeline...")
            return _run_direct_hdbscan_pipeline(config_path, data_path, output_dir, nest_hdbscan_results)
            
    except Exception as e:
        error_msg = f"Pipeline execution failed: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return _create_error_result(error_msg)


def _run_umap_hdbscan_pipeline(config_path: str, 
                              data_path: Optional[str],
                                                         output_dir: str,
                              nest_hdbscan_results: bool = True) -> Dict[str, Any]:
    """Run UMAP + HDBSCAN pipeline. Full wrapper + execution pipeline functionality."""
    try:
        # Import UMAP functionality
        try:
            from .UMAP_dim_reduction import umap_with_preprocessing
        except ImportError:
            from UMAP_dim_reduction import umap_with_preprocessing

        # Step 1: UMAP dimensionality reduction
        print("  📉 Applying UMAP dimensionality reduction...")
        umap_output_dir = os.path.join(output_dir, "umap_results")
        reduced_data, umap_model, preprocessed_data, preprocessing_info = umap_with_preprocessing(
            data_path=data_path,
            config_path=config_path,
            save_results=True,
            output_dir=umap_output_dir
        )

        # Step 2: HDBSCAN clustering on reduced data
        print("  🎯 Applying HDBSCAN clustering...")
        hdbscan_output_dir = output_dir
        hdbscan_results = hdbscan_clustering_pipeline(
            umap_data=reduced_data,
            config_path=config_path,
            evaluate_quality=True,
            create_visualizations=True,
            save_results=True,
            output_dir=hdbscan_output_dir
        )

        # Step 3: Combine results in standardized format
        return _create_standardized_result(
            umap_enabled=True,
            umap_results={
                'reduced_data': reduced_data,
                'umap_model': umap_model,
                'preprocessing_info': preprocessing_info,
                'n_original_features': preprocessed_data.shape[1] if preprocessed_data is not None else 0,
                'n_reduced_features': reduced_data.shape[1]
            },
            hdbscan_results=hdbscan_results,
            file_paths={
                'umap_results_dir': umap_output_dir,
                'hdbscan_results_dir': hdbscan_output_dir,
                'main_output_dir': output_dir
            }
        )

    except Exception as e:
        raise Exception(f"UMAP + HDBSCAN pipeline failed: {str(e)}")


def _run_direct_hdbscan_pipeline(config_path: str,
                                data_path: Optional[str], 
                                                             output_dir: str,
                                nest_hdbscan_results: bool = True) -> Dict[str, Any]:
    """Run direct HDBSCAN pipeline without UMAP. Full wrapper + execution pipeline functionality without UMAP."""
    try:
        # Import preprocessing functionality  
        try:
            from data.preprocess_cluster import preprocess_for_clustering
        except ImportError:
            sys.path.append(os.path.join(os.path.dirname(__file__), '../../data'))
            from preprocess_cluster import preprocess_for_clustering

        # Step 1: Load and preprocess data
        print("  📊 Loading and preprocessing data...")

        # Use the preprocessing function with config_path
        preprocessed_data, preprocessing_info = preprocess_for_clustering(
            data_path=data_path,
            config_path=config_path,
            apply_log_transform=True,
            apply_scaling=True
        )

        # Step 1.5: Apply column selection if specified (similar to UMAP logic)
        config = load_hdbscan_config(config_path)
        include_columns = config.get('umap', {}).get('include_columns', [])

        if include_columns:
            print(f"  🔍 Selecting specified columns: {include_columns}")
            # Validate that all specified columns exist
            missing_cols = [col for col in include_columns if col not in preprocessed_data.columns]
            if missing_cols:
                print(f"⚠️  WARNING: Missing columns: {missing_cols}")
                available_cols = [col for col in include_columns if col in preprocessed_data.columns]
                if available_cols:
                    include_columns = available_cols
                    print(f"  ✅ Using available columns: {include_columns}")
                else:
                    print("  ❌ No valid columns found, using all preprocessed data")
                    include_columns = []

            if include_columns:
                preprocessed_data = preprocessed_data[include_columns]
                print(f"  📊 Data shape after column selection: {preprocessed_data.shape}")
        else:
            print("  📊 Using all preprocessed columns")

        n_original_features = preprocessing_info.get('original_shape', (0, 0))[1]

        # Step 2: HDBSCAN clustering on preprocessed data
        print("  🎯 Applying HDBSCAN clustering...")
        hdbscan_output_dir = output_dir
        hdbscan_results = hdbscan_clustering_pipeline(
            umap_data=preprocessed_data,  # Use preprocessed data directly
            config_path=config_path,
            evaluate_quality=True,
            create_visualizations=True,
            save_results=True,
            output_dir=hdbscan_output_dir
        )

        # Step 3: Create standardized results
        return _create_standardized_result(
            umap_enabled=False,
            umap_results={
                'preprocessing_info': preprocessing_info,
                'preprocessed_data': preprocessed_data,
                'n_original_features': n_original_features,
                'n_reduced_features': preprocessed_data.shape[1]
            },
            hdbscan_results=hdbscan_results,
            file_paths={
                'hdbscan_results_dir': hdbscan_output_dir,
                'main_output_dir': output_dir
            }
        )

    except Exception as e:
        raise Exception(f"Direct HDBSCAN pipeline failed: {str(e)}")


def _create_standardized_result(umap_enabled: bool,
                              umap_results: Dict[str, Any],
                              hdbscan_results: Dict[str, Any],
                              file_paths: Dict[str, str]) -> Dict[str, Any]:
    """Create standardized result dictionary."""
    
    # Extract cluster information
    cluster_info = hdbscan_results.get('cluster_info', {})
    n_clusters = cluster_info.get('n_clusters', 0)
    
    # Extract data size information
    if 'reduced_data' in umap_results:
        total_points = umap_results['reduced_data'].shape[0]
    elif 'preprocessed_data' in umap_results:
        total_points = umap_results['preprocessed_data'].shape[0] 
    else:
        total_points = 0
    
    # Calculate noise points
    labels = hdbscan_results.get('cluster_labels', [])
    noise_points = len([l for l in labels if l == -1]) if labels is not None else 0
    
    return {
        'success': True,
        'umap_enabled': umap_enabled,
        'pipeline_info': {
            'n_original_features': umap_results.get('n_original_features', 0),
            'n_reduced_features': umap_results.get('n_reduced_features', 0),
            'n_clusters_found': n_clusters,
            'total_data_points': total_points,
            'noise_points': noise_points,
            'umap_applied': umap_enabled
        },
        'umap_results': umap_results,
        'hdbscan_results': hdbscan_results,
        'file_paths': file_paths
    }


def _create_error_result(error_message: str) -> Dict[str, Any]:
    """Create standardized error result."""
    return {
        'success': False,
        'umap_enabled': False,
        'pipeline_info': {
            'n_original_features': 0,
            'n_reduced_features': 0,
            'n_clusters_found': 0,
            'total_data_points': 0,
            'noise_points': 0,
            'umap_applied': False
        },
        'umap_results': {},
        'hdbscan_results': {},
        'file_paths': {},
        'error_message': error_message
    }


# Convenience functions for backward compatibility with existing scripts
def run_flexible_hdbscan_pipeline(data_path: Optional[str] = None,
                                config_path: Optional[str] = None,
                                output_dir: str = "flexible_hdbscan_results") -> Dict[str, Any]:
    """
    Backward-compatible wrapper for run_clustering_pipeline.
    Maintains the same interface as the original function.
    """
    result = run_clustering_pipeline(
        config_path=config_path,
        data_path=data_path,
        output_dir=output_dir,
        force_umap=None  # Use config setting
    )
    
    if not result['success']:
        raise Exception(result['error_message'])
    
    # Remove 'success' key for backward compatibility
    result.pop('success', None)
    result.pop('error_message', None)
    return result


def run_umap_hdbscan_pipeline(data_path: Optional[str] = None,
                            config_path: Optional[str] = None, 
                            output_dir: str = "umap_hdbscan_results") -> Dict[str, Any]:
    """
    Backward-compatible wrapper for run_clustering_pipeline with forced UMAP.
    Maintains the same interface as the original function.
    """
    result = run_clustering_pipeline(
        config_path=config_path,
        data_path=data_path,
        output_dir=output_dir,
        force_umap=True  # Force UMAP usage
    )
    
    if not result['success']:
        raise Exception(result['error_message'])
        
    # Remove 'success' key for backward compatibility
    result.pop('success', None) 
    result.pop('error_message', None)
    return result


if __name__ == "__main__":
    """Test the simplified clustering pipeline."""
    print("Simplified Clustering Pipeline - Test Run")
    print("=" * 50)
    
    try:
        # Test with auto-detection
        results = run_clustering_pipeline(
            output_dir="test_simplified_pipeline",
        )
        
        if results['success']:
            print("✅ Pipeline completed successfully!")
            print(f"📊 UMAP enabled: {results['umap_enabled']}")
            print(f"🎯 Clusters found: {results['pipeline_info']['n_clusters_found']}")
            print(f"📈 Total data points: {results['pipeline_info']['total_data_points']:,}")
            print(f"🔇 Noise points: {results['pipeline_info']['noise_points']:,}")
        else:
            print(f"❌ Pipeline failed: {results['error_message']}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        print("This is expected if you don't have data files available.")
