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
import sys

# Add the parent directory to path to import from source_code_package
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from data.preprocess_cluster import preprocess_for_clustering


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
        UMAP configuration dictionary
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../../config/config_cluster.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('umap', {})


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
    - Only numerical columns are used for UMAP
    - NaN values are handled by dropping rows with NaN
    """
    
    # Load UMAP configuration
    umap_config = load_umap_config(config_path)
    
    # Override config with any provided kwargs
    umap_params = {**umap_config, **kwargs}
    
    # Select only numerical columns
    numerical_data = data.select_dtypes(include=[np.number])
    
    # Handle NaN values
    if numerical_data.isnull().any().any():
        print(f"Warning: Found NaN values. Dropping {numerical_data.isnull().any(axis=1).sum()} rows with NaN.")
        numerical_data = numerical_data.dropna()
    
    print(f"Applying UMAP with parameters: {umap_params}")
    print(f"Input data shape: {numerical_data.shape}")
    
    # Initialize UMAP with configuration parameters
    umap_model = umap.UMAP(**umap_params)
    
    # Fit and transform the data
    reduced_data = umap_model.fit_transform(numerical_data)
    
    print(f"Output data shape: {reduced_data.shape}")
    print(f"Explained variance (approximation): {umap_model.explained_variance_ratio_[:2] if hasattr(umap_model, 'explained_variance_ratio_') else 'Not available'}")
    
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
    
    # Step 1: Preprocess the data
    print("\n=== PREPROCESSING STAGE ===")
    preprocessed_data, preprocessing_info = preprocess_for_clustering(
        data_path=data_path,
        config_path=config_path,
        apply_log_transform=apply_log_transform,
        apply_scaling=apply_scaling
    )
    
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


def evaluate_umap_quality(original_data: pd.DataFrame, reduced_data: np.ndarray, 
                         labels: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate the quality of UMAP dimensionality reduction.
    
    Parameters:
    -----------
    original_data : pd.DataFrame
        Original high-dimensional data
    reduced_data : np.ndarray
        UMAP reduced data
    labels : np.ndarray, optional
        Cluster labels for silhouette score calculation
    
    Returns:
    --------
    dict
        Dictionary containing quality metrics
    """
    
    metrics = {}
    
    # Calculate dimension reduction ratio
    original_dims = original_data.select_dtypes(include=[np.number]).shape[1]
    reduced_dims = reduced_data.shape[1]
    metrics['dimension_reduction_ratio'] = reduced_dims / original_dims
    
    # Calculate silhouette score if labels are provided
    if labels is not None and len(np.unique(labels)) > 1:
        try:
            silhouette_avg = silhouette_score(reduced_data, labels)
            metrics['silhouette_score'] = silhouette_avg
        except Exception as e:
            print(f"Could not calculate silhouette score: {e}")
            metrics['silhouette_score'] = None
    
    # Calculate variance preserved (approximate)
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(reduced_dims, original_dims))
        pca.fit(original_data.select_dtypes(include=[np.number]))
        metrics['variance_preserved_estimate'] = np.sum(pca.explained_variance_ratio_[:reduced_dims])
    except Exception as e:
        print(f"Could not estimate variance preserved: {e}")
        metrics['variance_preserved_estimate'] = None
    
    return metrics

#NEED TO UNDERSTAND THE METRICS BEING USED TO ASSESS QUALITY OF UMAP REDUCTION FROM THE ABOVE FUNCTION


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


#BELOW IS SOME FORM OF EXAMPLE USAGE FUNCTION THAT CAN BE USED TO RUN THE COMPLETE PIPELINE - CHECK NEED IN LINE WITH WHAT IS PLANNED ETC


# Example usage function
def run_umap_pipeline_example(config_path: Optional[str] = None):
    """
    Example function demonstrating the complete UMAP pipeline.
    """
    
    try:
        # Run the complete pipeline
        reduced_data, umap_model, preprocessed_data, preprocessing_info = umap_with_preprocessing(
            config_path=config_path
        )
        
        # Evaluate quality
        quality_metrics = evaluate_umap_quality(preprocessed_data, reduced_data)
        
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


if __name__ == "__main__":
    # Run example pipeline
    run_umap_pipeline_example()

