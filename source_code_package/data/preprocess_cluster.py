#This functionality is used to preprocess the data for clustering algorithms, such as scaling and log-transformation.
#This file also replicates the work from preprocess_AS_1.py, in regards to scaling. 
#However, this functionality is reproduced for modularity as well as the opportunity to edit functionality in the future.

import pandas as pd
import numpy as np
import yaml
import os
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler

#FUNCTIONS TO APPLY LOG TRANSFORMATION TO ALL NUMERICAL FEATURES IN THE DATASET EXCEPT FOR SPECIFIED COLUMNS

def log_transform_features(data_path: Optional[str] = None, config_path: Optional[str] = None, 
                          exclude_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Applies log transformation to all numerical features in the dataset except for specified columns.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the data file. If None, will use the path from config file.
    config_path : str, optional  
        Path to the config YAML file. If None, will use default config_cluster.yaml.
    exclude_columns : list, optional
        List of column names to exclude from log transformation. 
        Defaults to ['TX_PER_MONTH', 'ACTIVE_DURATION_DAYS'].
    
    Returns:
    --------
    tuple
        - DataFrame with log-transformed features
        - List of columns that were log-transformed
    
    Notes:
    ------
    - Only positive values are log-transformed. Zero and negative values are handled by adding 1 before transformation.
    - Non-numerical columns are automatically excluded.
    - The function uses natural logarithm (ln).
    """
    
    # Set default excluded columns
    if exclude_columns is None:
        exclude_columns = ['TX_PER_MONTH', 'ACTIVE_DURATION_DAYS']
    
    # Load configuration if config_path is provided
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../config/config_cluster.yaml')
    
    # Determine data path
    if data_path is None:
        # Find the project root (directory containing pyproject.toml)
        current_dir = os.path.abspath(os.path.dirname(__file__))
        while not os.path.exists(os.path.join(current_dir, 'pyproject.toml')):
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                raise FileNotFoundError('Could not find project root (pyproject.toml)')
            current_dir = parent_dir
        project_root = current_dir
        
        # Default to the raw data file for clustering
        data_path = os.path.join(project_root, 'data', 'raw_data', 'initial_raw_data_polygon.csv')
        data_path = os.path.normpath(data_path)
    elif not os.path.isabs(data_path):
        data_path = os.path.normpath(os.path.join(os.getcwd(), data_path))
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Create a copy to avoid modifying the original data
    df_transformed = df.copy()
    
    # Get numerical columns only
    numerical_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove excluded columns from the list of columns to transform
    cols_to_transform = [col for col in numerical_cols if col not in exclude_columns]
    
    # Apply log transformation
    transformed_columns = []
    for col in cols_to_transform:
        # Handle zero and negative values by adding 1 (log(x+1) transformation)
        if (df_transformed[col] <= 0).any():
            df_transformed[col] = np.log1p(df_transformed[col])
            print(f"Applied log1p transformation to {col} (contained zero/negative values)")
        else:
            df_transformed[col] = np.log(df_transformed[col])
            print(f"Applied log transformation to {col}")
        
        transformed_columns.append(col)
    
    print(f"\nLog transformation complete:")
    print(f"- Transformed {len(transformed_columns)} columns")
    print(f"- Excluded columns: {exclude_columns}")
    print(f"- Transformed columns: {transformed_columns}")
    
    return df_transformed, transformed_columns


def log_transform_features_from_config(config_path: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Applies log transformation based on configuration file settings.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to the config YAML file. If None, will use default config_cluster.yaml.
    
    Returns:
    --------
    tuple
        - DataFrame with log-transformed features  
        - List of columns that were log-transformed
    """
    
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../config/config_cluster.yaml')
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get preprocessing settings
    preprocessing_config = config.get('preprocessing', {})
    log_config = preprocessing_config.get('log_transformation', {})
    
    # Check if log transformation is enabled
    if not log_config.get('enabled', True):
        print("Log transformation is disabled in config file")
        return None, []
    
    # Get exclude columns from config
    exclude_columns = log_config.get('exclude_columns', ['TX_PER_MONTH', 'ACTIVE_DURATION_DAYS'])
    
    # Get data path from config
    data_config = config.get('data', {})
    raw_data_path = data_config.get('raw_data_path')
    
    if raw_data_path:
        # Convert relative path to absolute path
        current_dir = os.path.abspath(os.path.dirname(__file__))
        while not os.path.exists(os.path.join(current_dir, 'pyproject.toml')):
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                raise FileNotFoundError('Could not find project root (pyproject.toml)')
            current_dir = parent_dir
        project_root = current_dir
        data_path = os.path.join(project_root, raw_data_path)
        data_path = os.path.normpath(data_path)
    else:
        data_path = None
    
    return log_transform_features(data_path=data_path, config_path=config_path, exclude_columns=exclude_columns)


#FUNCTIONALITY TO APPLY SCALING TO ALL NUMERICAL FEATURES IN THE DATASET


def scale_features(data_path: Optional[str] = None, config_path: Optional[str] = None, 
                  exclude_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str], StandardScaler]:
    """
    Applies StandardScaler to all numerical features in the dataset except for specified columns.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the data file. If None, will use the path from config file.
    config_path : str, optional  
        Path to the config YAML file. If None, will use default config_cluster.yaml.
    exclude_columns : list, optional
        List of column names to exclude from scaling. 
        Defaults to non-numerical columns only.
    
    Returns:
    --------
    tuple
        - DataFrame with scaled features
        - List of columns that were scaled
        - Fitted StandardScaler object
    
    Notes:
    ------
    - Only numerical columns are scaled.
    - Non-numerical columns are preserved in their original form.
    - The fitted scaler is returned for potential use in transforming new data.
    """
    
    # Set default excluded columns (empty by default, will exclude non-numerical automatically)
    if exclude_columns is None:
        exclude_columns = []
    
    # Load configuration if config_path is provided
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../config/config_cluster.yaml')
    
    # Determine data path
    if data_path is None:
        # Find the project root (directory containing pyproject.toml)
        current_dir = os.path.abspath(os.path.dirname(__file__))
        while not os.path.exists(os.path.join(current_dir, 'pyproject.toml')):
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                raise FileNotFoundError('Could not find project root (pyproject.toml)')
            current_dir = parent_dir
        project_root = current_dir
        
        # Default to the raw data file for clustering
        data_path = os.path.join(project_root, 'data', 'raw_data', 'initial_raw_data_polygon.csv')
        data_path = os.path.normpath(data_path)
    elif not os.path.isabs(data_path):
        data_path = os.path.normpath(os.path.join(os.getcwd(), data_path))
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Create a copy to avoid modifying the original data
    df_scaled = df.copy()
    
    # Get numerical columns only
    numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove excluded columns from the list of columns to scale
    cols_to_scale = [col for col in numerical_cols if col not in exclude_columns]
    
    # Apply StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_scaled[cols_to_scale])
    
    # Replace the original columns with scaled versions
    df_scaled[cols_to_scale] = scaled_data
    
    # Print information about the scaling
    print(f"\nStandardScaler transformation complete:")
    print(f"- Scaled {len(cols_to_scale)} columns")
    print(f"- Excluded columns: {exclude_columns}")
    print(f"- Scaled columns: {cols_to_scale}")
    
    return df_scaled, cols_to_scale, scaler


def scale_features_from_config(config_path: Optional[str] = None) -> Tuple[pd.DataFrame, List[str], StandardScaler]:
    """
    Applies StandardScaler based on configuration file settings.
    
    Parameters:
    -----------
    config_path : str, optional
        Path to the config YAML file. If None, will use default config_cluster.yaml.
    
    Returns:
    --------
    tuple
        - DataFrame with scaled features  
        - List of columns that were scaled
        - Fitted StandardScaler object
    """
    
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../config/config_cluster.yaml')
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get preprocessing settings
    preprocessing_config = config.get('preprocessing', {})
    scaling_config = preprocessing_config.get('scaling', {})
    
    # Check if scaling is enabled
    if not scaling_config.get('enabled', True):
        print("Scaling is disabled in config file")
        return None, [], None
    
    # Get exclude columns from config (if any)
    exclude_columns = scaling_config.get('exclude_columns', [])
    
    # Get data path from config
    data_config = config.get('data', {})
    raw_data_path = data_config.get('raw_data_path')
    
    if raw_data_path:
        # Convert relative path to absolute path
        current_dir = os.path.abspath(os.path.dirname(__file__))
        while not os.path.exists(os.path.join(current_dir, 'pyproject.toml')):
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                raise FileNotFoundError('Could not find project root (pyproject.toml)')
            current_dir = parent_dir
        project_root = current_dir
        data_path = os.path.join(project_root, raw_data_path)
        data_path = os.path.normpath(data_path)
    else:
        data_path = None
    
    return scale_features(data_path=data_path, config_path=config_path, exclude_columns=exclude_columns)


#Final wrapper function to apply both log transformation and scaling for clustering purposes


def preprocess_for_clustering(data_path: Optional[str] = None, config_path: Optional[str] = None, 
                             apply_log_transform: bool = True, apply_scaling: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Comprehensive preprocessing function that applies both log transformation and scaling for clustering.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the data file. If None, will use the path from config file.
    config_path : str, optional  
        Path to the config YAML file. If None, will use default config_cluster.yaml.
    apply_log_transform : bool, default True
        Whether to apply log transformation.
    apply_scaling : bool, default True
        Whether to apply scaling.
    
    Returns:
    --------
    tuple
        - DataFrame with preprocessed features
        - Dictionary containing preprocessing information and fitted objects
    
    Notes:
    ------
    - Applies log transformation first, then scaling
    - Returns fitted objects for potential use in transforming new data
    """
    
    preprocessing_info = {
        'log_transformed_columns': [],
        'scaled_columns': [],
        'scaler': None,
        'steps_applied': []
    }
    
    # Start with original data
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../config/config_cluster.yaml')
    
    if data_path is None:
        # Find the project root and set default data path
        current_dir = os.path.abspath(os.path.dirname(__file__))
        while not os.path.exists(os.path.join(current_dir, 'pyproject.toml')):
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                raise FileNotFoundError('Could not find project root (pyproject.toml)')
            current_dir = parent_dir
        project_root = current_dir
        data_path = os.path.join(project_root, 'data', 'raw_data', 'initial_raw_data_polygon.csv')
        data_path = os.path.normpath(data_path)
    
    df_processed = pd.read_csv(data_path)
    
    # Step 1: Apply log transformation if requested
    if apply_log_transform:
        try:
            df_processed, log_transformed_cols = log_transform_features_from_config(config_path=config_path)
            preprocessing_info['log_transformed_columns'] = log_transformed_cols
            preprocessing_info['steps_applied'].append('log_transformation')
            print("Log transformation completed successfully.")
        except Exception as e:
            print(f"Warning: Log transformation failed: {str(e)}")
            apply_log_transform = False
    
    # Step 2: Apply scaling if requested  
    if apply_scaling:
        try:
            # Use the log-transformed data if available, otherwise use original
            temp_data_path = None  # Will use data from df_processed
            
            # Save the current dataframe to a temporary location for scaling
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                df_processed.to_csv(tmp_file.name, index=False)
                temp_data_path = tmp_file.name
            
            df_processed, scaled_cols, scaler = scale_features(data_path=temp_data_path, 
                                                             config_path=config_path)
            preprocessing_info['scaled_columns'] = scaled_cols
            preprocessing_info['scaler'] = scaler
            preprocessing_info['steps_applied'].append('scaling')
            print("Scaling completed successfully.")
            
            # Clean up temporary file
            os.unlink(temp_data_path)
            
        except Exception as e:
            print(f"Warning: Scaling failed: {str(e)}")
    
    print(f"\nPreprocessing pipeline complete. Applied steps: {preprocessing_info['steps_applied']}")
    
    return df_processed, preprocessing_info