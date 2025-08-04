#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) utility functions.

This module provides utility functions for data exploration, quality assessment,
and initial data analysis tasks commonly used in the wallet segmentation pipeline.

Author: Tom Davey
Date: August 2025
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns


def nan_checker(df: pd.DataFrame) -> None:
    """
    Check for NaN values in a DataFrame and report their locations.
    
    This function iterates through each cell in the DataFrame and prints
    the location of any NaN values found. If no NaN values are present,
    no output is generated.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check for NaN values
    
    Returns:
    --------
    None
        Prints locations of NaN values to console
    
    Notes:
    ------
    - For large DataFrames, this function may be slow as it checks every cell
    - Consider using df.isnull().sum() for a faster summary of NaN counts per column
    
    Example:
    --------
    >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
    >>> nan_checker(df)
    NaN value found at row 2, column 'A'
    NaN value found at row 1, column 'B'
    """
    for index, row in df.iterrows():
        for column in df.columns:
            if pd.isnull(row[column]):
                print(f"NaN value found at row {index}, column '{column}'")


def data_quality_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive data quality summary for a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    
    Returns:
    --------
    dict
        Dictionary containing data quality metrics:
        - shape: (rows, columns)
        - missing_values: Count and percentage of missing values per column
        - data_types: Data types of each column
        - duplicate_rows: Number of duplicate rows
        - memory_usage: Memory usage in MB
    
    Example:
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> summary = data_quality_summary(df)
    >>> print(summary['shape'])
    (3, 2)
    """
    summary = {}
    
    # Basic shape information
    summary['shape'] = df.shape
    summary['total_cells'] = df.shape[0] * df.shape[1]
    
    # Missing values analysis
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    summary['missing_values'] = {
        'counts': missing_counts.to_dict(),
        'percentages': missing_pct.to_dict(),
        'total_missing': missing_counts.sum(),
        'columns_with_missing': (missing_counts > 0).sum()
    }
    
    # Data types
    summary['data_types'] = df.dtypes.to_dict()
    
    # Duplicate rows
    summary['duplicate_rows'] = df.duplicated().sum()
    
    # Memory usage
    summary['memory_usage_mb'] = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    return summary


def numerical_summary_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generate detailed summary statistics for numerical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    columns : List[str], optional
        Specific columns to analyze. If None, analyzes all numerical columns
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with comprehensive statistics including skewness, kurtosis,
        and percentiles for numerical columns
    
    Notes:
    ------
    - Automatically identifies numerical columns if none specified
    - Includes additional statistics beyond pandas describe()
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Basic statistics
    stats_df = df[columns].describe().T
    
    # Additional statistics
    stats_df['skewness'] = df[columns].skew()
    stats_df['kurtosis'] = df[columns].kurtosis()
    stats_df['missing_count'] = df[columns].isnull().sum()
    stats_df['missing_pct'] = (df[columns].isnull().sum() / len(df)) * 100
    
    # Additional percentiles
    stats_df['p5'] = df[columns].quantile(0.05)
    stats_df['p95'] = df[columns].quantile(0.95)
    stats_df['p99'] = df[columns].quantile(0.99)
    
    return stats_df


def detect_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None,
                   method: str = 'iqr', multiplier: float = 1.5) -> Dict[str, Dict]:
    """
    Detect outliers in numerical columns using specified method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    columns : List[str], optional
        Columns to check for outliers. If None, checks all numerical columns
    method : str, default 'iqr'
        Method to use for outlier detection ('iqr' or 'zscore')
    multiplier : float, default 1.5
        Multiplier for outlier threshold (1.5 for IQR, 3 for z-score typically)
    
    Returns:
    --------
    dict
        Dictionary with outlier information for each column:
        - outlier_indices: Indices of outlier rows
        - outlier_count: Number of outliers
        - outlier_percentage: Percentage of data that are outliers
    
    Notes:
    ------
    - IQR method: outliers are values < Q1 - 1.5*IQR or > Q3 + 1.5*IQR
    - Z-score method: outliers are values with |z-score| > multiplier
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = {}
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > multiplier].index
        
        outlier_info[col] = {
            'outlier_indices': outliers.tolist(),
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(df)) * 100
        }
    
    return outlier_info


def correlation_analysis(df: pd.DataFrame, columns: Optional[List[str]] = None,
                        method: str = 'pearson', threshold: float = 0.7) -> Dict[str, Any]:
    """
    Perform correlation analysis on numerical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    columns : List[str], optional
        Columns to include in correlation analysis
    method : str, default 'pearson'
        Correlation method ('pearson', 'spearman', 'kendall')
    threshold : float, default 0.7
        Correlation threshold for identifying highly correlated pairs
    
    Returns:
    --------
    dict
        Dictionary containing:
        - correlation_matrix: Full correlation matrix
        - high_correlations: Pairs with correlation above threshold
        - avg_correlation: Average absolute correlation
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr(method=method)
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val >= threshold:
                high_corr_pairs.append({
                    'variable1': corr_matrix.columns[i],
                    'variable2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    # Calculate average absolute correlation
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    avg_correlation = upper_triangle.stack().abs().mean()
    
    return {
        'correlation_matrix': corr_matrix,
        'high_correlations': high_corr_pairs,
        'avg_correlation': avg_correlation
    }


def categorical_summary(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Generate summary statistics for categorical columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    columns : List[str], optional
        Categorical columns to analyze. If None, auto-detects categorical columns
    
    Returns:
    --------
    dict
        Dictionary with summary information for each categorical column:
        - unique_count: Number of unique values
        - most_frequent: Most frequent value and its count
        - value_counts: Full value counts
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    categorical_info = {}
    
    for col in columns:
        value_counts = df[col].value_counts()
        
        categorical_info[col] = {
            'unique_count': df[col].nunique(),
            'missing_count': df[col].isnull().sum(),
            'most_frequent': {
                'value': value_counts.index[0] if len(value_counts) > 0 else None,
                'count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'percentage': (value_counts.iloc[0] / len(df)) * 100 if len(value_counts) > 0 else 0
            },
            'value_counts': value_counts.to_dict()
        }
    
    return categorical_info