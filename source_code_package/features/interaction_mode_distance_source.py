"""
Module for calculating distance from feature medians with preprocessing, normalization, and proportionality weighting.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

def load_medians(median_csv_path: str, features: List[str]) -> pd.DataFrame:
    """
    Load feature medians from a CSV and return a DataFrame with only the specified features.
    """
    df = pd.read_csv(median_csv_path)
    return df[features]

def preprocess_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Apply log(1 + x) transformation to the specified features in the DataFrame.
    """
    df_proc = df.copy()
    for feat in features:
        df_proc[feat] = np.log1p(df_proc[feat])
    return df_proc  

def compute_distances(wallet_df: pd.DataFrame, median_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Compute signed distances from the median for each feature and wallet.
    For each wallet and feature, subtract the feature value from the median value (median - value).
    """
    dist = median_df.iloc[0][features] - wallet_df[features]
    return dist

def compute_mad(df: pd.DataFrame, features: List[str], median_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute the median absolute deviation (MAD) for each feature around the median value.
    """
    mad = {}
    for feat in features:
        mad[feat] = np.median(np.abs(df[feat] - median_df.iloc[0][feat]))
    return mad

def normalize_distances(abs_dist_df: pd.DataFrame, mad: Dict[str, float], features: List[str]) -> pd.DataFrame:
    """
    Normalize absolute distances by dividing by MAD for each feature.
    """
    norm_dist = abs_dist_df.copy()
    for feat in features:
        norm_dist[feat] = norm_dist[feat] / mad[feat] if mad[feat] != 0 else 0
    return norm_dist

def compute_proportionality_weights(wallet_df: pd.DataFrame, event_features: List[str]) -> pd.DataFrame:
    """
    For each wallet, calculate the proportion of each event feature.
    """
    event_sum = wallet_df[event_features].sum(axis=1)
    weights = wallet_df[event_features].div(event_sum, axis=0).fillna(0)
    return weights

def apply_proportionality_weighting(norm_dist_df: pd.DataFrame, weights_df: pd.DataFrame, event_features: List[str]) -> pd.DataFrame:
    """
    Multiply normalized distances by proportionality weights for event features.
    """
    weighted = norm_dist_df[event_features] * weights_df[event_features]
    return weighted
