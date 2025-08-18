"""
interaction_mode_distance_scores.py

Modular functions for computing distance scores from feature medians for interaction mode clusters.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List

FEATURES = ["DEX_EVENTS", "CEX_EVENTS", "BRIDGE_EVENTS", "DEFI_EVENTS"]

RAW_RESULTS_DIR = "data/raw_data/interaction_mode_results"


def load_reference_medians(dataset_name: str) -> Dict[str, float]:
    """
    Load the reference medians for each feature from the cluster results.
    """
    median_path = os.path.join(RAW_RESULTS_DIR, f"{dataset_name}_feature_medians.csv")
    medians = pd.read_csv(median_path, index_col=0)
    return medians["median"].to_dict()


def preprocess_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Apply log(1 + x) to the selected features.
    """
    df_proc = df.copy()
    for feat in features:
        df_proc[f"log1p_{feat}"] = np.log1p(df_proc[feat])
    return df_proc


def compute_absolute_distances(df_proc: pd.DataFrame, medians: Dict[str, float], features: List[str]) -> pd.DataFrame:
    """
    Compute absolute distances between wallet features and reference medians.
    """
    for feat in features:
        median_val = np.log1p(medians[feat])
        df_proc[f"abs_dist_{feat}"] = np.abs(df_proc[f"log1p_{feat}"] - median_val)
    return df_proc


def compute_mad(df_proc: pd.DataFrame, medians: Dict[str, float], features: List[str]) -> Dict[str, float]:
    """
    Compute the median absolute deviation (MAD) for each feature around the cluster median.
    """
    mad_dict = {}
    for feat in features:
        median_val = np.log1p(medians[feat])
        mad = np.median(np.abs(df_proc[f"log1p_{feat}"] - median_val))
        mad_dict[feat] = mad if mad > 0 else 1e-6
    return mad_dict


def normalise_distances(df_proc: pd.DataFrame, mad_dict: Dict[str, float], features: List[str]) -> pd.DataFrame:
    """
    Normalise distances by dividing by MAD.
    """
    for feat in features:
        df_proc[f"norm_dist_{feat}"] = df_proc[f"abs_dist_{feat}"] / mad_dict[feat]
    return df_proc


def compute_proportionality_weighting(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    For each wallet, compute the proportionality weighting for each feature.
    """
    sum_feats = df[features].sum(axis=1)
    for feat in features:
        df[f"prop_weight_{feat}"] = df[feat] / sum_feats.replace(0, np.nan)
    return df


def compute_final_scores(df_proc: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Multiply proportionality weighting by normalised distance for each feature.
    """
    for feat in features:
        df_proc[f"distance_score_{feat}"] = df_proc[f"prop_weight_{feat}"] * df_proc[f"norm_dist_{feat}"]
    return df_proc
