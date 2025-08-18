
"""
Script to compute and save interaction mode distance scores for each wallet in each dataset.
"""

import os
import pandas as pd
from source_code_package.features.interaction_mode_distance_scores import (
    FEATURES,
    load_reference_medians,
    preprocess_features,
    compute_absolute_distances,
    compute_mad,
    normalise_distances,
    compute_proportionality_weighting,
    compute_final_scores
)


PROCESSED_DATA_DIR = "data/raw_data/interaction_mode_results"
DATASETS = [
    "main_clustering",
    "cluster_0_clustering",
    "cluster_1_clustering"
]


def load_wallet_data(dataset_name):
    data_path = os.path.join(PROCESSED_DATA_DIR, f"{dataset_name}_feature_medians.csv")
    return pd.read_csv(data_path)


def process_dataset(dataset_name):
    print(f"Processing {dataset_name}...")
    medians = load_reference_medians(dataset_name)
    df = load_wallet_data(dataset_name)
    df_proc = preprocess_features(df, FEATURES)
    df_proc = compute_absolute_distances(df_proc, medians, FEATURES)
    mad_dict = compute_mad(df_proc, medians, FEATURES)
    df_proc = normalise_distances(df_proc, mad_dict, FEATURES)
    df_proc = compute_proportionality_weighting(df_proc, FEATURES)
    df_proc = compute_final_scores(df_proc, FEATURES)
    out_cols = ["wallet_address"] + [f"distance_score_{feat}" for feat in FEATURES]
    out_path = os.path.join(PROCESSED_DATA_DIR, f"{dataset_name}_distance_scores.csv")
    df_proc[out_cols].to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    for dataset in DATASETS:
        process_dataset(dataset)


if __name__ == "__main__":
    main()
