"""
Script to calculate distance from medians for each clustering dataset (cluster 0, cluster 1, main).
"""
import os
import pandas as pd
import numpy as np
from source_code_package.features.interaction_mode_distance_source import (
    load_medians, preprocess_features, compute_distances,
    compute_mad, normalize_distances, compute_proportionality_weights,
    apply_proportionality_weighting
)


# Define paths and features
BASE_PATH = "data/processed_data/interaction_mode_results/"
EVENT_FEATURES = ["DEX_EVENTS", "CEX_EVENTS", "BRIDGE_EVENTS", "DEFI_EVENTS"]
# You may want to adjust this list to match all features in the medians
FEATURES = EVENT_FEATURES  # Extend as needed

# Dynamically find all cluster folders and median files
cluster_dirs = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d)) and d.endswith('_clustering')]
median_files = {d: f"{d}_feature_medians.csv" for d in cluster_dirs}

for cluster in cluster_dirs:
    print(f"Processing {cluster}...")
    median_path = os.path.join(BASE_PATH, median_files[cluster])
    clustering_data_path = os.path.join(BASE_PATH, cluster, "clustered_data.csv")

    # Check if median file and clustered data exist
    if not os.path.exists(median_path):
        print(f"Warning: Median file not found for {cluster}: {median_path}")
        continue
    if not os.path.exists(clustering_data_path):
        print(f"Warning: Clustered data not found for {cluster}: {clustering_data_path}")
        continue

    # Load medians and wallet data
    medians_df = load_medians(median_path, FEATURES)
    wallet_full_df = pd.read_csv(clustering_data_path)
    wallet_df = wallet_full_df[FEATURES]

    # Preprocess
    #medians_proc = preprocess_features(medians_df, FEATURES)
    #wallet_proc = preprocess_features(wallet_df, FEATURES)

    medians_proc = medians_df
    wallet_proc = wallet_df

    # Compute signed distances
    dist = compute_distances(wallet_proc, medians_proc, FEATURES)
    abs_dist = np.abs(dist)

    # Compute MAD
    mad = compute_mad(wallet_proc, FEATURES, medians_proc)

    # Normalize distances
    norm_dist = normalize_distances(dist, mad, FEATURES)

    # Compute proportionality weights
    weights = compute_proportionality_weights(wallet_proc, EVENT_FEATURES)

    # Apply proportionality weighting
    weighted_dist = apply_proportionality_weighting(norm_dist, weights, EVENT_FEATURES)

    # Save to processed_data/interaction_mode_results/<cluster>/
    out_dir = os.path.join(BASE_PATH, cluster)
    os.makedirs(out_dir, exist_ok=True)


    # --- Full output for absolute (raw) distances ---
    abs_output = wallet_full_df.copy()
    for feat in FEATURES:
        abs_output[f"{feat}_MEDIAN"] = medians_df.iloc[0][feat]
    for feat in FEATURES:
        abs_output[f"{feat}_ABS_DIST"] = abs_dist[feat].values
    abs_output.to_csv(os.path.join(out_dir, "full_absolute_distances.csv"), index=False)

    # --- Full output for raw (absolute) distances ---
    raw_output = wallet_full_df.copy()
    for feat in FEATURES:
        raw_output[f"{feat}_MEDIAN"] = medians_df.iloc[0][feat]
    for feat in FEATURES:
        raw_output[f"{feat}_RAW_DIST"] = abs_dist[feat].values
    raw_output.to_csv(os.path.join(out_dir, "full_raw_distances.csv"), index=False)

    # --- Optionally, output signed distances ---
    signed_output = wallet_full_df.copy()
    for feat in FEATURES:
        signed_output[f"{feat}_MEDIAN"] = medians_df.iloc[0][feat]
    for feat in FEATURES:
        signed_output[f"{feat}_SIGNED_DIST"] = dist[feat].values
    signed_output.to_csv(os.path.join(out_dir, "full_signed_distances.csv"), index=False)

    # --- Full output for normalized distances ---
    norm_output = wallet_full_df.copy()
    for feat in FEATURES:
        norm_output[f"{feat}_MEDIAN"] = medians_df.iloc[0][feat]
    for feat in FEATURES:
        norm_output[f"{feat}_NORM_DIST"] = norm_dist[feat].values
    norm_output.to_csv(os.path.join(out_dir, "full_normalized_distances.csv"), index=False)

    # --- Full output for weighted distances ---
    weighted_output = wallet_full_df.copy()
    for feat in FEATURES:
        weighted_output[f"{feat}_MEDIAN"] = medians_df.iloc[0][feat]
    for feat in FEATURES:
        weighted_output[f"{feat}_WEIGHTED_DIST"] = weighted_dist[feat].values
    weighted_output.to_csv(os.path.join(out_dir, "full_weighted_distances.csv"), index=False)

    # Delete old single-score CSVs if they exist
    for fname in ["absolute_distances.csv", "distances.csv", "normalized_distances.csv", "weighted_distances.csv"]:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    print(f"Finished {cluster}. Full outputs saved to {out_dir}.")
