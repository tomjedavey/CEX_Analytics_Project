"""
Script to calculate distance from medians for each clustering dataset (cluster 0, cluster 1, main).
"""
import os
import pandas as pd
from source_code_package.features.distance_from_median import (
    load_medians, preprocess_features, compute_absolute_distances,
    compute_mad, normalize_distances, compute_proportionality_weights,
    apply_proportionality_weighting
)

# Define paths and features
BASE_PATH = "data/processed_data/interaction_mode_results/"
CLUSTERS = ["cluster_0_clustering", "cluster_1_clustering", "main_clustering"]
MEDIAN_FILES = {
    "cluster_0_clustering": "cluster_0_clustering_feature_medians.csv",
    "cluster_1_clustering": "cluster_1_clustering_feature_medians.csv",
    "main_clustering": "main_clustering_feature_medians.csv"
}
EVENT_FEATURES = ["DEX_EVENTS", "CEX_EVENTS", "BRIDGE_EVENTS", "DEFI_EVENTS"]
# You may want to adjust this list to match all features in the medians
FEATURES = EVENT_FEATURES  # Extend as needed

for cluster in CLUSTERS:
    print(f"Processing {cluster}...")
    median_path = os.path.join(BASE_PATH, MEDIAN_FILES[cluster])
    clustering_data_path = os.path.join(BASE_PATH, cluster, "clustered_data.csv")

    # Load medians and wallet data
    medians_df = load_medians(median_path, FEATURES)
    wallet_full_df = pd.read_csv(clustering_data_path)
    wallet_df = wallet_full_df[FEATURES]

    # Preprocess
    medians_proc = preprocess_features(medians_df, FEATURES)
    wallet_proc = preprocess_features(wallet_df, FEATURES)

    # Compute signed distances
    dist = compute_absolute_distances(wallet_proc, medians_proc, FEATURES)

    # Compute MAD
    mad = compute_mad(wallet_proc, FEATURES, medians_proc)

    # Normalize distances
    norm_dist = normalize_distances(dist, mad, FEATURES)

    # Compute proportionality weights
    weights = compute_proportionality_weights(wallet_proc, EVENT_FEATURES)

    # Apply proportionality weighting
    weighted_dist = apply_proportionality_weighting(norm_dist, weights, EVENT_FEATURES)

    # Save to processed_data/interaction_mode_results/<cluster>/
    out_dir = os.path.join("data/processed_data/interaction_mode_results", cluster)
    os.makedirs(out_dir, exist_ok=True)

    # --- Full output for absolute distances ---
    abs_output = wallet_full_df.copy()
    for feat in FEATURES:
        abs_output[f"{feat}_MEDIAN"] = medians_df.iloc[0][feat]
    for feat in FEATURES:
        abs_output[f"{feat}_ABS_DIST"] = dist[feat].values
    abs_output.to_csv(os.path.join(out_dir, "full_absolute_distances.csv"), index=False)

    # --- Full output for raw (signed) distances (same as absolute here, but can be changed if needed) ---
    raw_output = wallet_full_df.copy()
    for feat in FEATURES:
        raw_output[f"{feat}_MEDIAN"] = medians_df.iloc[0][feat]
    for feat in FEATURES:
        raw_output[f"{feat}_RAW_DIST"] = dist[feat].values
    raw_output.to_csv(os.path.join(out_dir, "full_raw_distances.csv"), index=False)

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
