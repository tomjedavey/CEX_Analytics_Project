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
BASE_PATH = "data/raw_data/interaction_mode_results/"
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
    wallet_df = pd.read_csv(clustering_data_path)
    wallet_df = wallet_df[FEATURES]

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

    # Save outputs
    out_dir = os.path.join(BASE_PATH, cluster)
    dist.to_csv(os.path.join(out_dir, "distances.csv"), index=False)
    norm_dist.to_csv(os.path.join(out_dir, "normalized_distances.csv"), index=False)
    weighted_dist.to_csv(os.path.join(out_dir, "weighted_distances.csv"), index=False)
    print(f"Finished {cluster}. Outputs saved to {out_dir}.")
