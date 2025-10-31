"""
Script to calculate distance from medians for each clustering dataset (cluster 0, cluster 1, main).
"""
import os
import sys
import pandas as pd
import numpy as np
import importlib.util

# Dynamically add the absolute path to Source_Code_Package to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
source_code_path = os.path.join(project_root, 'Source_Code_Package')

if not os.path.exists(source_code_path):
    raise ImportError(f"Could not find Source_Code_Package directory at {source_code_path}")

if source_code_path not in sys.path:
    sys.path.insert(0, source_code_path)

# Dynamically import the interaction_mode_distance_source module
features_path = os.path.join(source_code_path, 'features', 'interaction_mode_distance_source.py')
if not os.path.exists(features_path):
    raise ImportError(f"Could not find interaction_mode_distance_source.py at {features_path}")

spec = importlib.util.spec_from_file_location('interaction_mode_distance_source', features_path)
interaction_mode_distance_source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interaction_mode_distance_source)

# Import functions from the dynamically loaded module
load_medians = interaction_mode_distance_source.load_medians
compute_distances = interaction_mode_distance_source.compute_distances
compute_mad = interaction_mode_distance_source.compute_mad
normalize_distances = interaction_mode_distance_source.normalize_distances
compute_proportionality_weights = interaction_mode_distance_source.compute_proportionality_weights
apply_proportionality_weighting = interaction_mode_distance_source.apply_proportionality_weighting


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

    # Load medians for features from CSV
    medians_df = load_medians(median_path, FEATURES)
    medians_proc = medians_df[FEATURES]

    # Load wallet/clustered data
    wallet_df = pd.read_csv(clustering_data_path)
    wallet_full_df = wallet_df.copy()

    # DEBUG: Print median values being used, especially for DEFI_EVENTS
    print(f"  Median values being used for {cluster}:")
    for feat in FEATURES:
        median_val = medians_df.iloc[0][feat]
        print(f"    {feat}_MEDIAN: {median_val:.2f}")

    # Select rows based on cluster
    if 'cluster_label' in wallet_df.columns:
        wallet_proc = wallet_df[wallet_df['cluster_label'] == 1][FEATURES]  # Use cluster 1 for now
    else:
        wallet_proc = wallet_df[FEATURES]
    wallet_proc = wallet_df

    # Compute signed distances
    dist = compute_distances(wallet_proc, medians_proc, FEATURES)
    abs_dist = np.abs(dist)

    # DEBUG: Print distance statistics for DEFI_EVENTS specifically
    if "DEFI_EVENTS" in FEATURES:
        defi_dist = dist["DEFI_EVENTS"]
        print(f"  DEFI_EVENTS_SIGNED_DIST statistics:")
        print(f"    Min: {defi_dist.min():.2f}, Max: {defi_dist.max():.2f}")
        print(f"    Mean: {defi_dist.mean():.2f}, Median: {defi_dist.median():.2f}")
        print(f"    Wallets with <= 9: {(defi_dist <= 9).sum()} / {len(defi_dist)} ({(defi_dist <= 9).mean()*100:.2f}%)")

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
