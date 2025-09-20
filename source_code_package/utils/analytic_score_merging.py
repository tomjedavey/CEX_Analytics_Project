
import pandas as pd
import os

def merge_analytic_scores(
	clustered_data_path: str = "data/processed_data/clustering_results/clustered_data.csv",
	volatility_path: str = "data/processed_data/behavioural_volatility_features.csv",
	engagement_path: str = "data/processed_data/cross_domain_engagement_features.csv",
	revenue_path: str = "data/processed_data/revenue_proxy_features.csv",
	distances_path: str = "data/processed_data/interaction_mode_results/main_clustering/full_signed_distances.csv",
	output_path: str = "data/processed_data/merged_analytic_scores.csv"
) -> None:
	"""
	Merge analytic score columns from various processed_data CSVs into a single CSV.
	"""
	# Load main clustered data
	clustered_df = pd.read_csv(clustered_data_path)

	# Load and select required columns from each analytic score file
	volatility_df = pd.read_csv(volatility_path, usecols=["WALLET", "BEHAVIOURAL_VOLATILITY_SCORE"])
	engagement_df = pd.read_csv(engagement_path, usecols=["WALLET", "CROSS_DOMAIN_ENGAGEMENT_SCORE"])
	revenue_df = pd.read_csv(revenue_path, usecols=["WALLET", "REVENUE_SCORE_PROXY"])

	# Load and select required columns from distances file

	# Update SIGNED_DIST columns to INTERACTION_MODE
	distances_cols = [
		"WALLET",
		"DEX_EVENTS_MEDIAN", "CEX_EVENTS_MEDIAN", "BRIDGE_EVENTS_MEDIAN", "DEFI_EVENTS_MEDIAN",
		"DEX_EVENTS_SIGNED_DIST", "CEX_EVENTS_SIGNED_DIST", "BRIDGE_EVENTS_SIGNED_DIST", "DEFI_EVENTS_SIGNED_DIST"
	]
	distances_df = pd.read_csv(distances_path, usecols=distances_cols)

	# Rename *_SIGNED_DIST columns to *_INTERACTION_MODE
	rename_map = {col: col.replace("SIGNED_DIST", "INTERACTION_MODE") for col in distances_df.columns if col.endswith("SIGNED_DIST")}
	distances_df = distances_df.rename(columns=rename_map)

	# Merge all on WALLET
	merged = clustered_df.merge(volatility_df, on="WALLET", how="left") \
						.merge(engagement_df, on="WALLET", how="left") \
						.merge(revenue_df, on="WALLET", how="left") \
						.merge(distances_df, on="WALLET", how="left")

	# Rename 'cluster_label' to 'activity_cluster_label' if present
	if 'cluster_label' in merged.columns:
		merged = merged.rename(columns={'cluster_label': 'activity_cluster_label'})

	# Save to output
	merged.to_csv(output_path, index=False)

if __name__ == "__main__":
	merge_analytic_scores()

