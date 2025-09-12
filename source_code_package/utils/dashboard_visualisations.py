import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from typing import List, Optional

ANALYTIC_SCORE_COLUMNS = [
	"BEHAVIOURAL_VOLATILITY_SCORE",
	"CROSS_DOMAIN_ENGAGEMENT_SCORE",
	"REVENUE_SCORE_PROXY",
	"DEX_EVENTS_SIGNED_DIST",
	"CEX_EVENTS_SIGNED_DIST",
	"BRIDGE_EVENTS_SIGNED_DIST",
	"DEFI_EVENTS_SIGNED_DIST"
]


def plot_analytic_score_distributions(
	df: pd.DataFrame,
	columns: Optional[List[str]] = None,
	bins: int = 30,
	save_dir: Optional[str] = None,
	show: bool = True
) -> None:
	"""
	Plot distributions of analytic scores using Plotly histograms (all data, not separated by cluster).
	If save_dir is provided, saves each plot as an HTML file in that directory.
	If show is True, displays each plot interactively.
	"""
	if columns is None:
		columns = ANALYTIC_SCORE_COLUMNS
	for col in columns:
		fig = go.Figure()
		fig.add_trace(go.Histogram(
			x=df[col],
			nbinsx=bins,
			name=col,
			marker_color='blue',
			opacity=0.75
		))
		fig.update_layout(
			title=f"Distribution of {col}",
			xaxis_title=col,
			yaxis_title="Count",
			bargap=0.1
		)
		if save_dir:
			import os
			os.makedirs(save_dir, exist_ok=True)
			fig.write_html(os.path.join(save_dir, f"distribution_{col}.html"))
		if show:
			fig.show()


def plot_analytic_score_density_by_cluster(
	df: pd.DataFrame,
	cluster_col: str = "activity_cluster_label",
	columns: Optional[List[str]] = None,
	bins: int = 30,
	save_dir: Optional[str] = None,
	show: bool = True
) -> None:
	"""
	For each analytic score, plot density histograms overlayed for each cluster using Plotly.
	If save_dir is provided, saves each plot as an HTML file in that directory.
	If show is True, displays each plot interactively.
	"""
	if columns is None:
		columns = ANALYTIC_SCORE_COLUMNS
	clusters = sorted(df[cluster_col].dropna().unique())
	color_seq = pc.qualitative.Plotly
	for col in columns:
		fig = go.Figure()
		for i, cluster in enumerate(clusters):
			cluster_data = df[df[cluster_col] == cluster][col].dropna()
			fig.add_trace(go.Histogram(
				x=cluster_data,
				nbinsx=bins,
				name=f"Cluster {cluster}",
				histnorm='probability density',
				marker_color=color_seq[i % len(color_seq)],
				opacity=0.7
			))
		fig.update_layout(
			barmode='overlay',
			title=f"Density Distribution of {col} by Cluster",
			xaxis_title=col,
			yaxis_title="Density",
			bargap=0.1
		)
		if save_dir:
			import os
			os.makedirs(save_dir, exist_ok=True)
			fig.write_html(os.path.join(save_dir, f"density_by_cluster_{col}.html"))
		if show:
			fig.show()



def plot_filtered_wallets_analytic_score_distributions(
	df: pd.DataFrame,
	columns: Optional[List[str]] = None,
	bins: int = 30,
	save_dir: Optional[str] = None,
	show: bool = True
) -> None:
	"""
	Plots analytic score distributions for the whole dataset, overlays the distribution for wallets with:
	- BEHAVIOURAL_VOLATILITY_SCORE < 0.5
	- REVENUE_SCORE_PROXY above the 75th percentile
	Also produces visualisations of the cluster distribution (activity_cluster_label) for these filtered wallets.
	Plots are clearly labelled.
	"""
	if columns is None:
		columns = ANALYTIC_SCORE_COLUMNS
	# Calculate 75th percentile for REVENUE_SCORE_PROXY
	revenue_75th = df["REVENUE_SCORE_PROXY"].quantile(0.75)
	filtered = df[(df["BEHAVIOURAL_VOLATILITY_SCORE"] < 0.5) & (df["REVENUE_SCORE_PROXY"] > revenue_75th)]
	for col in columns:
		fig = go.Figure()
		# Plot full dataset
		fig.add_trace(go.Histogram(
			x=df[col],
			nbinsx=bins,
			name=f"All wallets",
			marker_color='lightgrey',
			opacity=0.5
		))
		# Overlay filtered wallets
		fig.add_trace(go.Histogram(
			x=filtered[col],
			nbinsx=bins,
			name=f"Filtered wallets (BV<0.5 & Revenue>75th pct)",
			marker_color='crimson',
			opacity=0.8
		))
		fig.update_layout(
			title=f"Distribution of {col} (All vs. Filtered Wallets)",
			xaxis_title=col,
			yaxis_title="Count",
			barmode='overlay',
			bargap=0.1,
			legend_title="Wallet Group"
		)
		if save_dir:
			import os
			os.makedirs(save_dir, exist_ok=True)
			fig.write_html(os.path.join(save_dir, f"filtered_overlay_{col}.html"))
		if show:
			fig.show()

	# Extra: For filtered wallets, show cluster distribution (bar plot)
	if "activity_cluster_label" in df.columns:
		import plotly.express as px
		cluster_counts = filtered["activity_cluster_label"].value_counts().sort_index()
		fig = px.bar(
			x=cluster_counts.index,
			y=cluster_counts.values,
			labels={"x": "Cluster Label", "y": "Wallet Count"},
			title="Cluster Distribution of Filtered Wallets (BV<0.5 & Revenue>75th pct)"
		)
		if save_dir:
			fig.write_html(os.path.join(save_dir, "filtered_wallets_cluster_distribution.html"))
		if show:
			fig.show()

		# For each analytic score, show distribution by cluster for filtered wallets
		color_seq = pc.qualitative.Plotly
		clusters = sorted(filtered["activity_cluster_label"].dropna().unique())
		for col in columns:
			fig = go.Figure()
			for i, cluster in enumerate(clusters):
				cluster_data = filtered[filtered["activity_cluster_label"] == cluster][col].dropna()
				fig.add_trace(go.Histogram(
					x=cluster_data,
					nbinsx=bins,
					name=f"Cluster {cluster}",
					histnorm='probability density',
					marker_color=color_seq[i % len(color_seq)],
					opacity=0.7
				))
			fig.update_layout(
				barmode='overlay',
				title=f"Filtered Wallets: Density of {col} by Cluster (BV<0.5 & Revenue>75th pct)",
				xaxis_title=col,
				yaxis_title="Density",
				bargap=0.1,
				legend_title="Cluster"
			)
			if save_dir:
				fig.write_html(os.path.join(save_dir, f"filtered_density_by_cluster_{col}.html"))
			if show:
				fig.show()
	# Removed redundant code that reloads the CSV and calls plotting functions again.


# Example execution (for direct script use)
if __name__ == "__main__":

	# Load merged analytic scores dataset
	df = pd.read_csv("data/processed_data/merged_analytic_scores.csv")
	output_dir = "artifacts/Dashboards"
	# Plot distributions for all analytic scores, save and show
	print("Saving and displaying analytic score distributions (all data) to artifacts/Dashboards ...")
	plot_analytic_score_distributions(df, save_dir=output_dir, show=True)
	# Plot density overlays by cluster, save and show
	print("Saving and displaying analytic score density distributions by cluster to artifacts/Dashboards ...")
	plot_analytic_score_density_by_cluster(df, save_dir=output_dir, show=True)
	# Plot filtered overlays and cluster analysis for filtered wallets
	print("Saving and displaying filtered wallet overlays and cluster analysis to artifacts/Dashboards ...")
	plot_filtered_wallets_analytic_score_distributions(df, save_dir=output_dir, show=True)