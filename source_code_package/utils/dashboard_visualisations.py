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
	bins: int = 30
) -> None:
	"""
	Plot distributions of analytic scores using Plotly histograms (all data, not separated by cluster).
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
		fig.show()

def plot_analytic_score_density_by_cluster(
	df: pd.DataFrame,
	cluster_col: str = "activity_cluster_label",
	columns: Optional[List[str]] = None,
	bins: int = 30
) -> None:
	"""
	For each analytic score, plot density histograms overlayed for each cluster using Plotly.
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
		fig.show()



# Example execution (for direct script use)
if __name__ == "__main__":
	# Load merged analytic scores dataset
	df = pd.read_csv("data/processed_data/merged_analytic_scores.csv")
	# Plot distributions for all analytic scores
	print("Plotting analytic score distributions (all data)...")
	plot_analytic_score_distributions(df)
	# Plot density overlays by cluster
	print("Plotting analytic score density distributions by cluster...")
	plot_analytic_score_density_by_cluster(df)