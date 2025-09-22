import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from typing import List, Optional
import os

ANALYTIC_SCORE_COLUMNS = [
	"BEHAVIOURAL_VOLATILITY_SCORE",
	"CROSS_DOMAIN_ENGAGEMENT_SCORE",
	"REVENUE_SCORE_PROXY",
	"DEX_EVENTS_INTERACTION_MODE",
	"CEX_EVENTS_INTERACTION_MODE",
	"BRIDGE_EVENTS_INTERACTION_MODE",
	"DEFI_EVENTS_INTERACTION_MODE"
]


def plot_analytic_score_distributions(
	df: pd.DataFrame,
	columns: Optional[List[str]] = None,
	bins: int = 30,
	save_dir: Optional[str] = None,
	show: bool = True,
	lower_percentile: float = 0,
	upper_percentile: float = 100,
	return_fig: bool = False
) -> 'Optional[go.Figure]':
	"""
	Plot distributions of analytic scores using Plotly histograms (all data, not separated by cluster).
	If save_dir is provided, saves each plot as an HTML file in that directory.
	If show is True, displays each plot interactively.
	"""
	if columns is None:
		columns = ANALYTIC_SCORE_COLUMNS
	figs = []
	for col in columns:
		lower = df[col].quantile(lower_percentile / 100)
		upper = df[col].quantile(upper_percentile / 100)
		filtered_data = df[(df[col] >= lower) & (df[col] <= upper)][col]
		fig = go.Figure()
		fig.add_trace(go.Histogram(
			x=filtered_data,
			nbinsx=bins,
			name=col,
			marker_color='blue',
			opacity=0.75
		))
		fig.update_layout(
			title=f"Distribution of {col} ({lower_percentile}th-{upper_percentile}th pct)",
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
		figs.append(fig)
	if return_fig:
		return figs[0] if len(figs) == 1 else figs


def plot_analytic_score_density_by_cluster(
	df: pd.DataFrame,
	cluster_col: str = "activity_cluster_label",
	columns: Optional[List[str]] = None,
	bins: int = 30,
	save_dir: Optional[str] = None,
	show: bool = True,
	lower_percentile: float = 0,
	upper_percentile: float = 100,
	return_fig: bool = False
) -> 'Optional[go.Figure]':
	"""
	For each analytic score, plot density histograms overlayed for each cluster using Plotly.
	If save_dir is provided, saves each plot as an HTML file in that directory.
	If show is True, displays each plot interactively.
	"""
	if columns is None:
		columns = ANALYTIC_SCORE_COLUMNS
	clusters = [c for c in sorted(df[cluster_col].dropna().unique()) if c != -1]
	# Use a consistent color scheme: shades of blue
	# Use archetype color scheme: blue for all wallets, orange for overlays
	color_seq = ['orange'] * 10
	figs = []
	for col in columns:
		lower = df[col].quantile(lower_percentile / 100)
		upper = df[col].quantile(upper_percentile / 100)
		all_wallets = df[(df[col] >= lower) & (df[col] <= upper)][col]
		for i, cluster in enumerate(clusters):
			cluster_wallets = df[(df[cluster_col] == cluster) & (df[col] >= lower) & (df[col] <= upper)][col]
			fig = go.Figure()
			fig.add_trace(go.Histogram(
				x=all_wallets,
				nbinsx=bins,
				name="All wallets",
				marker_color='blue',
				opacity=0.5,
				histnorm='probability density'
			))
			fig.add_trace(go.Histogram(
				x=cluster_wallets,
				nbinsx=bins,
				name=f"Cluster {cluster}",
				marker_color=color_seq[i % len(color_seq)],
				opacity=0.7,
				histnorm='probability density'
			))
			fig.update_layout(
				barmode='overlay',
				title=f"Density of {col} (All wallets vs. Cluster {cluster}, {lower_percentile}th-{upper_percentile}th pct)",
				xaxis_title=col,
				yaxis_title="Density",
				legend_title="Wallet Group",
				bargap=0.1
			)
			if save_dir:
				import os
				os.makedirs(save_dir, exist_ok=True)
				fig.write_html(os.path.join(save_dir, f"density_{col}_cluster_{cluster}.html"))
			if show:
				fig.show()
			figs.append(fig)
	if return_fig:
		return figs[0] if len(figs) == 1 else figs



def plot_stable_high_value_analytic_score_distributions(
	df: pd.DataFrame,
	columns: Optional[List[str]] = None,
	bins: int = 30,
	save_dir: Optional[str] = None,
	show: bool = True,
	lower_percentile: float = 0,
	upper_percentile: float = 100,
	return_fig: bool = False
) -> 'Optional[go.Figure]':
	"""
	Plots analytic score distributions for the whole dataset, overlays the distribution for the "Stable High-Value Wallets Archetype":
	- BEHAVIOURAL_VOLATILITY_SCORE < 0.5
	- REVENUE_SCORE_PROXY above the 75th percentile
	Also produces visualisations of the cluster distribution (activity_cluster_label) for these wallets.
	Plots are clearly labelled as "Stable High-Value Wallets Archetype".
	"""
	if columns is None:
		columns = ANALYTIC_SCORE_COLUMNS
	# Calculate 75th percentile for REVENUE_SCORE_PROXY
	revenue_75th = df["REVENUE_SCORE_PROXY"].quantile(0.75)
	stable_high_value_wallets = df[(df["BEHAVIOURAL_VOLATILITY_SCORE"] < 0.5) & (df["REVENUE_SCORE_PROXY"] > revenue_75th)]

	# 1. Plot cluster distribution with archetype proportions overlayed
	if "activity_cluster_label" in df.columns:
		import plotly.graph_objects as go
		all_cluster_counts = df["activity_cluster_label"].value_counts().sort_index()
		archetype_cluster_counts = stable_high_value_wallets["activity_cluster_label"].value_counts().sort_index()
		all_clusters = all_cluster_counts.index.tolist()
		archetype_props = [archetype_cluster_counts.get(cl, 0) / all_cluster_counts[cl] if cl in all_cluster_counts and all_cluster_counts[cl] > 0 else 0 for cl in all_clusters]
		fig = go.Figure()
		fig.add_trace(go.Bar(
			x=all_clusters,
			y=all_cluster_counts.values,
			name="All Wallets",
			marker_color='blue',
			opacity=0.5
		))
		fig.add_trace(go.Bar(
			x=all_clusters,
			y=[all_cluster_counts[cl] * archetype_props[i] for i, cl in enumerate(all_clusters)],
			name="Stable High-Value Wallets Archetype Proportion",
			marker_color='orange',
			opacity=0.8
		))
		fig.update_layout(
			title="Cluster Distribution: All Wallets with Stable High-Value Wallets Archetype Proportion Overlay",
			xaxis_title="Cluster Label",
			yaxis_title="Wallet Count",
			barmode='overlay',
			legend_title="Wallet Group"
		)
		if save_dir:
			import os
			fig.write_html(os.path.join(save_dir, "all_vs_stable_high_value_cluster_distribution.html"))
		if show:
			fig.show()

	# 2. For each analytic score, plot density overlays: all data vs. stable high value
	os.makedirs(save_dir, exist_ok=True) if save_dir else None
	figs = []
	for col in columns:
		# Filter outliers based on percentiles for the whole column
		lower = df[col].quantile(lower_percentile / 100)
		upper = df[col].quantile(upper_percentile / 100)
		all_data = df[(df[col] >= lower) & (df[col] <= upper)][col]
		stable_data = stable_high_value_wallets[(stable_high_value_wallets[col] >= lower) & (stable_high_value_wallets[col] <= upper)][col]
		fig = go.Figure()
		# Plot full dataset as density
		fig.add_trace(go.Histogram(
			x=all_data,
			nbinsx=bins,
			name="All Wallets",
			marker_color='blue',
			opacity=0.5,
			histnorm='probability density'
		))
		# Overlay stable high value wallets as density
		fig.add_trace(go.Histogram(
			x=stable_data,
			nbinsx=bins,
			name="Stable High-Value Wallets",
			marker_color='orange',
			opacity=0.7,
			histnorm='probability density'
		))
		fig.update_layout(
			barmode='overlay',
			title=f"Density of {col}: All Wallets vs. Stable High-Value Wallets",
			xaxis_title=col,
			yaxis_title="Density",
			legend_title="Wallet Group",
			bargap=0.1
		)
		if save_dir:
			fig.write_html(os.path.join(save_dir, f"density_{col}_all_vs_stable_high_value.html"))
		if show:
			fig.show()
		figs.append(fig)
	if return_fig:
		return figs[0] if len(figs) == 1 else figs



# New function for the "Erratic Speculator" archetype
def plot_erratic_speculator_analytic_score_distributions(
	df: pd.DataFrame,
	columns: Optional[List[str]] = None,
	bins: int = 30,
	save_dir: Optional[str] = None,
	show: bool = True,
	lower_percentile: float = 0,
	upper_percentile: float = 100,
	return_fig: bool = False
) -> 'Optional[go.Figure]':
	"""
	Plots analytic score distributions for the whole dataset, overlays the distribution for the "Erratic Speculator" archetype:
	- BEHAVIOURAL_VOLATILITY_SCORE > 0.6
	- CROSS_DOMAIN_ENGAGEMENT_SCORE < 0.25
	Also produces visualisations of the cluster distribution (activity_cluster_label) for these wallets.
	Plots are clearly labelled as "Erratic Speculator Archetype".
	"""
	if columns is None:
		columns = ANALYTIC_SCORE_COLUMNS
	# Filter for Erratic Speculator archetype
	erratic_speculator_wallets = df[(df["BEHAVIOURAL_VOLATILITY_SCORE"] > 0.6) & (df["CROSS_DOMAIN_ENGAGEMENT_SCORE"] < 0.25)]

	# 1. Plot cluster distribution with archetype proportions overlayed
	if "activity_cluster_label" in df.columns:
		all_cluster_counts = df["activity_cluster_label"].value_counts().sort_index()
		archetype_cluster_counts = erratic_speculator_wallets["activity_cluster_label"].value_counts().sort_index()
		all_clusters = all_cluster_counts.index.tolist()
		archetype_props = [archetype_cluster_counts.get(cl, 0) / all_cluster_counts[cl] if cl in all_cluster_counts and all_cluster_counts[cl] > 0 else 0 for cl in all_clusters]
		fig = go.Figure()
		fig.add_trace(go.Bar(
			x=all_clusters,
			y=all_cluster_counts.values,
			name="All Wallets",
			marker_color='blue',
			opacity=0.5
		))
		fig.add_trace(go.Bar(
			x=all_clusters,
			y=[all_cluster_counts[cl] * archetype_props[i] for i, cl in enumerate(all_clusters)],
			name="Erratic Speculator Archetype Proportion",
			marker_color='red',
			opacity=0.8
		))
		fig.update_layout(
			title="Cluster Distribution: All Wallets with Erratic Speculator Archetype Proportion Overlay",
			xaxis_title="Cluster Label",
			yaxis_title="Wallet Count",
			barmode='overlay',
			legend_title="Wallet Group"
		)
		if save_dir:
			os.makedirs(save_dir, exist_ok=True)
			fig.write_html(os.path.join(save_dir, "all_vs_erratic_speculator_cluster_distribution.html"))
		if show:
			fig.show()

	# 2. For each analytic score, plot density overlays: all data vs. erratic speculator
	os.makedirs(save_dir, exist_ok=True) if save_dir else None
	figs = []
	for col in columns:
		lower = df[col].quantile(lower_percentile / 100)
		upper = df[col].quantile(upper_percentile / 100)
		all_data = df[(df[col] >= lower) & (df[col] <= upper)][col]
		erratic_data = erratic_speculator_wallets[(erratic_speculator_wallets[col] >= lower) & (erratic_speculator_wallets[col] <= upper)][col]
		fig = go.Figure()
		fig.add_trace(go.Histogram(
			x=all_data,
			nbinsx=bins,
			name="All Wallets",
			marker_color='blue',
			opacity=0.5,
			histnorm='probability density'
		))
		fig.add_trace(go.Histogram(
			x=erratic_data,
			nbinsx=bins,
			name="Erratic Speculator Wallets",
			marker_color='red',
			opacity=0.7,
			histnorm='probability density'
		))
		fig.update_layout(
			barmode='overlay',
			title=f"Density of {col}: All Wallets vs. Erratic Speculator Wallets",
			xaxis_title=col,
			yaxis_title="Density",
			legend_title="Wallet Group",
			bargap=0.1
		)
		if save_dir:
			fig.write_html(os.path.join(save_dir, f"density_{col}_all_vs_erratic_speculator.html"))
		if show:
			fig.show()
		figs.append(fig)
	if return_fig:
		return figs[0] if len(figs) == 1 else figs


# New function for wallets with DEFI_EVENTS_INTERACTION_MODE <= 11
def plot_defi_power_users_analytic_score_distributions(
	df: pd.DataFrame,
	columns: Optional[List[str]] = None,
	bins: int = 30,
	save_dir: Optional[str] = None,
	show: bool = True,
	lower_percentile: float = 0,
	upper_percentile: float = 100,
	return_fig: bool = False
) -> 'Optional[go.Figure]':
	"""
	Plots analytic score distributions for the whole dataset, overlays the distribution for wallets with DEFI_EVENTS_INTERACTION_MODE <= 11.
	Also produces visualisations of the cluster distribution (activity_cluster_label) for these wallets.
	Plots are clearly labelled as "DeFi Power Users Archetype".
	"""
	if columns is None:
		columns = ANALYTIC_SCORE_COLUMNS
	# Filter for DeFi Power Users archetype
	defi_power_users_wallets = df[df["DEFI_EVENTS_INTERACTION_MODE"] <= 11]

	# 1. Plot cluster distribution with archetype proportions overlayed
	if "activity_cluster_label" in df.columns:
		all_cluster_counts = df["activity_cluster_label"].value_counts().sort_index()
		archetype_cluster_counts = defi_power_users_wallets["activity_cluster_label"].value_counts().sort_index()
		all_clusters = all_cluster_counts.index.tolist()
		archetype_props = [archetype_cluster_counts.get(cl, 0) / all_cluster_counts[cl] if cl in all_cluster_counts and all_cluster_counts[cl] > 0 else 0 for cl in all_clusters]
		fig = go.Figure()
		fig.add_trace(go.Bar(
			x=all_clusters,
			y=all_cluster_counts.values,
			name="All Wallets",
			marker_color='blue',
			opacity=0.5
		))
		fig.add_trace(go.Bar(
			x=all_clusters,
			y=[all_cluster_counts[cl] * archetype_props[i] for i, cl in enumerate(all_clusters)],
			name="DeFi Power Users Archetype Proportion",
			marker_color='green',
			opacity=0.8
		))
		fig.update_layout(
			title="Cluster Distribution: All Wallets with DeFi Power Users Archetype Proportion Overlay",
			xaxis_title="Cluster Label",
			yaxis_title="Wallet Count",
			barmode='overlay',
			legend_title="Wallet Group"
		)
		if save_dir:
			os.makedirs(save_dir, exist_ok=True)
			fig.write_html(os.path.join(save_dir, "all_vs_defi_power_users_cluster_distribution.html"))
		if show:
			fig.show()

	# 2. For each analytic score, plot density overlays: all data vs. DeFi Power Users
	os.makedirs(save_dir, exist_ok=True) if save_dir else None
	figs = []
	for col in columns:
		lower = df[col].quantile(lower_percentile / 100)
		upper = df[col].quantile(upper_percentile / 100)
		all_data = df[(df[col] >= lower) & (df[col] <= upper)][col]
		defi_power_data = defi_power_users_wallets[(defi_power_users_wallets[col] >= lower) & (defi_power_users_wallets[col] <= upper)][col]
		fig = go.Figure()
		fig.add_trace(go.Histogram(
			x=all_data,
			nbinsx=bins,
			name="All Wallets",
			marker_color='blue',
			opacity=0.5,
			histnorm='probability density'
		))
		fig.add_trace(go.Histogram(
			x=defi_power_data,
			nbinsx=bins,
			name="DeFi Power Users Wallets",
			marker_color='green',
			opacity=0.7,
			histnorm='probability density'
		))
		fig.update_layout(
			barmode='overlay',
			title=f"Density of {col}: All Wallets vs. DeFi Power Users Wallets",
			xaxis_title=col,
			yaxis_title="Density",
			legend_title="Wallet Group",
			bargap=0.1
		)
		if save_dir:
			fig.write_html(os.path.join(save_dir, f"density_{col}_all_vs_defi_power_users.html"))
		if show:
			fig.show()
		figs.append(fig)
	if return_fig:
		return figs[0] if len(figs) == 1 else figs


def plot_omnichain_explorers_analytic_score_distributions(
	df: pd.DataFrame,
	columns: Optional[List[str]] = None,
	bins: int = 30,
	save_dir: Optional[str] = None,
	show: bool = True,
	lower_percentile: float = 0,
	upper_percentile: float = 100,
	return_fig: bool = False
) -> 'Optional[go.Figure]':
	"""
	Plots analytic score distributions for the whole dataset, overlays the distribution for wallets with CROSS_DOMAIN_ENGAGEMENT_SCORE >= 0.1.
	Also produces visualisations of the cluster distribution (activity_cluster_label) for these wallets.
	Plots are clearly labelled as "Omnichain Explorers Archetype".
	"""
	if columns is None:
		columns = ANALYTIC_SCORE_COLUMNS
	# Filter for Omnichain Explorers archetype
	omnichain_explorers_wallets = df[df["CROSS_DOMAIN_ENGAGEMENT_SCORE"] >= 0.1]

	# 1. Plot cluster distribution with archetype proportions overlayed
	if "activity_cluster_label" in df.columns:
		all_cluster_counts = df["activity_cluster_label"].value_counts().sort_index()
		archetype_cluster_counts = omnichain_explorers_wallets["activity_cluster_label"].value_counts().sort_index()
		all_clusters = all_cluster_counts.index.tolist()
		archetype_props = [archetype_cluster_counts.get(cl, 0) / all_cluster_counts[cl] if cl in all_cluster_counts and all_cluster_counts[cl] > 0 else 0 for cl in all_clusters]
		fig = go.Figure()
		fig.add_trace(go.Bar(
			x=all_clusters,
			y=all_cluster_counts.values,
			name="All Wallets",
			marker_color='blue',
			opacity=0.5
		))
		fig.add_trace(go.Bar(
			x=all_clusters,
			y=[all_cluster_counts[cl] * archetype_props[i] for i, cl in enumerate(all_clusters)],
			name="Omnichain Explorers Archetype Proportion",
			marker_color='orange',
			opacity=0.8
		))
		fig.update_layout(
			title="Cluster Distribution: All Wallets with Omnichain Explorers Archetype Proportion Overlay",
			xaxis_title="Cluster Label",
			yaxis_title="Wallet Count",
			barmode='overlay',
			legend_title="Wallet Group"
		)
		if save_dir:
			os.makedirs(save_dir, exist_ok=True)
			fig.write_html(os.path.join(save_dir, "all_vs_omnichain_explorers_cluster_distribution.html"))
		if show:
			fig.show()

	# 2. For each analytic score, plot density overlays: all data vs. Omnichain Explorers
	os.makedirs(save_dir, exist_ok=True) if save_dir else None
	figs = []
	for col in columns:
		lower = df[col].quantile(lower_percentile / 100)
		upper = df[col].quantile(upper_percentile / 100)
		all_data = df[(df[col] >= lower) & (df[col] <= upper)][col]
		omni_data = omnichain_explorers_wallets[(omnichain_explorers_wallets[col] >= lower) & (omnichain_explorers_wallets[col] <= upper)][col]
		fig = go.Figure()
		fig.add_trace(go.Histogram(
			x=all_data,
			nbinsx=bins,
			name="All Wallets",
			marker_color='blue',
			opacity=0.5,
			histnorm='probability density'
		))
		fig.add_trace(go.Histogram(
			x=omni_data,
			nbinsx=bins,
			name="Omnichain Explorers Wallets",
			marker_color='orange',
			opacity=0.7,
			histnorm='probability density'
		))
		fig.update_layout(
			barmode='overlay',
			title=f"Density of {col}: All Wallets vs. Omnichain Explorers Wallets",
			xaxis_title=col,
			yaxis_title="Density",
			legend_title="Wallet Group",
			bargap=0.1
		)
		if save_dir:
			fig.write_html(os.path.join(save_dir, f"density_{col}_all_vs_omnichain_explorers.html"))
		if show:
			fig.show()
		figs.append(fig)
	if return_fig:
		return figs[0] if len(figs) == 1 else figs


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
	# Plot overlays and cluster analysis for Stable High-Value Wallets Archetype
	print("Saving and displaying Stable High-Value Wallets Archetype overlays and cluster analysis to artifacts/Dashboards ...")
	plot_stable_high_value_analytic_score_distributions(df, save_dir=output_dir, show=True)
	# Plot overlays and cluster analysis for Erratic Speculator Archetype
	print("Saving and displaying Erratic Speculator Archetype overlays and cluster analysis to artifacts/Dashboards ...")
	plot_erratic_speculator_analytic_score_distributions(df, save_dir=output_dir, show=True)
	# Plot overlays and cluster analysis for DeFi Power Users Archetype
	print("Saving and displaying DeFi Power Users Archetype overlays and cluster analysis to artifacts/Dashboards ...")
	plot_defi_power_users_analytic_score_distributions(df, save_dir=output_dir, show=True)
	# Plot overlays and cluster analysis for Omnichain Explorers Archetype
	print("Saving and displaying Omnichain Explorers Archetype overlays and cluster analysis to artifacts/Dashboards ...")
	plot_omnichain_explorers_analytic_score_distributions(df, save_dir=output_dir, show=True)

