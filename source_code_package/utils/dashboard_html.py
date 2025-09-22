
import os
import pandas as pd
import tempfile
from . import dashboard_statistics, dashboard_visualisations

def _fig_to_html(fig, title=None):
	# Helper to get HTML string from a plotly figure or list of figures
	html = ""
	if isinstance(fig, list):
		for f in fig:
			html += _fig_to_html(f)
	else:
		html += fig.to_html(full_html=False, include_plotlyjs='cdn')
	return html

def produce_dashboard_html(
	data_path: str = "data/processed_data/merged_analytic_scores.csv",
	output_path: str = "artifacts/Dashboards/combined_dashboard.html",
	stats_columns=None,
	cluster_col: str = "activity_cluster_label",
	bins: int = 30,
	lower_percentile: float = 5,
	upper_percentile: float = 95
) -> None:
	"""
	Produce a combined HTML dashboard with statistics and visualisations.
	Calls the relevant functions from dashboard_statistics and dashboard_visualisations directly,
	ensuring reproducibility with new data/configurations.
	"""
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	df = pd.read_csv(data_path)


	# --- Dashboard Structure Explanation ---
	structure_html = """
	<div style='background:#f0f4f8;padding:18px 24px 18px 24px;border-radius:8px;margin-bottom:32px;'>
	<h2>Dashboard Structure Overview</h2>
	<ol>
		<li><b>Analytic Score Descriptive Statistics</b> <br><span style='font-size:0.95em;color:#555;'>with Analytic Score Distributions (All Data) visualisations directly below</span></li>
		<li><b>Clustering Results</b>
			<ul>
				<li>Clustering summary statistics</li>
				<li>Clustering results visualisation</li>
			</ul>
		</li>
		<li><b>Cluster Analysis</b>
			<ul>
				<li>For each cluster: descriptive statistics, then density visualisations, with subheadings for each cluster</li>
			</ul>
		</li>
		<li><b>Archetype Visualisations</b>
			<ul>
				<li>Erratic Speculator Wallets</li>
				<li>DeFi Power Users Wallets</li>
				<li>Omnichain Explorers Wallets</li>
			</ul>
		</li>
	</ol>
	</div>
	"""


	# --- Section 1: Analytic Score Descriptive Statistics and Distributions ---
	section1_html = "<h2>Analytic Score Descriptive Statistics</h2>"
	stats_df = dashboard_statistics.analytic_score_descriptive_stats(df, columns=stats_columns)
	section1_html += stats_df.to_html(classes="stats-table", border=1)
	section1_html += "<h3>Analytic Score Distributions (All Data)</h3>"
	for col in (stats_columns or dashboard_visualisations.ANALYTIC_SCORE_COLUMNS):
		fig = dashboard_visualisations.plot_analytic_score_distributions(
			df, columns=[col], bins=bins, save_dir=None, show=False,
			lower_percentile=5, upper_percentile=95, return_fig=True
		)
		section1_html += _fig_to_html(fig)

	# --- Section 2: Clustering Summary and Visualisation ---
	clustering_summary_path = "data/processed_data/clustering_results/clustering_summary.txt"
	clustering_img_path = "data/processed_data/clustering_results/hdbscan_clustering_results.png"
	section2_html = ""
	if os.path.exists(clustering_summary_path):
		with open(clustering_summary_path, "r") as f:
			summary_text = f.read()
		section2_html += "<h2>Clustering Summary</h2>"
		section2_html += f'<pre style="background:#f8f8f8;padding:12px;border-radius:6px;">{summary_text}</pre>'
	else:
		section2_html += "<h2>Clustering Summary</h2><p><em>Summary file not found.</em></p>"
	import base64
	if os.path.exists(clustering_img_path):
		section2_html += "<h2>Clustering Results Visualisation</h2>"
		with open(clustering_img_path, "rb") as img_f:
			img_data = img_f.read()
			img_base64 = base64.b64encode(img_data).decode("utf-8")
		section2_html += f'<img src="data:image/png;base64,{img_base64}" alt="HDBSCAN Clustering Results" style="max-width:100%;border:1px solid #ccc;padding:8px;background:#fff;">'
	else:
		section2_html += "<h2>Clustering Results Visualisation</h2><p><em>Image file not found.</em></p>"

	# --- Section 3: Cluster Stats and Density Visualisations by Cluster ---
	section3_html = "<h2>Analytic Score Stats by Cluster</h2>"
	cluster_stats = dashboard_statistics.analytic_score_stats_by_cluster(df, cluster_col=cluster_col, columns=stats_columns)
	clusters = [c for c in cluster_stats.keys() if c != -1]
	for cluster in clusters:
		section3_html += f"<h3>Cluster {cluster}</h3>"
		cdf = cluster_stats[cluster]
		section3_html += cdf.to_html(classes="stats-table", border=1)
		section3_html += f"<h4>Analytic Score Density Visualisations for Cluster {cluster}</h4>"
		for col in (stats_columns or dashboard_visualisations.ANALYTIC_SCORE_COLUMNS):
			# Overlay cluster distribution on whole dataset's distribution
			fig = dashboard_visualisations.plot_analytic_score_density_by_cluster(
				df, cluster_col=cluster_col, columns=[col], bins=bins, save_dir=None, show=False,
				lower_percentile=5, upper_percentile=95, return_fig=True
			)
			# Only show the plot for the current cluster (figs is a list)
			if isinstance(fig, list):
				# Find the figure for the current cluster
				cluster_fig = None
				for f in fig:
					if hasattr(f, 'layout') and f.layout.title and str(cluster) in str(f.layout.title):
						cluster_fig = f
						break
				if cluster_fig:
					section3_html += _fig_to_html(cluster_fig)
			else:
				section3_html += _fig_to_html(fig)

	# --- Section 4: Archetype Visualisations (in specified order) ---
	section4_html = "<h2>Archetype Visualisations</h2>"
	archetype_funcs = [
		(dashboard_visualisations.plot_erratic_speculator_analytic_score_distributions, "Erratic Speculator Wallets"),
		(dashboard_visualisations.plot_defi_power_users_analytic_score_distributions, "DeFi Power Users Wallets"),
		(dashboard_visualisations.plot_omnichain_explorers_analytic_score_distributions, "Omnichain Explorers Wallets")
	]
	for func, label in archetype_funcs:
		section4_html += f"<h3>{label}</h3>"
		for col in (stats_columns or dashboard_visualisations.ANALYTIC_SCORE_COLUMNS):
			fig = func(
				df, columns=[col], bins=bins, save_dir=None, show=False,
				lower_percentile=5, upper_percentile=95, return_fig=True
			)
			section4_html += _fig_to_html(fig)

	# --- Combine all sections ---
	html = f"""
	<html>
	<head>
		<title>Combined Analytic Dashboard</title>
		<style>
			body {{ font-family: Arial, sans-serif; margin: 40px; }}
			.stats-table {{ border-collapse: collapse; margin-bottom: 40px; }}
			.stats-table th, .stats-table td {{ border: 1px solid #ccc; padding: 6px 10px; }}
			h2 {{ color: #2c3e50; }}
			h3 {{ color: #34495e; }}
		</style>
	</head>
	<body>
		<h1>Analytic Dashboard</h1>
		{structure_html}
		{section1_html}
		{section2_html}
		{section3_html}
		{section4_html}
	</body>
	</html>
	"""
	with open(output_path, "w") as f:
		f.write(html)
