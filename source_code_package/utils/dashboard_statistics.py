
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

ANALYTIC_SCORE_COLUMNS = [
	"BEHAVIOURAL_VOLATILITY_SCORE",
	"CROSS_DOMAIN_ENGAGEMENT_SCORE",
	"REVENUE_SCORE_PROXY",
	"DEX_EVENTS_SIGNED_DIST",
	"CEX_EVENTS_SIGNED_DIST",
	"BRIDGE_EVENTS_SIGNED_DIST",
	"DEFI_EVENTS_SIGNED_DIST"
]

def load_merged_analytic_scores(
	path: str = "data/processed_data/merged_analytic_scores.csv"
) -> pd.DataFrame:
	"""
	Load the merged analytic scores dataset from the processed_data folder.
	"""
	return pd.read_csv(path)

def analytic_score_descriptive_stats(
	df: pd.DataFrame,
	columns: Optional[List[str]] = None
) -> pd.DataFrame:
	"""
	Produce descriptive statistics for specified analytic score columns.
	"""
	if columns is None:
		columns = ANALYTIC_SCORE_COLUMNS
	stats = df[columns].describe().T
	stats["skewness"] = df[columns].skew()
	stats["kurtosis"] = df[columns].kurtosis()
	stats["missing_count"] = df[columns].isnull().sum()
	stats["missing_pct"] = (df[columns].isnull().sum() / len(df)) * 100
	stats["p5"] = df[columns].quantile(0.05)
	stats["p95"] = df[columns].quantile(0.95)
	stats["p99"] = df[columns].quantile(0.99)
	return stats

def analytic_score_stats_by_cluster(
	df: pd.DataFrame,
	cluster_col: str = "activity_cluster_label",
	columns: Optional[List[str]] = None
) -> Dict[Any, pd.DataFrame]:
	"""
	Produce descriptive statistics for analytic scores, separated by cluster.
	Returns a dict: {cluster_label: stats_df}
	"""
	if columns is None:
		columns = ANALYTIC_SCORE_COLUMNS
	cluster_stats = {}
	for cluster, group in df.groupby(cluster_col):
		cluster_stats[cluster] = analytic_score_descriptive_stats(group, columns)
	return cluster_stats

def all_column_descriptive_stats(
	df: pd.DataFrame
) -> pd.DataFrame:
	"""
	Produce descriptive statistics for all columns in the dataset.
	"""
	stats = df.describe(include='all').T
	# Add missing count and percent for all columns
	stats["missing_count"] = df.isnull().sum()
	stats["missing_pct"] = (df.isnull().sum() / len(df)) * 100
	return stats

def all_column_stats_by_cluster(
	df: pd.DataFrame,
	cluster_col: str = "activity_cluster_label"
) -> Dict[Any, pd.DataFrame]:
	"""
	Produce descriptive statistics for all columns, separated by cluster.
	Returns a dict: {cluster_label: stats_df}
	"""
	cluster_stats = {}
	for cluster, group in df.groupby(cluster_col):
		cluster_stats[cluster] = all_column_descriptive_stats(group)
	return cluster_stats
