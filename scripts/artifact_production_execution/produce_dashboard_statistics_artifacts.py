"""
Execution script to generate and save dashboard statistics artifacts.
"""

import os
import pandas as pd
from source_code_package.utils import dashboard_statistics as ds

# Output directory for dashboard artifacts
OUTPUT_DIR = "artifacts/Dashboards"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load merged analytic scores dataset
merged_df = ds.load_merged_analytic_scores()

# 1. Analytic score statistics (overall)
analytic_stats = ds.analytic_score_descriptive_stats(merged_df)
analytic_stats.to_csv(os.path.join(OUTPUT_DIR, "analytic_score_descriptive_stats.csv"))

# 2. Analytic score statistics by cluster
grouped_analytic_stats = ds.analytic_score_stats_by_cluster(merged_df)
for cluster, stats_df in grouped_analytic_stats.items():
    stats_df.to_csv(os.path.join(OUTPUT_DIR, f"analytic_score_stats_cluster_{cluster}.csv"))

# 3. All column statistics (overall)
all_col_stats = ds.all_column_descriptive_stats(merged_df)
all_col_stats.to_csv(os.path.join(OUTPUT_DIR, "all_column_descriptive_stats.csv"))

# 4. All column statistics by cluster
grouped_all_col_stats = ds.all_column_stats_by_cluster(merged_df)
for cluster, stats_df in grouped_all_col_stats.items():
    stats_df.to_csv(os.path.join(OUTPUT_DIR, f"all_column_stats_cluster_{cluster}.csv"))

print("Dashboard statistics artifacts saved to:", OUTPUT_DIR)

#** MOST LIKELY DELETE THIS FILE - THIS FUNCTIONALITY NEEDS TO BE TO BUILT INTO A DASHBOARD NOT JUST RAW FILES**