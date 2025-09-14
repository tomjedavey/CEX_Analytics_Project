import sys
import os

# Ensure the utils package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '../source_code_package/utils'))

from source_code_package.utils.dashboard_html import produce_dashboard_html

if __name__ == "__main__":
    # You can add arguments for input/output if needed
    produce_dashboard_html(
        data_path="data/processed_data/merged_analytic_scores.csv",
        output_path="artifacts/Dashboards/combined_dashboard.html"
    )
    print("Dashboard HTML created at artifacts/Dashboards/combined_dashboard.html")
