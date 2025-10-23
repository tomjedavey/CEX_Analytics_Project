import sys
import os

# Get the project root directory (two levels up from this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Add project root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import the module
from Source_Code_Package.utils.dashboard_html import produce_dashboard_html

if __name__ == "__main__":
    # You can add arguments for input/output if needed
    produce_dashboard_html(
        data_path="data/processed_data/merged_analytic_scores.csv",
        output_path="artifacts/Dashboards/combined_dashboard.html"
    )
    print("Dashboard HTML created at artifacts/Dashboards/combined_dashboard.html")
