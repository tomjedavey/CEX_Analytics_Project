import sys
import os

# Add the project root to Python path to ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

# Try importing with the correct case-sensitive directory name
try:
    from source_code_package.utils.dashboard_html import produce_dashboard_html
except ImportError:
    # Fallback for CI environments that might have lowercase
    sys.path.append(os.path.join(os.path.dirname(__file__), '../source_code_package/utils'))
    from source_code_package.utils.dashboard_html import produce_dashboard_html

if __name__ == "__main__":
    # You can add arguments for input/output if needed
    produce_dashboard_html(
        data_path="data/processed_data/merged_analytic_scores.csv",
        output_path="artifacts/Dashboards/combined_dashboard.html"
    )
    print("Dashboard HTML created at artifacts/Dashboards/combined_dashboard.html")
