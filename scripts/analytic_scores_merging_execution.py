import sys
import os

# Add the project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from Source_Code_Package.utils.analytic_score_merging import merge_analytic_scores

if __name__ == "__main__":
    merge_analytic_scores()
    print("Merged analytic scores CSV created at data/processed_data/merged_analytic_scores.csv")
