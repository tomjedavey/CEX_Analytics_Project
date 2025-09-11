import sys
import os

# Ensure the utils package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '../../source_code_package/utils'))

from source_code_package.utils.analytic_score_merging import merge_analytic_scores

if __name__ == "__main__":
    merge_analytic_scores()
    print("Merged analytic scores CSV created at artifacts/Dashboards/merged_analytic_scores.csv")
