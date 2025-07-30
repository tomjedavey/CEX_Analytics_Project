#!/usr/bin/env python3
"""
Execution script for preparing cluster-specific datasets for analytic algorithms + scores.
Utilises the results of UMAP + Clustering pipeline with the current configurations to produce clustered data from new_raw_data_polygon.csv.

This script serves as a simple execution wrapper around the core functionality
in source_code_package.data.cluster_datasets module.

Author: Tom Davey
Date: July 2025
"""

import os
import sys

# Add source_code_package to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'source_code_package'))

try:
    from source_code_package.data.cluster_datasets import prepare_cluster_datasets
except ImportError:
    # Alternative import approach
    current_dir = os.path.dirname(__file__)
    source_package_path = os.path.join(current_dir, 'source_code_package')
    sys.path.insert(0, source_package_path)
    from data.cluster_datasets import prepare_cluster_datasets


def main():
    """
    Main function to execute cluster dataset preparation.
    """
    print("CLUSTER DATASET PREPARATION")
    print("=" * 50)
    
    # Define paths relative to project root
    project_root = os.path.dirname(os.path.dirname(__file__))  # Go up two levels from scripts/
    original_data_path = os.path.join(project_root, 'data/raw_data/new_raw_data_polygon.csv')
    clustering_results_path = os.path.join(project_root, 'clustering_output/flexible_pipeline/hdbscan_results/cluster_labels.csv')
    output_directory = os.path.join(project_root, 'data/raw_data/cluster_datasets')
    
    # Execute the preparation pipeline
    results = prepare_cluster_datasets(
        original_data_path=original_data_path,
        clustering_results_path=clustering_results_path,
        output_directory=output_directory,
        noise_strategy='exclude'
    )
    
    # Report results
    if results['success']:
        print("\n" + "=" * 50)
        print("CLUSTER DATASET PREPARATION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"\nDatasets created in: {results['output_directory']}")
        print("Ready for AS_1 feature engineering and model training.")
        return True
    else:
        print(f"\nERROR: {results['error']}")
        print("Cluster dataset preparation failed. Check the error above.")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
