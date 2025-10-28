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
sys.path.append(os.path.join(os.path.dirname(__file__), 'source_code_package', '..'))

# Importing the neccessary function to prepare cluster datasets as csv files from cluster_datasets.py in source code package.
from source_code_package.data.cluster_datasets import prepare_cluster_datasets

def main():
    """
    Main function to execute cluster dataset preparation.
    """
    print("CLUSTER DATASET PREPARATION")
    print("=" * 50)
    
    # Robustly resolve project root (parent of parent of this script)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    print(f"[DEBUG] project_root: {project_root}")
    original_data_path = os.path.join(project_root, 'data/raw_data/new_raw_data_polygon.csv')
    clustering_results_path = os.path.join(project_root, 'data/processed_data/clustering_results/cluster_labels.csv')
    output_directory = os.path.join(project_root, 'data/processed_data/cluster_datasets')
    print(f"[DEBUG] original_data_path: {original_data_path}")
    print(f"[DEBUG] clustering_results_path: {clustering_results_path}")
    print(f"[DEBUG] output_directory: {output_directory}")

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
