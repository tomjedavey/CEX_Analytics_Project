# Interaction Mode Score HDBSCAN Clustering Pipeline Documentation

## Overview

The Interaction Mode Score HDBSCAN Clustering Pipeline is a specialized clustering system designed to process multiple datasets simultaneously to generate data for computing interaction mode scores. This pipeline builds upon the existing HDBSCAN clustering functionality and extends it to handle multiple related datasets with consistent configurations.

## Purpose

This pipeline was created to:
1. **Process Multiple Datasets**: Run HDBSCAN clustering on three different versions of the same underlying data
2. **Generate Interaction Mode Data**: Produce clustering results that can be used to compute interaction mode scores
3. **Maintain Consistency**: Use identical preprocessing and clustering configurations across all datasets
4. **Enable Score Computation**: Create the necessary foundation for building interaction mode scoring algorithms

## Datasets Processed

The pipeline processes three related datasets:

1. **Main Dataset** (`new_raw_data_polygon.csv`)
   - Complete dataset with all records
   - 20,174 data points, 22 columns
   - Used as the baseline for interaction mode analysis

2. **Cluster 0 Subset** (`new_raw_data_polygon_cluster_0.csv`)
   - Subset containing records from a specific cluster
   - 11,369 data points, 22 columns
   - Represents one interaction mode category

3. **Cluster 1 Subset** (`new_raw_data_polygon_cluster_1.csv`)
   - Subset containing records from another specific cluster
   - 8,647 data points, 22 columns
   - Represents another interaction mode category

## Configuration

The pipeline uses a specialized configuration file (`config_interaction_mode.yaml`) that includes:

### Key Features Selection
The pipeline focuses on four key features for interaction mode detection:
- `TX_PER_MONTH`: Transaction frequency per month
- `PROTOCOL_DIVERSITY`: Diversity of protocols used
- `ACTIVE_DURATION_DAYS`: Duration of activity in days
- `TOTAL_TRANSFER_USD`: Total transfer amount in USD

### HDBSCAN Parameters
Optimized for interaction mode detection:
- **Min Cluster Size**: 100 (smaller than default to capture finer interaction patterns)
- **Min Samples**: 25 (conservative setting for stable clusters)
- **Metric**: Euclidean distance
- **Cluster Selection Method**: Excess of Mass (EOM)

### Preprocessing
- **Log Transformation**: Applied to most features (excluding TX_PER_MONTH and ACTIVE_DURATION_DAYS)
- **Standard Scaling**: Applied to all features
- **UMAP Dimensionality Reduction**: Enabled with cosine metric for 2D projection

## Usage

### Basic Execution

Run the complete pipeline on all three datasets:
```bash
python3 scripts/clustering/run_interaction_mode_clustering.py
```

### Validation Only

Validate configuration and datasets without running clustering:
```bash
python3 scripts/clustering/run_interaction_mode_clustering.py --validate-only
```

### Process Specific Datasets

Process only selected datasets:
```bash
python3 scripts/clustering/run_interaction_mode_clustering.py --datasets main cluster_0
```

### Custom Output Directory

Specify a custom output directory:
```bash
python3 scripts/clustering/run_interaction_mode_clustering.py --output-dir my_interaction_results
```

### Force UMAP Settings

Override configuration UMAP setting:
```bash
# Force UMAP usage
python3 scripts/clustering/run_interaction_mode_clustering.py --force-umap

# Force direct HDBSCAN (no UMAP)
python3 scripts/clustering/run_interaction_mode_clustering.py --no-umap
```

## Results Structure

The pipeline generates the following output structure:

```
interaction_mode_results/
├── interaction_mode_pipeline_summary.txt     # Overall pipeline summary
├── main_clustering/                          # Main dataset results
│   └── hdbscan_results/
│       ├── cluster_labels.csv               # Cluster assignments
│       ├── clustered_data.csv              # Data with cluster labels
│       ├── clustering_summary.txt          # Clustering metrics
│       └── hdbscan_clustering_results.png  # Visualization
├── cluster_0_clustering/                     # Cluster 0 subset results
│   └── hdbscan_results/
│       ├── cluster_labels.csv
│       ├── clustered_data.csv
│       ├── clustering_summary.txt
│       └── hdbscan_clustering_results.png
└── cluster_1_clustering/                     # Cluster 1 subset results
    └── hdbscan_results/
        ├── cluster_labels.csv
        ├── clustered_data.csv
        ├── clustering_summary.txt
        └── hdbscan_clustering_results.png
```

## Clustering Results Summary

Based on the latest execution:

### Main Dataset Results
- **Clusters Found**: 2
- **Data Points**: 20,174
- **Noise Points**: 86 (0.4%)
- **Silhouette Score**: 0.458
- **Calinski-Harabasz Score**: 20,572.86
- **Davies-Bouldin Score**: 0.922

### Cluster 0 Subset Results
- **Clusters Found**: 2
- **Data Points**: 11,369
- **Noise Points**: 0 (0.0%)
- **Silhouette Score**: 0.103
- **Calinski-Harabasz Score**: 260.40
- **Davies-Bouldin Score**: 0.826

### Cluster 1 Subset Results
- **Clusters Found**: 3
- **Data Points**: 8,647
- **Noise Points**: 0 (0.0%)
- **Silhouette Score**: 0.644
- **Calinski-Harabasz Score**: 20,887.32
- **Davies-Bouldin Score**: 0.472

## Key Observations

1. **Different Cluster Patterns**: Each dataset exhibits different clustering patterns:
   - Main dataset: 2 clusters with minimal noise
   - Cluster 0 subset: 2 clusters, no noise, lower silhouette score (more homogeneous)
   - Cluster 1 subset: 3 clusters, no noise, highest silhouette score (most distinct clusters)

2. **Quality Metrics Variation**: The clustering quality varies significantly:
   - Cluster 1 subset shows the best separation (highest silhouette score: 0.644)
   - Cluster 0 subset shows the lowest internal separation (silhouette score: 0.103)
   - This suggests different interaction mode characteristics within each subset

3. **Noise Handling**: The smaller datasets (cluster subsets) show zero noise points, indicating more cohesive groupings within each subset

## Technical Implementation

### Core Components

1. **InteractionModeClusteringPipeline Class**: Main orchestrator that manages:
   - Configuration loading and validation
   - Dataset path management
   - Temporary configuration file creation for each dataset
   - Results aggregation and summary generation

2. **Integration with Existing Infrastructure**: 
   - Uses `simplified_clustering.py` for core clustering functionality
   - Leverages existing HDBSCAN and UMAP implementations
   - Maintains compatibility with existing configuration validation

3. **Error Handling and Validation**:
   - Comprehensive dataset validation before processing
   - Configuration validation using existing validators
   - Graceful error handling with detailed error messages

### Key Features

- **Parallel Processing Ready**: While currently sequential, the architecture supports future parallel processing
- **Flexible Dataset Selection**: Can process all datasets or specific subsets
- **Comprehensive Logging**: Detailed progress reporting and result summaries
- **Result Standardization**: Consistent output format across all datasets

## Future Enhancements

### Next Steps for Interaction Mode Score Development

1. **Score Computation Module**: Create algorithms to compute interaction mode scores based on:
   - Cross-dataset cluster comparisons
   - Cluster stability metrics
   - Pattern consistency analysis

2. **Feature Engineering**: Develop additional interaction-specific features:
   - Temporal interaction patterns
   - Cross-protocol interaction intensity
   - Behavioral consistency metrics

3. **Validation Framework**: Implement validation for interaction mode scores:
   - Cross-validation across datasets
   - Stability testing with different parameter sets
   - Correlation analysis with known interaction patterns

4. **Performance Optimization**: 
   - Parallel processing of multiple datasets
   - Memory optimization for large datasets
   - Caching of intermediate results

## Dependencies

- **Python Packages**: `pandas`, `numpy`, `scikit-learn`, `hdbscan`, `umap-learn`, `pyyaml`, `matplotlib`, `seaborn`
- **Existing Infrastructure**: HDBSCAN clustering functionality, UMAP dimensionality reduction, preprocessing pipeline
- **Configuration**: `config_interaction_mode.yaml` specialized configuration file

## Testing and Validation

A comprehensive testing framework is included:

### Validation Script
```bash
python3 scripts/clustering/test_interaction_mode_setup.py
```

This script validates:
- Configuration file loading and structure
- Dataset file existence and accessibility
- Module imports and dependencies
- Configuration parameter validation

## Maintenance Notes

- **Configuration Updates**: When modifying clustering parameters, update `config_interaction_mode.yaml`
- **Dataset Changes**: If input datasets change, update the dataset paths in the configuration
- **Parameter Tuning**: HDBSCAN parameters may need adjustment based on dataset characteristics
- **Performance Monitoring**: Monitor clustering quality metrics for significant changes

## Author and Date

- **Author**: Tom Davey
- **Created**: August 2025
- **Last Updated**: August 6, 2025
- **Version**: 1.0

This pipeline establishes the foundation for interaction mode score computation and can be extended as needed for specific scoring algorithms and analysis requirements.
