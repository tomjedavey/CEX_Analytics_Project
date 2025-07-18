# HDBSCAN Clustering Functionality Documentation

## Overview

This document explains the HDBSCAN clustering functionality built for the MLProject1. The implementation provides a comprehensive clustering solution that integrates with UMAP dimensionality reduction and includes extensive configuration options, quality evaluation metrics, and visualization capabilities.

## Files Created/Modified

### 1. `source_code_package/models/clustering_functionality/HBDSCAN_cluster.py`

**Purpose**: Core HDBSCAN clustering functionality with comprehensive features.

**Key Components**:

#### Configuration Management
- **`load_hdbscan_config()`**: Loads HDBSCAN parameters from YAML configuration file
- **`validate_hdbscan_config()`**: Validates configuration parameters and provides warnings/recommendations

#### Core Clustering Functions
- **`apply_hdbscan_clustering()`**: Main clustering function that applies HDBSCAN to input data
- **`predict_cluster_membership()`**: Predicts cluster membership for new data points

#### Quality Evaluation
- **`evaluate_clustering_quality()`**: Computes multiple clustering quality metrics:
  - Silhouette Score (measures cluster separation)
  - Calinski-Harabasz Score (measures cluster dispersion)
  - Davies-Bouldin Score (measures cluster compactness)
  - HDBSCAN-specific metrics (cluster persistence, probability statistics)

#### Visualization
- **`plot_clustering_results()`**: Creates comprehensive visualizations including:
  - Cluster scatter plots
  - Density plots
  - Cluster size distributions
  - Summary statistics

#### Pipeline Integration
- **`hdbscan_clustering_pipeline()`**: Complete pipeline for HDBSCAN clustering with evaluation and visualization
- **`run_umap_hdbscan_pipeline()`**: End-to-end pipeline integrating UMAP and HDBSCAN

### 2. `source_code_package/config/config_cluster.yaml` (Modified)

**Purpose**: Configuration file containing HDBSCAN parameters.

**Key HDBSCAN Parameters Added**:

#### Core Parameters
- **`min_cluster_size`** (15): Minimum number of samples in a cluster
  - Controls the minimum size of clusters
  - Larger values = fewer, larger clusters
  - Typical range: 5-100

- **`min_samples`** (5): Number of samples in neighborhood for a point to be considered core
  - Controls noise sensitivity
  - Should be ≤ min_cluster_size
  - Typical range: 1-30

- **`cluster_selection_epsilon`** (0.0): Distance threshold for cluster selection
  - 0.0 = automatic selection
  - Higher values = fewer clusters
  - Use with caution

#### Distance and Algorithm Parameters
- **`metric`** ("euclidean"): Distance metric for clustering
  - Options: "euclidean", "manhattan", "cosine", "minkowski"
  - Choose based on data characteristics

- **`algorithm`** ("best"): Algorithm implementation to use
  - "best" = automatic selection
  - Other options for specific performance needs

- **`alpha`** (1.0): Regularization strength for soft clustering
  - Higher values = softer cluster boundaries
  - Typical range: 0.1-2.0

#### Cluster Selection Parameters
- **`cluster_selection_method`** ("eom"): Method for selecting clusters
  - "eom" = Excess of Mass (recommended)
  - "leaf" = select leaf clusters

- **`allow_single_cluster`** (false): Whether to allow single cluster results
  - Usually false to avoid trivial clustering

- **`max_cluster_size`** (0): Maximum cluster size (0 = no limit)
  - Useful for preventing very large clusters

#### Performance Parameters
- **`prediction_data`** (true): Generate data for predicting new points
  - Enable for real-time prediction capabilities
  - Slightly increases memory usage

- **`core_dist_n_jobs`** (-1): Number of parallel jobs
  - -1 = use all available cores
  - Adjust based on system resources

- **`random_state`** (42): Random seed for reproducibility
  - Ensures consistent results across runs

### 3. `scripts/clustering/run_hdbscan_clustering.py`

**Purpose**: Example script demonstrating how to use the HDBSCAN functionality from the scripts folder.

**Key Features**:
- Configuration validation
- Clustering with existing UMAP data
- Complete UMAP + HDBSCAN pipeline
- Error handling and user feedback

### 4. `pyproject.toml` (Modified)

**Purpose**: Added required dependencies for HDBSCAN clustering.

**Dependencies Added**:
- `hdbscan>=0.8.29`: Core HDBSCAN algorithm
- `umap-learn>=0.5.0`: UMAP dimensionality reduction
- `pyyaml>=6.0`: YAML configuration file parsing

## How to Use the Functionality

### 1. Basic Usage from Scripts

```python
# Import the functionality
from source_code_package.models.clustering_functionality.HBDSCAN_cluster import (
    hdbscan_clustering_pipeline,
    run_umap_hdbscan_pipeline
)

# Option 1: Cluster existing UMAP data
results = hdbscan_clustering_pipeline(
    umap_data=your_umap_data,
    config_path="path/to/config_cluster.yaml",
    evaluate_quality=True,
    create_visualizations=True,
    save_results=True,
    output_dir="results"
)

# Option 2: Complete pipeline from raw data
results = run_umap_hdbscan_pipeline(
    data_path="path/to/raw_data.csv",
    config_path="path/to/config_cluster.yaml",
    output_dir="complete_results"
)
```

### 2. Configuration Customization

Modify the HDBSCAN section in `config_cluster.yaml`:

```yaml
hdbscan:
  min_cluster_size: 20        # Increase for fewer, larger clusters
  min_samples: 10            # Increase for less noise sensitivity
  metric: "cosine"           # Use cosine distance for text/high-dim data
  cluster_selection_method: "eom"  # Keep as "eom" for best results
```

### 3. Quality Evaluation

The functionality automatically computes multiple quality metrics:

- **Silhouette Score**: Higher is better (range: -1 to 1)
  - > 0.5 = Good clustering
  - 0.2-0.5 = Reasonable clustering
  - < 0.2 = Poor clustering

- **Calinski-Harabasz Score**: Higher is better
  - No fixed range, compare different configurations

- **Davies-Bouldin Score**: Lower is better
  - Values close to 0 indicate better clustering

### 4. Visualization Outputs

The functionality creates several visualizations:
- **Cluster Scatter Plot**: Shows clusters in 2D space
- **Density Plot**: Shows cluster density distribution
- **Cluster Size Distribution**: Bar chart of cluster sizes
- **Statistics Summary**: Text summary of clustering results

## Parameter Tuning Guide

### For Different Data Types:

**High-dimensional data (many features)**:
- Use `metric: "cosine"` or `metric: "manhattan"`
- Increase `min_cluster_size` to 20-50
- Set `min_samples` to 5-15

**Low-dimensional data (few features)**:
- Use `metric: "euclidean"`
- Keep `min_cluster_size` around 10-20
- Set `min_samples` to 3-10

**Noisy data**:
- Increase `min_samples` to reduce noise sensitivity
- Increase `min_cluster_size` for more stable clusters
- Consider using `cluster_selection_epsilon` > 0

**Clean data**:
- Use default parameters as starting point
- Decrease `min_cluster_size` for more fine-grained clusters
- Keep `min_samples` low (3-5)

### Performance Optimization:

**For large datasets**:
- Set `core_dist_n_jobs: -1` to use all CPU cores
- Consider using `algorithm: "boruvka_kdtree"` for better performance
- Increase `leaf_size` to 50-100

**For small datasets**:
- Use default `algorithm: "best"`
- Keep `leaf_size` at default (40)
- Single-core processing is usually sufficient

## Integration with Existing Project

The HDBSCAN functionality is designed to integrate seamlessly with the existing project structure:

1. **Config Integration**: Uses the same `config_cluster.yaml` file as UMAP
2. **Data Flow**: Accepts UMAP output directly as input
3. **Output Format**: Saves results in consistent CSV format
4. **Error Handling**: Comprehensive error handling and validation
5. **Logging**: Detailed progress information and warnings

## Example Workflow

1. **Data Preprocessing**: Use existing preprocessing functionality
2. **UMAP Reduction**: Apply UMAP dimensionality reduction
3. **HDBSCAN Clustering**: Apply HDBSCAN to UMAP output
4. **Evaluation**: Assess clustering quality using multiple metrics
5. **Visualization**: Create plots to understand results
6. **Prediction**: Use trained clusterer for new data points

## Error Handling

The functionality includes comprehensive error handling:
- **Configuration validation**: Checks parameter validity
- **Data validation**: Ensures input data is properly formatted
- **Runtime errors**: Graceful handling of clustering failures
- **Memory management**: Warnings for large datasets

## Future Enhancements

Potential improvements that could be added:
- **Hyperparameter optimization**: Automated parameter tuning
- **Interactive visualizations**: Web-based cluster exploration
- **Streaming clustering**: Real-time clustering of new data
- **Ensemble methods**: Combining multiple clustering results
- **Advanced metrics**: Additional clustering quality measures

## Troubleshooting

**Common Issues**:

1. **No clusters found**: Decrease `min_cluster_size` or `min_samples`
2. **Too many small clusters**: Increase `min_cluster_size`
3. **High noise percentage**: Increase `min_samples` or adjust `metric`
4. **Poor quality scores**: Try different `metric` or tune parameters
5. **Memory issues**: Reduce `core_dist_n_jobs` or use smaller datasets

**Performance Issues**:
- Large datasets: Use `algorithm: "boruvka_kdtree"` and increase `leaf_size`
- Slow clustering: Reduce `min_samples` or use fewer CPU cores
- Memory usage: Set `prediction_data: false` if not needed

This implementation provides a robust, configurable, and well-documented HDBSCAN clustering solution that integrates seamlessly with the existing UMAP functionality and project structure.
