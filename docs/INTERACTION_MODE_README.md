# Interaction Mode Score HDBSCAN Clustering Pipeline

## Quick Start

This pipeline runs HDBSCAN clustering on three datasets to generate data for interaction mode score computation.

### 1. Run the Complete Pipeline

```bash
python3 scripts/clustering/run_interaction_mode_clustering.py
```

### 2. Analyze Results

```bash
python3 scripts/clustering/analyze_interaction_mode_results.py
```

### 3. Validate Setup

```bash
python3 scripts/clustering/test_interaction_mode_setup.py
```

## What's New

### Configuration File
- **`source_code_package/config/config_interaction_mode.yaml`**: Specialized configuration for interaction mode clustering

### Execution Scripts
- **`scripts/clustering/run_interaction_mode_clustering.py`**: Main pipeline execution script
- **`scripts/clustering/test_interaction_mode_setup.py`**: Validation and testing script
- **`scripts/clustering/analyze_interaction_mode_results.py`**: Results analysis and visualization

### Documentation
- **`docs/Interaction_Mode_Clustering_Documentation.md`**: Comprehensive documentation

## Datasets Processed

1. **Main Dataset**: `new_raw_data_polygon.csv` (20,174 records)
2. **Cluster 0 Subset**: `new_raw_data_polygon_cluster_0.csv` (11,369 records)  
3. **Cluster 1 Subset**: `new_raw_data_polygon_cluster_1.csv` (8,647 records)

## Key Features

- **Consistent Configuration**: Same HDBSCAN parameters across all datasets
- **Specialized Feature Selection**: Focuses on interaction-relevant features
- **Comprehensive Results**: Clustering labels, metrics, and visualizations for each dataset
- **Analysis Framework**: Ready-to-use analysis tools for score development

## Results Summary

Latest execution results:
- **Main Dataset**: 2 clusters, 0.4% noise, Silhouette Score: 0.458
- **Cluster 0 Subset**: 2 clusters, 0% noise, Silhouette Score: 0.103  
- **Cluster 1 Subset**: 3 clusters, 0% noise, Silhouette Score: 0.644

## Next Steps

1. **Develop Scoring Algorithms**: Use the clustering results to compute interaction mode scores
2. **Feature Engineering**: Add interaction-specific features based on clustering insights
3. **Validation**: Implement validation framework for interaction mode scores
4. **Real-time Implementation**: Adapt for real-time interaction mode scoring

## Command Examples

```bash
# Validate only
python3 scripts/clustering/run_interaction_mode_clustering.py --validate-only

# Process specific datasets
python3 scripts/clustering/run_interaction_mode_clustering.py --datasets main cluster_0

# Force UMAP usage
python3 scripts/clustering/run_interaction_mode_clustering.py --force-umap

# Custom output directory
python3 scripts/clustering/run_interaction_mode_clustering.py --output-dir my_results

# Full help
python3 scripts/clustering/run_interaction_mode_clustering.py --help
```

## Integration

This pipeline integrates seamlessly with the existing HDBSCAN clustering infrastructure while providing specialized functionality for interaction mode analysis. All existing clustering tools and configurations remain fully functional.
