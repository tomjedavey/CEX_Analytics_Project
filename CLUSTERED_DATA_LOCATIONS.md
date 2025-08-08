# Clustered Data Output Locations - Complete Guide

## Overview

The interaction mode clustering pipeline saves clustered data in a completely separate directory structure from the original clustering pipeline, ensuring no conflicts or overwriting of existing results.

## Directory Structure

### Original Clustering Pipeline Outputs
**Location**: `clustering_output/`
```
clustering_output/
├── complete_pipeline/
│   └── hdbscan_results/
│       ├── cluster_labels.csv          # Original clustering labels
│       ├── clustered_data.csv          # Original clustered data
│       ├── clustering_summary.txt      # Original clustering metrics
│       └── hdbscan_clustering_results.png
├── flexible_pipeline/
├── standardized_pipeline/
├── hdbscan_only/
└── no_umap_example/
```

### NEW Interaction Mode Clustering Outputs
**Location**: `interaction_mode_results/`
```
interaction_mode_results/
├── main_clustering/                     # Main dataset (new_raw_data_polygon.csv)
│   └── hdbscan_results/
│       ├── cluster_labels.csv          # Cluster assignments for main dataset
│       ├── clustered_data.csv          # Full data with cluster labels
│       ├── clustering_summary.txt      # Clustering quality metrics
│       └── hdbscan_clustering_results.png # Visualization
├── cluster_0_clustering/                # Cluster 0 subset dataset
│   └── hdbscan_results/
│       ├── cluster_labels.csv          # Cluster assignments for cluster 0 subset
│       ├── clustered_data.csv          # Full data with cluster labels
│       ├── clustering_summary.txt      # Clustering quality metrics
│       └── hdbscan_clustering_results.png # Visualization
├── cluster_1_clustering/                # Cluster 1 subset dataset
│   └── hdbscan_results/
│       ├── cluster_labels.csv          # Cluster assignments for cluster 1 subset
│       ├── clustered_data.csv          # Full data with cluster labels
│       ├── clustering_summary.txt      # Clustering quality metrics
│       └── hdbscan_clustering_results.png # Visualization
├── interaction_mode_pipeline_summary.txt # Overall pipeline summary
└── interaction_mode_analysis_summary.png # Analysis visualization
```

## File Contents

### Cluster Labels Files
**Format**: CSV with cluster assignments
- `cluster_label`: Integer cluster ID (-1 for noise points)
- Row count matches input dataset size
- Example: 20,174 rows for main dataset

### Clustered Data Files  
⚠️ **Current Issue**: These files currently only contain cluster labels, not the full original data with cluster assignments.

**Should contain**: Original data columns + cluster_label column
**Currently contains**: Only cluster_label column

### Summary Files
- **Clustering quality metrics**: Silhouette score, Calinski-Harabasz score, Davies-Bouldin score
- **Cluster statistics**: Number of clusters, cluster sizes, noise point count
- **Configuration details**: HDBSCAN parameters used

## Dataset-Specific Results

### Main Dataset Results
- **File**: `interaction_mode_results/main_clustering/hdbscan_results/`
- **Source**: `new_raw_data_polygon.csv` (20,174 records)
- **Results**: 2 clusters, 86 noise points (0.4%)
- **Silhouette Score**: 0.458

### Cluster 0 Subset Results
- **File**: `interaction_mode_results/cluster_0_clustering/hdbscan_results/`
- **Source**: `new_raw_data_polygon_cluster_0.csv` (11,369 records)
- **Results**: 2 clusters, 0 noise points (0.0%)
- **Silhouette Score**: 0.103

### Cluster 1 Subset Results
- **File**: `interaction_mode_results/cluster_1_clustering/hdbscan_results/`
- **Source**: `new_raw_data_polygon_cluster_1.csv` (8,647 records)
- **Results**: 3 clusters, 0 noise points (0.0%)
- **Silhouette Score**: 0.644

## Accessing the Results

### Python Code Example
```python
import pandas as pd

# Load cluster labels for main dataset
main_labels = pd.read_csv('interaction_mode_results/main_clustering/hdbscan_results/cluster_labels.csv')

# Load cluster labels for cluster 0 subset
cluster_0_labels = pd.read_csv('interaction_mode_results/cluster_0_clustering/hdbscan_results/cluster_labels.csv')

# Load cluster labels for cluster 1 subset
cluster_1_labels = pd.read_csv('interaction_mode_results/cluster_1_clustering/hdbscan_results/cluster_labels.csv')

# Load original datasets and merge with cluster labels
main_data = pd.read_csv('data/raw_data/new_raw_data_polygon.csv')
main_data['cluster_label'] = main_labels['cluster_label']
```

### Command Line Access
```bash
# View cluster distribution for main dataset
cat interaction_mode_results/main_clustering/hdbscan_results/clustering_summary.txt

# Count clusters in each dataset
wc -l interaction_mode_results/*/hdbscan_results/cluster_labels.csv

# View overall pipeline results
cat interaction_mode_results/interaction_mode_pipeline_summary.txt
```

## Key Differences from Original Pipeline

### Separation
- **Original**: Uses `clustering_output/` directory
- **Interaction Mode**: Uses `interaction_mode_results/` directory
- **No conflicts**: Completely separate directory structures

### Multiple Datasets
- **Original**: Processes one dataset at a time
- **Interaction Mode**: Processes three related datasets simultaneously
- **Consistent config**: Same parameters across all three datasets

### Specialized Configuration
- **Original**: Uses `config_cluster.yaml`
- **Interaction Mode**: Uses `config_interaction_mode.yaml`
- **Feature focus**: Optimized for interaction mode detection

## Custom Output Directory

You can specify a custom output directory:

```bash
# Custom directory name
python3 scripts/clustering/run_interaction_mode_clustering.py --output-dir my_interaction_results

# Results will be saved to:
# my_interaction_results/main_clustering/
# my_interaction_results/cluster_0_clustering/
# my_interaction_results/cluster_1_clustering/
```

## Important Notes

1. **Data Preservation**: All results are preserved separately from original clustering outputs
2. **Reproducibility**: Each run creates timestamped results or overwrites previous runs in the same directory
3. **Configuration**: Uses event-focused features (DEX_EVENTS, GAMES_EVENTS, etc.) as specified in the updated config
4. **Analysis Ready**: Results can be immediately used with the analysis script: `scripts/clustering/analyze_interaction_mode_results.py`

## Next Steps for Score Development

The clustered data provides the foundation for interaction mode score computation:
1. **Cross-dataset comparison**: Compare clustering patterns between main dataset and subsets
2. **Stability analysis**: Measure consistency of cluster assignments
3. **Pattern detection**: Identify interaction mode signatures in cluster distributions
4. **Score formulation**: Develop scoring algorithms based on clustering results
