# HDBSCAN Parameter Optimization Documentation

## Overview

The `tune_hdbscan_parameters.py` module provides comprehensive functionality for optimizing HDBSCAN clustering parameters when used with UMAP dimensionality reduction. It implements multiple optimization strategies including grid search, random search, and Bayesian optimization.

## Features

### 🔍 **Multiple Optimization Methods**
- **Grid Search**: Exhaustive search over predefined parameter spaces
- **Random Search**: Efficient exploration of parameter space with random sampling
- **Bayesian Optimization**: Intelligent parameter search using Gaussian processes

### 📊 **Comprehensive Evaluation**
- **Multi-metric Scoring**: Combines silhouette score, Calinski-Harabasz, Davies-Bouldin
- **Composite Scoring**: Weighted combination with penalties/bonuses for cluster characteristics
- **Stability Assessment**: Includes HDBSCAN-specific stability metrics

### ⚙️ **Flexible Parameter Spaces**
- **Quick**: Small parameter space for fast testing (9-12 combinations)
- **Default**: Balanced exploration (80-160 combinations)
- **Comprehensive**: Extensive parameter coverage (525-2625 combinations)
- **Fine-tune**: Focused optimization around promising regions

### 📈 **Rich Output and Visualization**
- **Detailed Reports**: Markdown reports with top configurations
- **JSON Results**: Complete results for programmatic analysis
- **Visualization Plots**: Parameter space exploration and performance analysis
- **Config Integration**: Automatic update of configuration files

## Quick Start

### Basic Usage

```python
from source_code_package.models.clustering_functionality.tune_hdbscan_parameters import optimize_hdbscan_parameters

# Run quick optimization
results = optimize_hdbscan_parameters(
    method="grid_search",
    search_type="quick",
    save_results=True,
    output_dir="tuning_results"
)

# Get best parameters
best_umap_params = results[0]['umap_params']
best_hdbscan_params = results[0]['hdbscan_params']
print(f"Best score: {results[0]['composite_score']:.2f}")
```

### Advanced Usage

```python
from source_code_package.models.clustering_functionality.tune_hdbscan_parameters import HDBSCANParameterOptimizer

# Initialize optimizer
optimizer = HDBSCANParameterOptimizer(
    data_path="data/my_data.csv",
    config_path="config/config_cluster.yaml"
)

# Run comprehensive grid search
results = optimizer.grid_search_optimization(
    search_type="comprehensive",
    max_workers=4,  # Parallel processing
    save_results=True
)

# Update configuration with best parameters
optimizer.update_config_file()
```

## Optimization Methods

### 1. Grid Search (`grid_search_optimization`)

**Best for**: Systematic exploration, reproducible results
**Time**: Longer but thorough
**Recommended**: When you have computational resources and want comprehensive results

```python
results = optimizer.grid_search_optimization(
    search_type="default",  # or "quick", "comprehensive", "fine_tune"
    max_workers=4,          # Parallel processing
    save_results=True,
    output_dir="grid_search_results"
)
```

### 2. Random Search (`random_search_optimization`)

**Best for**: Large parameter spaces, time constraints
**Time**: Faster, configurable
**Recommended**: When grid search is too slow

```python
results = optimizer.random_search_optimization(
    n_trials=100,           # Number of random trials
    max_workers=4,
    save_results=True,
    output_dir="random_search_results"
)
```

### 3. Bayesian Optimization (`bayesian_optimization`)

**Best for**: Expensive evaluations, intelligent search
**Time**: Adaptive
**Recommended**: For fine-tuning around good regions

```python
results = optimizer.bayesian_optimization(
    n_initial=10,           # Initial random evaluations
    n_iterations=40,        # Optimization iterations
    save_results=True,
    output_dir="bayesian_results"
)
```

## Parameter Spaces

### Search Types

| Search Type | UMAP Combinations | HDBSCAN Combinations | Total | Use Case |
|-------------|-------------------|---------------------|-------|----------|
| `quick` | 6 | 12 | 72 | Fast testing |
| `default` | 24 | 20 | 480 | Balanced exploration |
| `comprehensive` | 75 | 105 | 7,875 | Thorough search |
| `fine_tune` | 36 | 28 | 1,008 | Refinement |

### UMAP Parameters

```python
# Example parameter spaces
umap_params = {
    'n_components': [5, 10, 15, 20],      # Dimensionality reduction target
    'n_neighbors': [15, 30, 50],          # Local neighborhood size
    'min_dist': [0.01, 0.1, 0.5],        # Minimum distance between points
    'metric': ['euclidean', 'cosine']     # Distance metric
}
```

### HDBSCAN Parameters

```python
# Example parameter spaces
hdbscan_params = {
    'min_cluster_size': [50, 100, 200],           # Minimum cluster size
    'min_samples': [10, 25, 40],                  # Noise sensitivity
    'metric': ['euclidean', 'cosine'],            # Distance metric
    'cluster_selection_method': ['eom'],          # Selection method
    'cluster_selection_epsilon': [0.0, 0.01]     # Distance threshold
}
```

## Evaluation Metrics

### Composite Scoring

The optimization uses a weighted composite score combining multiple metrics:

```python
composite_score = (
    40.0 * silhouette_score +                    # Quality of separation
    0.002 * min(calinski_harabasz, 50000) +      # Cluster dispersion
    -10.0 * davies_bouldin_score +               # Cluster compactness
    cluster_penalty +                            # Preference for 3-10 clusters
    noise_penalty +                              # Preference for <5% noise
    10.0 * stability_ratio                       # HDBSCAN stability
)
```

### Individual Metrics

- **Silhouette Score** (-1 to 1): Measures cluster separation quality
- **Calinski-Harabasz Score** (>0): Measures cluster dispersion
- **Davies-Bouldin Score** (>0): Measures cluster compactness (lower better)
- **Stability Ratio** (0 to 1): HDBSCAN cluster persistence measure

## Output Files

### 1. Detailed Results (JSON)
```
detailed_results_[method]_[timestamp].json
```
Complete results for programmatic analysis

### 2. Optimization Report (Markdown)
```
optimization_report_[method]_[timestamp].md
```
Human-readable report with top configurations and statistics

### 3. Visualization (PNG)
```
optimization_visualization_[method]_[timestamp].png
```
6-panel visualization showing:
- Score distribution
- Silhouette vs composite score
- Number of clusters vs score  
- Noise percentage vs score
- Parameter space exploration
- Top 10 configurations

## Configuration Integration

### Automatic Config Update

```python
# Update config file with best parameters
optimizer.update_config_file()
```

### Manual Parameter Application

```python
# Get best parameters
best_umap, best_hdbscan = optimizer.get_best_parameters()

# Apply manually to your pipeline
reducer = umap.UMAP(**best_umap)
clusterer = hdbscan.HDBSCAN(**best_hdbscan)
```

## Performance Considerations

### Parallel Processing

```python
# Enable parallel processing for faster evaluation
results = optimizer.grid_search_optimization(
    max_workers=4,  # Use 4 CPU cores
    search_type="comprehensive"
)
```

### Memory Usage

- Each evaluation requires fitting UMAP and HDBSCAN models
- Memory usage scales with data size and n_components
- Consider using `search_type="quick"` for initial exploration

### Time Estimates

| Search Type | Combinations | Est. Time (4 cores) |
|-------------|--------------|-------------------|
| quick | 72 | 5-10 minutes |
| default | 480 | 30-60 minutes |
| comprehensive | 7,875 | 8-16 hours |

## Best Practices

### 1. Start Small
```python
# Begin with quick search
quick_results = optimize_hdbscan_parameters(
    method="grid_search", 
    search_type="quick"
)
```

### 2. Progressive Refinement
```python
# Use fine_tune around promising regions
fine_results = optimize_hdbscan_parameters(
    method="grid_search",
    search_type="fine_tune"
)
```

### 3. Resource Management
```python
# Use appropriate parallelization
results = optimize_hdbscan_parameters(
    max_workers=min(4, os.cpu_count()),  # Don't exceed CPU count
    method="random_search",
    n_trials=100
)
```

### 4. Method Selection
- **Grid Search**: When you want comprehensive coverage
- **Random Search**: When parameter space is large
- **Bayesian**: When evaluations are expensive

## Example Workflows

### Workflow 1: Quick Exploration
```python
# 1. Quick exploration
quick_results = optimize_hdbscan_parameters(
    method="grid_search", search_type="quick"
)

# 2. Analyze results and refine
fine_results = optimize_hdbscan_parameters(
    method="grid_search", search_type="fine_tune"
)

# 3. Update configuration
optimizer = HDBSCANParameterOptimizer()
optimizer.results = fine_results
optimizer.update_config_file()
```

### Workflow 2: Comprehensive Search
```python
# 1. Random search for broad exploration
random_results = optimize_hdbscan_parameters(
    method="random_search", n_trials=200
)

# 2. Bayesian optimization for refinement
bayesian_results = optimize_hdbscan_parameters(
    method="bayesian", n_trials=50
)
```

### Workflow 3: Time-Constrained
```python
# Balance between coverage and time
results = optimize_hdbscan_parameters(
    method="random_search",
    n_trials=50,  # Adjust based on time available
    max_workers=4
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install scikit-learn hdbscan umap-learn matplotlib seaborn
   ```

2. **Memory Issues**: Reduce parameter space or data size
   ```python
   # Use smaller search space
   results = optimize_hdbscan_parameters(search_type="quick")
   ```

3. **No Successful Results**: Check data quality and parameter ranges
   ```python
   # Verify data loading
   optimizer = HDBSCANParameterOptimizer()
   data = optimizer.load_and_preprocess_data()
   print(f"Data shape: {data.shape}")
   ```

### Performance Issues

1. **Slow Evaluation**: Reduce data size or parameter space
2. **High Memory Usage**: Use fewer parallel workers
3. **Poor Results**: Check data preprocessing and parameter ranges

## Integration with Existing Pipeline

### Using with UMAP+HDBSCAN Pipeline

```python
# 1. Optimize parameters
results = optimize_hdbscan_parameters(update_config=True)

# 2. Run existing pipeline with optimized parameters
from source_code_package.models.clustering_functionality.HBDSCAN_cluster import run_flexible_hdbscan_pipeline

pipeline_results = run_flexible_hdbscan_pipeline(
    config_path="config/config_cluster.yaml",  # Now contains optimized parameters
    output_dir="optimized_clustering_results"
)
```

### Custom Parameter Application

```python
# Get optimized parameters
optimizer = HDBSCANParameterOptimizer()
results = optimizer.grid_search_optimization()
best_umap, best_hdbscan = optimizer.get_best_parameters()

# Apply to custom pipeline
import umap
import hdbscan

# Your data preprocessing
data = preprocess_for_clustering(data_path, config_path)

# Apply optimized UMAP
reducer = umap.UMAP(**best_umap)
umap_data = reducer.fit_transform(data)

# Apply optimized HDBSCAN
clusterer = hdbscan.HDBSCAN(**best_hdbscan)
labels = clusterer.fit_predict(umap_data)
```

This documentation provides comprehensive guidance for using the HDBSCAN parameter optimization functionality to improve clustering performance in your ML pipeline.
