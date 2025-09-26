# HDBSCAN Clustering with Optional UMAP Dimensionality Reduction

## Overview

The HDBSCAN clustering functionality has been enhanced to support optional UMAP dimensionality reduction. You can now configure the pipeline to run HDBSCAN directly on preprocessed features without applying UMAP dimensionality reduction first.

## Configuration

### Enabling/Disabling UMAP

In your `config_cluster.yaml` file, you can now control whether UMAP dimensionality reduction is applied:

```yaml
# UMAP Dimensionality Reduction Configuration
umap:
  # Enable/disable UMAP dimensionality reduction
  enabled: true             # Set to false to skip UMAP and run HDBSCAN directly on preprocessed features
  
  # Core UMAP parameters (only used when enabled: true)
  n_neighbors: 30
  n_components: 2
  metric: "cosine"
  # ... other UMAP parameters
```

### Configuration Options

- **`enabled: true`** (default): Run complete UMAP + HDBSCAN pipeline
- **`enabled: false`**: Skip UMAP and run HDBSCAN directly on preprocessed features

## Usage

### Method 1: Flexible Pipeline (Recommended)

Use the new `run_flexible_hdbscan_pipeline()` function which automatically respects the UMAP configuration:

```python
from models.clustering_functionality.HBDSCAN_cluster import run_flexible_hdbscan_pipeline

# This will automatically check config and run with or without UMAP
results = run_flexible_hdbscan_pipeline(
    data_path="data/my_data.csv",
    config_path="config/config_cluster.yaml",
    output_dir="results"
)

# Check what was actually run
if results['umap_enabled']:
    print("Pipeline used UMAP + HDBSCAN")
else:
    print("Pipeline used HDBSCAN directly on preprocessed features")
```

### Method 2: Existing Functions

You can still use the existing functions:

- `run_umap_hdbscan_pipeline()`: Always runs UMAP + HDBSCAN (ignores the enabled flag)
- `hdbscan_clustering_pipeline()`: Runs HDBSCAN on provided data (no UMAP)

## Pipeline Behavior

### When UMAP is Enabled (`enabled: true`)
1. Load and preprocess data (log transformation, scaling)
2. Apply UMAP dimensionality reduction
3. Run HDBSCAN clustering on UMAP-reduced data
4. Evaluate and visualize results

### When UMAP is Disabled (`enabled: false`)
1. Load and preprocess data (log transformation, scaling)
2. Run HDBSCAN clustering directly on preprocessed features
3. Evaluate and visualize results

## Feature Selection

When UMAP is disabled, HDBSCAN will cluster using all the features that would normally be input to UMAP. You can still control which features are used by modifying the `include_columns` setting in the UMAP configuration:

```yaml
umap:
  enabled: false
  # Even when disabled, these columns determine which features to use for clustering
  include_columns:
    - "TX_PER_MONTH"
    - "TOKEN_DIVERSITY"
    - "PROTOCOL_DIVERSITY"
    # ... other features
```

## Advantages and Considerations

### Benefits of Disabling UMAP

1. **Interpretability**: Clustering results directly relate to original features
2. **Performance**: Faster execution (no dimensionality reduction step)
3. **Feature Preservation**: All feature information is retained
4. **Debugging**: Easier to understand which features drive clustering

### When to Disable UMAP

- When you have a small number of features (< 20)
- When feature interpretability is crucial
- When UMAP is losing important information
- For comparative analysis with and without dimensionality reduction

### When to Keep UMAP Enabled

- With high-dimensional data (> 50 features)
- When visualization is important (UMAP enables 2D plots)
- When features are highly correlated
- When noise reduction is beneficial

## Example Scripts

### Basic Usage
```bash
# Run the enhanced clustering script
python scripts/clustering/run_hdbscan_clustering.py
```

### No-UMAP Example
```bash
# Run the specific no-UMAP example
python scripts/clustering/run_hdbscan_no_umap_example.py
```

## Output Structure

The results structure varies based on whether UMAP was used:

### With UMAP Enabled
```python
{
    'umap_enabled': True,
    'umap_results': { ... },
    'hdbscan_results': { ... },
    'pipeline_info': {
        'n_original_features': 15,
        'n_reduced_features': 2,
        'umap_applied': True,
        # ...
    }
}
```

### With UMAP Disabled
```python
{
    'umap_enabled': False,
    'preprocessing_results': { ... },
    'hdbscan_results': { ... },
    'pipeline_info': {
        'n_original_features': 15,
        'n_reduced_features': 15,  # Same as original
        'umap_applied': False,
        # ...
    }
}
```

## Migration Guide

### Updating Existing Scripts

Replace calls to `run_umap_hdbscan_pipeline()` with `run_flexible_hdbscan_pipeline()`:

```python
# Old way (always uses UMAP)
results = run_umap_hdbscan_pipeline(config_path=config_path)

# New way (respects configuration)
results = run_flexible_hdbscan_pipeline(config_path=config_path)
```

### Backward Compatibility

- Existing configurations without the `enabled` flag will default to `enabled: true`
- All existing functions continue to work as before
- No breaking changes to existing pipelines

## Troubleshooting

### Common Issues

1. **ImportError for preprocessing**: Ensure `preprocess_cluster.py` is available
2. **Configuration errors**: Check YAML syntax and file paths
3. **Data path issues**: Verify data file exists at specified location

### Testing Configuration

Use the configuration validation function:

```python
from models.clustering_functionality.HBDSCAN_cluster import validate_hdbscan_config, load_hdbscan_config

config = load_hdbscan_config(config_path)
validation = validate_hdbscan_config(config)
print("Valid:", validation['valid'])
```
