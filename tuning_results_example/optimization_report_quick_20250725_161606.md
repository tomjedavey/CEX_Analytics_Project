# HDBSCAN Parameter Optimization Report (quick)

Generated: 2025-07-25 16:16:06
Total combinations tested: 36
Data shape: (20174, 4)

## Top 10 Results

### Rank 1
**Composite Score:** 51.60

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 15
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 150
- min_samples: 25

**Results:**
- Clusters: 3
- Noise percentage: 0.1%
- Silhouette score: 0.554
- Total time: 13.3s

### Rank 2
**Composite Score:** 51.04

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 15
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 150
- min_samples: 40

**Results:**
- Clusters: 3
- Noise percentage: 0.1%
- Silhouette score: 0.554
- Total time: 13.4s

### Rank 3
**Composite Score:** 50.91

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 15
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 100
- min_samples: 40

**Results:**
- Clusters: 3
- Noise percentage: 0.1%
- Silhouette score: 0.554
- Total time: 13.2s

### Rank 4
**Composite Score:** 50.76

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 15
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 100
- min_samples: 25

**Results:**
- Clusters: 3
- Noise percentage: 0.1%
- Silhouette score: 0.554
- Total time: 13.2s

### Rank 5
**Composite Score:** 50.71

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 15
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 100
- min_samples: 10

**Results:**
- Clusters: 3
- Noise percentage: 0.1%
- Silhouette score: 0.554
- Total time: 13.1s

### Rank 6
**Composite Score:** 50.71

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 15
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 150
- min_samples: 10

**Results:**
- Clusters: 3
- Noise percentage: 0.1%
- Silhouette score: 0.554
- Total time: 13.5s

### Rank 7
**Composite Score:** 49.75

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 10
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 150
- min_samples: 10

**Results:**
- Clusters: 3
- Noise percentage: 0.0%
- Silhouette score: 0.542
- Total time: 12.0s

### Rank 8
**Composite Score:** 49.66

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 10
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 150
- min_samples: 25

**Results:**
- Clusters: 3
- Noise percentage: 0.0%
- Silhouette score: 0.542
- Total time: 13.7s

### Rank 9
**Composite Score:** 49.65

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 10
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 150
- min_samples: 40

**Results:**
- Clusters: 3
- Noise percentage: 0.0%
- Silhouette score: 0.542
- Total time: 12.0s

### Rank 10
**Composite Score:** 49.61

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 10
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 100
- min_samples: 25

**Results:**
- Clusters: 3
- Noise percentage: 0.0%
- Silhouette score: 0.542
- Total time: 12.0s

## Recommended Configuration

```yaml
umap:
  metric: euclidean
  min_dist: 0.1
  n_components: 15
  n_neighbors: 30

hdbscan:
  cluster_selection_method: eom
  metric: euclidean
  min_cluster_size: 150
  min_samples: 25
```

## Optimization Statistics

- Best score: 51.60
- Average score: 40.39
- Score std: 9.33
- Average evaluation time: 11.90s
- Total optimization time: 428.5s
