# HDBSCAN Parameter Optimization Report (quick)

Generated: 2025-07-26 09:30:39
Total combinations tested: 36
Data shape: (20174, 4)

## Top 10 Results

### Rank 1
**Composite Score:** 54.39

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 3
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 150
- min_samples: 10

**Results:**
- Clusters: 3
- Noise percentage: 0.2%
- Silhouette score: 0.557
- Total time: 11.1s

### Rank 2
**Composite Score:** 54.37

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 3
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 100
- min_samples: 10

**Results:**
- Clusters: 3
- Noise percentage: 0.2%
- Silhouette score: 0.557
- Total time: 11.2s

### Rank 3
**Composite Score:** 54.17

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 3
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 150
- min_samples: 25

**Results:**
- Clusters: 3
- Noise percentage: 0.0%
- Silhouette score: 0.554
- Total time: 11.1s

### Rank 4
**Composite Score:** 53.84

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 3
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 100
- min_samples: 40

**Results:**
- Clusters: 3
- Noise percentage: 0.0%
- Silhouette score: 0.554
- Total time: 11.2s

### Rank 5
**Composite Score:** 53.84

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 3
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 150
- min_samples: 40

**Results:**
- Clusters: 3
- Noise percentage: 0.0%
- Silhouette score: 0.554
- Total time: 11.3s

### Rank 6
**Composite Score:** 53.46

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 3
- n_neighbors: 30

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 100
- min_samples: 25

**Results:**
- Clusters: 3
- Noise percentage: 0.0%
- Silhouette score: 0.554
- Total time: 11.1s

### Rank 7
**Composite Score:** 45.29

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 3
- n_neighbors: 15

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 150
- min_samples: 25

**Results:**
- Clusters: 3
- Noise percentage: 0.1%
- Silhouette score: 0.489
- Total time: 9.2s

### Rank 8
**Composite Score:** 45.10

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 3
- n_neighbors: 15

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 150
- min_samples: 40

**Results:**
- Clusters: 3
- Noise percentage: 0.1%
- Silhouette score: 0.489
- Total time: 10.6s

### Rank 9
**Composite Score:** 44.72

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 3
- n_neighbors: 15

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 100
- min_samples: 40

**Results:**
- Clusters: 3
- Noise percentage: 0.1%
- Silhouette score: 0.489
- Total time: 9.2s

### Rank 10
**Composite Score:** 44.62

**UMAP Parameters:**
- metric: euclidean
- min_dist: 0.1
- n_components: 3
- n_neighbors: 15

**HDBSCAN Parameters:**
- cluster_selection_method: eom
- metric: euclidean
- min_cluster_size: 100
- min_samples: 25

**Results:**
- Clusters: 3
- Noise percentage: 0.1%
- Silhouette score: 0.489
- Total time: 9.0s

## Recommended Configuration

```yaml
umap:
  metric: euclidean
  min_dist: 0.1
  n_components: 3
  n_neighbors: 30

hdbscan:
  cluster_selection_method: eom
  metric: euclidean
  min_cluster_size: 150
  min_samples: 10
```

## Optimization Statistics

- Best score: 54.39
- Average score: 34.21
- Score std: 15.24
- Average evaluation time: 10.20s
- Total optimization time: 367.4s
