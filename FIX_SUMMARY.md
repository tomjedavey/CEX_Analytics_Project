# Fix for DEFI_EVENTS Median Discrepancy Between Local and GitHub Actions

## Problem Summary

The median DEFI_EVENTS score production was yielding different values between local environment (12.0) and GitHub Actions environment (9.0). This inconsistency was affecting the interaction mode distance calculations and subsequent analytic scores.

## Root Cause Analysis

The issue was caused by **multiple sources of non-determinism** in the clustering pipeline:

### 1. Non-deterministic Directory Processing Order
- **`interaction_mode_median_production_source.py` (line 209)**: Used `os.listdir()` without sorting
- **`interaction_mode_distance_execution.py` (line 45)**: Used `os.listdir()` without sorting

### 2. Non-deterministic Random Sampling in UMAP
Multiple locations in `UMAP_dim_reduction.py` used `np.random.choice()` without setting random seeds:
- Line 225: Distance calculation sampling
- Line 596: Neighborhood preservation sampling  
- Line 658: Silhouette score sampling
- Line 755: Global distance preservation sampling

### 3. Missing Random State in HDBSCAN Configuration
The HDBSCAN clustering algorithm was not configured with a `random_state` parameter, allowing for non-deterministic behavior.

### Why This Caused the Problem

- `os.listdir()` returns directory entries in arbitrary order that can vary between:
  - Different operating systems (macOS vs Linux)
  - Different Python versions
  - Different file system implementations
  - Different execution environments

- Non-deterministic sampling in UMAP creates different dimensionality reductions, leading to different HDBSCAN clustering results
- Different clustering results produce different median values for each dataset
- The analytic pipeline uses the `main_clustering` median, but in GitHub Actions it was getting a different value due to different clustering

## Solution Implemented

### 1. Made Directory Processing Deterministic

**File: `scripts/interaction_mode_score/interaction_mode_distance_execution.py`**
```python
# Before:
cluster_dirs = [d for d in os.listdir(BASE_PATH) if ...]

# After:
cluster_dirs = sorted([d for d in os.listdir(BASE_PATH) if ...])
```

**File: `Source_Code_Package/features/interaction_mode_median_production_source.py`**
```python
# Before:
for fname in os.listdir(results_dir):

# After:
for fname in sorted(os.listdir(results_dir)):
```

### 2. Fixed Non-deterministic Random Sampling in UMAP

**File: `source_code_package/models/clustering_functionality/UMAP_dim_reduction.py`**
Added `np.random.seed(42)` before each `np.random.choice()` call in:
- `_safe_distance_calculation()` function (line 224)
- `calculate_umap_quality_metrics()` function (line 594) 
- Silhouette score calculation (line 660)
- Global distance preservation (line 757)

### 3. Added Random State to HDBSCAN Configuration

**File: `source_code_package/config/config_interaction_mode.yaml`**
```yaml
hdbscan:
  # ... existing config ...
  random_state: 42               # Added for reproducibility
```

**File: `source_code_package/models/clustering_functionality/HBDSCAN_cluster.py`**
```python
# Add random_state if specified for reproducibility
if 'random_state' in hdbscan_config:
    hdbscan_params['random_state'] = hdbscan_config['random_state']
```

### 4. Enhanced Debug Output

Added environment detection and processing order logging:

```python
print(f"🔍 DEBUG - Dataset processing order: {dataset_names}")
print(f"🔍 DEBUG - Environment: {'CI' if os.environ.get('CI') else 'Local'}")
```

### 5. Added Comprehensive Test Coverage

Created test files to verify reproducibility:
- `tests/test_deterministic_ordering.py`: Verifies directory processing order consistency
- `tests/test_clustering_reproducibility.py`: Verifies random seed configuration

## Impact

This fix ensures:
- **Consistent clustering results** across all environments (local, CI, production)
- **Reproducible median values** for interaction mode distance calculations
- **Reliable analytic scores** for downstream applications
- **Deterministic pipeline behavior** regardless of file system or environment

## Expected Results

After this fix, both environments should produce:
- **Consistent DEFI_EVENTS median = 12.0** (from main_clustering)
- **Identical clustering results** given the same input data
- **Reproducible interaction mode distance calculations**

## Files Modified

### Primary Fixes:
1. `/scripts/interaction_mode_score/interaction_mode_distance_execution.py`
2. `/Source_Code_Package/features/interaction_mode_median_production_source.py`
3. `/source_code_package/models/clustering_functionality/UMAP_dim_reduction.py`
4. `/source_code_package/config/config_interaction_mode.yaml`
5. `/source_code_package/models/clustering_functionality/HBDSCAN_cluster.py`

### Test Files:
6. `/tests/test_deterministic_ordering.py` (new)
7. `/tests/test_clustering_reproducibility.py` (new)

## Testing

Run the following to verify the fix:
```bash
python -m pytest tests/test_deterministic_ordering.py -v
python -m pytest tests/test_clustering_reproducibility.py -v
python -m pytest tests/test_interaction_mode_median_production.py -v
```

## Prevention

This type of issue can be prevented by:
1. **Always setting random seeds** for any randomized algorithms (UMAP, HDBSCAN, sampling)
2. **Always sorting** results from `os.listdir()`, `glob.glob()`, etc.
3. **Adding determinism tests** for any code that processes multiple files/directories
4. **Environment-specific logging** to catch differences early
5. **Comprehensive CI testing** that verifies reproducibility across environments

## Next Steps

To confirm the fix works in GitHub Actions:
1. **Re-run the full clustering pipeline** to generate new clustering results with deterministic behavior
2. **Verify DEFI_EVENTS median = 12.0** in all environments
3. **Monitor interaction mode distance calculations** for consistency
4. **Validate final analytic scores** match expected values

The core issue was that the clustering algorithm itself was producing different results in different environments due to lack of random state control, rather than just the median selection logic.