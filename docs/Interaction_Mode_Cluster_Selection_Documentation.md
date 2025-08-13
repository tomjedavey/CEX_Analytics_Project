# Interaction Mode Score Cluster Selection Documentation

## Overview

The Interaction Mode Score cluster selection functionality identifies clusters that represent the "strongest display" of target features from the interaction mode clustering pipeline results. This is the first step in producing interaction mode scores for individual wallets.

## Methodology

### Target Features

The system analyzes four key interaction mode features from `config_interaction_mode.yaml`:
- `DEX_EVENTS`: Decentralized Exchange events
- `CEX_EVENTS`: Centralized Exchange events
- `DEFI_EVENTS`: DeFi protocol events  
- `BRIDGE_EVENTS`: Cross-chain bridge events

### Cluster Selection Formula

For each target feature, the algorithm selects the cluster with the highest selection score using an enhanced formula:

```
Selection Score = (cluster_size × density × feature_intensity) / feature_variance
```

Where:
- **cluster_size**: Number of wallets in the cluster
- **density**: Currently set to 1.0 (can be enhanced with HDBSCAN probabilities if available)
- **feature_intensity**: `log(1 + mean_feature_value)` to reward higher activity levels
- **feature_variance**: Variance of the target feature within the cluster

### Feature Intensity Calculation

The feature intensity component ensures that clusters with higher activity levels are preferred:
- For clusters with activity: `log(1 + mean_value)`
- For clusters with no activity: `0.001 × (1 + non_zero_proportion)` to provide minimal scoring

### Data Processing Pipeline

1. **Load Data**: Merges original dataset with cluster labels from HDBSCAN results
2. **Calculate Scores**: Computes selection score for each cluster and target feature
3. **Select Clusters**: Chooses the highest-scoring cluster for each feature
4. **Extract Medians**: Calculates median feature values from selected clusters
5. **Generate Summary**: Provides statistics across all datasets and features

## Results Structure

### Per Dataset Results
```yaml
datasets:
  main:
    total_points: 20174
    valid_clusters: 25
    noise_points: 2476
    feature_selections:
      BRIDGE_EVENTS:
        selected_cluster: 15
        median_value: 6.0
        cluster_size: 474
        selection_score: 922361.411
        feature_stats:
          mean: 6.0
          median: 6.0
          std: 0.0
          variance: 0.001
          intensity: 1.946
          non_zero_proportion: 1.0
```

### Summary Statistics
- Cross-dataset feature analysis
- Median value distributions
- Selection score comparisons
- Cluster usage patterns

## Results Analysis

### Problem with Original Algorithm (V1)

The original algorithm frequently selected clusters with median values of 0, despite being designed to find clusters with the "strongest display" of features. This occurred due to:

**Root Cause: Sparse Blockchain Data**
- DEX_EVENTS: Only 35.8% of wallets have any activity
- CEX_EVENTS: Only 5.4% of wallets have any activity  
- DEFI_EVENTS: Only 33.7% of wallets have any activity
- BRIDGE_EVENTS: Only 52.4% of wallets have any activity

**Algorithm Bias: Large, Consistent Clusters**
The original scoring formula `(cluster_size × density × feature_intensity) / variance` prioritized:
1. **Large clusters** (high cluster_size scores)
2. **Consistent clusters** (low variance scores)  
3. **Minimal feature_intensity** when mean = 0

This led to selection of large clusters where all wallets had 0 events (perfect consistency), rather than smaller clusters with meaningful activity levels.

**Example Problem:**
- DEX_EVENTS selected Cluster #5: 4,897 wallets, 0% activity, median = 0.0
- Algorithm found "most consistent" cluster, but it was consistently inactive

### Enhanced Algorithm Solution (V2)

**Key Innovation: Activity Threshold Requirement**
```python
# Reject clusters with insufficient activity
if non_zero_proportion < min_activity_threshold:
    return 0.0  # Exclude from consideration
```

**Enhanced Scoring Formula:**
```python
score = (cluster_size × density × enhanced_intensity × activity_bonus) / variance

where:
    enhanced_intensity = log(1 + mean_value) × (1 + non_zero_proportion)
    activity_bonus = 1 + (non_zero_proportion - threshold) × 2
```

**Fallback Strategy:**
If no clusters meet the activity threshold, select the cluster with the highest activity rate available.

### Current Results (Enhanced Algorithm)

**DEX_EVENTS & CEX_EVENTS**: 
- V1: Selected large inactive clusters (median = 0, activity = 0%)
- V2: Selects smaller but active clusters (median ≥ 0, activity ≥ 10-40%)
- Eliminates the "zero-median problem" by requiring meaningful participation

**DEFI_EVENTS & BRIDGE_EVENTS**: 
- Both algorithms perform well due to higher baseline activity rates
- V2 provides additional validation through activity rate reporting
- Maintains strong median values (3.0 and 6.0 respectively)

### Interpretation

**Enhanced Algorithm Benefits:**
1. **Meaningful Baselines**: Selected clusters represent actual behavioral patterns
2. **Activity Validation**: All selections verified to have meaningful participation rates  
3. **Behavioral Significance**: Median values reflect true interaction mode intensities
4. **Scoring Confidence**: Reference points based on demonstrably active wallet groups

**Practical Impact:**
The enhanced medians now serve as reliable reference points for individual wallet scoring:
- **DEX_EVENTS: 1.0+** → Identifies wallets matching active DEX user patterns
- **CEX_EVENTS: Variable** → Accounts for extreme sparsity while finding most active clusters
- **DEFI_EVENTS: 3.0** → Baseline for strong DeFi engagement
- **BRIDGE_EVENTS: 6.0** → Baseline for heavy cross-chain activity

## Usage

### Command Line Interface
```bash
# Basic usage
python3 scripts/interaction_mode_cluster_selection.py

# With custom paths
python3 scripts/interaction_mode_cluster_selection.py \
    --results-dir /path/to/interaction_mode_results \
    --output /path/to/output.yaml \
    --config /path/to/config.yaml

# Force overwrite existing results
python3 scripts/interaction_mode_cluster_selection.py --force --verbose
```

### Programmatic Usage
```python
from source_code_package.features.interaction_mode_features import (
    calculate_median_feature_values_for_clusters
)

results = calculate_median_feature_values_for_clusters(
    results_dir="/path/to/interaction_mode_results"
)
```

## Next Steps

1. **Individual Wallet Scoring**: Use the median values as reference points to calculate interaction mode scores for individual wallets
2. **Score Normalization**: Develop scaling methods based on the selected cluster characteristics
3. **Validation**: Compare cluster selections across different parameter settings
4. **Enhancement**: Integrate actual HDBSCAN probabilities for improved density calculations

## Files Generated

- `data/processed_data/interaction_mode_cluster_selections.yaml`: Complete results with cluster selections and statistics
- Console output with detailed progress and summary statistics

## Dependencies

- pandas: Data manipulation
- numpy: Numerical calculations
- PyYAML: Configuration and results storage
- Source cluster data from interaction mode clustering pipeline
