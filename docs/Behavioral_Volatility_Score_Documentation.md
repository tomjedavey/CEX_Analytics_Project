# Behavioral Volatility Score Implementation Summary

## Overview

The Behavioral Volatility Score is a comprehensive analytic feature that measures the inconsistency and unpredictability of wallet behavior across three key dimensions:

1. **Financial Volatility (35% weight)**: Measures transfer amount volatility
2. **Activity Volatility (40% weight)**: Measures inconsistency in activity patterns  
3. **Exploration Volatility (25% weight)**: Measures exploration intensity relative to activity

## Implementation Structure

Following the modular structure used throughout the repository, the implementation consists of:

### Configuration File
- **Path**: `source_code_package/config/config_behavioral_volatility.yaml`
- **Purpose**: Centralized configuration for all parameters, weights, and settings
- **Key Settings**:
  - Component weights: Financial (35%), Activity (40%), Exploration (25%)
  - Activity sub-component weights: CV (40%), Variance Ratio (30%), Gini (30%)
  - Normalization: Min-max scaling to [0,1] range

### Core Feature Module
- **Path**: `source_code_package/features/behavioral_volatility_features.py`
- **Purpose**: Core implementation of all calculation logic
- **Key Functions**:
  - `calculate_financial_volatility()`: USD_TRANSFER_STDDEV / AVG_TRANSFER_USD
  - `calculate_activity_volatility()`: Composite of CV, variance ratio, and Gini coefficient
  - `calculate_exploration_volatility()`: Exploration intensity with sqrt transformation
  - `behavioral_volatility_pipeline()`: Complete end-to-end pipeline

### Execution Script
- **Path**: `scripts/behavioral_volatility_feature_engineering.py`
- **Purpose**: Command-line interface for running the feature engineering
- **Features**: Argument parsing, verbose output, error handling

### Test Suite
- **Path**: `tests/test_behavioral_volatility.py`
- **Purpose**: Validation of component calculations and edge cases
- **Coverage**: Unit tests for all core functions, edge cases, real data validation

## Detailed Component Calculations

### 1. Financial Volatility (35% weight)
```
Financial_Volatility = USD_TRANSFER_STDDEV / AVG_TRANSFER_USD
```
- **Purpose**: Captures variability in transfer amounts
- **Interpretation**: Higher values indicate inconsistent transfer patterns
- **Edge Cases**: Returns 0 when AVG_TRANSFER_USD is 0 or missing

### 2. Activity Volatility (40% weight)
Composite score from three sub-components applied to activity event counts:

#### Component A: Coefficient of Variance (40% of activity score)
```
CV = std(activity_counts) / mean(activity_counts)
```

#### Component B: Variance Ratio from Uniform Distribution (30% of activity score)
```
Variance_Ratio = actual_variance / expected_uniform_variance
```

#### Component C: Gini Coefficient (30% of activity score)
```
Gini = (2 * Σ(i * sorted_counts)) / (n * Σ(counts)) - (n+1)/n
```

**Activity Event Types Used**:
- DEX_EVENTS, GAMES_EVENTS, CEX_EVENTS, DAPP_EVENTS
- CHADMIN_EVENTS, DEFI_EVENTS, BRIDGE_EVENTS, NFT_EVENTS
- TOKEN_EVENTS, FLOTSAM_EVENTS

### 3. Exploration Volatility (25% weight)
```
Exploration_Intensity = avg(diversity_features) / TX_PER_MONTH
Exploration_Volatility = sqrt(Exploration_Intensity)
```

**Diversity Features Used**:
- PROTOCOL_DIVERSITY, INTERACTION_DIVERSITY, TOKEN_DIVERSITY

## Final Score Calculation

### Raw Score
```
Behavioral_Volatility_Raw = 
    0.35 * Financial_Volatility + 
    0.40 * Activity_Volatility + 
    0.25 * Exploration_Volatility
```

### Normalized Score
Applied Min-Max scaling to transform to [0,1] range:
```
Behavioral_Volatility_Score = (Raw - Min) / (Max - Min)
```

## Usage Examples

### Basic Usage
```bash
python3 scripts/behavioral_volatility_feature_engineering.py
```

### Custom Input/Output
```bash
python3 scripts/behavioral_volatility_feature_engineering.py \
    --input data/custom_input.csv \
    --output data/custom_output.csv \
    --verbose
```

### Custom Configuration
```bash
python3 scripts/behavioral_volatility_feature_engineering.py \
    --config custom_config.yaml
```

## Output Features

The implementation generates the following columns in the output dataset:

1. **FINANCIAL_VOLATILITY**: Raw financial volatility component
2. **ACTIVITY_VOLATILITY**: Raw activity volatility component  
3. **EXPLORATION_VOLATILITY**: Raw exploration volatility component
4. **BEHAVIORAL_VOLATILITY_SCORE_RAW**: Weighted composite before normalization
5. **BEHAVIORAL_VOLATILITY_SCORE**: Final normalized score [0,1]

## Validation Results

✅ **All tests passed successfully**:
- Component calculations verified against manual calculations
- Edge cases handled properly (zero activity, missing values)
- Real data validation confirms no null/infinite values
- Normalization produces proper [0,1] range
- Composite score matches weighted sum exactly

## Business Interpretation

### High Volatility (Score → 1.0)
- Unpredictable transfer amounts
- Inconsistent activity patterns across event types
- High exploration relative to activity rate
- **Use Cases**: Risk assessment, fraud detection

### Low Volatility (Score → 0.0)  
- Consistent transfer amounts
- Stable activity patterns
- Low exploration relative to activity
- **Use Cases**: Stable user identification, predictable behavior modeling

### Applications
- **Risk Assessment**: High volatility may indicate unpredictable behavior
- **User Segmentation**: Stable vs volatile behavioral patterns
- **Fraud Detection**: Unusual volatility patterns may signal anomalies
- **Product Recommendations**: Different approaches for stable vs volatile users

## Files Generated

### Primary Output
- **Path**: `data/processed_data/behavioral_volatility_features.csv`
- **Records**: 20,174 wallet records
- **Columns**: Original 22 columns + 5 new behavioral volatility features

### Summary Statistics
- **Mean Score**: 0.033258
- **Standard Deviation**: 0.017853  
- **Range**: [0.000000, 1.000000]
- **No missing values or invalid calculations**

## Integration with Existing Codebase

The Behavioral Volatility Score follows the same modular architecture as:
- Cross-Domain Engagement Score (Shannon entropy-based)
- Revenue Score Proxy (financial behavior analysis)

This ensures consistency in:
- Configuration management
- Error handling
- Pipeline structure
- Output formatting
- Documentation standards

The feature is ready for integration into clustering pipelines, analysis notebooks, and downstream modeling workflows.
