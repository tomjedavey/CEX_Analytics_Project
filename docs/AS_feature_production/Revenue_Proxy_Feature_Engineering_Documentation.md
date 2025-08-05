# Revenue Proxy Feature Engineering Documentation

## Overview

This document describes the **REVENUE_SCORE_PROXY** feature engineering system, which estimates potential revenue contribution of cryptocurrency wallets for Centralized Exchange (CEX) analytics. The system follows the established modular design pattern used throughout the MLProject1 codebase.

## Business Context

### Purpose
The REVENUE_SCORE_PROXY provides CEXs with a quantitative measure to:
- **Segment users** by potential revenue contribution
- **Prioritize customer engagement** strategies
- **Identify high-value users** for premium services
- **Optimize fee structures** based on user behavior patterns

### Formula Rationale
The revenue proxy combines three key behavioral indicators that correlate with CEX profitability:

```
REVENUE_SCORE_PROXY = 0.4 × (AVG_TRANSFER_USD × TX_PER_MONTH) + 
                     0.35 × (DEX_EVENTS + DEFI_EVENTS) × AVG_TRANSFER_USD + 
                     0.25 × BRIDGE_TOTAL_VOLUME_USD
```

#### Component Breakdown:
1. **Transaction Activity (40% weight)**: `AVG_TRANSFER_USD × TX_PER_MONTH`
   - Measures consistent trading behavior and volume
   - Primary revenue driver through trading fees
   - Higher weight reflects fundamental importance of transaction frequency

2. **DEX/DeFi Activity (35% weight)**: `(DEX_EVENTS + DEFI_EVENTS) × AVG_TRANSFER_USD`
   - Captures sophisticated DeFi users who typically generate higher fees
   - These users often engage in complex trading strategies
   - Strong indicator of user engagement and platform stickiness

3. **Bridge Activity (25% weight)**: `BRIDGE_TOTAL_VOLUME_USD`
   - Direct measure of cross-chain value transfer
   - Bridge users often handle substantial amounts
   - Indicates multi-chain sophistication and higher transaction values

## Architecture

### Modular Design Pattern
Following the established MLProject1 methodology:

```
source_code_package/features/revenue_proxy_features.py    # Core functionality
source_code_package/config/config_revenue_proxy.yaml     # Configuration
scripts/revenue_proxy_feature_engineering.py             # Execution wrapper
scripts/validate_revenue_proxy.py                        # Validation & analysis
```

### File Structure and Responsibilities

#### 1. Core Functionality: `revenue_proxy_features.py`
**Location**: `source_code_package/features/revenue_proxy_features.py`

**Key Functions**:
- `validate_required_columns()`: Ensures all necessary input columns are present
- `handle_missing_values()`: Manages missing data with configurable strategies
- `calculate_revenue_score_proxy()`: Core calculation function
- `revenue_proxy_feature_pipeline()`: Complete end-to-end pipeline

**Features**:
- Configuration-driven weights and parameters
- Comprehensive data validation
- Missing value handling (zero, mean, median strategies)
- Component decomposition for analysis
- Detailed logging and statistics

#### 2. Configuration: `config_revenue_proxy.yaml`
**Location**: `source_code_package/config/config_revenue_proxy.yaml`

**Configuration Sections**:
- **Data paths**: Input and output file specifications
- **Feature weights**: Customizable formula components (0.4, 0.35, 0.25)
- **Missing value handling**: Strategy and method selection
- **Quality checks**: Validation parameters and thresholds
- **Metadata**: Documentation and business rationale

#### 3. Execution Script: `revenue_proxy_feature_engineering.py`
**Location**: `scripts/revenue_proxy_feature_engineering.py`

**Capabilities**:
- Command-line interface with argument parsing
- Flexible input/output path specification
- Configuration file override options
- Comprehensive progress reporting
- Top wallet identification and statistics

**Usage Examples**:
```bash
# Basic execution with default configuration
python3 scripts/revenue_proxy_feature_engineering.py

# Custom input file
python3 scripts/revenue_proxy_feature_engineering.py --input data/custom_data.csv

# Custom output location
python3 scripts/revenue_proxy_feature_engineering.py --output data/my_output.csv

# Custom configuration
python3 scripts/revenue_proxy_feature_engineering.py --config custom_config.yaml

# Test mode (no file saving)
python3 scripts/revenue_proxy_feature_engineering.py --no-save
```

#### 4. Validation Script: `validate_revenue_proxy.py`
**Location**: `scripts/validate_revenue_proxy.py`

**Analysis Functions**:
- **Formula validation**: Verifies mathematical correctness
- **Distribution analysis**: Statistical characterization of scores
- **Component contribution analysis**: Breakdown of formula components
- **CEX revenue validity assessment**: Business-relevant segmentation analysis

## Output Features

The pipeline generates the following features in the output CSV:

### Primary Feature
- **`REVENUE_SCORE_PROXY`**: Main revenue estimation score (weighted formula result)

### Component Features (for analysis)
- **`REVENUE_PROXY_TRANSACTION_COMPONENT`**: Raw transaction activity component
- **`REVENUE_PROXY_DEX_DEFI_COMPONENT`**: Raw DEX/DeFi activity component  
- **`REVENUE_PROXY_BRIDGE_COMPONENT`**: Raw bridge activity component

*Note: Component features store unweighted values for analytical purposes.*

## Results and Validation

### Dataset Statistics (Polygon blockchain data)
- **Total wallets analyzed**: 20,174
- **Mean revenue proxy**: $46,784.50
- **Median revenue proxy**: $299.83
- **Standard deviation**: $3,084,413.65
- **Range**: $0.00 - $399,231,793.29

### User Segmentation Analysis
| Segment | Wallet Count | Avg TX/Month | Avg Transfer USD | Avg DEX+DeFi Events |
|---------|--------------|--------------|------------------|---------------------|
| Low Revenue (≤25th percentile) | 5,044 | 18.6 | $6.66 | 2.0 |
| Medium Revenue (25-75th percentile) | 10,086 | 21.2 | $110.44 | 13.0 |
| High Revenue (>75th percentile) | 5,044 | 75.1 | $5,306.13 | 62.6 |
| Top Revenue (>95th percentile) | 1,009 | 236.7 | $24,534.07 | 208.3 |

### Correlation Analysis
The revenue proxy shows strong correlations with key activity metrics:
- **Total Transfer USD**: 0.961 (very strong)
- **Average Transfer USD**: 0.778 (strong)
- **Bridge Total Volume**: 0.132 (moderate)
- **Transaction Per Month**: 0.010 (weak, due to value weighting)

## Business Validity Assessment

### Why This Is a Valid Revenue Proxy for CEXs

1. **Transaction-Based Revenue Model Alignment**
   - CEXs primarily generate revenue through trading fees
   - Formula emphasizes transaction frequency and volume
   - Higher activity users typically generate more fees

2. **Sophisticated User Identification**
   - DEX/DeFi component identifies advanced users
   - These users typically engage in higher-value, more frequent trading
   - Strong correlation with customer lifetime value

3. **Cross-Chain Activity Capture**
   - Bridge activity indicates multi-chain engagement
   - Such users often manage larger portfolios
   - Higher probability of platform loyalty and engagement

4. **Practical Segmentation Results**
   - Clear differentiation between user tiers
   - Actionable insights for customer management
   - Aligns with typical CEX user classification systems

### Limitations and Considerations

1. **On-Chain Data Only**: Does not capture off-chain trading behavior
2. **Single Blockchain**: Currently Polygon-only (expandable to multi-chain)
3. **Static Weights**: Formula weights may need adjustment based on specific CEX business models
4. **Missing Internal Metrics**: Cannot incorporate actual CEX revenue data for validation

## Integration with Existing Pipeline

The revenue proxy feature engineering integrates seamlessly with the existing clustering pipeline:

1. **Input Compatibility**: Uses same raw data format as clustering analysis
2. **Output Format**: Compatible with existing feature engineering patterns
3. **Configuration Driven**: Follows established YAML configuration approach
4. **Modular Design**: Maintains separation of concerns principle

### Next Steps for Integration
1. **Include in clustering analysis**: Add REVENUE_SCORE_PROXY to clustering features
2. **Customer segmentation**: Use revenue proxy for customer tier classification
3. **Predictive modeling**: Incorporate as target variable for LTV prediction models
4. **Multi-blockchain expansion**: Extend to Ethereum, Bitcoin, and other chains

## Usage in CEX Analytics

### Customer Segmentation
```python
# Load revenue proxy features
df = pd.read_csv('data/processed_data/revenue_proxy_features.csv')

# Define revenue tiers
low_tier = df[df['REVENUE_SCORE_PROXY'] <= df['REVENUE_SCORE_PROXY'].quantile(0.25)]
high_tier = df[df['REVENUE_SCORE_PROXY'] > df['REVENUE_SCORE_PROXY'].quantile(0.75)]

# Targeted engagement strategies
```

### A/B Testing Frameworks
- Use revenue proxy scores to stratify test groups
- Ensure balanced revenue potential across test cohorts
- Measure impact of features on high-value user segments

### Customer Lifetime Value Prediction
- Revenue proxy as input feature for LTV models
- Correlation analysis with actual CEX revenue data
- Refinement of formula weights based on business outcomes

---

**Author**: Tom Davey  
**Date**: August 2025  
**Version**: 1.0
