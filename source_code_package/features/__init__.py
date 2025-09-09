"""
Feature engineering modules for MLProject1.

This module contains:
- AS_1 feature engineering functions
- Feature transformation utilities
- Revenue proxy calculations
- Behavioral volatility feature engineering
- Cross-domain engagement features
- Interaction mode score cluster selection

Available modules:
- revenue_proxy_features: REVENUE_SCORE_PROXY calculation and pipeline
- behavioral_volatility_features: Behavioral volatility score calculation
- cross_domain_engagement_features: Shannon entropy cross-domain engagement scores
- interaction_mode_features: Cluster selection for interaction mode scoring

Author: Tom Davey
Date: July 2025
"""

from .revenue_proxy_features import (
    calculate_revenue_score_proxy,
    revenue_proxy_feature_pipeline
)

from .behavioural_volatility_features import (
    calculate_behavioural_volatility_score,
    behavioural_volatility_pipeline
)

from .cross_domain_engagement_features import (
    calculate_cross_domain_engagement_score,
    cross_domain_engagement_pipeline
)

from .interaction_mode_median_production_source import (
    calculate_median_feature_values_for_clusters,
    print_cluster_selection_summary,
    save_cluster_selection_results
)

__all__ = [
    # Revenue proxy features
    'calculate_revenue_score_proxy',
    'revenue_proxy_feature_pipeline',
    
    # Behavioral volatility features  
    'calculate_behavioral_volatility_score',
    'behavioral_volatility_pipeline',
    
    # Cross-domain engagement features
    'calculate_cross_domain_engagement_score', 
    'cross_domain_engagement_pipeline',
    
    # Interaction mode features v2
    'calculate_median_feature_values_for_clusters_v2',
    'print_cluster_selection_summary_v2',
    'save_cluster_selection_results'
]
