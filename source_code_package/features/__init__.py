"""
Feature engineering modules for MLProject1.

This module contains:
- AS_1 feature engineering functions
- Feature transformation utilities
- Revenue proxy calculations

Available modules:
- revenue_proxy_features: REVENUE_SCORE_PROXY calculation and pipeline

Author: Tom Davey
Date: July 2025
"""

from .revenue_proxy_features import (
    calculate_revenue_score_proxy,
    revenue_proxy_feature_pipeline
)

__all__ = [
    'calculate_revenue_score_proxy',
    'revenue_proxy_feature_pipeline'
]
