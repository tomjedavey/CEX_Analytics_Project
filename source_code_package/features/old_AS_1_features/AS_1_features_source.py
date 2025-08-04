#!/usr/bin/env python3
"""
AS_1 Feature Engineering Source Module.

This module provides feature engineering functionality for creating revenue contribution
scores and related features from raw wallet activity data. The engineered features
support linear regression modeling for revenue prediction and wallet segmentation.

Author: Tom Davey
Date: August 2025
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
import sys
import os

# Building Features for Analytic Score 1: Revenue Contribution Score (with linear regression)

data_1 = pd.read_csv('data/raw_data/initial_raw_data_polygon.csv')


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer comprehensive features from wallet activity data for revenue modeling.
    
    This function creates a wide range of derived features from raw wallet data
    including revenue proxies, sophistication metrics, engagement indicators,
    and behavioral flags to support revenue contribution score modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing raw wallet activity data with columns:
        - AVG_TRANSFER_USD: Average transfer amount in USD
        - TX_PER_MONTH: Number of transactions per month
        - DEX_EVENTS, DEFI_EVENTS: DeFi activity indicators
        - BRIDGE_EVENTS, BRIDGE_TOTAL_VOLUME_USD: Cross-chain activity
        - PROTOCOL_DIVERSITY, INTERACTION_DIVERSITY: Complexity metrics
        - ACTIVE_DURATION_DAYS: Wallet lifetime
        - Various event type columns (NFT_EVENTS, TOKEN_EVENTS, etc.)
    
    Returns:
    --------
    pd.DataFrame
        Enhanced DataFrame with all original columns plus 20+ engineered features:
        
        Core Features:
        - REVENUE_PROXY: Weighted revenue estimation target variable
        - ESTIMATED_TOTAL_VOLUME: Volume-based activity metric
        - TRADING_EVENTS_TOTAL: Combined DeFi trading activity
        - TOTAL_EVENTS: Sum of all event types
        
        Sophistication Features:
        - PROTOCOL_EXPERTISE: Protocol diversity normalized by duration
        - METHOD_SOPHISTICATION: Interaction complexity per transaction
        - TRADING_INTENSITY: Trading events per transaction ratio
        
        Behavioral Features:
        - WALLET_MATURITY_SCORE: Log-scaled wallet age metric
        - ACTIVITY_VELOCITY: Events per day ratio
        - CROSS_CHAIN_INTENSITY: Bridge usage intensity
        - BRIDGE_EFFICIENCY: Average volume per bridge transaction
        - DOMAIN_BREADTH: Count of different activity domains
        
        Binary Indicators:
        - IS_BRIDGE_USER: Cross-chain activity flag
        - IS_ADVANCED_USER: High protocol diversity flag
        - IS_HIGH_FREQUENCY: Above-median transaction frequency flag
        - IS_MULTI_DOMAIN: Multiple domain interaction flag
        
        Log-Transformed Features:
        - LOG_AVG_TRANSFER: Log-scaled average transfer amount
        - LOG_TOTAL_VOLUME: Log-scaled total volume estimate
        - LOG_BRIDGE_VOLUME: Log-scaled bridge volume
    
    Notes:
    ------
    - Uses numpy.maximum() to prevent division by zero errors
    - Applies log1p transformation to handle zero values in log features
    - Revenue proxy uses weighted combination (40% volume, 35% trading, 25% bridge)
    - Binary flags use median thresholds for classification
    - All derived metrics are designed for linear regression compatibility
    
    Example:
    --------
    >>> raw_data = pd.read_csv('wallet_data.csv')
    >>> enriched_data = engineer_features(raw_data)
    >>> print(f"Added {enriched_data.shape[1] - raw_data.shape[1]} new features")
    """

    # Revenue proxy (target variable)
    df["REVENUE_PROXY"] = (
        0.4 * df["AVG_TRANSFER_USD"] * df["TX_PER_MONTH"] +
        0.35 * (df["DEX_EVENTS"] + df["DEFI_EVENTS"]) * df["AVG_TRANSFER_USD"] +
        0.25 * df["BRIDGE_TOTAL_VOLUME_USD"]
    )
    
    # Base features
    df["ESTIMATED_TOTAL_VOLUME"] = df["AVG_TRANSFER_USD"] * df["TX_PER_MONTH"]
    df["TRADING_EVENTS_TOTAL"] = df["DEX_EVENTS"] + df["DEFI_EVENTS"]
    df["TOTAL_EVENTS"] = (df["DEX_EVENTS"] + df["DEFI_EVENTS"] + 
                         df["BRIDGE_EVENTS"] + df["NFT_EVENTS"] + 
                         df["TOKEN_EVENTS"] + df["FLOTSAM_EVENTS"])
    
    # Sophistication features
    df["PROTOCOL_EXPERTISE"] = df["PROTOCOL_DIVERSITY"] / np.maximum(df["ACTIVE_DURATION_DAYS"] / 30, 1)
    df["METHOD_SOPHISTICATION"] = df["INTERACTION_DIVERSITY"] / np.maximum(df["TX_PER_MONTH"], 1)
    df["TRADING_INTENSITY"] = df["TRADING_EVENTS_TOTAL"] / np.maximum(df["TX_PER_MONTH"], 1)
    
    # Stability features
    df["WALLET_MATURITY_SCORE"] = np.log1p(df["ACTIVE_DURATION_DAYS"] / 30)
    df["ACTIVITY_VELOCITY"] = df["TOTAL_EVENTS"] / np.maximum(df["ACTIVE_DURATION_DAYS"], 1)
    
    # Cross-chain features
    df["CROSS_CHAIN_INTENSITY"] = df["BRIDGE_EVENTS"] / np.maximum(df["TX_PER_MONTH"], 1)
    df["BRIDGE_EFFICIENCY"] = df["BRIDGE_TOTAL_VOLUME_USD"] / np.maximum(df["BRIDGE_EVENTS"], 1)
    
    # Engagement features
    df["DOMAIN_BREADTH"] = ((df["DEX_EVENTS"] > 0).astype(int) +
                           (df["DEFI_EVENTS"] > 0).astype(int) +
                           (df["NFT_EVENTS"] > 0).astype(int) +
                           (df["BRIDGE_EVENTS"] > 0).astype(int))
    
    # Binary indicators
    df["IS_BRIDGE_USER"] = (df["BRIDGE_EVENTS"] > 0).astype(int)
    df["IS_ADVANCED_USER"] = (df["PROTOCOL_DIVERSITY"] >= 5).astype(int)
    df["IS_HIGH_FREQUENCY"] = (df["TX_PER_MONTH"] > df["TX_PER_MONTH"].median()).astype(int)
    df["IS_MULTI_DOMAIN"] = (df["DOMAIN_BREADTH"] >= 3).astype(int)
    
    # Log transformations for skewed features
    df["LOG_AVG_TRANSFER"] = np.log1p(df["AVG_TRANSFER_USD"])
    df["LOG_TOTAL_VOLUME"] = np.log1p(df["ESTIMATED_TOTAL_VOLUME"])
    df["LOG_BRIDGE_VOLUME"] = np.log1p(df["BRIDGE_TOTAL_VOLUME_USD"])
    
    return df

#Calling the function to engineer features on the intial dataset
data_2_rcs = engineer_features(data_1)

