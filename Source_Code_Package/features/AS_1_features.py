#Importing necessary libraries
import pandas as pd
import numpy as np
import sys
import os

#Building Features for Analytic Score 1: Revenue Contribution Score (with linear regression)

data_1 = pd.read_csv('data/raw_data/initial_raw_data_polygon.csv')

#1.1 - caulculating a Proxy of Revenue (large part of inaccuaracy here overall - looking for a good indicator etc)

# Create comprehensive feature set
def engineer_features(df):
    '''
    Function to engineer new features from the initial dataset in order to be able to produce a revenue contribution score using linear regression.
    Returns a new dataframe with the engineered features which has been saved to processed data folder.

    New features include:

    - REVENUE_PROXY: A proxy for revenue based on transaction volume, events, and bridge volume.
    - ESTIMATED_TOTAL_VOLUME: Estimated total volume based on average transfer and transactions per month.
    - TRADING_EVENTS_TOTAL: Total trading events combining DEX and DeFi events.
    - TOTAL_EVENTS: Total number of events across all categories.
    - PROTOCOL_EXPERTISE: Diversity of protocols used relative to active duration.
    - METHOD_SOPHISTICATION: Diversity of interaction methods relative to transaction volume.
    - TRADING_INTENSITY: Ratio of trading events to transactions per month.
    - WALLET_MATURITY_SCORE: Logarithmic score based on active duration.
    - ACTIVITY_VELOCITY: Ratio of total events to active duration.
    - CROSS_CHAIN_INTENSITY: Ratio of bridge events to transactions per month.
    - BRIDGE_EFFICIENCY: Average volume per bridge event.
    - DOMAIN_BREADTH: Count of different event types the wallet has interacted with.
    - IS_BRIDGE_USER: Binary indicator if the wallet has used bridges.
    - IS_ADVANCED_USER: Binary indicator if the wallet has high protocol diversity.
    - IS_HIGH_FREQUENCY: Binary indicator if the wallet has above median transaction frequency.
    - IS_MULTI_DOMAIN: Binary indicator if the wallet has interacted with multiple event domains.
    - LOG_AVG_TRANSFER: Log-transformed average transfer amount.
    - LOG_TOTAL_VOLUME: Log-transformed estimated total volume.
    - LOG_BRIDGE_VOLUME: Log-transformed total bridge volume.
    '''

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

def engineer_features_1(df):
    '''
    Wrapper for engineer_features for import in other scripts.
    '''
    return engineer_features(df)

#Calling the function to engineer features on the intial dataset
data_2_rcs = engineer_features_1(data_1)

# Saving the processed data to a CSV file and adding it to the processed_data folder as the first processing made to initial dataset

# Ensure the processed_data directory exists
os.makedirs('data/processed_data', exist_ok=True)

# Save the processed DataFrame to CSV
data_2_rcs.to_csv('data/processed_data/initial_processed_data.csv', index=False)