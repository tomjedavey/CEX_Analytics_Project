#Importing necessary libraries
import pandas as pd
import numpy as np
import sys
import os

#Building Features for Analytic Score 1: Revenue Contribution Score (with linear regression)

data_1 = pd.read_csv('data/raw_data/initial_raw_data_polygon.csv')

#1.1 - caulculating a Proxy of Revenue (large part of inaccuaracy here overall - looking for a good indicator etc)

# Primary volume proxy
estimated_total_volume = data_1['AVG_TRANSFER_USD'] * data_1['TX_PER_MONTH'] #**COME BACK TO THIS - IS TX_PER_MONTH SAME AS TRANSACTION FREQUENCY?**

# Trading volume proxy
trading_volume_proxy = (data_1['DEX_EVENTS'] + data_1['DEFI_EVENTS']) * data_1['AVG_TRANSFER_USD']

# Cross-chain volume (already available)
bridge_volume = data_1['BRIDGE_TOTAL_VOLUME_USD']

# Token diversity volume proxy
token_complexity_volume = data_1['TOKEN_DIVERSITY'] * data_1['AVG_TRANSFER_USD']

revenue_proxy = (
    # Base transaction volume (50%)
    0.5 * estimated_total_volume +
    
    # Specialized trading volume (30%)
    0.3 * trading_volume_proxy +
    
    # Cross-chain volume premium (20%)
    0.2 * bridge_volume * 1.15  # 15% premium for bridge users
)

# Apply log transformation
revenue_proxy_log = np.log1p(revenue_proxy)

