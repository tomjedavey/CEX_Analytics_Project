#Data pipeline not as maybe would be expected in this file of a github repository because data already processed initially and downloaded in csv.
#Therefore, this file serves the purpose of utilising logic and functionality from source_code_package/features in order to add new features in the
#form of new datasets (in csv) to the data/processed_data folder.

#Processing data to add new features for Analytic Score 1 (AS_1) - Revenue Contribution Score (RCS)

import sys
import os
import pandas as pd

# Add the Source_Code_Package to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Source_Code_Package', 'features')))

from source_code_package import engineer_features_1

# Read the raw data
raw_data = pd.read_csv(os.path.join('..', 'data', 'raw_data', 'initial_raw_data_polygon.csv'))

# Engineer features
processed_data = engineer_features_1(raw_data)

# Ensure the processed_data directory exists
os.makedirs(os.path.join('..', 'data', 'processed_data'), exist_ok=True)

# Save the processed data to CSV
processed_data.to_csv(os.path.join('..', 'data', 'processed_data', 'new_features_data.csv'), index=False)