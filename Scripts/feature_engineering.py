#Data pipeline not as maybe would be expected in this file of a github repository because data already processed initially and downloaded in csv.
#Therefore, this file serves the purpose of utilising logic and functionality from source_code_package/features in order to add new features in the
#form of new datasets (in csv) to the data/processed_data folder.

import pandas as pd
import numpy as np
import sys
import os

#Processing data to add new features for Analytic Score 1 (AS_1) - Revenue Contribution Score (RCS)

#Importing the feature engineering function from source_code_package/features/AS_1_features.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from source_code_package.features.AS_1_features import engineer_features

#Loading the initial raw data
data = pd.read_csv('data/raw_data/initial_raw_data_polygon.csv')

#Calling the feature engineering function to create new features
AS_1_feature_data = engineer_features(data)

#Saving the processed data with new features to the processed_data folder
AS_1_feature_data.to_csv('data/processed_data/AS_1_feature_data.csv', index=False)