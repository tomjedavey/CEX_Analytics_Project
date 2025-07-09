#Data pipeline not as maybe would be expected in this file of a github repository because data already processed initially and downloaded in csv.
#Therefore, this file serves the purpose of utilising logic and functionality from source_code_package/features in order to add new features in the
#form of new datasets (in csv) to the data/processed_data folder.

#Processing data to add new features for Analytic Score 1 (AS_1) - Revenue Contribution Score (RCS)


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source_code_package.features.AS_1_features import my_function