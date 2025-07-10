#Script to execute the training of the linear regression model to produce the Revenue Contribution Score (RCS) for each wallet.
#Training section of the test train split as specified in config.yaml, main functionality executed here can be seen in train_model.py

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import yaml
from source_code_package.data.preprocess import scale_features_from_config
import pandas as pd
import os

#Importing the train_linear_regression function from the train_model.py file:
from source_code_package.models.AS_1_functionality.train_model import train_linear_regression

#Calling of the train_linear_regression function to train the model:
train_linear_regression(config_path='source_code_package/config/config.yaml')