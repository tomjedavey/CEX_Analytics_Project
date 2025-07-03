#CODE TO GO THROUGH ORIGINAL FORM OF THE FLIPSIDE DATA WITH INITIAL FEATURES AND CHECK FOR ERRORS, MISSING VALUES ETC

#
import pandas as pd
import numpy as np

#pd.read_csv code to read the csv file
df = pd.read_csv('Data/raw_data/most_recent_flipside_query.csv')
print(df)

#importing neccessary functions to clean and process the data properly - functions can be seen in the utils.py file