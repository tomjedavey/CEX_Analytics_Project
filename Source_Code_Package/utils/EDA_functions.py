import pandas as pd

def nan_checker(df):
    """
    Function to check whether NaN values are present. 
    Inputted argument is a dataframe and if Nan value - identifies column and row number
    If no NaN values are found, nothing will be outputted
    """
    for index, row in df.iterrows():
        for column in df.columns:
            if pd.isnull(row[column]):
                print(f"NaN value found at row {index}, column '{column}'")