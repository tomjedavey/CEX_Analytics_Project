#DON'T REALLY HAVE TO WORK ON THIS JUST YET AS NO DATA CLEANING + PROCESSING NEEDED FOR FLIPSIDE DATA FROM SNOWFLAKE MARKETPLACE

from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml
import pandas as pd
import os

def scale_features(features):
    '''
    Function to scale features. Aimed at scaling the indepenent variables.
    '''

    new_features = StandardScaler(features)

    return new_features

def scale_features_from_config(data_path=None, config_path=None):
    '''
    Loads config.yaml, reads the independent variables, loads the data, and applies StandardScaler to those features.
    Returns the scaled features as a DataFrame and the fitted scaler.
    '''
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    features = config['features']['independent_variables']
    if data_path is None:
        data_path = config['data']['processed_data_path']
    # Adjust path if running from a different working directory
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), '../../../', data_path)
    df = pd.read_csv(data_path)
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df.index)
    return X_scaled_df, scaler