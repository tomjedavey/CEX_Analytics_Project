#Functionality for the training of the linear regression model 

'''Function to train the linear regression model for the purpose of building a Revenue Contribution Score (RCS) for each wallet. 
    This Revenue Contribution Score is produced from the predicted value of the revenue proxy by the linear regression model,
    the independent variables being used in the model are wallet features seen in config.yaml.
    df is the dataframe containing all features (independent variables) and the revenue proxy for each wallet (dependent variable).
    X is the independent variables to be used in the model (wallet features).
    Y is the dependent variable to be used in the model (revenue proxy).'''

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import yaml
from source_code_package.data.preprocess import scale_features_from_config
import pandas as pd
import os

def train_linear_regression(config_path=None):
    """
    Trains a linear regression model, optionally scaling features based on config.yaml as well as choosing the variables in config.yaml, 
    for the purpose of building a Revenue Contribution Score (RCS) for each wallet.
    This Revenue Contribution Score is produced from the predicted value of the revenue proxy by the linear regression model,
    the independent variables being used in the model are wallet features seen in config.yaml.
    df is the dataframe containing all features (independent variables) and the revenue proxy for each wallet (dependent variable).
    X is the independent variables to be used in the model (wallet features).
    Y is the dependent variable to be used in the model (revenue proxy).
    """

    #Code to scale the features if specified in config.yaml

    # Load config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get dependent variable
    target_col = config['features']['dependent_variable']

    # Check if scaling is enabled
    use_scaling = config.get('preprocessing', {}).get('use_scaling', False)
    if use_scaling:
        X, scaler = scale_features_from_config(config_path=config_path)
    else:
        # Load data and select features without scaling
        data_path = config['data']['processed_data_path']
        if not os.path.isabs(data_path):
            data_path = os.path.join(os.path.dirname(__file__), '../../../', data_path)
        df = pd.read_csv(data_path)
        X = df[config['features']['independent_variables']]

    # Load target variable
    y = df[target_col] if not use_scaling else pd.read_csv(data_path)[target_col] #see "Get dependent variable" comment above

    #Actual code to train the linear regression model, given the independent and dependent variables specified in config.yaml

    # Get train/test split parameters from config
    split_cfg = config.get('train_test_split', {})
    test_size = split_cfg.get('test_size', 0.2)
    random_state = split_cfg.get('random_state', 42)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    ) #because random_state is set to 42 in config.yaml, the test/train split will be reproducible when producing the functon in test_model.py

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    #The concept of what is being attempted is correct - not sure if the code works as intended so check when testing the model

    # Save the model (and scaler if used)
    model_path = config.get('output', {}).get('model_path', 'linear_regression_model.pkl')
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(__file__), '../../../', model_path)
    import joblib
    joblib.dump({'model': model, 'scaler': scaler if use_scaling else None}, model_path)

    # Optionally, return the model and test data for evaluation elsewhere
    return model, X_test, y_test