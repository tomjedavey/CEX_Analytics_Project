#Functionality for the training of the linear regression model 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import yaml
from sklearn.preprocessing import StandardScaler
import os

def train_linear_model(df, X, Y):
    '''
    Function to train the linear regression model for the purpose of building a Revenue Contribution Score (RCS) for each wallet. 
    This Revenue Contribution Score is produced from the predicted value of the revenue proxy by the linear regression model,
    the independent variables being used in the model are wallet features seen in config.yaml.
    df is the dataframe containing all features (independent variables) and the revenue proxy for each wallet (dependent variable).
    X is the independent variables to be used in the model (wallet features).
    Y is the dependent variable to be used in the model (revenue proxy).
    '''
    X = df[X]
    y = df[Y]

    model = LinearRegression()
    model.fit(X, y)

    return model

def train_linear_model_from_config(df, config_path=None):
    '''
    Trains a linear regression model using configuration from a YAML file.
    - Scales X features
    - Uses train/test split from config
    - Uses model params from config
    Returns: trained model, scaler, X_train, X_test, y_train, y_test
    '''
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../../config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    features_cfg = config['features']
    model_cfg = config['model']
    split_cfg = config['train_test_split']

    X_cols = features_cfg['independent_variables']
    y_col = features_cfg['dependent_variable']

    X = df[X_cols]
    y = df[y_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=split_cfg.get('test_size', 0.2),
        random_state=split_cfg.get('random_state', 42)
    )

    model = LinearRegression(
        fit_intercept=model_cfg.get('fit_intercept', True),
        # 'normalize' is deprecated in sklearn >=1.0, so we ignore it if present
    )
    model.fit(X_train, y_train)

    return model, scaler, X_train, X_test, y_train, y_test