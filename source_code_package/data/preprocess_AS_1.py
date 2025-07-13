#DON'T REALLY HAVE TO WORK ON THIS JUST YET AS NO DATA CLEANING + PROCESSING NEEDED FOR FLIPSIDE DATA FROM SNOWFLAKE MARKETPLACE

from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml
import pandas as pd
import os


def scale_features_from_config(data_path=None, config_path=None):
    '''
    Loads config.yaml, reads the independent variables and dependent variable, loads the data, and applies StandardScaler to those features.
    Returns the scaled features as a DataFrame, the target variable, and the fitted scaler.
    Because the data path is already specified in config.yaml, this function will use that path to load the data, no need to input data_path itself.
    '''
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    features = config['features']['independent_variables']
    target_col = config['features']['dependent_variable']
    if data_path is None:
        # Find the project root (directory containing pyproject.toml)
        current_dir = os.path.abspath(os.path.dirname(__file__))
        while not os.path.exists(os.path.join(current_dir, 'pyproject.toml')):
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                raise FileNotFoundError('Could not find project root (pyproject.toml)')
            current_dir = parent_dir
        project_root = current_dir
        data_path = os.path.join(project_root, config['data']['processed_data_path'])
        data_path = os.path.normpath(data_path)
    elif not os.path.isabs(data_path):
        data_path = os.path.normpath(os.path.join(os.getcwd(), data_path))
    df = pd.read_csv(data_path)
    X = df[features]
    y = df[target_col]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df.index)
    return X_scaled_df, y, scaler  #now scaling function returns both features and target variable, therefore alignment guarenteed.
#**CHANGE THIS BACK TO NOT HAVING PRINT STATEMENTS IN THE RETURN**

#**ALSO IMPORTANT TO REMEMBER FOR SCALING:
#"TEST FUNCTION MUST KNOW WHETHER TO USE THE SCALER (BY CHECKING IF IT EXISTS IN THE LOADED OBJECT)" - RE-EVALUATE THIS LATER