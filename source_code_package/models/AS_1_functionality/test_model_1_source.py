#Functionality for the testing of the linear regression model (given the test train split specified in config.yaml)

import joblib
import pandas as pd
import os
import yaml

def test_linear_regression(config_path=None):

    '''
    Function to test the linear regression model, making sure the train test split is being used as specified in config.yaml.
    This function loads the model and scaler, in order to apply the same preprocessing as during training specified in congig.yaml,
    and evaluates the model on the test set to display the accuracy of the model in producing the Revenue Contribution Score (RCS) for each wallet.
    The independent variables being used in the model are wallet features seen in config.yaml.
    The dependent variable is the revenue proxy for each wallet, which is also specified in config.yaml
    The function returns the true values and predicted values for the test set.
    **NEED TO FIGURE OUT THE EVALUATION METRICS LATER ON ONCE IT IS CONFIRMED THAT THE FUNCTIONALITY OF BOTH FILES WORKS AS INTENDED**  
    '''

    # Load config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), '../config/config_AS_1.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load model and scaler
    model_path = config.get('output', {}).get('model_path', 'linear_regression_model.pkl')
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(__file__), '../../../', model_path)
    saved = joblib.load(model_path)
    model = saved['model']
    scaler = saved.get('scaler', None)

    # Load test data (should match the split used in training)
    data_path = config['data']['processed_data_path']
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), '../../../', data_path)
    df = pd.read_csv(data_path)
    features = config['features']['independent_variables']
    target_col = config['features']['dependent_variable']

    # Reproduce the split
    from sklearn.model_selection import train_test_split
    X = df[features]
    y = df[target_col]
    split_cfg = config.get('train_test_split', {})
    test_size = split_cfg.get('test_size', 0.2)
    random_state = split_cfg.get('random_state', 42)
    X_train, X_test, y_train_, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale if needed
    if scaler is not None:
        X_test = scaler.transform(X_test)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    # Add evaluation metrics as needed (e.g., mean squared error, R^2, etc.) - WHERE TO ADD THE EVALUATION METRICS ONCE ALL IS WORKING / RUNNING AS INTENTED
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # If you want to return indices
    indices = X_test.index if hasattr(X_test, 'index') else None


    return y_test, y_pred, mse, r2, indices