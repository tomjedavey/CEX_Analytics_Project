#Script to test the trained model for AS_1 functionality, importing the test_model_function from the test_model.py file.
# This script will load the test data, load the trained model, and evaluate the model using the test data.
#**NEED TO IMPROVE THE EVALUATION MEASURES WRITTEN IN THE test_model.py FILE - CURRENTLY JUST PRINTS THE RESULTS, NEED TO ADD MORE FUNCTIONALITY**

# Import necessary modules
import yaml
from source_code_package.models.AS_1_functionality.test_model import test_linear_regression  # Replace with actual function name
import pandas as pd
import joblib

# Load test data
#Reading the data path from config.yaml (meaning any changes to the data path in config.yaml will be reflected here):
with open('source_code_package/config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
data_path = config['data']['processed_data_path']  # Adjust the key as per your config.yaml structure

# Preprocess test data if needed (reuse code from train_model_AS_1.py)

# Evaluate model using the function from test_model.py
results = test_linear_regression('source_code_package/config/config.yaml')
y_test, y_pred, mse, r2, indices = results
results_df = pd.DataFrame({'index': indices, 'y_true': y_test, 'y_pred': y_pred})
results_df.to_csv('data/scores/AS_1_test_results.csv', index=False)
print(f"MSE: {mse}, R2: {r2}")
print(results_df)

#DON'T THINK THAT THE INDEX IS WORKING IN THIS CODE PROPERLY - NEED TO CHECK THIS
