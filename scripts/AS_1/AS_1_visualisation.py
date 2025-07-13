#Script to visualise the results of the trained model for AS_1 functionality

#**MAYBE WANT TO CHANGE IN FUTURE SO THAT THE SOURCE CODE FOR THIS IS PRE-BUILT IN SOURCE_CODE_PACKAGE AND THIS IS JUST AN EXECUTION SCRIPT**

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load test results
results_df = pd.read_csv('data/scores/AS_1_test_results.csv')

# Scatter plot: Actual vs Predicted Revenue
plt.figure(figsize=(8, 6))
sns.scatterplot(x=results_df['y_true'], y=results_df['y_pred'], alpha=0.6)
plt.xlabel('Actual Revenue Proxy')
plt.ylabel('Predicted Revenue Proxy')
plt.title('Actual vs Predicted Revenue Proxy (Test Set)')
plt.plot([results_df['y_true'].min(), results_df['y_true'].max()],
         [results_df['y_true'].min(), results_df['y_true'].max()],
         color='red', linestyle='--', label='Ideal Fit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

