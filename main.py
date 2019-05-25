# Import libraries
from models import decision_tree
import pandas as pd

# Import the data and read from csv file
df_train = pd.read_csv('banking_training.csv')
df_test = pd.read_csv('banking_testing.csv')

# Set up features and targets for prediction
feature_cols = ['age', 'duration', 'campaign', 'pdays', 'previous']
target_col = 'Final_Y'
output_df = pd.DataFrame({
    'row ID': df_train['row ID'],
    'Final_Y': df_train['Final_Y'],
})

# Write the data frame to a csv file
output_df.to_csv('output.csv', index=False)
