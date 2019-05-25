# Import libraries
from models import decision_tree
import pandas as pd

# Import the data and read from csv file
df_train = pd.read_csv('banking_training.csv')
df_test = pd.read_csv('banking_testing.csv')

# Set up features and targets for prediction
feature_cols = ['age', 'duration', 'campaign', 'pdays', 'previous']
target_col = 'Final_Y'

# Build a decision tree from training data
dt = decision_tree.build_tree(df_train[feature_cols], df_train[target_col])

# Create a combined data frame for output of predictions
output_df = pd.DataFrame({
    'row ID': df_test['row ID'],
    'Final_Y': dt.predict(df_test[feature_cols]),
})

# Write the data frame to a csv file
output_df.to_csv('fda_a3_12582589.csv', index=False)
