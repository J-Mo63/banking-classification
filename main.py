# Import libraries
from models import decision_tree
import pandas as pd

# Import the data and read from csv file
df_train = pd.read_csv('banking_training.csv')
df_test = pd.read_csv('banking_testing.csv')

# Set up features and targets for prediction
target_col = 'Final_Y'

# Process the data sets for use
processed_train = decision_tree.process_data(df_train, target_col, include_target=True)
processed_test = decision_tree.process_data(df_test, target_col)

# Build a decision tree from training data
dt = decision_tree.build_tree(processed_train.drop(['Final_Y'], axis=1), processed_train[target_col])

# Create a combined data frame for output of predictions
output_df = pd.DataFrame({
    'row ID': df_test['row ID'],
    'Final_Y': dt.predict(processed_test)
})

# Write the data frame to a csv file
output_df.to_csv('fda_a3_12582589.csv', index=False)
