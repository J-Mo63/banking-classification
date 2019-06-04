# Import libraries
from models import decision_tree, support_vector_machine, neural_network
from utils import pre_processing as prep
import pandas as pd
import numpy as np
from scipy import stats

# Import the data and read from csv file
df_train = pd.read_csv('banking_training.csv')
df_test = pd.read_csv('banking_testing.csv')

# Set up features and targets for prediction
target_col = 'Final_Y'

# #Process the data sets for use
# processed_train = decision_tree.process_data(df_train, target_col, include_target=True)
# processed_test = decision_tree.process_data(df_test, target_col)
#
# # processed_train = processed_train[(np.abs(stats.zscore(processed_train)) < 3).all(axis=1)]
#
# # Build a decision tree from training data
# clf = decision_tree.build_dt(
#     prep.remove_target(processed_train, target_col),
#     processed_train[target_col])


# # Process the data sets for use
# processed_train = support_vector_machine.process_data(df_train, target_col, include_target=True)
# processed_test = support_vector_machine.process_data(df_test, target_col)
#
# # Build a support vector machine from training data
# clf = support_vector_machine.build_svm(
#     prep.remove_target(processed_train, target_col),
#     processed_train[target_col])

#Process the data sets for use
processed_train = neural_network.process_data(df_train, target_col, include_target=True)
processed_test = neural_network.process_data(df_test, target_col)

clf = neural_network.build_nn(
    prep.remove_target(processed_train, target_col),
    processed_train[target_col])

# Create a combined data frame for output of predictions
# output_df = pd.DataFrame({
#     'row ID': df_test['row ID'],
#     'Final_Y': clf.predict(processed_test)
# })
#
# # Write the data frame to a csv file
# output_df.to_csv('fda_a3_12582589.csv', index=False)
