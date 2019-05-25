# Import libraries
import pandas as pd

# Import the data and read from csv file
df_train = pd.read_csv('banking_training.csv')
df_test = pd.read_csv('banking_testing.csv')

# Create a combined data frame for output
output_df = pd.DataFrame({
    'row ID': df_train['row ID'],
    'Final_Y': df_train['Final_Y'],
})

# Write the data frame to a csv file
output_df.to_csv('output.csv', index=False)
