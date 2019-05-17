# Import libraries
import pandas as pd

# Import the data and read from Excel file
df = pd.read_csv('banking_training.csv')

# Create a combined data frame for output
output_df = pd.DataFrame({
    'row ID': df['row ID'],
    'Final_Y': df['Final_Y'],
})

# Write the data frame to a csv file
output_df.to_csv('output.csv', index=False)
