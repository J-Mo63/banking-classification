# Import libraries
from scripts import pre_processing as prep
import pandas as pd

# Import the data and read from Excel file
df = pd.read_csv('banking_campaign.csv')

# Create a combined data frame for output
output_df = pd.DataFrame({
    # 'campaign.equi-width': binned_equi_width_campaign,
})

# Merge and write the data frame to an excel file
prep.write_to_xls(pd.concat([df, output_df], axis=1, sort=False))