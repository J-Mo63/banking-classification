import pandas as pd

def write_to_xls(df):
    # Write and save the data to an excel document
    writer = pd.ExcelWriter('output.xls', engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()