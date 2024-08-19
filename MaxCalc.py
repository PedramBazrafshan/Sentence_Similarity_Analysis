"""Code developed and presented by Pedram Bazrafshan"""


import pandas as pd

# File path to your Excel file
file_path = r"SentenceSimScore.xlsx"

# Read from the specified sheet
df = pd.read_excel(file_path, sheet_name='Human 1 - Online - Score', header=0)

# Assuming the first column is the group number and we need to calculate the max from columns D to S
group_column = df.columns[0]  # First column (Group number)
columns_to_max = df.columns[3:19]  # Columns D to S

# Group by the group number and calculate the max for each group
max_values = df.groupby(group_column)[columns_to_max].max()

# Create a Pandas Excel writer using openpyxl as the engine
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
    # Add the new sheet with the max values
    max_values.to_excel(writer, sheet_name='Human 3 - Private - Score_max')

print("Max values for each group have been calculated and saved.")
