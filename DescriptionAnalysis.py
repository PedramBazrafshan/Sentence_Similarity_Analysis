"""Code developed and presented by Pedram Bazrafshan"""


import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from openpyxl import load_workbook
import torch

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# File path to your Excel file
file_path = r"DescriptionSimScore.xlsx"

# Read from the specified sheet
df = pd.read_excel(file_path, sheet_name='Human 1 - Online - Data', usecols=[0, 1], header=0)

# Models for similarity calculation
Models = ["distilbert-base-nli-mean-tokens", "bert-base-uncased", "all-MiniLM-L12-v2", 
          "multi-qa-MiniLM-L6-cos-v1", "paraphrase-multilingual-mpnet-base-v2", 
          "paraphrase-multilingual-MiniLM-L12-v2"]


results = []

# Define the maximum row index to process
max_row_index = 70

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    print("index =", index)
    if index > max_row_index:
        break
    
    description1 = row.iloc[0]
    description2 = row.iloc[1]

    row_scores = []
    for model_type in Models[:-1]:  # Exclude "IoU" from Models list for this loop
        model = SentenceTransformer(model_type).to(device)
        encoding1 = model.encode(description1)
        encoding2 = model.encode(description2)
        similarity = np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
        row_scores.append(similarity)
        # print (f"Similarity Score of {model_type} = {similarity:.2f}")
    
    results.append(row_scores)


# Column letters for Excel (B, D, F, ...)
excel_columns = ['B', 'C', 'D', 'E', 'F', 'G']

# Load workbook and sheet
workbook = load_workbook(file_path)
sheet = workbook['Human 1 - Online - Score']

# Writing data to specified columns
for row_idx, (row_data) in enumerate(results, start=2):
    for col_idx, data in enumerate(row_data):
        cell = f"{excel_columns[col_idx]}{row_idx}"
        sheet[cell] = data

# Save the workbook
output_path = r"DescriptionSimScore.xlsx"
workbook.save(output_path)

print("Similarity scores have been written to 'DescriptionSimScore.xlsx', in specific columns.")
