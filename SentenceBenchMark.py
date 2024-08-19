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
file_path = r"BenchmarkSimScore.xlsx"

# Read from the specified sheet
df = pd.read_excel(file_path, sheet_name='BenchMark', usecols=[0,1], header=0)

# Models for similarity calculation
Models = ["all-mpnet-base-v2", "distilbert-base-nli-mean-tokens", "bert-base-uncased", "multi-qa-mpnet-base-dot-v1", "all-distilroberta-v1", "all-MiniLM-L12-v2", 
          "multi-qa-distilbert-cos-v1", "all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1", "paraphrase-multilingual-mpnet-base-v2", 
          "paraphrase-albert-small-v2", "paraphrase-multilingual-MiniLM-L12-v2", "paraphrase-MiniLM-L3-v2", 
          "distiluse-base-multilingual-cased-v1", "distiluse-base-multilingual-cased-v2", "IoU"]


results = []

# Define the maximum row index to process
max_row_index = 6

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    print("index =", index)
    if index > max_row_index:
        break
    
    sentence1 = row.iloc[0]
    sentence2 = row.iloc[1]

    row_scores = []
    for model_type in Models[:-1]:  # Exclude "IoU" from Models list for this loop
        model = SentenceTransformer(model_type).to(device)
        encoding1 = model.encode(sentence1)
        encoding2 = model.encode(sentence2)
        similarity = np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
        row_scores.append(similarity)
        # print (f"Similarity Score of {model_type} = {similarity:.2f}")
    
    # IoU score calculations for lexical similarity analysis
    set1 = set(sentence1.split())
    set2 = set(sentence2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    iou_score = len(intersection) / len(union) if len(union) != 0 else 0
    row_scores.append(iou_score)
    # print (f"Similarity Score of IoU = {iou_score:.2f}")
    results.append(row_scores)

# Column letters for Excel (B, D, F, ...)
excel_columns = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']

# Load workbook and sheet
workbook = load_workbook(file_path)
sheet = workbook['BenchMark']

# Writing data to specified columns starting from row 4 (1-based index, hence row 3 in 0-based)
for row_idx, (row_data) in enumerate(results, start=2):
    for col_idx, data in enumerate(row_data):
        cell = f"{excel_columns[col_idx]}{row_idx}"
        sheet[cell] = data

# Save the workbook
output_path = r"BenchmarkSimScore.xlsx"
workbook.save(output_path)

print("Similarity scores have been written to 'BenchmarkSimScore.xlsx', in specific columns.")
