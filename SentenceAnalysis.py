"""Code developed and presented by Pedram Bazrafshan"""



import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
import torch

nltk.download('punkt')  # Ensure the Punkt tokenizer models are downloaded

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# File path to your Excel file
file_path = r"SentenceSimScore.xlsx"

# Read from the specified sheet
df = pd.read_excel(file_path, sheet_name='Human 1 - Online - Data', usecols=[0, 1], header=0)

# Models for similarity calculation
models = ["distilbert-base-nli-mean-tokens", "bert-base-uncased", "all-MiniLM-L12-v2", 
          "multi-qa-MiniLM-L6-cos-v1", "paraphrase-multilingual-mpnet-base-v2", 
          "paraphrase-multilingual-MiniLM-L12-v2"]

all_results = []

# Define the maximum row index to process
max_row_index = 70

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    print("index =", index)
    if index > max_row_index:
        break
    
    description1 = row.iloc[0]
    description2 = row.iloc[1]

    # Tokenize descriptions into sentences using nltk
    sentences1 = nltk.sent_tokenize(description1)
    sentences2 = nltk.sent_tokenize(description2)

    for sentence1 in sentences1:
        for sentence2 in sentences2:
            row_data = [index + 1, sentence1, sentence2]  # Initialize row data with sentences
            for model_name in models:
                model = SentenceTransformer(model_name).to(device)
                encoding1 = model.encode(sentence1)
                encoding2 = model.encode(sentence2)
                similarity = np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
                row_data.append(similarity)  # Append each similarity score to the row data
                # print (f"Similarity Score of {model_name} = {similarity:.2f}")

            all_results.append(row_data)  # Add the complete row data to results
            

# Create DataFrame from the results
column_names = ['Row Number', 'Sentence1', 'Sentence2'] + models + ['IoU']
results_df = pd.DataFrame(all_results, columns=column_names)

# Use ExcelWriter to write to a specific sheet
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    results_df.to_excel(writer, sheet_name='Human 1 - Online - Score', index=False)

print("Similarity scores have been written to 'SentenceSimScore.xlsx'")
