


"""Code developed and presented by Pedram Bazrafshan"""



import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from openpyxl import load_workbook
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
# import nltk

# # Download necessary NLTK resources
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# File path to your Excel file
file_path = r"BenchmarkSimScore.xlsx"

# Read from the specified sheet
df = pd.read_excel(file_path, sheet_name='BenchMark', usecols=[0,1], header=0)

# Models for similarity calculation (SentenceTransformers only)
SBERT_Models = [
    "all-mpnet-base-v2", "distilbert-base-nli-mean-tokens", "bert-base-uncased",
    "multi-qa-mpnet-base-dot-v1", "all-distilroberta-v1", "all-MiniLM-L12-v2",
    "multi-qa-distilbert-cos-v1", "all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1",
    "paraphrase-multilingual-mpnet-base-v2", "paraphrase-albert-small-v2",
    "paraphrase-multilingual-MiniLM-L12-v2", "paraphrase-MiniLM-L3-v2",
    "distiluse-base-multilingual-cased-v1", "distiluse-base-multilingual-cased-v2"
]


results = []

# Define the maximum row index to process
max_row_index = 12

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    print("Processing row index =", index)
    if index > max_row_index:
        break

    # description1 versus description2 similarity analysis from the first four columns
    description1 = row.iloc[0]
    description2 = row.iloc[1]

    row_scores = []

    # Calculate similarities using SentenceTransformer models
    for model_type in SBERT_Models:
        model = SentenceTransformer(model_type).to(device)
        encoding_human = model.encode(description1)
        encoding_vlm = model.encode(description2)
        similarity = np.dot(encoding_human, encoding_vlm) / (np.linalg.norm(encoding_human) * np.linalg.norm(encoding_vlm))
        row_scores.append(similarity)

    # Compute BLEU score with smoothing to avoid zero n-gram overlaps
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu([description1.split()], description2.split(), smoothing_function=smoothing_function)
    row_scores.append(bleu_score)

    # Compute ROUGE-L score
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_score = rouge_scorer_instance.score(description1, description2)['rougeL'].fmeasure
    row_scores.append(rouge_l_score)

    # Compute METEOR score (requires tokenized reference and hypothesis)
    meteor = meteor_score([description1.split()], description2.split())  # Reference as a list of lists
    row_scores.append(meteor)
    
    # Compute IoU similarity
    human_set = set(description1.split())
    vlm_set = set(description2.split())
    intersection = human_set.intersection(vlm_set)
    union = human_set.union(vlm_set)
    iou_score = len(intersection) / len(union) if len(union) != 0 else 0
    row_scores.append(iou_score)

    results.append(row_scores)


# Column letters for Excel (B, D, F, ...)
excel_columns = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V']

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
