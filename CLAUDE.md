# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for "Semantic and Lexical Analysis of Pre-Trained Vision Language Artificial Intelligence Models for Automated Image Descriptions in Civil Engineering" by Pedram Bazrafshan et al.

The project performs semantic and lexical similarity analysis comparing human-written and AI-generated (ChatGPT) image descriptions using multiple transformer models and NLP metrics.

## Core Architecture

### Data Flow Pipeline

The analysis follows a specific execution order:

1. **Benchmark Analysis** (`SentenceBenchMark.py`)
   - Independent analysis of benchmark sentences
   - Input: `BenchmarkSimScore.xlsx` (sheet: 'BenchMark', columns A-B)
   - Applies 15 SentenceTransformer models + BLEU + ROUGE-L + METEOR + IoU
   - Writes results to columns D-V

2. **Description-Level Analysis** (`DescriptionAnalysis.py`)
   - Whole-description similarity comparison
   - Input: `DescriptionSimScore.xlsx` (configurable: 'Online - Data' or 'Private - Data', columns 0-1)
   - Uses 5 core semantic models
   - Writes to sheet 'Human 1 - Online - Score' (columns B-F)

3. **Sentence-Level Analysis** (`SentenceAnalysis.py`)
   - Sentence-by-sentence pairwise comparison
   - Input: `SentenceSimScore.xlsx` (configurable sheet/columns like DescriptionAnalysis)
   - Tokenizes descriptions into sentences using NLTK
   - Creates all pairwise combinations between human/ChatGPT sentences
   - Uses 5 core semantic models
   - Output sheet name pattern: 'Human X - [Dataset] - Score'

4. **Max Score Calculation** (`MaxCalc.py`)
   - Post-processes sentence-level results
   - Finds maximum similarity score across all sentence pairs per description
   - Reads from sentence-level score sheets (columns D-H)
   - Creates new sheets with "_max" suffix
   - **Note**: Will error if output sheet already exists

### Model Configuration

**Benchmark Models** (15 SentenceTransformers):
```
all-mpnet-base-v2, distilbert-base-nli-mean-tokens, bert-base-uncased,
multi-qa-mpnet-base-dot-v1, all-distilroberta-v1, all-MiniLM-L12-v2,
multi-qa-distilbert-cos-v1, all-MiniLM-L6-v2, multi-qa-MiniLM-L6-cos-v1,
paraphrase-multilingual-mpnet-base-v2, paraphrase-albert-small-v2,
paraphrase-multilingual-MiniLM-L12-v2, paraphrase-MiniLM-L3-v2,
distiluse-base-multilingual-cased-v1, distiluse-base-multilingual-cased-v2
```

**Core Analysis Models** (5 SentenceTransformers):
```
distilbert-base-nli-mean-tokens, bert-base-uncased,
multi-qa-mpnet-base-dot-v1, paraphrase-multilingual-mpnet-base-v2,
paraphrase-multilingual-MiniLM-L12-v2
```

**Additional Metrics**:
- BLEU (with smoothing)
- ROUGE-L (F-measure)
- METEOR
- IoU (Intersection over Union)

### GPU Utilization

All scripts automatically detect and use GPU if available via PyTorch:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Models are moved to the device with `.to(device)` before encoding.

### Key Configuration Points

Each script contains hard-coded configuration that may need adjustment:

**DescriptionAnalysis.py & SentenceAnalysis.py**:
- `sheet_name`: Toggle between 'Online - Data' and 'Private - Data'
- `usecols=[0, 1]`: Column indices for different human annotators
- `max_row_index`: Number of rows to process (currently 82)
- Output sheet naming pattern: 'Human X - [Dataset] - Score'

**MaxCalc.py**:
- Input sheet name (line 12)
- Output sheet name (line 24)
- Column ranges (line 16: columns 3:8 = columns D-H in Excel)

**SentenceBenchMark.py**:
- `max_row_index`: Currently 12 for benchmark data

## Running the Code

### Dependencies

```bash
pip install numpy pandas openpyxl torch sentence-transformers nltk rouge-score
```

For NLTK first-time setup, uncomment and run in Python:
```python
import nltk
nltk.download('punkt')      # For sentence tokenization
nltk.download('wordnet')     # For METEOR
nltk.download('omw-1.4')     # For METEOR
```

### Execution

Run scripts directly with Python 3.11+:

```bash
# Independent benchmark analysis
python3 SentenceBenchMark.py

# Description-level similarity
python3 DescriptionAnalysis.py

# Sentence-level pairwise similarity
python3 SentenceAnalysis.py

# Calculate max scores from sentence-level results
python3 MaxCalc.py
```

**Important**: Before running `MaxCalc.py`, manually delete the output sheet if it exists in the Excel file to avoid save errors.

### Switching Datasets/Annotators

To analyze different datasets or annotators in `DescriptionAnalysis.py` or `SentenceAnalysis.py`:

1. Change `sheet_name='Online - Data'` to `sheet_name='Private - Data'`
2. Adjust `usecols=[0, 1]` to select different annotator columns
3. Update the output sheet name in the ExcelWriter section (line 76 in SentenceAnalysis, line 71 in DescriptionAnalysis)

## Excel File Structure

- **BenchmarkSimScore.xlsx**: Benchmark sentence pairs with similarity scores
- **DescriptionSimScore.xlsx**: Complete human/ChatGPT descriptions for Online/Private datasets
- **SentenceSimScore.xlsx**: Same descriptions as above, processed sentence-by-sentence

All Excel files contain both input data and computed results. Scripts use `openpyxl` to read and write to specific cells/sheets without overwriting unrelated data.

## Code Patterns

### Similarity Calculation
All semantic similarity uses cosine similarity via dot product:
```python
similarity = np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
```

### Excel I/O Pattern
1. Read with pandas: `pd.read_excel(file_path, sheet_name=..., usecols=...)`
2. Write with openpyxl: Load workbook, access sheet, write to specific cells
3. Save: `workbook.save(output_path)`

### Result Storage
- SentenceBenchMark/DescriptionAnalysis: Write to specific columns in existing sheet
- SentenceAnalysis: Create/replace entire sheet with DataFrame
- MaxCalc: Append new sheet to workbook

## License

Non-commercial use only under Creative Commons Attribution-NonCommercial 4.0 International.
