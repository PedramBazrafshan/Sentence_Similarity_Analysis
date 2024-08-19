### **Sentence Similarity Analysis**
---


### **Overview**
---
This is the official repository of the Research Manuscript _Performance analysis of pre-trained visual language models in describing images within the civil engineering domain: a case study of visual ChatGPT-4_ by Kris Melag et al.


### **Get Started**
---
1. Two types of semantic similarity analysis and lexical similarity analysis are performed in this research. SentenceTransfformers is used for the semantic analysis, and Intersection over Union (IoU) is used for the lexical analysis.

2. There are three Excel files, one for the benchmark analysis (BenchmarkSimScore.xlsx), and two for the description (DescriptionSimScore.xlsx) and sentence pair-wise analysis (SentenceSimScore.xlsx).

3. If "SentenceBenchMark.py" is executed, the code will open "BenchmarkSimScore.xlsx", which contains the benchmark sentences. Then, the semantic and lexical similarity analyses are performed and the results are saved in the same Excel file.

4. If "DescriptionAnalysis.py" is executed, the code will open "DescriptionSimScore.xlsx", which contains the human and ChatGPT descriptions for the online and private datasets. Then, the semantic similarity analysis is performed, and the results are saved in the same Excel file. This code performs the similarity analysis between the whole description of the human and that of ChatGPT.

5. If "SentenceAnalysis.py" is executed, the code will open "SentenceSimScore.xlsx", which contains the human and ChatGPT descriptions for the online and private datasets. Then, the semantic similarity analysis is performed, and the results are saved in the same Excel file. This code performs the similarity analysis between each sentence of the human description and that of ChatGPT.

6. When "SentenceAnalysis.py" is executed, the code calculates and saves the sentence pair-wise similarity scores for each of the human and ChatGPT descriptions. To report the maximum similarity score for each of the descriptions, "MaxCalc.py" should be executed. This code calculates the maximum similarity score among all of the sentences pair-wise for each description for all of the semantic models. The results are saved in the same Excel file. Please note that if an Excel Sheet with the name is already exist, a saving error will be encountered.


### **Developers**
---
The codes are developed by Pedram Bazrafshan.

### **License**
---
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only. Any commercial use should get formal permission first.


### **Inquiries**
---
For inquiries, please contact:  
pb669@drexel.edu
