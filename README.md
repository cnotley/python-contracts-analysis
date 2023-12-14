# Advanced Contract Analysis AI Project

## Overview
This project leverages AI and NLP techniques to analyze contract documents. By extracting, processing, and classifying key terms from contract PDFs, it aims to streamline the contract review process, enhancing both efficiency and accuracy.

## Objectives
- **Automate Key Term Extraction and Classification**: Automatically extract and classify significant terms from contract documents.
- **Apply Advanced NLP Techniques**: Utilize transformer models like BERT for sophisticated natural language understanding.
- **Incorporate Active Learning**: Enhance model performance over time through targeted data sampling.
- **Enable Model Explainability**: Provide insights into model predictions using SHAP and LIME.

## Project Structure
```
python-contracts-analysis/
│
├── src/                        # Source code directory
│   ├── term_extraction_and_classification.py # Extracts and classifies terms from PDFs
│   ├── data_utils.py           # Data handling and augmentation
│   ├── models.py               # Deep learning models for NLP
│   ├── training.py             # Training routines for the models
│   ├── active_learning.py      # Active learning implementations
│   ├── explainability.py       # SHAP and LIME explainability methods
│   └── main.py                 # Main script for end-to-end execution
│
├── data/                       # Data directory
│   ├── pdfs/                   # PDF documents for analysis
│   └── ...                     # Other data files (CSVs, etc.)
│
├── models/                     # Trained model files
│
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Installation
1. **Set Up Your Environment**:
   - Create and activate a Python virtual environment.
   - Install dependencies: `pip install -r requirements.txt`.

2. **Prepare Your Data**:
   - Place your contract PDFs in `data/pdfs`.

## Running the Project

### Initial Data Extraction and Classification
```bash
python src/term_extraction_and_classification.py
```
- Extracts and classifies terms from PDFs.
- Outputs initial data in `data/extracted_terms.csv`.

### Full Pipeline Execution
```bash
python src/main.py --data_dir data
```
- Processes and augments data.
- Trains and evaluates the models.
- Applies active learning.
- Generates model explanations.

### Expected Outputs
- Augmented data: `data/augmented_terms.csv`.
- Active learning selections: `data/selected_data_for_labeling.csv`.
- Explainability insights in the console.

## Technologies Used
- TensorFlow for deep learning models.
- Transformers for utilizing pre-trained BERT models.
- NLTK for data augmentation.
- SHAP and LIME for model explainability.