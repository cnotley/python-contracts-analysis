import fitz  # PyMuPDF
import os
import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def preprocess_for_bert(text, tokenizer, max_len=512):
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_len,
                                   truncation=True, padding="max_length", return_attention_mask=True, return_tensors="pt")
    return inputs

def extract_terms_with_legal_bert(model, tokenizer, text):
    inputs = preprocess_for_bert(text, tokenizer)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    extracted_terms = [tokens[i] for i in range(len(tokens)) if predictions[0][i] == 1]
    return extracted_terms

def main():
    tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = BertForTokenClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")

    data_dir = "data/pdfs"
    all_extracted_terms = []

    for file in os.listdir(data_dir):
        if file.endswith('.pdf'):
            text = extract_text_from_pdf(os.path.join(data_dir, file))
            terms = extract_terms_with_legal_bert(model, tokenizer, text)
            all_extracted_terms.extend(terms)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(all_extracted_terms)

    y = [1 if "contract" in term else 0 for term in all_extracted_terms]

    clf = RandomForestClassifier()
    clf.fit(X, y)

    extracted_data = pd.DataFrame({'term': all_extracted_terms, 'label': y})
    extracted_data.to_csv('data/extracted_terms.csv', index=False)

    print(f"Extracted terms saved to data/extracted_terms.csv")

if __name__ == "__main__":
    main()
