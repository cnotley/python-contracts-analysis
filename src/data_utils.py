import os
import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def load_data(data_dir, tokenizer_name):
    """
    Loads contract data from a directory, tokenizes and preprocesses it.

    Args:
        data_dir: Path to the directory containing contract files.
        tokenizer_name: Name of the pre-trained tokenizer to use.

    Returns:
        A list of dictionaries where each dictionary represents a contract and contains the following keys:
          - text: The tokenized and preprocessed text of the contract.
          - label: The key terms extracted from the contract.
    """
    data = []
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, encoding="utf-8") as f:
            contract_data = json.load(f)
            text = contract_data["text"]
            tokenized_text = tokenizer(text, truncation=True, padding=True, return_tensors="pt")

            data.append({
                "text": tokenized_text,
                "label": contract_data["label"]
            })

    return data

def split_data(data, test_size=0.2):
    """
    Splits data into training and test sets.

    Args:
        data: A list of dictionaries representing contracts.
        test_size: The proportion of data to be used for testing.

    Returns:
        A tuple containing two lists:
          - train_data: Training data.
          - test_data: Test data.
    """
    train_data, test_data = train_test_split(data, test_size=test_size)
    return train_data, test_data

def get_labeled_data(data):
    """
    Extracts labeled data from a list of contracts.

    Args:
        data: A list of dictionaries representing contracts.

    Returns:
        A tuple containing two lists:
          - texts: The texts of all contracts.
          - labels: The corresponding key terms for each contract.
    """
    texts = [contract["text"] for contract in data]
    labels = [contract["label"] for contract in data]
    return texts, labels

def get_unlabeled_data(data):
    """
    Extracts unlabeled data from a list of contracts.

    Args:
        data: A list of dictionaries representing contracts.

    Returns:
        A list of dictionaries containing the text of unlabeled contracts.
    """
    unlabeled_data = []
    for contract in data:
        if not contract["label"]:
            unlabeled_data.append({"text": contract["text"]})
    return unlabeled_data

def augment_data(data, augmentation_function, num_augmented_per_original=2):
    augmented_data = []
    for item in data:
        augmented_texts = augmentation_function(item["text"], num_augmented_per_original)
        for text in augmented_texts:
            augmented_data.append({"text": text, "label": item["label"]})
    return data + augmented_data
