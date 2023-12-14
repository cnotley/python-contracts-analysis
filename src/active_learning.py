import numpy as np
from sklearn.utils import shuffle
from typing import List, Tuple, Callable
from transformers import BertTokenizer
import tensorflow as tf
import pandas as pd

def uncertainty_sampling(model, file_path, sample_size: int):
    """
    Selects data points with the highest uncertainty (lowest confidence) for labeling using a trained model.

    Args:
        model: The trained model.
        file_path: Path to the CSV file containing data for sampling.
        sample_size: The number of data points to select.

    Returns:
        A DataFrame with selected data points.
    """
    data = pd.read_csv(file_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    inputs = tokenizer(data['term'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="tf")
    predictions = model.predict(inputs)

    uncertainties = 1 - np.max(predictions, axis=1)

    indices = np.argsort(uncertainties)[-sample_size:]
    selected_data = data.iloc[indices]

    return selected_data

if __name__ == "__main__":
    example_model = None
    file_path = 'data/augmented_terms.csv'
    selected_data = uncertainty_sampling(example_model, file_path, sample_size=10)
    selected_data.to_csv('data/selected_data_for_labeling.csv', index=False)
    print("Selected data saved to data/selected_data_for_labeling.csv")
