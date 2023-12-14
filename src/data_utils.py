import pandas as pd
from sklearn.model_selection import train_test_split
import random
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_extracted_data(file_path):
    """
    Loads the data extracted and saved by text_extraction.py.

    Args:
        file_path: Path to the CSV file containing extracted terms and labels.

    Returns:
        A DataFrame containing the terms and labels.
    """
    return pd.read_csv(file_path)

def augment_data(data, augmentation_factor=0.1):
    """
    Augments the given data using synonym replacement.

    Args:
        data: The original data (DataFrame with 'term' and 'label' columns).
        augmentation_factor: The proportion of data to augment.

    Returns:
        The augmented data as a DataFrame.
    """
    augmented_data = []
    for _, row in data.iterrows():
        term = row['term']
        label = row['label']
        words = nltk.word_tokenize(term)
        num_words_to_augment = int(len(words) * augmentation_factor)
        random_indices = random.sample(range(len(words)), num_words_to_augment)

        for i in random_indices:
            synonyms = wordnet.synsets(words[i])
            if synonyms:
                synonym = random.choice(synonyms).lemmas()[0].name()
                words[i] = synonym

        new_term = ' '.join(words)
        augmented_data.append({'term': new_term, 'label': label})

    augmented_df = pd.DataFrame(augmented_data)
    return pd.concat([data, augmented_df], ignore_index=True)

# Example usage
if __name__ == "__main__":
    extracted_data_file = 'data/extracted_terms.csv'
    extracted_data = load_extracted_data(extracted_data_file)
    augmented_data = augment_data(extracted_data)
    augmented_data.to_csv('data/augmented_terms.csv', index=False)
    print("Augmented data saved to data/augmented_terms.csv")
