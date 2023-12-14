import shap
import lime
from lime import lime_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
import pandas as pd

def get_shap_explanation(model, tokenizer, text, label):
    ''' Generates a SHAP explanation for the model's prediction on a given text.
    
    Args:
      model: The trained model.
      text: The text to explain.
      label: The predicted label for the text.
      tokenizer: The tokenizer used for preprocessing.

    Returns:
      SHAP explanation visualization.
    '''
    # Tokenize the input text and convert to tensor
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # Create a SHAP explainer and get the shap values
    explainer = shap.Explainer(model, tokenizer)
    shap_values = explainer(inputs)

    # Plot the shap values
    shap.plots.text(shap_values[:,:,label])

def get_lime_explanation(model, text, label, tokenizer_name):
    ''' Generates a LIME explanation for the model's prediction on a given text.

    Args:
      model: The trained model.
      text: The text to explain.
      label: The predicted label for the text.
      tokenizer_name: The name of the tokenizer used for preprocessing.

    Returns:
      A list of tuples where each tuple contains a word and its corresponding weight in the decision process.
    '''
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def predict_proba(texts):
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            predictions = model(**tokens)
        return predictions.logits.softmax(dim=-1).numpy()

    explainer = lime_text.LimeTextExplainer(class_names=[0, 1])

    exp = explainer.explain_instance(text, predict_proba, num_features=10)
    exp.show_in_notebook(text=True)
    
    return exp.as_list(label=label)

if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained("your-model-path")
    tokenizer_name = "bert-base-uncased"
    
    data = pd.read_csv('data/extracted_terms.csv')
    example_text = data['term'].iloc[0]
    predicted_label = data['label'].iloc[0]
    
    # Get explanations
    shap_explanation = get_shap_explanation(model, tokenizer, example_text, predicted_label)
    lime_explanation = get_lime_explanation(model, example_text, predicted_label, tokenizer_name)
