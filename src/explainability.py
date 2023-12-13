from lime import lime_text
from shap import Explainer

def get_lime_explanation(model, text, label, tokenizer):
  """
  Generates a LIME explanation for the model's prediction on a given text.

  Args:
    model: The trained model.
    text: The text to explain.
    label: The predicted label for the text.
    tokenizer: The tokenizer used for preprocessing.

  Returns:
    A list of tuples where each tuple contains a word and its importance score.
  """
  text_encoded = tokenizer(text, return_tensors="tf")
  explainer = lime.lime_text.LimeTextExplainer(class_names=[str(label)])
  explanation = explainer.explain_instance(text, model.predict, labels=[0, 1], num_samples=100)

  return explanation.as_list()

def get_shap_explanation(model, text, label, tokenizer):
  """
  Generates a SHAP explanation for the model's prediction on a given text.

  Args:
    model: The trained model.
    text: The text to explain.
    label: The predicted label for the text.
    tokenizer: The tokenizer used for preprocessing.

  Returns:
    A list of tuples where each tuple contains a word and its SHAP value.
  """
  text_encoded = tokenizer(text, return_tensors="tf")
  explainer = shap.Explainer(model.predict, text_encoded)
  shap_values = explainer([text_encoded])

  word_importance = []
  for i, word in enumerate(tokenizer.convert_ids_to_tokens(text_encoded['input_ids'][0])):
    word_importance.append((word, shap_values[0][0, i]))

  return word_importance

def explain_prediction(model, text, label, tokenizer, explainer_type="lime"):
  """
  Provides an explanation for the model's prediction on a given text using LIME or SHAP.

  Args:
    model: The trained model.
    text: The text to explain.
    label: The predicted label for the text.
    tokenizer: The tokenizer used for preprocessing.
    explainer_type: The type of explainer to use ("lime" or "shap").

  Returns:
    A list of tuples where each tuple contains a word and its explanation importance score.
  """
  if explainer_type == "lime":
    return get_lime_explanation(model, text, label, tokenizer)
  elif explainer_type == "shap":
    return get_shap_explanation(model, text, label, tokenizer)
  else:
    raise ValueError("Invalid explainer type:", explainer_type)
