from sklearn.utils import shuffle

def uncertainty_sampling(model, unlabeled_data, sample_size):
  """
  Selects data points with the highest uncertainty (lowest confidence) for labeling.

  Args:
    model: The trained model.
    unlabeled_data: The unlabeled data.
    sample_size: The number of data points to select.

  Returns:
    A list of selected data points.
  """
  predictions = model.predict(unlabeled_data["text"])
  uncertainties = abs(predictions - 0.5)
  sorted_indices = np.argsort(uncertainties)
  selected_indices = sorted_indices[-sample_size:]
  return [unlabeled_data[i] for i in selected_indices]

def margin_sampling(model, unlabeled_data, sample_size):
  """
  Selects data points closest to the decision boundary (margin) for labeling.

  Args:
    model: The trained model.
    unlabeled_data: The unlabeled data.
    sample_size: The number of data points to select.

  Returns:
    A list of selected data points.
  """
  predictions = model.predict(unlabeled_data["text"])
  margins = predictions - np.minimum(predictions, 1 - predictions)
  sorted_indices = np.argsort(margins)
  selected_indices = sorted_indices[-sample_size:]
  return [unlabeled_data[i] for i in selected_indices]

def active_learning(model, unlabeled_data, n_iterations=5, active_learning_strategy=uncertainty_sampling):
  """
  Performs active learning by iteratively selecting data points for labeling.

  Args:
    model: The trained model.
    unlabeled_data: The unlabeled data.
    n_iterations: The number of iterations.
    active_learning_strategy: The strategy to select data points (uncertainty_sampling or margin_sampling).

  Returns:
    A tuple containing two lists:
      - labeled_data: The labeled data.
      - unlabeled_data: The remaining unlabeled data.
  """
  labeled_data = []
  for _ in range(n_iterations):
    unlabeled_data, selected_data = shuffle(unlabeled_data)
    selected_data = active_learning_strategy(model, selected_data, sample_size=10)
    labeled_data.extend(selected_data)
    unlabeled_data = [data for data in unlabeled_data if data not in selected_data]
  return labeled_data, unlabeled_data
