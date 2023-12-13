from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, GlobalMaxPool1D, Dropout, Dense
from transformers import AutoTokenizer, TFAutoModel

class BiLSTM_ResNet(tf.keras.Model):
  """
  A BiLSTM with residual connections for key term extraction.
  """
  def __init__(self, embedding_dim=128, filters=64, num_blocks=2):
    super(BiLSTM_ResNet, self).__init__()
    self.embedding_layer = Embedding(max_features, embedding_dim)
    self.lstm_layer = Bidirectional(LSTM(units=filters, return_sequences=True))
    self.residual_layers = [tf.keras.layers.Dense(filters, activation="relu") for _ in range(num_blocks)]
    self.global_max_pool_layer = GlobalMaxPool1D()
    self.dropout_layer = Dropout(0.5)
    self.dense_layer = Dense(1, activation="sigmoid")

  def call(self, inputs, **kwargs):
    x = self.embedding_layer(inputs)
    residual = x
    for layer in self.residual_layers:
      x = self.lstm_layer(x)
      x = layer(x)
      x = x + residual
      residual = x
    x = self.global_max_pool_layer(x)
    x = self.dropout_layer(x)
    return self.dense_layer(x)

class BERT_Model(tf.keras.Model):
  """
  A pre-trained BERT model fine-tuned for key term extraction.
  """
  def __init__(self, bert_model_name="bert-base-uncased", num_classes=1):
    super(BERT_Model, self).__init__()
    self.bert_model = TFAutoModel.from_pretrained(bert_model_name)
    self.dropout_layer = Dropout(0.5)
    self.dense_layer = Dense(num_classes, activation="sigmoid")

  def call(self, inputs, **kwargs):
    outputs = self.bert_model(inputs)
    x = outputs.last_hidden_state[:, 0, :]
    x = self.dropout_layer(x)
    return self.dense_layer(x)

class EnsembleModel(tf.keras.Model):
  """
  An ensemble model combining BiLSTM_ResNet and BERT_Model.
  """
  def __init__(self, models):
    super(EnsembleModel, self).__init__()
    self.models = models

  def call(self, inputs, **kwargs):
    outputs = []
    for model in self.models:
      outputs.append(model(inputs))
    return tf.keras.backend.mean(outputs, axis=0)
