import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, GlobalMaxPool1D, Dropout, Dense
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class BiLSTM_ResNet(tf.keras.Model):
    """
    A BiLSTM with ResNet architecture for key term extraction.
    """
    def __init__(self, max_features, embedding_dim=128, lstm_units=64, num_res_blocks=2):
        super(BiLSTM_ResNet, self).__init__()
        self.embedding_layer = Embedding(max_features, embedding_dim)
        self.bilstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=True))
        self.res_blocks = [self._build_residual_block(lstm_units) for _ in range(num_res_blocks)]
        self.pooling_layer = GlobalMaxPool1D()
        self.dense_layer = Dense(1, activation='sigmoid')

    def _build_residual_block(self, units):
        def layer(x):
            lstm_out = Bidirectional(LSTM(units, return_sequences=True))(x)
            return tf.keras.layers.add([x, lstm_out])
        return layer

    def call(self, x):
        x = self.embedding_layer(x)
        x = self.bilstm_layer(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.pooling_layer(x)
        return self.dense_layer(x)

class BertForTermExtraction(tf.keras.Model):
    """
    BERT model for key term extraction in contract documents.
    """
    def __init__(self, model_name='bert-base-uncased'):
        super(BertForTermExtraction, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name)
        self.dense_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        outputs = self.bert(inputs)[1]
        return self.dense_layer(outputs)

class EnsembleModel(tf.keras.Model):
    """
    Ensemble of BiLSTM_ResNet and BERT for robust key term extraction.
    """
    def __init__(self, max_features, embedding_dim, lstm_units, num_res_blocks, bert_model_name):
        super(EnsembleModel, self).__init__()
        self.bilstm_resnet = BiLSTM_ResNet(max_features, embedding_dim, lstm_units, num_res_blocks)
        self.bert_model = BertForTermExtraction(bert_model_name)
        self.combined_dense = Dense(1, activation='sigmoid')

    def call(self, inputs):
        bilstm_input, bert_input = inputs
        bilstm_output = self.bilstm_resnet(bilstm_input)
        bert_output = self.bert_model(bert_input)
        combined_output = tf.concat([bilstm_output, bert_output], axis=1)
        return self.combined_dense(combined_output)
