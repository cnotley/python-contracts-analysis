import argparse
from data_utils import load_extracted_data, augment_data
from models import BiLSTM_ResNet, BertForTermExtraction, EnsembleModel
from training import train_model
from active_learning import uncertainty_sampling
from explainability import get_shap_explanation, get_lime_explanation
import pandas as pd
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str, help="Path to the data directory")
    parser.add_argument("--model_name", default='bert-base-uncased', type=str, help="Name of the BERT model")
    parser.add_argument("--max_length", default=512, type=int, help="Maximum sequence length for the model")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs for training")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate for training")

    args = parser.parse_args()

    extracted_data = load_extracted_data(f'{args.data_dir}/extracted_terms.csv')
    augmented_data = augment_data(extracted_data)
    augmented_data_file = f'{args.data_dir}/augmented_terms.csv'
    augmented_data.to_csv(augmented_data_file, index=False)

    max_features = 10000
    bilstm_resnet_model = BiLSTM_ResNet(max_features, embedding_dim=128, lstm_units=64, num_res_blocks=2)
    bert_model = BertForTermExtraction(args.model_name)
    ensemble_model = EnsembleModel(max_features, embedding_dim=128, lstm_units=64, num_res_blocks=2, bert_model_name=args.model_name)
    trained_model, training_history = train_model(ensemble_model, augmented_data_file, args.max_length, args.batch_size, args.epochs, args.learning_rate)

    selected_data = uncertainty_sampling(trained_model, augmented_data_file, sample_size=10)
    selected_data_file = f'{args.data_dir}/selected_data_for_labeling.csv'
    selected_data.to_csv(selected_data_file, index=False)

    example_text = extracted_data['term'].iloc[0]
    predicted_label = trained_model.predict(tf.convert_to_tensor([example_text]))[0][0]
    shap_explanation = get_shap_explanation(trained_model, bert_model.tokenizer, example_text, predicted_label)
    lime_explanation = get_lime_explanation(trained_model, example_text, predicted_label, args.model_name)

    print("SHAP explanation:", shap_explanation)
    print("LIME explanation:", lime_explanation)

if __name__ == "__main__":
    main()
