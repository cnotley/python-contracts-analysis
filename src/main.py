import argparse
import os
from data_utils import load_data, split_data, get_labeled_data, get_unlabeled_data
from models import BiLSTM_ResNet, BERT_Model, EnsembleModel
from training import train_model, load_model
from active_learning import uncertainty_sampling, active_learning
from explainability import explain_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str, help="Path to the data directory")
    parser.add_argument("--model_type", choices=["bilstm", "bert", "ensemble"], default="bilstm", help="Type of model to train")
    parser.add_argument("--pre_trained_model", type=str, help="Name of the pre-trained model for transfer learning")
    parser.add_argument("--use_active_learning", action="store_true", help="Use active learning for data selection")
    parser.add_argument("--explainer_type", choices=["lime", "shap"], default="lime", help="Type of explainer to use")
    args = parser.parse_args()

    data = load_data(args.data_dir)
    train_data, test_data = split_data(data)
    texts, labels = get_labeled_data(train_data)

    if args.model_type == "bilstm":
        model = BiLSTM_ResNet()
    elif args.model_type == "bert":
        model = BERT_Model(args.pre_trained_model)
    elif args.model_type == "ensemble":
        model = EnsembleModel([BiLSTM_ResNet(), BERT_Model(args.pre_trained_model)])

    if args.use_active_learning:
        unlabeled_data = get_unlabeled_data(train_data)
        labeled_data, unlabeled_data = active_learning(model, unlabeled_data, active_learning_strategy=uncertainty_sampling)
        texts, labels = get_labeled_data(labeled_data)

    trained_model = train_model(model, texts, labels)

    test_texts, test_labels = get_labeled_data(test_data)

    test_loss, test_accuracy = trained_model.evaluate(test_texts, test_labels)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    trained_model.save("trained_model.hdf5")


if __name__ == "__main__":
    main()
