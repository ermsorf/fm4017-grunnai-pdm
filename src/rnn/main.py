from predict import make_predictions
from preprocessing import preprocess_data
from train import train_model
from validate import validate_model


def run(data_directory="data/raw/2024-06-01_2024-10-31", model_path="models/rnn_model.pkl"):
    data = preprocess_data(data_directory)
    model = train_model(data_directory, model_path)
    metrics = validate_model(model, data)
    predictions = make_predictions(model, data)
    return metrics, predictions


if __name__ == "__main__":
    metrics, predictions = run()
    print(metrics)
    print(f"Generated {len(predictions)} predictions")