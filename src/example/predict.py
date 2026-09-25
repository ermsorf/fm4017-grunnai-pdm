from src.data_loader import DataLoader
from src.model import Model
import numpy as np

def make_predictions():
    # Load the validation data
    data_loader = DataLoader()
    validation_data = data_loader.load_validation_data()

    # Load the trained model
    model = Model()
    model.load('models/trained_model.h5')  # Adjust the path as necessary

    # Make predictions
    predictions = model.predict(validation_data)

    # Return predictions
    return predictions

if __name__ == "__main__":
    preds = make_predictions()
    print(preds)