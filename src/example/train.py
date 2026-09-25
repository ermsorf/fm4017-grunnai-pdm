from src.data_loader import DataLoader
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.model import Model
import os

def train_model():
    # Load the training data
    data_loader = DataLoader()
    train_data = data_loader.load_train_data(os.path.join('data', 'train', 'lerp_10s_G2'))
    
    # Preprocess the data
    processed_data = preprocess_data(train_data)
    
    # Create features for the model
    features, labels = create_features(processed_data)
    
    # Initialize and train the model
    model = Model()
    model.train(features, labels)
    
    # Save the trained model
    model.save(os.path.join('models', 'trained_model.pkl'))

if __name__ == "__main__":
    train_model()