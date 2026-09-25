# Configuration settings for the vibration prediction project

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'train') # subject to change
VALIDATION_DATA_PATH = os.path.join(BASE_DIR, 'data', 'validation') # subject to change
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models')

# Model parameters
MODEL_TYPE = 'RandomForest'  # Example model type
N_ESTIMATORS = 100
MAX_DEPTH = None

# Hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 100

# Other configurations
RANDOM_SEED = 42
LOGGING_LEVEL = 'INFO'  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL