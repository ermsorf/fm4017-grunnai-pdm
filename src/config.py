# Configuration settings for the vibration prediction project

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'train') # subject to change
VALIDATION_DATA_PATH = os.path.join(BASE_DIR, 'data', 'validation') # subject to change
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models')
