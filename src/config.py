# Configuration settings for the vibration prediction project

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, 'data')  # subject to change
DATA_RAW_2024 = os.path.join(DATA_FOLDER, 'raw', '2024-06-01_2024-10-31')  # subject to change
DATA_RAW_2025 = os.path.join(DATA_FOLDER, 'raw', '2025-06-01_2025-10-31')  # subject to change
DATA_PROCESSED = os.path.join(DATA_FOLDER, 'processed')  # subject to change

TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'train') # subject to change
VALIDATION_DATA_PATH = os.path.join(BASE_DIR, 'data', 'validation') # subject to change
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "src", 'models')
