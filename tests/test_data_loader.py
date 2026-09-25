import os
import pandas as pd
import pytest
from src.data_loader import DataLoader

@pytest.fixture
def data_loader():
    return DataLoader()

def test_load_train_data(data_loader):
    train_data_path = os.path.join('data', 'train', 'lerp_10s_G2')
    train_data = data_loader.load_train_data(train_data_path)
    assert isinstance(train_data, pd.DataFrame)
    assert not train_data.empty
    assert 'process_value' in train_data.columns
    assert 'vibration_value' in train_data.columns

def test_load_validation_data(data_loader):
    validation_data_path = os.path.join('data', 'validation', 'lerp_10s_G2')
    validation_data = data_loader.load_validation_data(validation_data_path)
    assert isinstance(validation_data, pd.DataFrame)
    assert not validation_data.empty
    assert 'vibration_value' in validation_data.columns