import pandas as pd
import pytest
from src.data_loader import DataLoader

@pytest.fixture
def data_loader():
    data_path = 'data/raw/2024-06-01_2024-10-31/Scada.GRUN.AGG1.G1.BLAGER.M_LAGTMP.csv'
    return DataLoader(data_path, data_path)

def test_load_train_data(data_loader):
    train_data = data_loader.load_train_data()
    assert isinstance(train_data, pd.DataFrame)
    assert not train_data.empty
    assert 'identifier' in train_data.columns

def test_load_validation_data(data_loader):
    validation_data = data_loader.load_validation_data()
    assert isinstance(validation_data, pd.DataFrame)
    assert not validation_data.empty
    assert 'identifier' in validation_data.columns
