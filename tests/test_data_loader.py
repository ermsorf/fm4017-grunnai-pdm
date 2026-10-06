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


def _write_csv(path):
    path.write_text(
        "identifier,value\nunit,unit\n2024-01-01 00:00:00,1\n",
        encoding="utf-8",
    )


def test_load_data_selects_directory_files(tmp_path):
    _write_csv(tmp_path / "Scada.AGG1.signal.csv")
    _write_csv(tmp_path / "Scada.AGG2.signal.csv")
    _write_csv(tmp_path / "Hydrocord.AGG2.signal.csv")

    data = DataLoader().load_data(
        tmp_path,
        filename_contains=["AGG2"],
        filename_exclude="Hydrocord",
    )

    assert len(data) == 2
    assert data.iloc[0, 0] == "unit"


def test_load_data_supports_regex_and_recursive_selection(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_csv(nested / "Scada.AGG2.signal.csv")

    data = DataLoader().load_data(
        tmp_path,
        filename_matches=r"AGG2\.signal\.csv$",
        recursive=True,
    )

    assert len(data) == 2
