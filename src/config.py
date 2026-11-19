# Configuration settings for the vibration prediction project

import os
from dataclasses import dataclass, field
from collections.abc import Mapping

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(BASE_DIR, 'data')  # subject to change
DATA_RAW_2024 = os.path.join(DATA_FOLDER, 'raw', '2024-06-01_2024-10-31')  # subject to change
DATA_RAW_2025 = os.path.join(DATA_FOLDER, 'raw', '2025-06-01_2025-10-31')  # subject to change
DATA_PROCESSED = os.path.join(DATA_FOLDER, 'processed')  # subject to change

TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'train') # subject to change
VALIDATION_DATA_PATH = os.path.join(BASE_DIR, 'data', 'validation') # subject to change
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "src", 'models')


@dataclass(frozen=True)
class RecurrentModelConfig:
    """Shared settings used by the LSTM and GRU training/evaluation scripts."""

    training_data_folder: str = os.path.join(
        DATA_PROCESSED, "2024-06-02_2024-10-31_2NI"
    )
    evaluation_data_folder: str = os.path.join(
        DATA_PROCESSED, "2025-06-02_2025-10-31_2NI"
    )
    validation_data_folder: str | None = None
    # os.path.join(
    #     DATA_PROCESSED, "2025-06-02_2025-10-31_stepinterp_2s"
    # )
    sequence_length: int = 4
    hidden_size: int = 16
    num_layers: int = 4
    num_epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 0.00001
    early_stopping_patience: int | None = 10
    predictor_contains: tuple[str, ...] = ("MW", "NALER", "TURT", "VANNSTR")
    predictor_exclude: tuple[str, ...] = ("VIBR", "LAGTMP6")
    predictor_features: tuple[str, ...] = (
        "value",
        # "mask",
        # "age"
        )
    reference_contains: tuple[str, ...] = ("VIBR1")
    reference_exclude: tuple[str, ...] = ()
    exclude_below: Mapping[str, float] | None = None #field(
        # default_factory=lambda: {
            # "VIBR1": 0.1,
            # "VIBR2": 0.1,
            # "TURT": 90.0,
            # "MW": 0.5,
        # }
    # )
    exclude_before_samples: int = 0
    exclude_after_samples: int = 0
    exclude_before: str | None = None
    exclude_after: str | None = None
    training_normalization_parameters: str | None = os.path.join(
        DATA_PROCESSED,
        "2024-06-02_2024-10-31_2NI",
        "normalization_parameters.json",
    )
    evaluation_normalization_parameters: str | None = None


SEQUENCE_MODEL_CONFIG = RecurrentModelConfig()


@dataclass(frozen=True)
class XGBoostModelConfig:
    """Shared data and estimator settings used by XGBoost training."""

    training_data_folder: str = os.path.join(
        DATA_PROCESSED, "2024-06-02_2024-10-31_2NI"
    )
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    predictor_contains: tuple[str, ...] = ("AGG2")
    predictor_exclude: tuple[str, ...] = ("VIBR", "TMP")
    predictor_features: tuple[str, ...] = ("value",)
    reference_contains: tuple[str, ...] = ("VIBR1","VIBR2")
    reference_exclude: tuple[str, ...] = ()
    exclude_below: Mapping[str, float] | None = None # field(
    #     default_factory=lambda: {
    #         "VIBR1": 0.1,
    #         "VIBR2": 0.1,
    #         "TURT": 90.0,
    #     }
    # )
    n_estimators: int = 512
    max_depth: int = 64
    learning_rate: float = 0.0001
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    objective: str = "reg:squarederror"
    tree_method: str = "hist"
    device: str = "cuda"
    random_state: int = 42


XGBOOST_MODEL_CONFIG = XGBoostModelConfig()


@dataclass(frozen=True)
class ANNModelConfig:
    """Settings for the non-recurrent per-timestep ANN baseline."""

    training_data_folder: str = os.path.join(
        DATA_PROCESSED, "2024-06-02_2024-10-31_stepinterp_2s"
    )
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    # An empty include list means all available processed signals.
    predictor_contains: tuple[str, ...] = ("NALER", "MW", "synth")
    # Keep vibration targets out of the predictors to avoid target leakage.
    predictor_exclude: tuple[str, ...] = ("VIBR",)
    predictor_features: tuple[str, ...] = ("value",)
    reference_contains: tuple[str, ...] = ("VIBR2",)
    reference_exclude: tuple[str, ...] = ()
    exclude_below: Mapping[str, float] | None = None #field(
    #     default_factory=lambda: {"VIBR2": 0.1}
    # )
    hidden_sizes: tuple[int, ...] = (16,16,8,8)
    dropout: float = 0.0
    num_epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 0.001
    early_stopping_patience: int | None = 10
    device: str | None = None


ANN_MODEL_CONFIG = ANNModelConfig()
