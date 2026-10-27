from pathlib import Path
from math import sqrt
import tensorflow as tf
import joblib


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.data_analysis.data_pre_processing import load_signal

DATA_FOLDER = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "2024_new_interpol"
)

PROCESS_SIGNALS = [
    "Scada.GRUN.AGG2.G2.MV.M_P_MW",
    # "Scada.GRUN.AGG2.G2.MAGN.M_I",
    # "Scada.GRUN.AGG2.TURB2.REG.M_TURT",
    # "Scada.GRUN.AGG2.TURB2.PADRAG.M_WCKT_POS",
    # "Scada.GRUN.AGG2.G2.LAGER.M_LAGTMP1",
    # "Scada.GRUN.AGG2.G2.LAGER.M_LAGTMP4",
    # "Scada.GRUN.AGG2.G2.LAGER.M_LAGTMP5",
    # "Scada.GRUN.AGG2.G2.LAGER.M_LAGTMP6",
    # "Scada.GRUN.AGG2.G2.LAGER.M_OLTMP1",
    # "Scada.GRUN.AGG2.G2.LAGER.M_OLTMP2",
    # "Scada.GRUN.AGG2.G2.KJOL.M_KJVANN",
    "Scada.GRUN.AGG2.VANN.SJAKT.M_TRYKK",
    "Scada.GRUN.AGG2.TURB2.REG.M_VANNSTR",
    # "Scada.GRUN.AGG2.TURB2.TRYKKOL.M_OLTMP",
    "Scada.GRUN.AGG2.TURB2.TRYKKOL.M_OLTRYKSW",
    "Scada.GRUN.AGG2.TURB2.NALER.M_NAL1",
    "Scada.GRUN.AGG2.TURB2.NALER.M_NAL2",
    "Scada.GRUN.AGG2.TURB2.NALER.M_NAL3",
    "Scada.GRUN.AGG2.TURB2.NALER.M_NAL4",
    "Scada.GRUN.AGG2.TURB2.NALER.M_NAL5"
]
TARGET_SIGNALS = [
    # "Scada.GRUN.AGG2.G2.LAGER.M_VIBR1",
    "Scada.GRUN.AGG2.G2.LAGER.M_VIBR2",
]

TARGET_NAME = TARGET_SIGNALS[0].split("_")[-1].lower()

# def create_segment_dataset(
#     signals_dataframe: pd.DataFrame,
#     process_signals: list[str],
#     target_signals: list[str],
#     change_points: list[int],
# ) -> pd.DataFrame:
#     selected = signals_dataframe[process_signals + target_signals].dropna()
#     boundaries = [0] + sorted(set(change_points)) + [len(selected)]
#     rows = []

#     for segment_id, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
#         segment = selected.iloc[start:end]
#         if segment.empty:
#             continue

#         row = {"segment_id": segment_id}
#         for signal_name in process_signals:
#             row[f"{signal_name}_mean"] = segment[signal_name].mean()
#             row[f"{signal_name}_std"] = segment[signal_name].std()
#         for signal_name in target_signals:
#             row[f"{signal_name}_target"] = segment[signal_name].mean()
#         rows.append(row)

#     return pd.DataFrame(rows)


# def create_rolling_dataset(
#     signals_dataframe: pd.DataFrame,
#     process_signals: list[str],
#     target_signals: list[str],
#     window_size: int = 30,
# ) -> pd.DataFrame:
#     """Create rolling process features and current target values."""
#     if window_size < 2:
#         raise ValueError("window_size must be at least 2.")

#     selected = signals_dataframe[process_signals + target_signals].dropna()
#     previous_values = selected[process_signals].shift(1)
#     features = {}

#     for signal_name in process_signals:
#         rolling_signal = previous_values[signal_name].rolling(window_size)
#         features[f"{signal_name}_mean"] = rolling_signal.mean()
#         features[f"{signal_name}_std"] = rolling_signal.std()

#     rolling_dataset = pd.DataFrame(features, index=selected.index)
#     rolling_dataset["segment_id"] = range(len(selected))
#     for signal_name in target_signals:
#         rolling_dataset[f"{signal_name}_target"] = selected[signal_name]

#     return rolling_dataset.dropna().reset_index(drop=True)


# def train_model(segment_dataset: pd.DataFrame):
#     feature_columns = [
#         column
#         for column in segment_dataset.columns
#         if column != "segment_id" and not column.endswith("_target")
#     ]
#     target_columns = [f"{signal_name}_target" for signal_name in TARGET_SIGNALS]
#     dataset = segment_dataset.dropna(subset=feature_columns + target_columns)

#     if len(dataset) < 5:
#         raise ValueError("At least 5 complete operating segments are required.")

#     split_index = max(1, int(len(dataset) * 0.8))
#     if split_index >= len(dataset):
#         split_index = len(dataset) - 1

#     train_data = dataset.iloc[:split_index]
#     test_data = dataset.iloc[split_index:]
#     model = RandomForestRegressor(max_depth= 5, n_estimators=200, random_state=42)
#     model.fit(train_data[feature_columns], train_data[target_columns])

#     predictions = model.predict(test_data[feature_columns])
#     metrics = {
#         target_column: {
#             "MAE": mean_absolute_error(test_data[target_column], predictions[:, index]),
#             "RMSE": sqrt(
#                 mean_squared_error(test_data[target_column], predictions[:, index])
#             ),
#             "R2": r2_score(test_data[target_column], predictions[:, index]),
#         }
#         for index, target_column in enumerate(target_columns)
#     }
#     return model, metrics


# def train_xgboost_model(segment_dataset: pd.DataFrame):
#     """Train XGBoost models and evaluate predictions on the final data split."""
#     feature_columns = [
#         column
#         for column in segment_dataset.columns
#         if column != "segment_id" and not column.endswith("_target")
#     ]
#     target_columns = [f"{signal_name}_target" for signal_name in TARGET_SIGNALS]
#     dataset = segment_dataset.dropna(subset=feature_columns + target_columns)

#     if len(dataset) < 5:
#         raise ValueError("At least 5 complete operating segments are required.")

#     split_index = max(1, int(len(dataset) * 0.8))
#     if split_index >= len(dataset):
#         split_index = len(dataset) - 1

#     train_data = dataset.iloc[:split_index]
#     test_data = dataset.iloc[split_index:]
#     model = MultiOutputRegressor(
#         XGBRegressor(
#             max_depth=5,
#             n_estimators=200,
#             learning_rate=0.05,
#             objective="reg:squarederror",
#             random_state=42,
#         )
#     )
#     model.fit(train_data[feature_columns], train_data[target_columns])

#     predictions = model.predict(test_data[feature_columns])
#     metrics = {
#         target_column: {
#             "MAE": mean_absolute_error(test_data[target_column], predictions[:, index]),
#             "RMSE": sqrt(
#                 mean_squared_error(test_data[target_column], predictions[:, index])
#             ),
#             "R2": r2_score(test_data[target_column], predictions[:, index]),
#         }
#         for index, target_column in enumerate(target_columns)
#     }
#     return model, metrics





def create_rnn_sequences(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    window_size: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ordered rows into look-back sequences and next-step targets."""
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")
    complete = dataset.dropna(subset=feature_columns + target_columns)
    values = complete[feature_columns].to_numpy(dtype=np.float32)
    targets = complete[target_columns].to_numpy(dtype=np.float32)
    if len(complete) <= window_size:
        raise ValueError(
            f"At least {window_size + 1} complete rows are required for an RNN."
        )
    sequences = np.stack(
        [values[index - window_size : index] for index in range(window_size, len(values))]
    ).astype(np.float32)
    return sequences, targets[window_size:]


def train_rnn_model(
    segment_dataset: pd.DataFrame,
    window_size: int = 60,
    hidden_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 0.001,
):  
    print("Entering train_rnn_model function")

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.losses import Huber
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from math import sqrt

    """Train and evaluate a many-to-one RNN on the chronological segments."""
    feature_columns = PROCESS_SIGNALS
    target_columns = TARGET_SIGNALS

    print("Creating Sequence...")

    values = segment_dataset[feature_columns].to_numpy(dtype=np.float32)
    targets = segment_dataset[target_columns].to_numpy(dtype=np.float32)

    # sequences, targets = create_rnn_sequences(
    #     segment_dataset, feature_columns, target_columns, window_size
    # )

    print("Sequences created")
    print("Sequences shape:", values.shape)
    print("Targets shape:", targets.shape)

    print("Splitting data...")

    split_index = max(1, int(len(values) * 0.8))

    X_train_values = values[:split_index]
    y_train_values = targets[:split_index]

    X_test_values = values[split_index:]
    y_test_values = targets[split_index:]

    print("X_train memory GB:", X_train_values.nbytes / 1024**3)
    print("Scaling data...")

    from sklearn.preprocessing import StandardScaler

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_values = feature_scaler.fit_transform(
        X_train_values
    ).astype(np.float32)

    X_test_values = feature_scaler.transform(
        X_test_values
    ).astype(np.float32)

    y_train_values = target_scaler.fit_transform(
        y_train_values
    ).astype(np.float32)

    y_test_values_scaled = target_scaler.transform(
        y_test_values
    ).astype(np.float32)


    train_dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=X_train_values,
        targets=y_train_values[window_size:],
        sequence_length=window_size,
        sequence_stride=1,
        batch_size=128,
    )

    test_dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=X_test_values,
        targets=y_test_values_scaled[window_size:],
        sequence_length=window_size,
        sequence_stride=1,
        batch_size=128,
    )
    y_test = y_test_values[window_size:]

    # X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    # X_test_2d = X_test.reshape(-1, X_test.shape[-1])

    # X_train = feature_scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    # X_test = feature_scaler.transform(X_test_2d).reshape(X_test.shape)
    # X_train = X_train.astype(np.float32)
    # X_test = X_test.astype(np.float32)

    # y_train = target_scaler.fit_transform(y_train).astype(np.float32)
    # y_test_scaled = target_scaler.transform(y_test)

    print("Building model...")

    model = Sequential([
        LSTM(
            hidden_size,
            input_shape=(window_size, len(feature_columns))
        ),
        Dense(16, activation="relu"),
        #Dense(len(target_columns))
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        # loss=Huber(),
        metrics=["mae"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    print("Starting fit...")
    
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )

    # np.save("src/rnnModel/X_test.npy", X_test)
    # np.save("src/rnnModel/y_test.npy", y_test)
    
    predictions_scaled = model.predict(test_dataset,verbose=0)

    predictions = target_scaler.inverse_transform(predictions_scaled)

    metrics = {
        target_column: {
            "MAE": mean_absolute_error(
                y_test[:, index],
                predictions[:, index]
            ),
            "RMSE": sqrt(
                mean_squared_error(
                    y_test[:, index],
                    predictions[:, index]
                )
            ),
            "R2": r2_score(
                y_test[:, index],
                predictions[:, index]
            ),
        }
        for index, target_column in enumerate(target_columns)
    }

    # with open("src/rnnModel/metrics_vib2.json","w") as f:
    #     json.dump(metrics, f, indent=4)

    model.save(f"src/rnnModel/hydro_lstm_model_{TARGET_NAME}.keras")

    joblib.dump(
        feature_scaler,
        f"src/rnnModel/feature_scaler_{TARGET_NAME}.pkl"
    )

    joblib.dump(
        target_scaler,
        f"src/rnnModel/target_scaler_{TARGET_NAME}.pkl"
    )

    return model, metrics


if __name__ == "__main__":

    print("Num GPUs Available:",
      len(tf.config.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices('GPU'))

    required_signals = PROCESS_SIGNALS + TARGET_SIGNALS

    frames = []

    for signal in required_signals:
        path = DATA_FOLDER / f"{signal}.csv"

        print(f"Loading: {path.name}")

        signal_data = load_signal(path)

        print(f"Loaded: {path.name} Shape={signal_data.shape}")

        frames.append(signal_data)

    print("Starting concat...")

    signals_dataframe = pd.concat(frames, axis=1).sort_index()

    print("Concat complete")
    print(signals_dataframe.shape)

    lstm_dataset = (
        signals_dataframe[
            PROCESS_SIGNALS + TARGET_SIGNALS
        ]
        .dropna()
        .reset_index(drop=True)
    )

    print("Final rows:", len(lstm_dataset))
    print("Final shape:", lstm_dataset.shape)

    _, rnn_metrics = train_rnn_model(
        lstm_dataset,
        window_size=120,
        hidden_size=32)
    print(rnn_metrics)
    