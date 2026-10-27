import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path


from tensorflow.keras.models import load_model
from src.data_analysis.data_pre_processing import load_signal

WINDOW_SIZE = 120

ROOT_DIRECTORY = Path(__file__).resolve().parents[2]

TEST_FOLDER = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "2025_new_interpol"
)

YEAR = TEST_FOLDER.name.split("_")[0]

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

def plot_resp(row,test,pred):
    plt.figure(figsize=(15,5))

    plt.plot(
        test[:1000, row],
        label="Actual"
    )

    plt.plot(
        pred[:1000, row],
        label="Predicted"
    )

    plt.title(f"{row} Actual vs Predicted")
    plt.xlabel("Sample")
    plt.ylabel("Vibration")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"vibr_{row+1}.png", dpi=300)
    plt.close()




if __name__ == "__main__":
    model = load_model(f"src/rnnModel/hydro_lstm_model_{TARGET_NAME}.keras")
    feature_scaler = joblib.load(f"src/rnnModel/feature_scaler_{TARGET_NAME}.pkl")
    target_scaler = joblib.load(f"src/rnnModel/target_scaler_{TARGET_NAME}.pkl")

    required_signals = PROCESS_SIGNALS + TARGET_SIGNALS

    frames = []

    for signal in required_signals:

        path = TEST_FOLDER / f"{signal}.csv"

        signal_data = load_signal(path)

        frames.append(signal_data)

    signals_dataframe = pd.concat(
        frames,
        axis=1
    ).sort_index()  

    test_df = (
        signals_dataframe[
            PROCESS_SIGNALS + TARGET_SIGNALS
        ]
        .dropna()
    )

    values = test_df[
        PROCESS_SIGNALS
    ].to_numpy(dtype=np.float32)

    targets = test_df[
        TARGET_SIGNALS
    ].to_numpy(dtype=np.float32)

    values = feature_scaler.transform(
        values
    ).astype(np.float32)

    import tensorflow as tf

    test_dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=values,
        targets=None,
        sequence_length=WINDOW_SIZE,
        sequence_stride=1,
        batch_size=128,
    )

    predictions_scaled = model.predict(
        test_dataset,
        verbose=1
    )

    predictions = target_scaler.inverse_transform(
        predictions_scaled
    )

    y_test = targets[WINDOW_SIZE-1:]
    timestamps = test_df.index[WINDOW_SIZE - 1:]

    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )

    from math import sqrt

    metrics = {
        TARGET_SIGNALS[0]: {
            "MAE": mean_absolute_error(
                y_test[:, 0],
                predictions[:, 0]
            ),
            "RMSE": sqrt(
                mean_squared_error(
                    y_test[:, 0],
                    predictions[:, 0]
                )
            ),
            "R2": r2_score(
                y_test[:, 0],
                predictions[:, 0]
            ),
        }
    }

    print(metrics)

    with open(
        f"{ROOT_DIRECTORY}/Results/LSTM/Metrics/metrics_{TARGET_NAME}_{YEAR}.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    results = pd.DataFrame(
        {
            "Actual": y_test[:, 0],
            "Predicted": predictions[:, 0]
        },
        index=timestamps
    )


    plt.figure(figsize=(20,8))

    plt.plot(
        results.index,
        results["Actual"],
        label=f"Actual {TARGET_NAME.upper()}",
        linewidth=0.8
    )

    plt.plot(
        results.index,
        results["Predicted"],
        label=f"Predicted {TARGET_NAME.upper()}",
        linewidth=0.8
    )

    plt.legend()

    plt.xlabel("Time")
    plt.ylabel("Vibration")

    plt.title(f"{TARGET_NAME.upper()} Actual vs Predicted")

    plt.grid(True)

    plt.savefig(
        f"{ROOT_DIRECTORY}/Results/LSTM/Plots/{TARGET_NAME}_timeline_{YEAR}.png",
        dpi=300
    )


