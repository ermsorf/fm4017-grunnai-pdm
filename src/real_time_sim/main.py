import numpy as np
import pandas as pd
import time
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
from src.data_analysis.data_pre_processing import load_signal

TEST_FOLDER = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "2025_new_interpol"
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
    "Scada.GRUN.AGG2.G2.LAGER.M_VIBR1",
    "Scada.GRUN.AGG2.G2.LAGER.M_VIBR2",
]

if __name__ == "__main__":

    # model = load_model(f"src/rnnModel/hydro_lstm_model_{}.keras")
    # feature_scaler = joblib.load(f"src/rnnModel/feature_scaler_{}.pkl")
    # target_scaler = joblib.load(f"src/rnnModel/target_scaler_{}.pkl")

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

    print(test_df.head())
