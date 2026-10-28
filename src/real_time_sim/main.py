import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from tensorflow.keras.models import load_model
from src.data_analysis.data_pre_processing import load_signal
from src.real_time_sim.replay_engine import ReplayEngine
from src.real_time_sim.prediction_engine import PredictionEngine
from src.real_time_sim.health_monitor import HealthMonitor
from src.real_time_sim.simulator_state import SimulatorState


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

    vibr1_model = load_model(f"src/rnnModel/hydro_lstm_model_vibr1.keras")
    vibr2_model = load_model(f"src/rnnModel/hydro_lstm_model_vibr2.keras")

    feature_scaler_vib1 = joblib.load(f"src/rnnModel/feature_scaler_vibr1.pkl")
    feature_scaler_vib2 = joblib.load(f"src/rnnModel/feature_scaler_vibr2.pkl")


    target_scaler_vib1 = joblib.load(f"src/rnnModel/target_scaler_vibr1.pkl")
    target_scaler_vib2 = joblib.load(f"src/rnnModel/target_scaler_vibr2.pkl")

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

    print(f"Testing Dataset Build: \n{test_df.head()}")

    engine = ReplayEngine(dataframe=test_df, speed=100)

    predictor = PredictionEngine(
        window_size=120,
        process_signals=PROCESS_SIGNALS,

        vibr1_model=vibr1_model,
        vibr2_model=vibr2_model,

        feature_scaler_vibr1=feature_scaler_vib1,
        feature_scaler_vibr2=feature_scaler_vib2,

        target_scaler_vibr1=target_scaler_vib1,
        target_scaler_vibr2=target_scaler_vib2
    )

    monitor = HealthMonitor(
        warning_threshold=0.5,
        alarm_threshold=1.0
    )

    state = SimulatorState()

    print("Data & Model Loaded.")
    print("Simulator Starting...")

    for timestamp, row in engine.stream():

        prediction = predictor.update(
            row
        )

        if prediction is None:
            continue

        actual_vibr1 = row[
            "Scada.GRUN.AGG2.G2.LAGER.M_VIBR1"
        ]

        actual_vibr2 = row[
            "Scada.GRUN.AGG2.G2.LAGER.M_VIBR2"
        ]

        residual_vibr1 = abs(
            actual_vibr1
            - prediction["vibr1"]
        )

        residual_vibr2 = abs(
            actual_vibr2
            - prediction["vibr2"]
        )

        health_score, status = monitor.evaluate(
            residual_vibr1,
            residual_vibr2
        )

        latest_data = {
            "timestamp": str(timestamp),

            "actual_vibr1": float(actual_vibr1),
            "predicted_vibr1": float(prediction["vibr1"]),

            "actual_vibr2": float(actual_vibr2),
            "predicted_vibr2": float(prediction["vibr2"]),

            "health_score": float(health_score),
            "status": status
        }

        with open(
            "src/real_time_sim/realtime.json",
            "w"
        ) as f:

            json.dump(
                latest_data,
                f,
                indent=4
            )


