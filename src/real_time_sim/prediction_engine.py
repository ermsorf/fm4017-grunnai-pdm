from collections import deque
import numpy as np


class PredictionEngine:

    def __init__(
        self,
        window_size,
        process_signals,

        vibr1_model,
        vibr2_model,

        feature_scaler_vibr1,
        feature_scaler_vibr2,

        target_scaler_vibr1,
        target_scaler_vibr2
    ):

        self.window_size = window_size

        self.process_signals = process_signals

        self.buffer = deque(
            maxlen=window_size
        )

        self.vibr1_model = vibr1_model
        self.vibr2_model = vibr2_model

        self.feature_scaler_vibr1 = feature_scaler_vibr1
        self.feature_scaler_vibr2 = feature_scaler_vibr2

        self.target_scaler_vibr1 = target_scaler_vibr1
        self.target_scaler_vibr2 = target_scaler_vibr2

    def update(self, row):

        features = row[
            self.process_signals
        ].to_numpy(dtype=np.float32)

        self.buffer.append(features)

        if len(self.buffer) < self.window_size:
            return None

        window = np.array(
            self.buffer,
            dtype=np.float32
        )

        vib1_window = (
            self.feature_scaler_vibr1
            .transform(window)
        )

        vib2_window = (
            self.feature_scaler_vibr2
            .transform(window)
        )

        vib1_window = np.expand_dims(
            vib1_window,
            axis=0
        )

        vib2_window = np.expand_dims(
            vib2_window,
            axis=0
        )

        vibr1_pred = (
            self.target_scaler_vibr1
            .inverse_transform(
                self.vibr1_model.predict(
                    vib1_window,
                    verbose=0
                )
            )
        )[0, 0]

        vibr2_pred = (
            self.target_scaler_vibr2
            .inverse_transform(
                self.vibr2_model.predict(
                    vib2_window,
                    verbose=0
                )
            )
        )[0, 0]

        return {
            "vibr1": vibr1_pred,
            "vibr2": vibr2_pred
        }