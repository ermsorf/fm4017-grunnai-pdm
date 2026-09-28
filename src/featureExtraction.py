"""Small, runnable regression template for the Grunnai signal data.

Run from this directory with:

	python model.py

Change TARGET_SIGNAL and FEATURE_SIGNALS below to train on another set of
signals.
"""

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt



DATA_FOLDER = (
	Path(__file__).parent.parent
	/ "data"
	/ "2024-06-01_2024-10-31"
)
MODEL_PATH = Path(__file__).parent.parent / "trained_model.joblib"


def load_signal(file_path: Path) -> pd.Series:
	"""Load one exported signal as a timestamp-indexed numeric series."""
	signal = pd.read_csv(
		file_path,
		header=None,
		skiprows=2,
		names=["timestamp", file_path.stem],
	)
	signal["timestamp"] = pd.to_datetime(signal["timestamp"], errors="coerce")
	signal[file_path.stem] = pd.to_numeric(signal[file_path.stem], errors="coerce")
	return signal.dropna().set_index("timestamp")[file_path.stem].sort_index()


def detect_change_points(sensor_signal: pd.Series) -> list[int]:
	"""Detect signal change points using kernel-based change-point detection."""
	detector = rpt.KernelCPD(
		kernel="rbf",
		min_size=50,
		jump=10,
	).fit(sensor_signal.to_numpy())
	return detector.predict(n_bkps=5)
	


# def build_dataset() -> pd.DataFrame:
# 	"""Read and timestamp-align the target and feature signals."""
# 	target = load_signal(DATA_FOLDER / TARGET_SIGNAL).rename("target")
# 	features = [load_signal(DATA_FOLDER / name) for name in FEATURE_SIGNALS]
# 	dataset = (
# 		pd.concat([target, *features], axis=1)
# 		.sort_index()
# 		.dropna()
# 	)

# 	if dataset.empty:
# 		raise ValueError("No complete rows were found after aligning the signals.")
# 	return dataset

def plot_signals_change_points(datasignal: pd.Series, change_points: list[int]) -> None:
	"""Plot one signal and its detected change points."""
	plt.figure(figsize=(12, 6))
	plt.plot(datasignal.index, datasignal.values, label=datasignal.name)
	for point in change_points[:-1]:
		plt.axvline(datasignal.index[point], color="red", linestyle="--")
	plt.title("Aligned Signals")
	plt.xlabel("Time")
	plt.ylabel("Value")
	plt.legend()
	plt.grid()
	plt.show()

# def train_model(dataset: pd.DataFrame) -> RandomForestRegressor:
# 	"""Train on the first 80 percent of time-ordered samples and evaluate the rest."""
# 	split_index = int(len(dataset) * 0.8)
# 	if split_index < 1 or split_index >= len(dataset):
# 		raise ValueError("The dataset needs at least two complete rows.")

# 	features = dataset.drop(columns="target")
# 	target = dataset["target"]
# 	train_features, test_features = features.iloc[:split_index], features.iloc[split_index:]
# 	train_target, test_target = target.iloc[:split_index], target.iloc[split_index:]

# 	model = RandomForestRegressor(
# 		n_estimators=100,
# 		random_state=42,
# 		n_jobs=-1,
# 	)
# 	model.fit(train_features, train_target)
# 	predictions = model.predict(test_features)

# 	print(f"Rows: {len(dataset):,} | train: {len(train_features):,} | test: {len(test_features):,}")
# 	print(f"MAE: {mean_absolute_error(test_target, predictions):.4f}")
# 	print(f"R2:  {r2_score(test_target, predictions):.4f}")
# 	return model


if __name__ == "__main__":
	sensor_signal = load_signal(DATA_FOLDER / "Scada.GRUN.AGG1.G1.MV.M_I.csv")
	change_points = detect_change_points(sensor_signal)
	plot_signals_change_points(sensor_signal, change_points)
	
	
