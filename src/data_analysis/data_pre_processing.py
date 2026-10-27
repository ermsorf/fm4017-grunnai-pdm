from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


DATA_FOLDER = (
	Path(__file__).parent.parent.parent
	/ "data"
	/ "raw"
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


def detect_change_points(
	sensor_signal: pd.Series,
	n_bkps: int = 5,
) -> list[int]:
	# detector = rpt.Pelt(
	# 		model="l2",
	# 		min_size=50,
	# 		jump=20,
	# 	).fit(sensor_signal.to_numpy())
	
	# return detector.predict(pen=penalty)
	detector = rpt.KernelCPD(
		kernel="rbf",
		min_size=50,
		jump=20,
	).fit(sensor_signal.to_numpy())
	# detector = rpt.Binseg(
    #     model="l2",
    #     min_size=50,
    #     jump=20,
    # ).fit(sensor_signal.to_numpy())
	return detector.predict(n_bkps=n_bkps)




def calculate_segments_features(
		power: pd.Series,
		speed: pd.Series,
		upper_bearing_temp: pd.Series,
		lower_bearing_temp: pd.Series,
		voltage: pd.Series,
		change_points: list[int]
	) -> pd.DataFrame:
	signals = pd.concat(
		{
			"power": power,
			"speed": speed,
			"upper_bearing_temp": upper_bearing_temp,
			"lower_bearing_temp": lower_bearing_temp,
			"voltage": voltage
		}, axis=1).sort_index()

	bounds = [0] + change_points
	rows = []

	for segments_id, (start,end) in enumerate(zip(bounds[:-1],bounds[1:])):
		start_time = power.index[start]
		end_time = power.index[end-1]

		segment = signals.loc[start_time:end_time]

		rows.append({
			"segment_id": segments_id,
			"start_time": start_time,
			"end_time": end_time,
			"power_mean": segment["power"].mean(),
			"power_std": segment["power"].std(),
			"speed_mean": segment["speed"].mean(),
			"speed_std": segment["speed"].std(),
			"upper_b_temp_mean": segment["upper_bearing_temp"].mean(),
			"upper_b_temp_std": segment["upper_bearing_temp"].std(),
			"lower_b_temp_mean": segment["lower_bearing_temp"].mean(),
			"lower_b_temp_std": segment["lower_bearing_temp"].std(),
			"voltage_mean":segment["voltage"].mean(),
			"voltage_std": segment["voltage"].std()
		})
	return pd.DataFrame(rows)


def detect_segment_anomalies(
	segments_features: pd.DataFrame,
	contamination: float = 0.2,
) -> pd.DataFrame:
	"""Flag unusual operating segments using their calculated features."""
	feature_columns = [
		"power_mean",
		"power_std",
		"speed_mean",
		"speed_std",
		"upper_b_temp_mean",
		"upper_b_temp_std",
		"lower_b_temp_mean",
		"lower_b_temp_std",
		"voltage_mean",
		"voltage_std"
	]
	valid = segments_features.dropna(subset=feature_columns).copy()

	if len(valid) < 10:
		print("Warning: fewer than 10 segments; anomaly results are exploratory only.")

	scaler = StandardScaler()
	features_scaled = scaler.fit_transform(valid[feature_columns])
	detector = IsolationForest(
		contamination=contamination,
		n_estimators=200,
		random_state=42,
	)

	valid["anomaly"] = detector.fit_predict(features_scaled)
	valid["anomaly_score"] = -detector.score_samples(features_scaled)
	return valid

		


def plot_signals_change_points(datasignal: pd.Series, change_points: list[int]) -> None:
	"""Plot the complete signal with every detected segment highlighted."""
	fig, ax = plt.subplots(figsize=(16, 7))
	plt.plot(datasignal.index, datasignal.values, label=datasignal.name)

	bounds = [0] + change_points
	for segment_id, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
		start_time = datasignal.index[start]
		end_time = datasignal.index[end - 1]
		if segment_id % 2 == 0:
			ax.axvspan(start_time, end_time, color="lightblue", alpha=0.2)
		if end < len(datasignal):
			ax.axvline(
				datasignal.index[end],
				color="red",
				linestyle="--",
				linewidth=1,
			)
		ax.text(
			(start_time + (end_time - start_time) / 2),
			0.98,
			f"S{segment_id}",
			transform=ax.get_xaxis_transform(),
			ha="center",
			va="top",
		)

	ax.set_title("Signal Segments Detected by KernelCPD")
	ax.set_xlabel("Time")
	ax.set_ylabel("Value")
	ax.legend()
	ax.grid()
	fig.autofmt_xdate()
	plt.show()


if __name__ == "__main__":
	power = load_signal(DATA_FOLDER / "Scada.GRUN.AGG1.G1.MV.M_P.csv")
	speed = load_signal(DATA_FOLDER / "Scada.GRUN.AGG1.TURB1.MV.M_TURTALL.csv")
	upper_bearing_temp = load_signal(DATA_FOLDER / "Scada.GRUN.AGG1.G1.SSSTYLAG.M_LAGTMP.csv")
	lower_bearing_temp = load_signal(DATA_FOLDER / "Scada.GRUN.AGG1.G1.DSSTYLAG.M_LAGTMP.csv")
	voltage = load_signal(DATA_FOLDER / "Scada.GRUN.AGG1.G1.MV.M_U.csv")

	change_points = detect_change_points(power)

	segments_features = calculate_segments_features(power, speed, upper_bearing_temp, lower_bearing_temp, voltage, change_points)

	# segments_features.to_csv("segments_features.csv", index=False)
	segments_with_anomalies = detect_segment_anomalies(segments_features)
	print("\nSegment anomaly results:")
	print(
		segments_with_anomalies[
			["segment_id", "start_time", "end_time", "anomaly", "anomaly_score"]
		].to_string(index=False)
	)
	print(f"\nThe length of the segments_with_anomalies DataFrame is: {len(segments_with_anomalies)}")

	plot_signals_change_points(power, change_points)