from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
# import ruptures as rpt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


DATA_FOLDER = (
	Path(__file__).parent.parent.parent
	/ "data"
	/ "raw"
	/ "2025-06-01_2025-10-31"
)

REQUIRED_SIGNALS = [
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
	"Scada.GRUN.AGG2.G2.LAGER.M_VIBR1",
	"Scada.GRUN.AGG2.G2.LAGER.M_VIBR2",
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

OUTPUT_FOLDER = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "2025_new_interpol"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

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


def remove_zeros(series:pd.Series) -> pd.Series:

	return series[series != 0.0]


def step_interpolate(signal:pd.Series,sample_time) -> pd.Series:

	time = f"{sample_time}"
	signal = (
		signal
		.resample(time)
		.ffill()
	)

	return signal



# def detect_change_points(
# 	sensor_signal: pd.Series,
# 	penalty: float = 10.0,
# ) -> list[int]:
# 	detector = rpt.Pelt(
# 			model="l2",
# 			jump=10,
# 		).fit(sensor_signal.to_numpy())
	
# 	return detector.predict(pen=penalty)
# 	# detector = rpt.KernelCPD(
# 	# 	kernel="rbf",
# 	# 	min_size=50,
# 	# 	jump=20,
# 	# ).fit(sensor_signal.to_numpy())
# 	# detector = rpt.Binseg(
#     #     model="l2",
#     #     min_size=50,
#     #     jump=20,
#     # ).fit(sensor_signal.to_numpy())
# 	# return detector.predict(n_bkps=n_bkps)

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

	for file_path in sorted(DATA_FOLDER.glob("*.csv")):

		if file_path.stem not in REQUIRED_SIGNALS:
			continue

		print(f"Processing {file_path.name}")

		signal = load_signal(file_path)

		# Only remove zeros from vibration signals
		if "M_VIBR" in file_path.stem:
			signal = remove_zeros(signal)

		inter_signal = step_interpolate(
			signal=signal,
			sample_time="2s"
		)

		output_path = OUTPUT_FOLDER / file_path.name

		(
			inter_signal
			.to_frame()
			.reset_index()
			.to_csv(
				output_path,
				index=False
			)
		)

	
