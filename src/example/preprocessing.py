import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
import os

def preprocess_data(data_directory):
    all_data = []
    
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_directory, filename)
            data = pd.read_csv(file_path)
            all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True)

    # Data cleaning: remove rows with NaN values
    cleaned_data = combined_data.dropna()

    # Normalization: scale features
    scaler = StandardScaler()
    feature_columns = cleaned_data.columns[:-1]  # Assuming last column is the target
    cleaned_data[feature_columns] = scaler.fit_transform(cleaned_data[feature_columns])

    return cleaned_data

def samples_per_day_count(file_path: Path) -> pd.Series:
    raw_signal = pd.read_csv(
            file_path,
            header=None,
            skiprows=2,
            names=["timestamp", file_path.stem],
        )
    raw_signal["timestamp"] = pd.to_datetime(raw_signal["timestamp"], errors="coerce")
    raw_signal[file_path.stem] = pd.to_numeric(raw_signal[file_path.stem], errors="coerce")
    signal = raw_signal.dropna().set_index("timestamp")[file_path.stem].sort_index()
    return signal.resample("D").size()


if __name__ == "__main__":

    DATA_FOLDER = (
	Path(__file__).parent.parent.parent
	/ "data"
	/ "raw"
	/ "2024-06-01_2024-10-31"
    )
    signal = samples_per_day_count(DATA_FOLDER / "Scada.GRUN.AGG1.G1.MV.M_P.csv")
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(signal.index, signal.values)
    print(signal)
    plt.show()

    
