import os
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path

import pandas as pd
import src.config
import matplotlib.pyplot as plt
from src.data_loader import load_signal, load_signals, select_data_files


SignalFrames = Mapping[Path, pd.DataFrame]


def count_samples_per_day(signal: pd.DataFrame) -> dict[str, int]:
    timestamps = signal["timestamp"].dropna()
    counts = timestamps.dt.strftime("%Y-%m-%d").value_counts().sort_index()
    return counts.astype(int).to_dict()


def get_all_sample_counts(signals: SignalFrames) -> dict[str, dict[str, int]]:
    return {
        file_path.name: count_samples_per_day(signal)
        for file_path, signal in signals.items()
    }


def plot_sample_counts(sample_counts: dict[str, dict[str, int]]) -> None:
    plt.figure(figsize=(12, 6))
    for file_name, counts in sample_counts.items():
        dates = list(counts.keys())
        counts_values = list(counts.values())
        plt.scatter(dates, counts_values, alpha=0.1, label=file_name, color='blue', edgecolors='none')
    
    plt.xticks(rotation=90)
    plt.xlabel("Date")
    plt.ylabel("Number of Samples")
    plt.title("Sample Counts per Day for Each CSV File")
    plt.yscale("log")  # Use logarithmic scale for better visibility
    # plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sample_counts_for_signals(signals: SignalFrames) -> None:
    sample_counts = get_all_sample_counts(signals)
    plot_sample_counts(sample_counts)


def plot_single_signal_sample_counts(file_path: Path, signal: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sample_counts = count_samples_per_day(signal)
    dates = list(sample_counts.keys())
    counts_values = list(sample_counts.values())
    plt.plot(dates, counts_values, label=file_path, color='blue')
    
    plt.xticks(rotation=90)
    plt.xlabel("Date")
    plt.ylabel("Number of Samples")
    plt.title(f"Sample Counts per Day for {file_path}")
    plt.tight_layout()
    plt.show()

def get_sampling_statistics(sample_counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    statistics = {}
    for file_name, counts in sample_counts.items():
        counts_values = list(counts.values())
        if counts_values:
            min_count = min(counts_values)
            max_count = max(counts_values)
            avg_count = sum(counts_values) / len(counts_values)
            statistics[file_name] = {
                "min": min_count,
                "max": max_count,
                "avg": avg_count
            }
    return statistics

def get_sample_count_for_signal(signal: pd.DataFrame) -> int:
    return sum(count_samples_per_day(signal).values())


def analyze_daily_sampling_trend(
    signals: SignalFrames,
) -> pd.DataFrame:
    """Return per-sensor daily sample counts normalized by sensor mean.

    Each sensor contributes a value of ``1`` on a day with its average
    number of samples.  The ``normalized_total`` column is the sum of those
    per-sensor values, so sensors with higher native sampling rates do not
    dominate the combined trend.
    """
    daily_counts = {}
    sensor_means = {}

    for file_path, signal in signals.items():
        timestamps = signal["timestamp"].dropna()
        counts = timestamps.dt.normalize().value_counts().sort_index()
        sensor_name = Path(file_path).name
        daily_counts[sensor_name] = counts
        sensor_means[sensor_name] = counts.mean()

    if not daily_counts:
        return pd.DataFrame(columns=["normalized_total"])

    raw_counts = pd.DataFrame(daily_counts).fillna(0).sort_index()
    normalized_counts = raw_counts.copy()
    for sensor_name, mean_count in sensor_means.items():
        normalized_counts[sensor_name] = (
            raw_counts[sensor_name] / mean_count if mean_count else 0.0
        )

    normalized_counts["normalized_total"] = normalized_counts.sum(axis=1)
    normalized_counts.index.name = "date"
    return normalized_counts


def compare_sampling_covariance(
    signals: SignalFrames, time_bin: str = "1min"
) -> pd.DataFrame:
    """Compare whether sensors change their sample rates at the same times."""
    sample_counts = {}
    for file_path, signal in signals.items():
        timestamps = signal["timestamp"].dropna()
        counts = timestamps.dt.floor(time_bin).value_counts().sort_index()
        sample_counts[str(file_path)] = counts

    results = []
    for left_file, right_file in combinations(sample_counts, 2):
        paired_counts = pd.concat(
            [sample_counts[left_file], sample_counts[right_file]],
            axis=1,
            join="outer",
        ).fillna(0)
        paired_counts.columns = ["left_rate", "right_rate"]
        covariance = paired_counts["left_rate"].cov(paired_counts["right_rate"])
        if paired_counts["left_rate"].nunique() > 1 and paired_counts["right_rate"].nunique() > 1:
            correlation = paired_counts["left_rate"].corr(paired_counts["right_rate"])
        else:
            correlation = float("nan")
        results.append(
            {
                "file_left": left_file,
                "file_right": right_file,
                "shared_time_bins": len(paired_counts),
                "sampling_rate_covariance": covariance,
                "sampling_rate_correlation": correlation,
            }
        )

    return pd.DataFrame(results)


def sampling_rate_correlation_matrix(
    signals: SignalFrames, time_bin: str = "1min"
) -> pd.DataFrame:
    sample_counts = {}
    for file_path, signal in signals.items():
        timestamps = signal["timestamp"].dropna()
        sample_counts[Path(file_path).name] = (
            timestamps.dt.floor(time_bin).value_counts().sort_index()
        )

    if not sample_counts:
        return pd.DataFrame()

    sampling_rates = pd.DataFrame(sample_counts).fillna(0)
    return sampling_rates.corr()


def plot_sampling_rate_correlation_matrix(correlation_matrix: pd.DataFrame) -> None:
    if correlation_matrix.empty:
        return

    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(correlation_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(correlation_matrix.columns)))
    axis.set_yticks(range(len(correlation_matrix.index)))
    axis.set_xticklabels(correlation_matrix.columns, rotation=90)
    axis.set_yticklabels(correlation_matrix.index)
    axis.set_title("Sampling-rate correlation by sensor")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    plt.show()


def plot_daily_sampling_trend(
    daily_sampling_trend: pd.DataFrame,
    power_signal: pd.DataFrame | None = None,
) -> None:
    """Plot daily normalized sample counts and optional daily mean power."""
    if daily_sampling_trend.empty or "normalized_total" not in daily_sampling_trend:
        return

    sensor_count = len(daily_sampling_trend.columns) - 1
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(
        daily_sampling_trend.index,
        daily_sampling_trend["normalized_total"],
        color="tab:blue",
        linewidth=1.5,
        marker=".",
        label="Normalized samples",
    )
    axis.axhline(
        sensor_count,
        color="tab:orange",
        linestyle="--",
        label="Expected average",
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Combined normalized samples", color="tab:blue")
    axis.set_title("Daily sampling trend across sensors")
    axis.tick_params(axis="y", labelcolor="tab:blue")

    if power_signal is not None:
        daily_power = (
            power_signal.dropna(subset=["timestamp", "value"])
            .assign(date=lambda frame: frame["timestamp"].dt.normalize())
            .groupby("date")["value"]
            .mean()
        )

        power_axis = axis.twinx()
        power_axis.plot(
            daily_power.index,
            daily_power,
            color="tab:green",
            linewidth=1.5,
            marker=".",
            label="Mean power generation",
        )
        power_axis.set_ylabel("Mean power generation (MW)", color="tab:green")
        power_axis.tick_params(axis="y", labelcolor="tab:green")
        lines, labels = axis.get_legend_handles_labels()
        power_lines, power_labels = power_axis.get_legend_handles_labels()
        axis.legend(lines + power_lines, labels + power_labels)
    else:
        axis.legend()

    axis.grid(axis="y", alpha=0.3)
    figure.autofmt_xdate()
    figure.tight_layout()
    plt.show()


def analyze_threshold_change(signals: SignalFrames) -> None:
    # Dictionary to store threshold change for each sensor
    threshold_changes = {}

    for file_path, signal in signals.items():
        file_name = file_path.name
        try:
            df = signal.dropna(subset=["timestamp", "value"]).set_index("timestamp")

            # Calculate the absolute difference between consecutive values
            df["diff"] = df["value"].diff().abs()

            # Determine the threshold change (e.g., mean or median of differences)
            threshold_change = df["diff"].median()   # You can also use mean() or median() statistics
            threshold_changes[file_name] = threshold_change

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Print the threshold changes for each sensor
    for sensor, threshold in threshold_changes.items():
        print(f"Sensor: {sensor}, Threshold Change: {threshold}")

def analyze_apparent_sampling_rate(signals: SignalFrames) -> None:
    # Dictionary to store apparent sampling interval statistics for each sensor
    apparent_sampling_intervals = {}

    for file_path, signal in signals.items():
        file_name = file_path.name
        try:
            df = signal.dropna(subset=["timestamp"]).set_index("timestamp")

            # Calculate the time differences between consecutive timestamps
            df["time_diff"] = df.index.to_series().diff().dt.total_seconds()

            # Compare the typical interval with the shortest observed interval
            apparent_sampling_intervals[file_name] = {
                "median": df["time_diff"].median(),
                "min": df["time_diff"].min(),
            }

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Print both interval statistics for each sensor
    for sensor, intervals in apparent_sampling_intervals.items():
        print(
            f"Sensor: {sensor}, Sampling Interval (seconds) - "
            f"Median: {intervals['median']}, Min: {intervals['min']}"
        )



if __name__ == "__main__":
    # data_folder = os.path.join(src.config.DATA_FOLDER, "raw", "2024-06-01_2024-10-31")
    data_folder = src.config.DATA_RAW_2025
    # data_folder = "C:/github/fm4017-grunnai-pdm/data/processed/2024/2024-06-01_2024-10-31_1min/2024-06-01_2024-10-31"
    # sample_counts = get_all_sample_counts(data_folder)
    # plot_sample_counts(sample_counts)

    # print("Sampling Statistics:")
    # statistics = get_sampling_statistics(sample_counts)
    # for file_name, stats in statistics.items():
    #     print(f"{file_name}: Min: {stats['min']}, Max: {stats['max']}, Avg: {stats['avg']:.2f}")
    # file_path = os.path.join(data_folder, "Scada.GRUN.AGG1.G1.MV.M_P.csv")
    # plot_single_file_sample_counts(file_path)
    
    # plot file by file
    # for file_name in os.listdir(data_folder):
    #     if file_name.endswith(".csv"):
    #         file_path = os.path.join(data_folder, file_name)
    #         plot_single_file_sample_counts(file_path)

    # total_samples = {}
    # for file_name in os.listdir(data_folder):
    #     if file_name.endswith(".csv"):
    #         file_path = os.path.join(data_folder, file_name)
    #         sample_count = get_sample_count_for_file(file_path)
    #         print(f"Total samples for {file_name}: {sample_count}")
    #     total_samples[file_name] = sample_count

    # total_samples_sorted = dict(sorted(total_samples.items(), key=lambda item: item[1], reverse=True))

    # filename_contains = "AGG2"
    # csv_files = select_data_files(data_folder, filename_contains=filename_contains)
    # signals = load_signals(csv_files)
    # correlation_matrix = sampling_rate_correlation_matrix(csv_files)
    # # print("Sampling-rate correlation matrix:")
    # # # print(correlation_matrix.to_string())
    # # plot_sampling_rate_correlation_matrix(correlation_matrix)

    # daily_sampling_trend = analyze_daily_sampling_trend(signals)
    # print("Days with the lowest normalized sampling totals:")
    # print(daily_sampling_trend["normalized_total"].sort_values().head(20).to_string())
    # power_file_path = os.path.join(
    #     src.config.DATA_FOLDER,
    #     "processed",
    #     "2024",
    #     "2024-06-01_2024-10-31_1min",
    #     "2024-06-01_2024-10-31",
    #     "Scada.GRUN.AGG2.G2.MV.M_P_MW.csv",
    # )
    # plot_daily_sampling_trend(daily_sampling_trend, load_signal(power_file_path))

    # # plot_sample_counts_for_keywords(data_folder, filename_contains=["G2.MAGN.M_I", "NAL1"])
    # # plot_sample_counts_for_keywords(data_folder, filename_contains=["AGG2"])
    # analyze_threshold_change(signals)

    print("\n\nNow analyzing 2024 data...\n\n")

    data_folder = src.config.DATA_RAW_2024
    filename_contains = "AGG2"
    csv_files = select_data_files(data_folder, filename_contains=filename_contains)
    signals = load_signals(csv_files)
    analyze_apparent_sampling_rate(signals)

    print("\n\nNow analyzing 2025 data...\n\n")

    data_folder = src.config.DATA_RAW_2025
    filename_contains = "AGG2"
    csv_files = select_data_files(data_folder, filename_contains=filename_contains)
    signals = load_signals(csv_files)
    analyze_apparent_sampling_rate(signals)
