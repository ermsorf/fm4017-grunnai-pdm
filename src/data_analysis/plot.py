"""Print metrics and plot each prediction CSV separately.

Run from the repository root with:

    python -m src.data_analysis.plot
"""

from pathlib import Path
import src.config

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FOLDER = ROOT / "plots" / "report"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

vibr_train = src.config.SEQUENCE_MODEL_CONFIG.training_data_folder

# vibr_train = pd.read_csv(vibr_train)

# Load data
vibr1 = pd.read_csv(ROOT / "src" / "models" / "lstm" / "vibr1" / "lstm_predictions.csv")
vibr2 = pd.read_csv(ROOT / "src" / "models" / "lstm" / "vibr2" / "lstm_predictions.csv")
vibr1_nal1 = pd.read_csv(
    ROOT / "src" / "models" / "lstm" / "vibr1_nal1" / "lstm_predictions.csv"
)
vibr2_nal1 = pd.read_csv(
    ROOT / "src" / "models" / "lstm" / "vibr2_nal1" / "lstm_predictions.csv"
)
vibr1["timestamp"] = pd.to_datetime(vibr1["timestamp"])
vibr2["timestamp"] = pd.to_datetime(vibr2["timestamp"])
vibr1_nal1["timestamp"] = pd.to_datetime(vibr1_nal1["timestamp"])
vibr2_nal1["timestamp"] = pd.to_datetime(vibr2_nal1["timestamp"])


# Calculate and print metrics
vibr1_actual = vibr1["Scada.GRUN.AGG2.G2.LAGER.M_VIBR1.csv_actual"]
vibr1_predicted = vibr1["Scada.GRUN.AGG2.G2.LAGER.M_VIBR1.csv_predicted"]
print("vibr1")
print("MAE:", mean_absolute_error(vibr1_actual, vibr1_predicted))
print("RMSE:", mean_squared_error(vibr1_actual, vibr1_predicted) ** 0.5)
print("R2:", r2_score(vibr1_actual, vibr1_predicted))

vibr2_actual = vibr2["Scada.GRUN.AGG2.G2.LAGER.M_VIBR2.csv_actual"]
vibr2_predicted = vibr2["Scada.GRUN.AGG2.G2.LAGER.M_VIBR2.csv_predicted"]
print("\nvibr2")
print("MAE:", mean_absolute_error(vibr2_actual, vibr2_predicted))
print("RMSE:", mean_squared_error(vibr2_actual, vibr2_predicted) ** 0.5)
print("R2:", r2_score(vibr2_actual, vibr2_predicted))

vibr1_nal1_actual = vibr1_nal1["Scada.GRUN.AGG2.G2.LAGER.M_VIBR1.csv_actual"]
vibr1_nal1_predicted = vibr1_nal1["Scada.GRUN.AGG2.G2.LAGER.M_VIBR1.csv_predicted"]
print("\nvibr1_nal1")
print("MAE:", mean_absolute_error(vibr1_nal1_actual, vibr1_nal1_predicted))
print("RMSE:", mean_squared_error(vibr1_nal1_actual, vibr1_nal1_predicted) ** 0.5)
print("R2:", r2_score(vibr1_nal1_actual, vibr1_nal1_predicted))

vibr2_nal1_actual = vibr2_nal1["Scada.GRUN.AGG2.G2.LAGER.M_VIBR2.csv_actual"]
vibr2_nal1_predicted = vibr2_nal1["Scada.GRUN.AGG2.G2.LAGER.M_VIBR2.csv_predicted"]
print("\nvibr2_nal1")
print("MAE:", mean_absolute_error(vibr2_nal1_actual, vibr2_nal1_predicted))
print("RMSE:", mean_squared_error(vibr2_nal1_actual, vibr2_nal1_predicted) ** 0.5)
print("R2:", r2_score(vibr2_nal1_actual, vibr2_nal1_predicted))


# Use every row for metrics, but only plot this many points per line.
MAX_PLOT_POINTS = 10_000_000
vibr1_plot = vibr1.iloc[::max(1, len(vibr1) // MAX_PLOT_POINTS)]
vibr2_plot = vibr2.iloc[::max(1, len(vibr2) // MAX_PLOT_POINTS)]
vibr1_nal1_plot = vibr1_nal1.iloc[::max(1, len(vibr1_nal1) // MAX_PLOT_POINTS)]
vibr2_nal1_plot = vibr2_nal1.iloc[::max(1, len(vibr2_nal1) // MAX_PLOT_POINTS)]

plt.rcParams.update({'font.size': 14})

# Plot vibr1
plt.figure(figsize=(14, 6))
plt.plot(vibr1_plot["timestamp"], vibr1_plot[vibr1_actual.name], label="Actual")
plt.plot(vibr1_plot["timestamp"], vibr1_plot[vibr1_predicted.name], label="Predicted")
plt.axhline()
plt.title("Actual vs Predicted DE Vibration")
plt.xlabel("Time")
plt.ylabel("RMS Vibration [mm/s]")
plt.legend()
plt.grid(True)
plt.xticks()
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "vibr1_pva.png")
plt.close()

# Plot vibr1 zoom from sept to oct 2025
plt.figure(figsize=(14, 6))
plt.plot(vibr1_plot["timestamp"], vibr1_plot[vibr1_actual.name], label="Actual")
plt.plot(vibr1_plot["timestamp"], vibr1_plot[vibr1_predicted.name], label="Predicted")
plt.title("Actual vs Predicted DE Vibration (July 2025)")
plt.xlabel("Time")
plt.ylabel("RMS Vibration [mm/s]")
plt.xlim(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-07-31"))
plt.ylim(bottom=0, top=1.0)
plt.legend()
plt.grid(True)
plt.xticks()
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "vibr1_pva_zoom.png")
plt.close()
# plt.show()

# Plot vibr2
plt.figure(figsize=(14, 6))
plt.plot(vibr2_plot["timestamp"], vibr2_plot[vibr2_actual.name], label="Actual")
plt.plot(vibr2_plot["timestamp"], vibr2_plot[vibr2_predicted.name], label="Predicted")
plt.title("Actual vs Predicted NDE Vibration")
plt.xlabel("Time")
plt.ylabel("RMS Vibration [mm/s]")

plt.legend()
plt.grid(True)
plt.xticks()
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "vibr2_pva.png")
plt.close()

# Plot vibr2 zoom
plt.figure(figsize=(14, 6))
plt.plot(vibr2_plot["timestamp"], vibr2_plot[vibr2_actual.name], label="Actual")
plt.plot(vibr2_plot["timestamp"], vibr2_plot[vibr2_predicted.name], label="Predicted")
plt.title("Actual vs Predicted NDE Vibration")
plt.xlabel("Time")
plt.ylabel("RMS Vibration [mm/s]")
plt.ylim(bottom=0, top=4.0)
plt.xlim(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-07-31"))
plt.legend()
plt.grid(True)
plt.xticks()
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "vibr2_pva_zoom.png")
plt.close()

# Plot vibr1_nal1
plt.figure(figsize=(14, 6))
plt.plot(vibr1_nal1_plot["timestamp"], vibr1_nal1_plot[vibr1_nal1_actual.name], label="Actual")
plt.plot(vibr1_nal1_plot["timestamp"], vibr1_nal1_plot[vibr1_nal1_predicted.name], label="Predicted")
plt.title("Actual vs Predicted DE Vibration w/o needle 1")
plt.xlabel("Time")
plt.ylabel("RMS Vibration [mm/s]")
plt.legend()
plt.grid(True)
plt.xticks()
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "vibr1_nal1_pva.png")
plt.close()

# Plot vibr2_nal1
plt.figure(figsize=(14, 6))
plt.plot(vibr2_nal1_plot["timestamp"], vibr2_nal1_plot[vibr2_nal1_actual.name], label="Actual")
plt.plot(vibr2_nal1_plot["timestamp"], vibr2_nal1_plot[vibr2_nal1_predicted.name], label="Predicted")
plt.title("Actual vs Predicted NDE Vibration w/o needle 1")
plt.xlabel("Time")
plt.ylabel("RMS Vibration [mm/s]")
plt.legend()
plt.grid(True)
plt.xticks()
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "vibr2_nal1_pva.png")
plt.close()


residual_vibr1 = vibr1_plot[vibr1_actual.name] - vibr1_plot[vibr1_predicted.name]
residual_vibr2 = vibr2_plot[vibr2_actual.name] - vibr2_plot[vibr2_predicted.name]

# plt.figure(figsize=(14, 6))
# plt.subplot(2, 1, 1)
# plt.plot(vibr1_plot["timestamp"], residual_vibr1, label="Residual")
# plt.title("Residual DE Vibration")
# plt.xlabel("Time")
# plt.ylabel("Residual RMS Vibration [mm/s]")
# plt.axhline(0, color="black", linestyle="--", linewidth=1)
# plt.ylim(bottom=-0.5, top=1)
# plt.legend()
# plt.grid(True)
# plt.subplot(2, 1, 2)
# plt.plot(vibr2_plot["timestamp"], residual_vibr2, label="Residual")
# plt.title("Residual NDE Vibration")
# plt.xlabel("Time")
# plt.ylabel("Residual RMS Vibration [mm/s]")
# plt.axhline(0, color="black", linestyle="--", linewidth=1)
# plt.ylim(bottom=-2, top=4)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(OUTPUT_FOLDER / "vibr1_vibr2_residuals_time.png")
# plt.close()

#
# residuals vibr1
plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)
plt.scatter(residual_vibr1,vibr1_predicted, alpha=0.1)
# plt.axline((0, 0), slope=1, color='red', linestyle='--', label="Perfect Prediction")
plt.title("Residual vs Predicted DE Vibration")
plt.xlabel("Residual RMS Vibration [mm/s]")
plt.ylabel("Predicted RMS Vibration [mm/s]")
# plt.xlim(left=0, right=6.0)
# plt.ylim(bottom=0, top=6.0)
plt.legend()
plt.grid(True)
plt.subplot(1,2,2)
plt.scatter(residual_vibr2,vibr2_predicted, alpha=0.1)
plt.axline((0, 0), slope=1, color='red', linestyle='--', label="Perfect Prediction")
plt.title("Residual vs Predicted NDE Vibration")
plt.xlabel("Residual RMS Vibration [mm/s]")
plt.ylabel("Predicted RMS Vibration [mm/s]")
# plt.xlim(left=0, right=6.0)
# plt.ylim(bottom=0, top=6.0)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "vibr1_vibr2_residual.png")
plt.close()


# pvr plot vibr1
plt.title("Actual vs Predicted DE Vibration")
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(vibr1_actual,vibr1_predicted, alpha=0.1)
plt.axline((0, 0), slope=1, color='red', linestyle='--', label="Perfect Prediction")
plt.xlabel("Actual RMS Vibration [mm/s]")
plt.ylabel("Predicted RMS Vibration [mm/s]")
# plt.xlim(left=0, right=0.8)
# plt.ylim(bottom=0, top=0.8)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.subplot(1, 2, 2)
plt.scatter(vibr1_actual,vibr1_predicted, alpha=0.1)
plt.axline((0, 0), slope=1, color='red', linestyle='--', label="Perfect Prediction")
plt.xlabel("Actual RMS Vibration [mm/s]")
plt.ylabel("Predicted RMS Vibration [mm/s]")
plt.xlim(left=0, right=0.8)
# plt.ylim(bottom=0, top=0.8)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "vibr1_pvr.png")
plt.close()

# pvr plot vibr2
plt.figure(figsize=(14, 6))
plt.subplot(1,2,1)
plt.scatter(vibr2_actual,vibr2_predicted, alpha=0.1)
plt.axline((0, 0), slope=1, color='red', linestyle='--', label="Perfect Prediction")
plt.title("Actual vs Predicted NDE Vibration")
plt.xlabel("Actual RMS Vibration [mm/s]")
plt.ylabel("Predicted RMS Vibration [mm/s]")
# plt.xlim(left=0, right=6.0)
# plt.ylim(bottom=0, top=6.0)
plt.legend()
plt.grid(True)
plt.subplot(1,2,2)
plt.scatter(vibr2_actual,vibr2_predicted, alpha=0.1)
plt.axline((0, 0), slope=1, color='red', linestyle='--', label="Perfect Prediction")
plt.title("Actual vs Predicted NDE Vibration")
plt.xlabel("Actual RMS Vibration [mm/s]")
plt.ylabel("Predicted RMS Vibration [mm/s]")
plt.xlim(left=0, right=6.0)
# plt.ylim(bottom=0, top=6.0)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "vibr2_pvr.png")
plt.close()

# C:\github\fm4017-grunnai-pdm\plots\xgboost_variable_influence
# target,variable,mean_importance,std_importance
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR1,Scada.GRUN.AGG2.TURB2.NALER.M_NAL3_value,0.03789349210666919,3.3292281581305153e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR1,Scada.GRUN.AGG2.TURB2.NALER.M_NAL4_value,0.025429511398817682,0.0003530664323446453
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR1,Scada.GRUN.AGG2.VANN.SJAKT.M_TRYKK_value,0.015915664460282884,2.220849615217806e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR1,Scada.GRUN.AGG2.G2.MV.M_P_MW_value,0.010491359486114382,3.444209216709649e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR1,Scada.GRUN.AGG2.TURB2.REG.M_VANNSTR_value,0.009604880449163166,1.9035488314609343e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR1,Scada.GRUN.AGG2.TURB2.REG.M_TURT_value,0.0045593174404707,5.276661967006163e-06
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR1,Scada.GRUN.AGG2.G2.MAGN.M_I_value,0.0036879733574298524,1.9634388136775194e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR1,Scada.GRUN.AGG2.G2.KJOL.M_KJVANN_value,0.0003507091734242573,3.8123774617445636e-06
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR1,Scada.GRUN.AGG2.TURB2.TRYKKOL.M_OLTRYKSW_value,0.000348864708331853,3.9241917398005076e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR1,Scada.GRUN.AGG2.TURB2.PADRAG.M_WCKT_POS_value,0.00028474409614069883,6.797657848942762e-06
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR1,Scada.GRUN.AGG2.TURB2.NALER.M_NAL2_value,-0.0007824608118253362,2.5127942084192865e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR1,Scada.GRUN.AGG2.TURB2.NALER.M_NAL5_value,-0.0015889372602844842,1.4278059936311991e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR1,Scada.GRUN.AGG2.TURB2.NALER.M_NAL1_value,-0.008749879584710818,3.472354948086292e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR2,Scada.GRUN.AGG2.TURB2.NALER.M_NAL3_value,0.06157661453895684,5.357541853697243e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR2,Scada.GRUN.AGG2.TURB2.REG.M_TURT_value,0.01956376022375548,1.707466183300086e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR2,Scada.GRUN.AGG2.TURB2.NALER.M_NAL1_value,0.002145816214003915,9.59571992404889e-06
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR2,Scada.GRUN.AGG2.G2.KJOL.M_KJVANN_value,0.0020655667637262897,3.7399243214420525e-06
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR2,Scada.GRUN.AGG2.TURB2.REG.M_VANNSTR_value,0.0018674887102260174,2.6778806169441578e-06
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR2,Scada.GRUN.AGG2.TURB2.NALER.M_NAL4_value,0.0016802762476814115,4.308095874942138e-06
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR2,Scada.GRUN.AGG2.VANN.SJAKT.M_TRYKK_value,0.0008687986998314967,7.999264735010308e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR2,Scada.GRUN.AGG2.G2.MV.M_P_MW_value,0.000503704163893981,1.348054982041376e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR2,Scada.GRUN.AGG2.TURB2.PADRAG.M_WCKT_POS_value,0.00048602009934901603,1.7391593984315587e-06
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR2,Scada.GRUN.AGG2.TURB2.TRYKKOL.M_OLTRYKSW_value,0.0003514767129794372,4.716943353135567e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR2,Scada.GRUN.AGG2.TURB2.NALER.M_NAL2_value,-0.00012998924635831965,6.640690373307278e-06
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR2,Scada.GRUN.AGG2.G2.MAGN.M_I_value,-0.00375180406368103,2.1886531960783647e-05
# Scada.GRUN.AGG2.G2.LAGER.M_VIBR2,Scada.GRUN.AGG2.TURB2.NALER.M_NAL5_value,-0.004474559912572129,4.2464924657109634e-05

XGBOOST_INFLUENCE_FOLDER = ROOT / "plots" / "xgboost_variable_influence"
for report_path in sorted(XGBOOST_INFLUENCE_FOLDER.glob("*_variable_influence.csv")):
    if report_path.name == "all_targets_variable_influence.csv":
        continue

    report = pd.read_csv(report_path)
    required_columns = {"variable", "mean_importance", "std_importance"}
    missing_columns = required_columns.difference(report.columns)
    if missing_columns:
        raise ValueError(
            f"{report_path} is missing required columns: {sorted(missing_columns)}"
        )

    report = report.sort_values("mean_importance", ascending=True)
    figure_height = max(6, 0.35 * len(report))
    figure, axis = plt.subplots(figsize=(14, figure_height))
    axis.barh(
        report["variable"],
        report["mean_importance"],
        xerr=report["std_importance"],
        capsize=3,
    )
    target = str(report["target"].iloc[0]) if "target" in report else report_path.stem
    axis.set_title(f"XGBoost variable influence: {target}")
    axis.set_xlabel("Mean decrease in R2")
    axis.set_ylabel("Variable")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT_FOLDER / f"{report_path.stem}.png")
    plt.close(figure)


# lstm vibr1
#  1. Scada.GRUN.AGG2.TURB2.REG.M_TURT.csv: MSE increase=0.003737
#  2. Scada.GRUN.AGG2.TURB2.NALER.M_NAL3.csv: MSE increase=0.002879
#  3. Scada.GRUN.AGG2.TURB2.NALER.M_NAL2.csv: MSE increase=0.002610
#  4. Scada.GRUN.AGG2.G2.MV.M_P_MW.csv: MSE increase=0.002104
#  5. Scada.GRUN.AGG2.TURB2.REG.M_VANNSTR.csv: MSE increase=0.001448
#  6. Scada.GRUN.AGG2.TURB2.NALER.M_NAL1.csv: MSE increase=0.001291
#  7. Scada.GRUN.AGG2.TURB2.NALER.M_NAL4.csv: MSE increase=0.001212
#  8. Scada.GRUN.AGG2.TURB2.NALER.M_NAL5.csv: MSE increase=0.001031

lstm_vibr1_importance = {
    "RPM": 0.003737,
    "Needle 3": 0.002879,
    "Needle 2": 0.002610,
    "Power": 0.002104,
    "Water flow": 0.001448,
    "Needle 1": 0.001291,
    "Needle 4": 0.001212,
    "Needle 5": 0.001031,
}
lstm_vibr2_importance = {
    "RPM": 0.030996,
    "Needle 5": 0.023083,
    "Needle 3": 0.010965,
    "Needle 2": 0.001524,
    "Needle 1": 0.000539,
    "Water flow": -0.000556,
    "Needle 4": -0.000993,
    "Power": -0.001232,
}


#  1. Scada.GRUN.AGG2.TURB2.REG.M_TURT.csv: MSE increase=0.030996
#  2. Scada.GRUN.AGG2.TURB2.NALER.M_NAL5.csv: MSE increase=0.023083
#  3. Scada.GRUN.AGG2.TURB2.NALER.M_NAL3.csv: MSE increase=0.010965
#  4. Scada.GRUN.AGG2.TURB2.NALER.M_NAL2.csv: MSE increase=0.001524
#  5. Scada.GRUN.AGG2.TURB2.NALER.M_NAL1.csv: MSE increase=0.000539
#  6. Scada.GRUN.AGG2.TURB2.REG.M_VANNSTR.csv: MSE increase=-0.000556
#  7. Scada.GRUN.AGG2.TURB2.NALER.M_NAL4.csv: MSE increase=-0.000993
#  8. Scada.GRUN.AGG2.G2.MV.M_P_MW.csv: MSE increase=-0.001232