# Data folder
This is the data folder, where all raw and processed data are placed. This data is not included in the GitHub Repo, but will have to be acquired elsewhere and placed in this folder according to this README, and the subfolders. 

## Sensor Documentation

Sensor files located under `data/raw` for the periods:

- `2024-06-01_2024-10-31`
- `2025-06-01_2025-10-31`

Unless otherwise noted, each sensor is available for both periods.

## Hydrocord

| File | Description | Notes |
|---|---|---|
| `Hydrocord.Eta.G2_ETA_Head(=C1=JB2=BCG1).csv` | Water head | Calculated head used in the efficiency calculation. Available for 2025 only. |
| `Hydrocord.Eta.G2_ETA_Value(=D1=LE1=LE1).csv` | Turbine or generator efficiency | Values are faulty for the last three years and should be ignored. |
| `Hydrocord.StandardData.G2_Sjakt_Trykk(=A2=HB1=BPA1)Mean.csv` | Turbine inlet pressure, mean | Hydrocord measurement. |
| `Hydrocord.StandardData.G2_Utlop_Trykk(=A2=JB1=BPA1)Mean.csv` | Outlet pressure, mean | Hydrocord measurement. |

## Generator 1

| File | Description | Notes |
|---|---|---|
| `Scada.GRUN.AGG1.G1.BLAGER.M_LAGTMP.csv` | Support-bearing temperature | Generator 1 support bearing. |
| `Scada.GRUN.AGG1.G1.DSSTYLAG.M_LAGTMP.csv` | Lower guide-bearing temperature |  |
| `Scada.GRUN.AGG1.G1.KJOL.M_KJVANN.csv` | Cooling-water temperature |  |
| `Scada.GRUN.AGG1.G1.MV.M_I.csv` | Current |  |
| `Scada.GRUN.AGG1.G1.MV.M_P.csv` | Active power | Negative values near standstill can be ignored. |
| `Scada.GRUN.AGG1.G1.MV.M_U.csv` | Voltage |  |
| `Scada.GRUN.AGG1.G1.SSSTYLAG.M_LAGTMP.csv` | Upper guide-bearing temperature |  |
| `Scada.GRUN.AGG1.TURB1.MV.M_TURTALL.csv` | Turbine rotational speed | Stored as a percentage. Short-term runaway values above 100% can occur. |

### Generator 1 injector openings

The following files contain the openings of the five Pelton turbine injectors:

| File |
|---|
| `Scada.GRUN.AGG1.TURB1.NALER.M_NAL1POS.csv` |
| `Scada.GRUN.AGG1.TURB1.NALER.M_NAL2POS.csv` |
| `Scada.GRUN.AGG1.TURB1.NALER.M_NAL3POS.csv` |
| `Scada.GRUN.AGG1.TURB1.NALER.M_NAL4POS.csv` |
| `Scada.GRUN.AGG1.TURB1.NALER.M_NAL5POS.csv` |

## Generator 2

| File | Description | Notes |
|---|---|---|
| `Scada.GRUN.AGG2.G2.KJOL.M_KJVANN.csv` | Cooling-water temperature |  |
| `Scada.GRUN.AGG2.G2.LAGER.M_LAGTMP1.csv` | Bearing temperature 1 | Generator 2, NDE axial. |
| `Scada.GRUN.AGG2.G2.LAGER.M_LAGTMP4.csv` | Bearing temperature 4 | Generator 2, NDE axial. |
| `Scada.GRUN.AGG2.G2.LAGER.M_LAGTMP5.csv` | Bearing temperature 5 | Generator 2, NDE radial. |
| `Scada.GRUN.AGG2.G2.LAGER.M_LAGTMP6.csv` | Bearing temperature 6 | Generator 2, NDE radial. |
| `Scada.GRUN.AGG2.G2.LAGER.M_OLTMP1.csv` | Oil temperature 1 | NDE. |
| `Scada.GRUN.AGG2.G2.LAGER.M_OLTMP2.csv` | Oil temperature 2 | DE. |
| `Scada.GRUN.AGG2.G2.LAGER.M_VIBR1.csv` | Bearing vibration 1 | DE. |
| `Scada.GRUN.AGG2.G2.LAGER.M_VIBR2.csv` | Bearing vibration 2 | NDE. |
| `Scada.GRUN.AGG2.G2.MAGN.M_I.csv` | Magnetization current |  |
| `Scada.GRUN.AGG2.G2.MV.M_P_MW.csv` | Active power |  |

### Generator 2 injector openings

These files contain the openings of the five Pelton turbine injectors:

| File |
|---|
| `Scada.GRUN.AGG2.TURB2.NALER.M_NAL1.csv` |
| `Scada.GRUN.AGG2.TURB2.NALER.M_NAL2.csv` |
| `Scada.GRUN.AGG2.TURB2.NALER.M_NAL3.csv` |
| `Scada.GRUN.AGG2.TURB2.NALER.M_NAL4.csv` |
| `Scada.GRUN.AGG2.TURB2.NALER.M_NAL5.csv` |

## Turbine 2 and plant measurements

| File | Description | Notes |
|---|---|---|
| `Scada.GRUN.AGG2.TURB2.PADRAG.M_WCKT_POS.csv` | Combined injector-opening value | Combined value based on all five injector openings; despite its name, this is not a conventional wicket-gate position. |
| `Scada.GRUN.AGG2.TURB2.REG.M_TURT.csv` | Turbine rotational speed | Supplier naming convention differs from Generator 1. |
| `Scada.GRUN.AGG2.TURB2.REG.M_VANNSTR.csv` | Volume flow rate |  |
| `Scada.GRUN.AGG2.TURB2.TRYKKOL.M_OLTMP.csv` | Turbine 2 pressure-oil system temperature |  |
| `Scada.GRUN.AGG2.TURB2.TRYKKOL.M_OLTRYKSW.csv` | Turbine 2 pressure-oil pressure |  |
| `Scada.GRUN.AGG2.VANN.SJAKT.M_TRYKK.csv` | Water pressure | Measurement from the control system. This differs from the Hydrocord penstock-pressure sensor, which also calculates an average. |

## Data-quality notes

- The efficiency values in `Hydrocord.Eta.G2_ETA_Value(=D1=LE1=LE1).csv` should be excluded from analysis.
- Generator 1 active power values slightly below zero during standstill can be ignored.
- Generator 1 turbine speed is represented as a percentage, not RPM.
- Short-term runaway values above 100% may occur in the Generator 1 turbine-speed signal.
- Both turbines are Pelton turbines with five injector-opening signals, not wicket gates.

