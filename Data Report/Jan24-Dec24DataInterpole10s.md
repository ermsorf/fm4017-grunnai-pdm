# Jan24-Dec24 Data, 10s Interpolated Data findings

## Overview
Analysis was performed on SCADA data from a hydropower generating unit for the period January-December 2024.

---

## Signals Analyzed

- Bearing Temperature (`M_LAGTMP3`)
- Oil Temperature (`M_OLTMP2`)
- Vibration (`M_VIBR1`)
- Current (`M_I`)
- Power Output (`M_P_MW`)
- Power Setpoint (`M_PSP`)
- Wicket Gate Position (`M_WCKT_POS`)
- Turbine Speed (`M_TURT`)
- Water Pressure (`M_TRYKK`)

---

## Bearing Temperature Findings

- Bearing temperature generally operated between 40°C and 60°C.
- Temperature increased gradually from winter into summer.
- Peak temperatures occurred during periods of high power production.
- Multiple isolated drops to 0°C were observed and are likely invalid measurements or SCADA communication issues.
- No evidence of progressive overheating was observed.

### Conclusion
Bearing temperature appears load-dependent and consistent with normal operation.

---

## Oil Temperature Findings

- Oil temperature closely follows bearing temperature.
- Typical operating range was approximately 35°C to 50°C.
- Oil remained several degrees cooler than the bearing, consistent with expected lubrication system behavior.
- Similar invalid spikes and drops were present.

### Conclusion
Lubrication and heat transfer behavior appear normal.

---

## Vibration Findings

- Baseline vibration remains very low throughout most of the year.
- Several isolated spikes were observed.
- A significant vibration spike (~70 units) occurred during February.
- No long-term increase in vibration trend was observed.

### Conclusion
No evidence of progressive mechanical degradation. The February event warrants further investigation.

---

## Current Findings

- Current remained relatively stable for most of the year.
- Very large spikes (~1200+) were observed around May-June.
- These spikes are significantly larger than normal operating values and may represent transient events or data anomalies.

### Conclusion
The underlying trend appears stable, but extreme outliers should be investigated separately.

---

## Power Output Findings

- Power output varied between approximately 1.5 MW and 10 MW.
- High production periods occurred primarily from spring through autumn.
- Actual power closely followed the power setpoint.
- Frequent periods of low or zero generation likely correspond to unit shutdowns or reduced operational demand.

### Conclusion
Power production behavior appears normal and governed effectively.

---

## Wicket Gate Position Findings

- Wicket gate position generally varied between approximately 20% and 90%.
- Larger openings corresponded to increased power generation.
- A single extreme spike of approximately 405 was observed.

### Interpretation of the 405 Spike

- A value of 405% opening is physically impossible.
- The event coincides with the large vibration anomaly observed in February.
- Most likely explanations:
  - SCADA/data acquisition anomaly
  - Scaling error
  - Communication fault during a transient operating event

### Conclusion
Wicket gate position generally behaves as expected and correlates strongly with power generation.

---

## Turbine Speed Findings

- Turbine speed remained near 100% for almost the entire year.
- Brief drops to 0 correspond to shutdown periods.
- Short overshoots above 100% were observed during transient events.

### Conclusion
The governor appears to regulate turbine speed effectively.

---

## Water Pressure Findings

- Water pressure remained remarkably constant around 37-38 units.
- Small fluctuations appear related to operational changes.
- Several instantaneous drops to 0 were observed and are not physically realistic.

### Conclusion
The hydraulic system appears stable. Zero-pressure events are likely instrumentation or data issues.

---

## February Anomaly Investigation

The February event is characterized by:

- Large vibration spike (~70)
- Wicket gate spike (~405)
- Turbine speed disturbance
- Water pressure anomaly
- No corresponding long-term increase in temperature

### Assessment

A real operational transient likely occurred, such as:

- Unit startup
- Unit shutdown
- Governor action
- Load transition

During this event, one or more sensors may also have recorded invalid values.

---

## Overall System Assessment

The analyzed hydropower unit appears mechanically healthy.

### Evidence

- Bearing temperature follows load.
- Oil temperature follows bearing temperature.
- Turbine speed remains tightly regulated.
- Water pressure remains stable.
- Vibration baseline remains low.
- Power output follows the power setpoint.

### No Evidence Found For

- Progressive bearing failure
- Lubrication system degradation
- Sustained vibration growth
- Long-term thermal deterioration

---

## Key System Relationship Identified

The strongest relationship observed in the dataset is:

Wicket Gate Opening
→ Increased Water Flow
→ Increased Power Output
→ Increased Current
→ Increased Bearing Temperature
→ Increased Oil Temperature

This relationship is consistent with normal hydropower plant operation.

---

## Final Conclusion

The generating unit demonstrates normal operational behavior throughout most of 2024. Seasonal variations in load explain the observed temperature changes. Isolated anomalies were observed in vibration, current, wicket gate position, and pressure signals, but these events do not indicate ongoing degradation and are more consistent with transient operating conditions, instrumentation issues, or SCADA data anomalies.
