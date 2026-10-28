import json
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st_autorefresh(
    interval=1000,
    key="realtime_refresh"
)

with open(
        "src/real_time_sim/realtime.json"
    ) as f:

        data = json.load(f)

st.set_page_config(
    page_title="Hydropower Predictive Maintenance",
    layout="wide"
)

st.title(
    "Hydropower Predictive Maintenance"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Health Status",
        data["status"]
    )

with col2:

    st.metric(
        "Predicted VIBR1",
        f"{data['predicted_vibr1']:.3f}"
    )

with col3:

    st.metric(
        "Predicted VIBR2",
        f"{data['predicted_vibr2']:.3f}"
    )

col4, col5, col6 = st.columns(3)

with col4:

    st.metric(
        "Actual VIBR1",
        f"{data['actual_vibr1']:.3f}"
    )

with col5:

    st.metric(
        "Actual VIBR2",
        f"{data['actual_vibr2']:.3f}"
    )

with col6:

    st.metric(
        "Health Score",
        f"{data['health_score']:.3f}"
    )

if data["status"] == "HEALTHY":

    st.success("HEALTHY")

elif data["status"] == "WARNING":

    st.warning("WARNING")

else:

    st.error("ALARM")

st.write(
    f"Timestamp: {data['timestamp']}"
)

