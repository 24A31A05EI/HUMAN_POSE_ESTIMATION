import pandas as pd
import streamlit as st


def render_session_dashboard():
    st.title("Session Analytics")

    if "session_metrics" not in st.session_state:
        st.info("Session metrics appear after you process frames in image, video, webcam, or posture workflows.")
        return

    metrics = st.session_state.session_metrics
    summary = metrics.summary()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames Processed", summary["frames_processed"])
    c2.metric("People Detected", summary["people_detected"])
    c3.metric("Average Confidence", f"{summary['average_confidence']:.1%}")
    c4.metric("Latest Posture", summary["latest_posture"])

    if metrics.angle_history:
        angle_df = pd.DataFrame(list(metrics.angle_history))
        numeric = angle_df.drop(columns=["frame"]).select_dtypes(include="number")
        if not numeric.empty:
            st.markdown("### Joint-angle timeline")
            st.line_chart(numeric.set_axis(angle_df["frame"]))
        st.markdown("### Session records")
        st.dataframe(angle_df, use_container_width=True)

    if metrics.posture_history:
        posture_counts = pd.Series(metrics.posture_history).value_counts()
        st.markdown("### Posture distribution")
        st.bar_chart(posture_counts)

    if st.button("Export Session Data", type="primary"):
        paths = metrics.export()
        st.success("Session report created")
        with open(paths["json"], "rb") as file:
            st.download_button("Download JSON summary", file.read(), "session_summary.json", "application/json")
        with open(paths["csv"], "rb") as file:
            st.download_button("Download angle history CSV", file.read(), "angle_history.csv", "text/csv")
