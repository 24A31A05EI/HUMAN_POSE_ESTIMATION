import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from models.pose_model import load_pose_model
from modules.image_processing import process_image
from modules.angle_calculation import calculate_joint_angles
from modules.posture_analysis import classify_posture


def render_posture_analysis():
    st.title("Posture Analysis")
    uploaded = st.file_uploader("Upload an image for posture analysis", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Upload an image containing a visible person.")
        return

    image = Image.open(uploaded).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    try:
        model, model_name, device = load_pose_model()
        annotated, people = process_image(model, image_bgr)
        st.caption(f"Model: {model_name} · Device: {device}")
        if not people:
            st.warning("No person was detected.")
            return

        person_index = 0
        if len(people) > 1:
            person_index = st.selectbox("Select person", range(1, len(people)+1), format_func=lambda x: f"Person {x}") - 1
        points = people[person_index]["keypoints"]
        angles = calculate_joint_angles(points)
        posture, explanation = classify_posture(points, angles)

        a, b = st.columns([1.2, 1])
        with a:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        with b:
            st.metric("Detected Posture", posture)
            st.write(explanation)
            st.caption("Educational estimate, not a medical diagnosis.")
            angle_df = pd.DataFrame({
                "Joint": list(angles),
                "Angle (degrees)": pd.Series(
                    [float(v) if v is not None else np.nan for v in angles.values()],
                    dtype="float64",
                ),
            })
            st.dataframe(angle_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
