import cv2
import numpy as np
import streamlit as st
from PIL import Image

from models.pose_model import load_pose_model
from modules.image_processing import process_image
from modules.angle_calculation import calculate_joint_angles
from modules.exercise_counter import RepetitionCounter


def render_exercise_tracking():
    st.title("Exercise Tracking")
    exercise = st.radio("Exercise", ["Squats", "Bicep Curls"], horizontal=True)
    uploaded = st.camera_input("Capture a frame")

    if "counter_exercise" not in st.session_state:
        st.session_state.counter_exercise = exercise
        st.session_state.counter = RepetitionCounter(exercise=exercise)
    elif st.session_state.counter_exercise != exercise:
        st.session_state.counter_exercise = exercise
        st.session_state.counter = RepetitionCounter(exercise=exercise)

    if st.button("Reset Counter"):
        st.session_state.counter.reset()

    if not uploaded:
        st.info("Capture or upload a camera frame to analyze an exercise movement.")
        return

    image = Image.open(uploaded).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    try:
        model, model_name, device = load_pose_model()
        annotated, people = process_image(model, image_bgr)
        if not people:
            st.warning("No person detected in this frame.")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            return

        person = people[0]
        angles = calculate_joint_angles(person["keypoints"])
        result = st.session_state.counter.update(angles)

        a, b = st.columns([1.2, 1])
        with a:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        with b:
            m1, m2 = st.columns(2)
            m1.metric("Repetitions", result["count"])
            m2.metric("Status", result["status"])
            angle_label = "Average knee angle" if exercise == "Squats" else "Average elbow angle"
            st.metric(angle_label, "N/A" if result["angle"] is None else f"{result['angle']:.1f}°")
    except Exception as exc:
        st.error(f"Exercise tracking failed: {exc}")
