import cv2
import streamlit as st

from models.pose_model import load_pose_model
from modules.keypoint_detection import extract_people
from modules.visualization import draw_pose
from modules.webcam_processing import open_webcam, release_webcam


def render_webcam_detection():
    st.title("Live Webcam Detection")
    c1, c2 = st.columns(2)
    conf = c1.slider("Detection confidence", 0.10, 0.90, 0.25, 0.05)
    kpt_conf = c2.slider("Keypoint confidence", 0.10, 0.90, 0.25, 0.05)

    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    b1, b2 = st.columns(2)
    if b1.button("Start Webcam", type="primary"):
        st.session_state.webcam_running = True
    if b2.button("Stop Webcam"):
        st.session_state.webcam_running = False

    placeholder = st.empty()
    if not st.session_state.webcam_running:
        st.info("Press Start Webcam to begin. Stop the loop to release camera resources.")
        return

    camera = None
    try:
        model, model_name, device = load_pose_model()
        camera = open_webcam(0)
        st.caption(f"Model: {model_name} · Device: {device}")

        frame_count = 0
        while st.session_state.webcam_running:
            ok, frame = camera.read()
            if not ok:
                placeholder.error("Could not read a frame from the webcam.")
                break

            results = model.predict(frame, conf=conf, verbose=False)
            people = extract_people(results, kpt_conf_threshold=kpt_conf)
            annotated = draw_pose(frame, people, kpt_threshold=kpt_conf)
            placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                              channels="RGB", use_container_width=True,
                              caption=f"People detected: {len(people)}")
            frame_count += 1
            if frame_count >= 150:
                st.warning("Preview stopped after 150 frames. Press Stop or Start again.")
                st.session_state.webcam_running = False
    except Exception as exc:
        st.error(f"Webcam error: {exc}")
        st.session_state.webcam_running = False
    finally:
        release_webcam(camera)
