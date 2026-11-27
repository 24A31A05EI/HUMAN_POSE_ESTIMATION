"""Human Pose Estimation Streamlit application."""
from pathlib import Path
import streamlit as st

from pages.image_detection import render_image_detection
from pages.video_detection import render_video_detection
from pages.webcam_detection import render_webcam_detection
from pages.posture_analysis import render_posture_analysis
from pages.exercise_tracking import render_exercise_tracking

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Human Pose Estimation Using Deep Learning",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = ROOT / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

st.sidebar.title("🏃 Pose AI")
st.sidebar.caption("Human Pose Estimation")
page = st.sidebar.radio(
    "Navigation",
    ["Home Dashboard", "Image Pose Detection", "Video Pose Detection",
     "Live Webcam Detection", "Posture Analysis", "Exercise Tracking", "About Project"],
)

if page == "Home Dashboard":
    st.markdown("# Human Pose Estimation Using Deep Learning")
    st.markdown("<p class='subtitle'>AI-powered 17-keypoint body pose detection for images, videos, and live streams.</p>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Keypoints", "17", "COCO format")
    c2.metric("Input Modes", "3", "Image · Video · Webcam")
    c3.metric("Joint Angles", "6", "Elbows · Knees · Shoulders")
    c4.metric("Tracking", "2", "Squats · Curls")

    st.markdown("### Project Overview")
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("""
        This application uses a pretrained Ultralytics YOLO pose model to locate human joints,
        connect them into a skeleton, calculate joint angles, and provide basic posture and exercise
        repetition insights.

        **Objectives**
        - Detect one or multiple people in visual data
        - Visualize accurate joint and skeleton information
        - Analyze simple body geometry and movement states
        - Present results through a responsive professional dashboard
        """)
    with right:
        st.info("**Technology**\n\nYOLO Pose · OpenCV · NumPy · Streamlit · Pandas · Matplotlib")

    st.markdown("### Main Features")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("<div class='feature-card'>🖼️<h4>Image Detection</h4><p>Upload and analyze JPG, JPEG, or PNG images.</p></div>", unsafe_allow_html=True)
    with f2:
        st.markdown("<div class='feature-card'>🎞️<h4>Video Detection</h4><p>Process frames and export an annotated MP4.</p></div>", unsafe_allow_html=True)
    with f3:
        st.markdown("<div class='feature-card'>📷<h4>Live Detection</h4><p>Use a webcam for real-time pose overlays.</p></div>", unsafe_allow_html=True)

    st.warning("Educational posture and exercise estimates are not medical diagnoses.")

elif page == "Image Pose Detection":
    render_image_detection()
elif page == "Video Pose Detection":
    render_video_detection()
elif page == "Live Webcam Detection":
    render_webcam_detection()
elif page == "Posture Analysis":
    render_posture_analysis()
elif page == "Exercise Tracking":
    render_exercise_tracking()
else:
    st.title("About Project")
    st.markdown("""
    ## Human Pose Estimation Using Deep Learning

    This project combines pretrained deep learning pose estimation with a Streamlit dashboard.
    It does not train a model from scratch and keeps model loading in `models/pose_model.py`.

    ### Limitations
    - Accuracy depends on image quality, lighting, occlusion, and camera angle.
    - Low-confidence joints are omitted rather than interpreted.
    - Posture labels and repetition counts are simple geometric heuristics.
    - They must not be used for clinical, diagnostic, or rehabilitation decisions.
    """)
