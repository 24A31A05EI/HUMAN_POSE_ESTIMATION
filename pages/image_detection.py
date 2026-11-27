import time
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from models.pose_model import load_pose_model
from modules.image_processing import process_image
from modules.angle_calculation import calculate_joint_angles
from modules.keypoint_detection import keypoint_table
from utils.config import COCO_KEYPOINT_NAMES, IMAGE_EXTENSIONS, DEFAULT_CONFIDENCE


def render_image_detection():
    st.title("Image Pose Detection")
    uploaded = st.file_uploader("Upload an image", type=list(IMAGE_EXTENSIONS))

    c1, c2, c3 = st.columns(3)
    conf = c1.slider("Detection confidence", 0.10, 0.90, DEFAULT_CONFIDENCE, 0.05)
    kpt_conf = c2.slider("Keypoint confidence", 0.10, 0.90, 0.25, 0.05)
    imgsz = c3.select_slider("Inference size", options=[320, 480, 640, 800], value=640)

    if not uploaded:
        st.info("Upload a JPG, JPEG, or PNG image to begin.")
        return

    try:
        pil_image = Image.open(uploaded).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        st.error(f"Invalid or unreadable image: {exc}")
        return

    if st.button("Detect Pose", type="primary"):
        try:
            start = time.time()
            model, model_name, device = load_pose_model()
            annotated, people = process_image(
                model, image_bgr, conf=conf, kpt_conf=kpt_conf, imgsz=imgsz
            )
            elapsed = time.time() - start

            st.caption(f"Model: {model_name} · Device: {device} · Processing time: {elapsed:.2f}s")
            original = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            processed = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.image(original, caption="Original Image", use_container_width=True)
            with col2:
                st.image(processed, caption="Pose Overlay", use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("People Detected", len(people))
            confidence = [k["confidence"] for p in people for k in p["keypoints"]]
            m2.metric("Average Keypoint Confidence", f"{(np.mean(confidence) if confidence else 0):.1%}")
            m3.metric("Processing Time", f"{elapsed:.2f}s")

            if people:
                st.markdown("### Keypoint Coordinates")
                st.dataframe(pd.DataFrame(keypoint_table(people, COCO_KEYPOINT_NAMES)), use_container_width=True)

                person_index = 0
                if len(people) > 1:
                    person_index = st.selectbox("Select person for angles", range(len(people)),
                                                format_func=lambda i: f"Person {i + 1}")
                angles = calculate_joint_angles(people[person_index]["keypoints"], kpt_conf)
                st.markdown("### Joint Angles")
                angle_df = pd.DataFrame({
                    "Joint": list(angles.keys()),
                    "Angle (degrees)": pd.Series(
                        [float(v) if v is not None else np.nan for v in angles.values()],
                        dtype="float64",
                    ),
                })
                st.dataframe(angle_df, use_container_width=True)

            ok, encoded = cv2.imencode(".jpg", annotated)
            if ok:
                st.download_button(
                    "Download processed image",
                    data=encoded.tobytes(),
                    file_name="pose_estimation_result.jpg",
                    mime="image/jpeg",
                )
            if not people:
                st.warning("No person was detected. Try a clearer image or lower the confidence threshold.")
        except Exception as exc:
            st.error(f"Image processing failed: {exc}")
