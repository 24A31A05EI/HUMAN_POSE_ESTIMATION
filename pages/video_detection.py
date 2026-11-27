import time
from pathlib import Path
import streamlit as st

from models.pose_model import load_pose_model
from modules.video_processing import process_video
from utils.config import VIDEO_EXTENSIONS
from utils.helpers import make_temp_file


def render_video_detection():
    st.title("Video Pose Detection")
    uploaded = st.file_uploader("Upload a video", type=list(VIDEO_EXTENSIONS))
    c1, c2 = st.columns(2)
    conf = c1.slider("Detection confidence", 0.10, 0.90, 0.25, 0.05)
    kpt_conf = c2.slider("Keypoint confidence", 0.10, 0.90, 0.25, 0.05)

    if not uploaded:
        st.info("Upload an MP4, AVI, or MOV video to begin.")
        return

    suffix = Path(uploaded.name).suffix.lower()
    input_path = make_temp_file(suffix)
    output_path = str(Path("outputs") / f"pose_{int(time.time())}.mp4")
    Path(input_path).write_bytes(uploaded.getbuffer())

    if st.button("Process Video", type="primary"):
        try:
            model, model_name, device = load_pose_model()
            st.caption(f"Model: {model_name} · Device: {device}")
            progress = st.progress(0)
            status = st.empty()

            def update(done, total):
                if total:
                    progress.progress(min(done / total, 1.0))
                status.text(f"Processed {done} / {total or '?'} frames")

            stats = process_video(model, input_path, output_path, conf=conf,
                                  kpt_conf=kpt_conf, progress_callback=update)
            progress.progress(1.0)
            status.success("Video processing complete")

            a, b, c = st.columns(3)
            a.metric("Frames Processed", stats["frames"])
            b.metric("People Detected (sum)", stats["detected_people"])
            c.metric("Average Confidence", f"{stats['average_confidence']:.1%}")

            generated_path = Path(stats["output_path"])
            if not generated_path.exists() or generated_path.stat().st_size == 0:
                raise RuntimeError("Processing completed but no valid output video was created.")

            st.success(f"Generated video: {generated_path.name} ({generated_path.stat().st_size / 1024 / 1024:.2f} MB)")
            if generated_path.suffix.lower() == ".mp4":
                st.video(str(generated_path), format="video/mp4")
                mime = "video/mp4"
            else:
                st.video(str(generated_path), format="video/x-msvideo")
                mime = "video/x-msvideo"

            with open(generated_path, "rb") as video:
                st.download_button("Download processed video", video.read(),
                                   file_name=generated_path.name, mime=mime)
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
