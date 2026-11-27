
"""Model loading and inference wrapper for Ultralytics pose models."""

from pathlib import Path
from typing import Optional

import streamlit as st
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

@st.cache_resource(show_spinner="Loading pose estimation model...")
def load_pose_model(
    model_name: str = "yolo26n-pose.pt",
    device: Optional[str] = None
):
    """Load and cache the YOLO pose model."""

    selected_device = device or (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # Check local checkpoint first.
    local_model = ROOT / model_name

    candidates = [
        str(local_model) if local_model.exists() else model_name,
        "yolo11n-pose.pt",
        "yolov8n-pose.pt",
    ]

    last_error = None

    for candidate in dict.fromkeys(candidates):
        try:
            model = YOLO(candidate)
            model.to(selected_device)

            return model, candidate, selected_device

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Unable to load YOLO pose model. "
        f"Last error: {last_error}"
    )


def run_inference(model, frame, conf=0.25, imgsz=640):
    """Run pose inference on an OpenCV BGR frame."""

    results = model.predict(
        source=frame,
        conf=conf,
        imgsz=imgsz,
        device="cpu" if not torch.cuda.is_available() else 0,
        verbose=False
    )

    return results
