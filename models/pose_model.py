"""Modular loading and inference wrapper for Ultralytics pose models."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
import torch


@st.cache_resource(show_spinner=False)
def load_pose_model(model_name: str = "yolo26n-pose.pt", device: Optional[str] = None):
    """Load a pose model once per Streamlit process.

    YOLO26 Pose is preferred. When that checkpoint is unavailable in an older
    Ultralytics release, this wrapper falls back to the lightweight YOLO11/YOLOv8
    pose checkpoints. Ultralytics downloads recognized checkpoints automatically.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc

    selected_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    candidates = [model_name, "yolo11n-pose.pt", "yolov8n-pose.pt"]
    last_error: Optional[Exception] = None
    tried = []

    for candidate in dict.fromkeys(candidates):
        tried.append(candidate)
        try:
            model = YOLO(candidate)
            model.to(selected_device)
            return model, candidate, selected_device
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to load a YOLO pose checkpoint. Tried: " + ", ".join(tried) +
        f". Check internet access and Ultralytics compatibility. Last error: {last_error}"
    )


def run_inference(model, frame, conf: float = 0.25, imgsz: int = 640):
    """Run pose inference on an OpenCV BGR frame and return Ultralytics results."""
    return model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)
