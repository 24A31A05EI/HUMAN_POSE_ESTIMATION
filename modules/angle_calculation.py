"""Calculate joint angles from 17 COCO keypoints using NumPy."""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees (0-180)."""
    ba = a - b
    bc = c - b
    norm_product = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm_product == 0:
        return float("nan")
    cosine = np.dot(ba, bc) / norm_product
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _get_point(points: list, index: int, threshold: float):
    if index >= len(points):
        return None
    kpt = points[index]
    if not kpt["visible"] or kpt["confidence"] < threshold:
        return None
    if not np.isfinite(kpt["x"]) or not np.isfinite(kpt["y"]):
        return None
    return np.array([kpt["x"], kpt["y"]], dtype=np.float32)


def calculate_joint_angles(keypoints: list, confidence_threshold: float = 0.25) -> Dict[str, Optional[float]]:
    """Calculate elbow, knee, and shoulder angles for one person."""
    joints = {
        "Left Elbow": (5, 7, 9),
        "Right Elbow": (6, 8, 10),
        "Left Knee": (11, 13, 15),
        "Right Knee": (12, 14, 16),
        "Left Shoulder": (7, 5, 11),
        "Right Shoulder": (8, 6, 12),
    }
    angles = {}
    for name, (a_idx, b_idx, c_idx) in joints.items():
        a = _get_point(keypoints, a_idx, confidence_threshold)
        b = _get_point(keypoints, b_idx, confidence_threshold)
        c = _get_point(keypoints, c_idx, confidence_threshold)
        if a is None or b is None or c is None:
            angles[name] = None
        else:
            value = calculate_angle(a, b, c)
            angles[name] = None if not np.isfinite(value) else round(value, 1)
    return angles
