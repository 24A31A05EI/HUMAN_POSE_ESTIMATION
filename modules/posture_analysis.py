"""Basic, non-medical heuristic posture classification."""
from typing import Dict, Tuple, Optional
import numpy as np


def _valid(value: Optional[float]) -> bool:
    return value is not None and np.isfinite(value)


def classify_posture(keypoints: list, angles: Dict[str, Optional[float]]) -> Tuple[str, str]:
    """Classify Standing, Sitting, Arms raised, Squatting, or Bending.

    The function is educational and intentionally not a medical assessment.
    """
    left_knee = angles.get("Left Knee")
    right_knee = angles.get("Right Knee")
    left_shoulder = angles.get("Left Shoulder")
    right_shoulder = angles.get("Right Shoulder")

    try:
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_shoulder_kpt = keypoints[5]
        right_shoulder_kpt = keypoints[6]
        nose = keypoints[0]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
    except IndexError:
        return "Uncertain", "Not enough keypoints were detected."

    wrist_above = (
        left_wrist["visible"] and right_wrist["visible"] and
        left_shoulder_kpt["visible"] and right_shoulder_kpt["visible"] and
        left_wrist["y"] < left_shoulder_kpt["y"] - 15 and
        right_wrist["y"] < right_shoulder_kpt["y"] - 15
    )
    if wrist_above:
        return "Arms raised", "Both wrists are above the shoulders."

    knee_values = [v for v in (left_knee, right_knee) if _valid(v)]
    if knee_values and np.mean(knee_values) < 125:
        return "Squatting", f"Average knee angle is {np.mean(knee_values):.1f}° (bent posture)."

    if left_wrist["visible"] and right_wrist["visible"] and nose["visible"]:
        hip_ys = [p["y"] for p in (left_hip, right_hip) if p["visible"]]
        if hip_ys:
            hip_y = float(np.mean(hip_ys))
            if nose["y"] > hip_y - 20:
                return "Bending", "Head/upper body is low relative to the hip line."

    return "Standing", "No strong bent-knee, arm-raised, or lowered-head pattern was detected."
