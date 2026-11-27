"""Metrics and aggregate statistics helpers."""
from typing import Iterable
import numpy as np


def average_confidence(keypoints) -> float:
    if not keypoints:
        return 0.0
    confidences = [float(k[2]) for k in keypoints if len(k) >= 3 and np.isfinite(k[2])]
    return float(np.mean(confidences)) if confidences else 0.0


def mean_list(values: Iterable[float]) -> float:
    valid = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(valid)) if valid else 0.0
