"""Lightweight temporal smoothing and tracking helpers for keypoints."""
from __future__ import annotations

from typing import Dict, List
import numpy as np


class KeypointSmoother:
    """Smooth keypoint coordinates with a bounded EMA across frames."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self.previous: Dict[int, np.ndarray] = {}

    def update(self, people: List[Dict]) -> List[Dict]:
        smoothed = []
        for person_index, person in enumerate(people):
            points = []
            for index, kpt in enumerate(person["keypoints"]):
                key = person_index * 100 + index
                current = np.array([kpt["x"], kpt["y"]], dtype=np.float32)
                if kpt["visible"] and np.isfinite(current).all():
                    if key in self.previous:
                        current = self.alpha * current + (1.0 - self.alpha) * self.previous[key]
                    self.previous[key] = current
                    kpt = dict(kpt, x=float(current[0]), y=float(current[1]))
                points.append(kpt)
            smoothed.append(dict(person, keypoints=points))
        return smoothed

    def reset(self) -> None:
        self.previous.clear()
