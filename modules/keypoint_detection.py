"""Convert Ultralytics pose results into a consistent data structure."""
from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def extract_people(results: Any, kpt_conf_threshold: float = 0.25) -> List[Dict]:
    """Extract people and their 17 COCO keypoints from one Ultralytics result."""
    people = []
    if not results:
        return people

    result = results[0]
    if result.keypoints is None:
        return people

    xy = result.keypoints.xy.cpu().numpy()
    conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
    boxes = result.boxes

    for person_idx in range(len(xy)):
        points = []
        for kpt_idx, (x, y) in enumerate(xy[person_idx]):
            score = float(conf[person_idx, kpt_idx]) if conf is not None else 1.0
            visible = bool(score >= kpt_conf_threshold and np.isfinite(x) and np.isfinite(y))
            points.append({"x": float(x), "y": float(y), "confidence": score, "visible": visible})

        box = None
        if boxes is not None and len(boxes) > person_idx:
            box = [float(v) for v in boxes.xyxy[person_idx].cpu().numpy()]
        people.append({"box": box, "keypoints": points})
    return people


def keypoint_table(people: List[Dict], keypoint_names: Dict[int, str]):
    """Flatten detections into a pandas-friendly list."""
    rows = []
    for person_index, person in enumerate(people, start=1):
        for kpt_index, kpt in enumerate(person["keypoints"]):
            rows.append({
                "Person": person_index,
                "Keypoint": keypoint_names.get(kpt_index, str(kpt_index)),
                "X": round(kpt["x"], 1),
                "Y": round(kpt["y"], 1),
                "Confidence": round(kpt["confidence"], 3),
            })
    return rows
