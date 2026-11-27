"""OpenCV rendering utilities for keypoints and skeletons."""
from typing import Dict, List
import cv2

from utils.config import COCO_SKELETON


def draw_pose(frame, people: List[Dict], kpt_threshold: float = 0.25) -> object:
    """Draw a clear 17-keypoint skeleton on a copy of a BGR frame."""
    annotated = frame.copy()

    for person_index, person in enumerate(people):
        points = person["keypoints"]

        for a, b in COCO_SKELETON:
            if a >= len(points) or b >= len(points):
                continue
            pa, pb = points[a], points[b]
            if pa["confidence"] >= kpt_threshold and pb["confidence"] >= kpt_threshold:
                cv2.line(annotated, (int(pa["x"]), int(pa["y"])),
                         (int(pb["x"]), int(pb["y"])), (0, 220, 255), 3, cv2.LINE_AA)

        for kpt_index, kpt in enumerate(points):
            if kpt["confidence"] >= kpt_threshold:
                cv2.circle(annotated, (int(kpt["x"]), int(kpt["y"])), 5,
                           (255, 105, 180), -1, cv2.LINE_AA)
                cv2.circle(annotated, (int(kpt["x"]), int(kpt["y"])), 7,
                           (255, 255, 255), 1, cv2.LINE_AA)

        if person.get("box"):
            x1, y1, x2, y2 = map(int, person["box"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (91, 157, 255), 2)
            cv2.putText(annotated, f"Person {person_index + 1}", (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (91, 157, 255), 2, cv2.LINE_AA)
    return annotated
