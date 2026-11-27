"""Image pose-processing workflow."""
from typing import Any, Dict, List, Tuple
import cv2

from models.pose_model import run_inference
from modules.keypoint_detection import extract_people
from modules.visualization import draw_pose


def process_image(model, image_bgr, conf: float = 0.25, kpt_conf: float = 0.25,
                  imgsz: int = 640) -> Tuple[Any, List[Dict]]:
    results = run_inference(model, image_bgr, conf=conf, imgsz=imgsz)
    people = extract_people(results, kpt_conf_threshold=kpt_conf)
    annotated = draw_pose(image_bgr, people, kpt_threshold=kpt_conf)
    return annotated, people
