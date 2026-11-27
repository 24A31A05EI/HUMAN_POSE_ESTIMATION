"""Frame-by-frame video processing."""
from pathlib import Path
from typing import Optional
import math
import shutil
import subprocess

import cv2

from models.pose_model import run_inference
from modules.keypoint_detection import extract_people
from modules.visualization import draw_pose


def _ffmpeg_convert(input_path: str, output_path: str) -> bool:
    """Convert an OpenCV output to H.264 MP4 when FFmpeg is available."""
    executable = shutil.which("ffmpeg")
    if executable:
        command = [
            executable, "-y", "-i", input_path,
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", output_path,
        ]
    else:
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return False
        command = [
            executable, "-y", "-i", input_path,
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", output_path,
        ]

    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, check=False)
        return result.returncode == 0 and Path(output_path).stat().st_size > 0
    except Exception:
        return False


def process_video(model, input_path: str, output_path: str, conf: float = 0.25,
                  kpt_conf: float = 0.25, imgsz: int = 640,
                  progress_callback=None, max_frames: Optional[int] = None):
    """Process a video and return output path plus aggregate statistics."""
    input_file = Path(input_path)
    if not input_file.exists() or input_file.stat().st_size == 0:
        raise RuntimeError("The uploaded video file is empty or missing.")

    capture = cv2.VideoCapture(str(input_file))
    if not capture.isOpened():
        raise RuntimeError("Could not open the uploaded video. Try a standard MP4 or AVI H.264 file.")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if not math.isfinite(fps) or fps <= 0:
        fps = 24.0
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("The uploaded video has invalid frame dimensions.")

    width -= width % 2
    height -= height % 2

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = destination.with_name(destination.stem + "_raw.avi")
    temporary_path.unlink(missing_ok=True)
    writer = cv2.VideoWriter(str(temporary_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not initialize the video output writer.")

    frames = 0
    people_total = 0
    confidence_values = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frames is not None and frames >= max_frames:
                break

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            results = run_inference(model, frame, conf=conf, imgsz=imgsz)
            people = extract_people(results, kpt_conf_threshold=kpt_conf)
            annotated = draw_pose(frame, people, kpt_threshold=kpt_conf)
            writer.write(annotated)
            frames += 1
            people_total += len(people)

            for person in people:
                confidence_values.extend(
                    k["confidence"] for k in person["keypoints"] if k["confidence"] > 0
                )

            if progress_callback:
                progress_callback(frames, total)
    finally:
        capture.release()
        writer.release()

    if frames == 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("No video frames could be read. The source may be corrupted or use an unsupported codec.")

    converted = _ffmpeg_convert(str(temporary_path), str(destination))
    if not converted:
        # If FFmpeg is unavailable, return the raw AVI as a usable fallback.
        fallback = destination.with_suffix(".avi")
        temporary_path.replace(fallback)
        destination = fallback

    if destination.stat().st_size == 0:
        raise RuntimeError("The processed video file is empty.")

    if temporary_path.exists():
        temporary_path.unlink(missing_ok=True)

    average_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    return {
        "frames": frames,
        "detected_people": people_total,
        "average_confidence": average_confidence,
        "output_path": str(destination),
    }
