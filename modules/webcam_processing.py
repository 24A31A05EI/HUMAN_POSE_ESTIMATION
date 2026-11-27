"""Webcam lifecycle helpers."""
import cv2


def open_webcam(index: int = 0):
    """Open the first available webcam and verify that it is readable."""
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Webcam unavailable. Check that it is connected, not used by another app, "
            "and allowed by Windows privacy settings."
        )
    return camera


def release_webcam(camera) -> None:
    if camera is not None:
        camera.release()
