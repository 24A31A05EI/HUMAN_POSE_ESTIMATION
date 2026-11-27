import numpy as np

from modules.angle_calculation import calculate_angle, calculate_joint_angles
from modules.exercise_counter import RepetitionCounter
from modules.temporal_tracking import KeypointSmoother
from modules.posture_analysis import classify_posture


def test_calculate_angle():
    assert np.isclose(calculate_angle(np.array([0, 1]), np.array([0, 0]), np.array([1, 0])), 90.0)


def test_missing_keypoints():
    points = [{"x": 0.0, "y": 0.0, "confidence": 0.0, "visible": False} for _ in range(17)]
    assert all(value is None for value in calculate_joint_angles(points).values())


def test_repetition_state_requires_contraction():
    counter = RepetitionCounter("Squats")
    counter.update({"Left Knee": 110, "Right Knee": 110})
    counter.update({"Left Knee": 110, "Right Knee": 110})
    result = counter.update({"Left Knee": 160, "Right Knee": 160})
    assert result["count"] == 1


def test_posture_missing_data_is_uncertain():
    points = [{"x": 0.0, "y": 0.0, "confidence": 0.0, "visible": False} for _ in range(17)]
    posture, _ = classify_posture(points, {})
    assert posture == "Standing"


def test_keypoint_smoothing():
    smoother = KeypointSmoother(alpha=0.5)
    people = [{"keypoints": [{"x": 0.0, "y": 0.0, "confidence": 1.0, "visible": True}]}]
    first = smoother.update(people)
    people[0]["keypoints"][0]["x"] = 20
    second = smoother.update(people)
    assert first[0]["keypoints"][0]["x"] == 0.0
    assert second[0]["keypoints"][0]["x"] == 10.0
