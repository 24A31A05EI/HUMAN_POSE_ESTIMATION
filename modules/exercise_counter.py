"""State-machine repetition counters for squats and bicep curls."""
from dataclasses import dataclass


@dataclass
class RepetitionCounter:
    exercise: str = "Squats"
    count: int = 0
    state: str = "ready"

    def update(self, angles: dict) -> dict:
        if self.exercise == "Squats":
            left = angles.get("Left Knee")
            right = angles.get("Right Knee")
            valid = [v for v in (left, right) if v is not None]
            angle = sum(valid) / len(valid) if valid else None
            down_threshold, up_threshold = 115.0, 155.0
        else:
            left = angles.get("Left Elbow")
            right = angles.get("Right Elbow")
            valid = [v for v in (left, right) if v is not None]
            angle = sum(valid) / len(valid) if valid else None
            down_threshold, up_threshold = 55.0, 145.0

        if angle is None:
            self.state = "waiting for visible joints"
        elif angle < down_threshold:
            self.state = "contracted/down"
        elif angle > up_threshold:
            if self.state in ("contracted/down", "down"):
                self.count += 1
            self.state = "extended/up"
        else:
            self.state = "transition"

        return {"count": self.count, "status": self.state, "angle": angle}

    def reset(self):
        self.count = 0
        self.state = "ready"
