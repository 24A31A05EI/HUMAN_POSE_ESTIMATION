"""Session-level metrics and export helpers."""
from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict
import json

import pandas as pd


class SessionMetrics:
    def __init__(self, history_size: int = 300):
        self.frames = 0
        self.people_detected = 0
        self.confidence_sum = 0.0
        self.confidence_count = 0
        self.angle_history: Deque[Dict] = deque(maxlen=history_size)
        self.posture_history: Deque[str] = deque(maxlen=history_size)

    def add_frame(self, people: list, confidence_values: list, angles: dict = None,
                  posture: str = None) -> None:
        self.frames += 1
        self.people_detected += len(people)
        values = [float(v) for v in confidence_values if v is not None]
        self.confidence_sum += sum(values)
        self.confidence_count += len(values)
        if angles:
            self.angle_history.append({"frame": self.frames, **angles})
        if posture:
            self.posture_history.append(posture)

    @property
    def average_confidence(self) -> float:
        return self.confidence_sum / self.confidence_count if self.confidence_count else 0.0

    def summary(self) -> Dict:
        return {
            "frames_processed": self.frames,
            "people_detected": self.people_detected,
            "average_confidence": round(self.average_confidence, 4),
            "latest_posture": self.posture_history[-1] if self.posture_history else "No data",
        }

    def export(self, output_dir: str = "outputs") -> Dict[str, str]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = directory / f"pose_session_{stamp}.json"
        csv_path = directory / f"pose_angles_{stamp}.csv"

        json_path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        pd.DataFrame(list(self.angle_history)).to_csv(csv_path, index=False)
        return {"json": str(json_path), "csv": str(csv_path)}
