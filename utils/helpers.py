"""Small reusable utility functions."""
from pathlib import Path
import tempfile
import uuid


def safe_extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def make_temp_file(suffix: str) -> str:
    temp_dir = tempfile.gettempdir()
    path = Path(temp_dir) / f"hpe_{uuid.uuid4().hex}{suffix}"
    return str(path)


def ensure_project_folders(root: Path) -> None:
    for folder in ("outputs", "dataset"):
        (root / folder).mkdir(parents=True, exist_ok=True)
