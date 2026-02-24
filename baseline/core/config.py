from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ULTRALYTICS_ROOT = REPO_ROOT / "ultralytics"


@dataclass
class BaselineConfig:
    model: str
    tracker: str
    conf: float
    iou: float
    imgsz: int
    classes: list[int]
    device: str
    resize: str


def ensure_local_ultralytics_path() -> None:
    if LOCAL_ULTRALYTICS_ROOT.exists():
        local_path = str(LOCAL_ULTRALYTICS_ROOT)
        if local_path not in sys.path:
            sys.path.insert(0, local_path)


def default_model_path() -> Path | str:
    local_weight = REPO_ROOT / "tmp" / "yolo11n.pt"
    if local_weight.exists():
        return local_weight.resolve()
    return "yolo11n.pt"


def default_tracker_path() -> Path:
    return (Path(__file__).resolve().parents[1] / "botsort_baseline.yaml").resolve()
