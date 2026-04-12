from __future__ import annotations
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

@dataclass
class RaftGMCConfig:
    model_name: str = "small"
    device: str | None = None
    mixed_precision: bool = False
    scale_gmc: float = 1
    scale: float = 1
    sample_step: int = 8
    ransac_reproj_threshold: float = 3

def default_model_path() -> Path | str:
    local_weight = REPO_ROOT / "models" / "detector" / "yolo11n.pt"
    if local_weight.exists():
        return local_weight.resolve()
    return "yolo11n.pt"


def default_tracker_path() -> Path:
    return (REPO_ROOT / "configs" / "botsort_baseline.yaml").resolve()
