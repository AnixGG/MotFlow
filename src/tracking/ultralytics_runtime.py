from __future__ import annotations

import sys

from contextlib import contextmanager
from typing import Any

from utils.config import LOCAL_ULTRALYTICS_ROOT


def ensure_local_ultralytics() -> None:
    if LOCAL_ULTRALYTICS_ROOT.exists():
        local_path = str(LOCAL_ULTRALYTICS_ROOT)
        if local_path not in sys.path:
            sys.path.insert(0, local_path)


def get_ultralytics_version() -> str:
    ensure_local_ultralytics()
    try:
        import ultralytics
    except Exception as e:
        return "unknown: exception"
    return str(getattr(ultralytics, "__version__", "unknown"))

def load_yolo_model(model_path: str) -> Any:
    ensure_local_ultralytics()
    from ultralytics import YOLO

    return YOLO(model_path)
