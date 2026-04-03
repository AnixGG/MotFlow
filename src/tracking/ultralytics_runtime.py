from __future__ import annotations

import sys

from contextlib import contextmanager
from typing import Iterator, Any

from gmc.raft_gmc import RaftGMC
from utils.config import RaftGMCConfig, LOCAL_ULTRALYTICS_ROOT


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

@contextmanager
def patch_botsort_gmc(raft_config: RaftGMCConfig) -> Iterator[None]:
    ensure_local_ultralytics()
    import ultralytics.trackers.bot_sort as bot_sort_module
    from ultralytics.trackers.utils.gmc import GMC as DefaultGMC

    original_gmc = bot_sort_module.GMC

    def get_gmc(method: str = "sparseOptFlow", downscale: int = 2):
        if str(method).lower() == "raft":
            divisor = max(1, int(downscale))
            scale_gmc = 1 / float(divisor)
            return RaftGMC(method=method, scale_gmc=scale_gmc, config=raft_config)
        return DefaultGMC(method=method, downscale=downscale)

    bot_sort_module.GMC = get_gmc
    try:
        yield
    finally:
        bot_sort_module.GMC = original_gmc

