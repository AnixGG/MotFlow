from __future__ import annotations

import cv2
import numpy as np

from raft.raft_wrapper import RAFTWrapper
from utils.config import RaftGMCConfig


class RaftGMC:
    def __init__(self, method: str = "raft", scale_gmc: float = 0.5, config: RaftGMCConfig | None = None) -> None:
        self.method = method
        self.scale_gmc = float(scale_gmc)
        self.config = config or RaftGMCConfig()

        self.raft = RAFTWrapper(
            name=self.config.model_name,
            device=self.config.device,
            mixed_precision=self.config.mixed_precision,
            scale=self.config.scale,
        )
        self.prev_frame: np.ndarray | None = None
        self.initializedFirstFrame = False

    def reset_params(self) -> None:
        self.prev_frame = None
        self.initializedFirstFrame = False

    def apply(self, raw_frame: np.ndarray, detections: list | np.ndarray | None = None) -> np.ndarray:
        return raw_frame