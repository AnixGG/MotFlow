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

    def _build_static_mask(self, h: int, w: int, detections: list | np.ndarray | None) -> np.ndarray:
        if detections is None:
            return np.ones((h, w), dtype=bool)
        
        det_array = np.asarray(detections)
        mask = np.ones((h, w), dtype=bool)

        for det in det_array:
            x1, y1, x2, y2 = det[:4]

            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = False

        return mask

    def apply(self, raw_frame: np.ndarray, detections: list | np.ndarray | None = None) -> np.ndarray:
        warp = np.eye(2, 3, dtype=np.float32)
        
        if raw_frame is None or raw_frame.size == 0:
            return warp

        frame = raw_frame
        detections_scaled = detections

        if not self.initializedFirstFrame:
            self.prev_frame = frame.copy()
            self.initializedFirstFrame = True
            return warp
        
        flow = self.raft(self.prev_frame, frame)  # [H, W, 2]
        h, w = flow.shape[:2]

        step = max(2, int(self.config.sample_step))
        ys = np.arange(0, h, step, dtype=int)
        xs = np.arange(0, w, step, dtype=int)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")

        p0 = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
        flow_samples = flow[grid_y, grid_x].reshape(-1, 2).astype(np.float32)
        valid = np.isfinite(flow_samples).all(axis=1)

        static_mask = self._build_static_mask(h, w, detections_scaled)
        valid &= static_mask[grid_y, grid_x].reshape(-1)

        p0 = p0[valid]
        flow_samples = flow_samples[valid]

        if p0.shape[0] > 0:
            shift = np.median(flow_samples, axis=0)
            warp[:, 2] = shift.astype(np.float32)

        self.prev_frame = frame.copy()
        return warp
