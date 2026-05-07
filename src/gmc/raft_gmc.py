from __future__ import annotations

import time

import cv2
import numpy as np

from raft.raft_wrapper import RAFTWrapper
from utils.config import RaftGMCConfig


class RaftGMC:
    def __init__(self, method: str = "raft", scale_gmc: float = 0.5, config: RaftGMCConfig | None = None) -> None:
        self.method = method
        self.scale_gmc = float(scale_gmc)
        self.config = config or RaftGMCConfig()
        self.image_size = 128
        self.last_timing_ms = None
        self.last_timing_breakdown_ms = {}

        self.raft = RAFTWrapper(
            name=self.config.model_name,
            device=self.config.device,
            mixed_precision=self.config.mixed_precision,
        )
        self.prev_frame: np.ndarray | None = None
        self.initializedFirstFrame = False

    def _prepare_frame(self, raw_frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        frame = raw_frame
        scale_x = 1.0
        scale_y = 1.0

        if self.scale_gmc != 1:
            h0, w0 = frame.shape[:2]
            new_w = max(1, int(round(w0 * self.scale_gmc)))
            new_h = max(1, int(round(h0 * self.scale_gmc)))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scale_x /= self.scale_gmc
            scale_y /= self.scale_gmc

        h, w = frame.shape[:2]
        size = max(128, self.image_size)
        if size % 8:
            size = ((size + 7) // 8) * 8

        scale_x *= w / size
        scale_y *= h / size
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        return frame, scale_x, scale_y

    def reset_params(self) -> None:
        self.prev_frame = None
        self.initializedFirstFrame = False
        self.last_timing_ms = None
        self.last_timing_breakdown_ms = {}

    def apply(self, raw_frame: np.ndarray, detections: list | np.ndarray | None = None) -> np.ndarray:
        total_start = time.perf_counter()
        warp = np.eye(2, 3, dtype=np.float32)

        if raw_frame is None or raw_frame.size == 0:
            self.last_timing_ms = 0
            self.last_timing_breakdown_ms = {"total": 0}
            return warp

        prepare_start = time.perf_counter()

        frame, scale_x, scale_y = self._prepare_frame(raw_frame)
        prepare_ms = (time.perf_counter() - prepare_start) * 1000

        if not self.initializedFirstFrame:
            self.prev_frame = frame.copy()
            self.initializedFirstFrame = True

            total_ms = (time.perf_counter() - total_start) * 1000
            self.last_timing_ms = total_ms
            self.last_timing_breakdown_ms = {
                "prepare": prepare_ms,
                "raft_infer": 0,
                "sample_filter": 0,
                "warp_estimation": 0,
                "total": total_ms,
            }
            return warp

        raft_start = time.perf_counter()
        flow = self.raft(self.prev_frame, frame)  # low-res [H/8, W/8, 2], values in resized-frame pixels
        raft_infer_ms = (time.perf_counter() - raft_start) * 1000

        sample_start = time.perf_counter()

        flow_flat = flow.reshape(-1, 2).astype(np.float32)
        valid = np.isfinite(flow_flat).all(axis=1)
        flow_samples = flow_flat[valid]

        sample_filter_ms = (time.perf_counter() - sample_start) * 1000

        warp_start = time.perf_counter()

        if flow_samples.shape[0] > 0:
            shift = np.median(flow_samples, axis=0)
            warp[0, 2] = float(shift[0]) * scale_x
            warp[1, 2] = float(shift[1]) * scale_y

        warp_estimation_ms = (time.perf_counter() - warp_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000
        self.last_timing_ms = total_ms
        self.last_timing_breakdown_ms = {
            "prepare": prepare_ms,
            "raft_infer": raft_infer_ms,
            "sample_filter": sample_filter_ms,
            "warp_estimation": warp_estimation_ms,
            "total": total_ms,
        }

        self.prev_frame = frame.copy()
        return warp
