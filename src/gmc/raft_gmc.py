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
        self.last_timing_ms: float | None = None
        self.last_timing_breakdown_ms: dict[str, float] = {}

        self.raft = RAFTWrapper(
            name=self.config.model_name,
            device=self.config.device,
            mixed_precision=self.config.mixed_precision,
        )
        self.prev_frame: np.ndarray | None = None
        self.initializedFirstFrame = False

    def reset_params(self) -> None:
        self.prev_frame = None
        self.initializedFirstFrame = False
        self.last_timing_ms = None
        self.last_timing_breakdown_ms = {}

    def _build_static_mask(self, h: int, w: int, detections: list | np.ndarray | None) -> np.ndarray:
        if detections is None:
            return np.ones((h, w), dtype=bool)
        
        det_array = np.asarray(detections)
        mask = np.ones((h, w), dtype=bool)

        for det in det_array:
            x1, y1, x2, y2 = det[:4]

            x1_clip = np.clip(int(np.floor(x1)), 0, w)
            y1_clip = np.clip(int(np.floor(y1)), 0, h)

            x2_clip = np.clip(int(np.ceil(x2)), 0, h)
            y2_clip = np.clip(int(np.ceil(y2)), 0, h)

            if x2_clip > x1_clip and y2_clip > y1_clip:
                mask[y1_clip:y2_clip, x1_clip:x2_clip] = False

        return mask

    def apply(self, raw_frame: np.ndarray, detections: list | np.ndarray | None = None) -> np.ndarray:
        total_start = time.perf_counter()
        warp = np.eye(2, 3, dtype=np.float32)
        
        if raw_frame is None or raw_frame.size == 0:
            self.last_timing_ms = 0.0
            self.last_timing_breakdown_ms = {"total": 0.0}
            return warp

        frame = raw_frame
        detections_scaled = detections
        prepare_start = time.perf_counter()

        if self.scale_gmc != 1:
            h0, w0 = frame.shape[:2]

            new_w = max(1, int(round(w0 * self.scale_gmc)))
            new_h = max(1, int(round(h0 * self.scale_gmc)))

            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            if detections is not None:
                det_array = np.asarray(detections, dtype=np.float32)
                if det_array.ndim == 1:
                    det_array = det_array.reshape(1, -1)
                    
                det_array = det_array.copy()
                det_array[:, :4] *= self.scale_gmc
                detections_scaled = det_array

        prepare_ms = (time.perf_counter() - prepare_start) * 1000.0

        if not self.initializedFirstFrame:
            self.prev_frame = frame.copy()
            self.initializedFirstFrame = True
            total_ms = (time.perf_counter() - total_start) * 1000.0
            self.last_timing_ms = total_ms
            self.last_timing_breakdown_ms = {
                "prepare": prepare_ms,
                "raft_infer": 0.0,
                "sample_filter": 0.0,
                "warp_estimation": 0.0,
                "total": total_ms,
            }
            return warp
        
        raft_start = time.perf_counter()
        flow = self.raft(self.prev_frame, frame)  # [H, W, 2]
        raft_infer_ms = (time.perf_counter() - raft_start) * 1000.0

        sample_start = time.perf_counter()
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
        
        sample_filter_ms = (time.perf_counter() - sample_start) * 1000.0

        warp_start = time.perf_counter()
        if p0.shape[0] > 0:
            shift = np.median(flow_samples, axis=0)
            warp[:, 2] = shift.astype(np.float32)

        if self.scale_gmc != 1:
            warp[0, 2] /= self.scale_gmc
            warp[1, 2] /= self.scale_gmc
        warp_estimation_ms = (time.perf_counter() - warp_start) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0
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
