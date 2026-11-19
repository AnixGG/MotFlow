from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small

class RAFTWrapper: # считает dense optical flow между двумя кадрами
    def __init__(
        self,
        name: Literal["small", "large"] = "small",
        device: str | None = None,
        mixed_precision: bool = False,
        num_flow_updates=1,
        onnx_model: Path | None = None,
    ) -> None:
        if name not in {"small", "large"}:
            raise ValueError(f"Unsupported RAFT size")

        self.name = name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mixed_precision = bool(mixed_precision and self.device.type == "cuda")
        self.num_flow_updates = num_flow_updates

        weights = Raft_Small_Weights.DEFAULT if name == "small" else Raft_Large_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.onnx = None
        if onnx_model is not None:
            if mixed_precision or num_flow_updates != 1:
                raise ValueError("The RAFT ONNX graph requires FP32 and one flow update")
            from raft.onnx_flow import ONNXFlow

            self.onnx = ONNXFlow(onnx_model, self.device)
        else:
            build = raft_small if name == "small" else raft_large
            self.model = build(weights=weights, progress=False)
            self.model.mask_predictor = None
            self.model = self.model.to(self.device).eval()

    def _frame_to_tensor(self, frame: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(frame, torch.Tensor):
            tensor = frame.detach().clone()
            if tensor.ndim != 3:
                raise ValueError(f"Expected [C,H,W], got {tuple(tensor.shape)}")
            if tensor.shape[0] != 3:
                raise ValueError(f"Expected 3 channels, got {tuple(tensor.shape)}")
            if tensor.max() > 1:
                tensor = tensor / 255
            return tensor

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected [H,W,3], got {frame.shape}")

        # BGR -> RGB
        rgb = frame[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255
        return tensor

    def prepare_pair(self, frame_prev: np.ndarray | torch.Tensor, frame_curr: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image1 = self._frame_to_tensor(frame_prev).unsqueeze(0).to(self.device)
        image2 = self._frame_to_tensor(frame_curr).unsqueeze(0).to(self.device)

        if image1.shape != image2.shape:
            raise ValueError(f"image1.shape != image2.shape {tuple(image1.shape)} != {tuple(image2.shape)}")

        return self.preprocess(image1, image2)

    def infer(self, frame_prev: np.ndarray | torch.Tensor, frame_curr: np.ndarray | torch.Tensor) -> np.ndarray:
        image1, image2 = self.prepare_pair(frame_prev, frame_curr)
        with torch.inference_mode():
            if self.onnx is not None:
                flow = self.onnx(image1, image2)
            elif self.mixed_precision:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    flow = self.model(image1, image2, num_flow_updates=self.num_flow_updates)[-1]
            else:
                flow = self.model(image1, image2, num_flow_updates=self.num_flow_updates)[-1]

        flow = flow[0] # [2,h,w]
        return flow.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    def __call__(self, frame_prev: np.ndarray | torch.Tensor, frame_curr: np.ndarray | torch.Tensor) -> np.ndarray:
        return self.infer(frame_prev, frame_curr)
