from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small

class RAFTWrapper: # считает dense optical flow между двумя кадрами
    def __init__(
        self,
        name: Literal["small", "large"] = "small",
        device: str | None = None,
        mixed_precision: bool = False,
        num_flow_updates=1,
    ) -> None:
        if name not in {"small", "large"}:
            raise ValueError(f"Unsupported RAFT model size: {name}. Use 'small' or 'large'.")

        self.name = name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mixed_precision = bool(mixed_precision and self.device.type == "cuda")
        self.num_flow_updates = num_flow_updates

        if name == "small":
            weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights, progress=False)
        else:
            weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=weights, progress=False)

        self.model.mask_predictor = None
        self.preprocess = weights.transforms()
        self.model = self.model.to(self.device).eval()

    def _frame_to_tensor(self, frame: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(frame, torch.Tensor):
            tensor = frame.detach().clone()
            if tensor.ndim != 3:
                raise ValueError(f"Expected 3D tensor [C,H,W], got shape={tuple(tensor.shape)}")
            if tensor.shape[0] != 3:
                raise ValueError(f"Expected 3 channels, got shape={tuple(tensor.shape)}")
            if tensor.max() > 1:
                tensor = tensor / 255
            return tensor

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected frame with shape [H,W,3], got shape={frame.shape}")

        # BGR -> RGB
        rgb = frame[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255
        return tensor

    def infer(self, frame_prev: np.ndarray | torch.Tensor, frame_curr: np.ndarray | torch.Tensor) -> np.ndarray:
        image1 = self._frame_to_tensor(frame_prev).unsqueeze(0).to(self.device)
        image2 = self._frame_to_tensor(frame_curr).unsqueeze(0).to(self.device)

        if image1.shape != image2.shape:
            raise ValueError(f"Frame shapes must match: {tuple(image1.shape)} vs {tuple(image2.shape)}")

        image1, image2 = self.preprocess(image1, image2)

        with torch.inference_mode():
            if self.mixed_precision:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    flow_predictions = self.model(image1, image2, num_flow_updates=self.num_flow_updates)
            else:
                flow_predictions = self.model(image1, image2, num_flow_updates=self.num_flow_updates)

        flow = flow_predictions[-1] # [1,2,h,w]

        flow = flow[0] # [2,h,w]
        return flow.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    def __call__(self, frame_prev: np.ndarray | torch.Tensor, frame_curr: np.ndarray | torch.Tensor) -> np.ndarray:
        return self.infer(frame_prev, frame_curr)
