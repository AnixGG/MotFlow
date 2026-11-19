"""ONNX execution for the fixed-size RAFT flow graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class ONNXFlow:
    def __init__(self, model: Path, device: torch.device, profile_prefix: Path | None = None) -> None:
        import onnxruntime as ort

        self.device = device
        options = ort.SessionOptions()
        if profile_prefix is not None:
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_prefix)
        providers = ["CPUExecutionProvider"]
        if device.type == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("ONNX Runtime CUDAExecutionProvider is unavailable")
            providers = [("CUDAExecutionProvider", {
                "device_id": device.index if device.index is not None else torch.cuda.current_device(),
                "user_compute_stream": str(torch.cuda.current_stream(device).cuda_stream),
                "use_tf32": "0",
            }), "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
        if device.type == "cuda" and self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError("ONNX Runtime failed to initialize CUDA for RAFT")
        self.session.disable_fallback()

    def __call__(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cpu":
            result = self.session.run(["flow"], {"previous": previous.numpy(), "current": current.numpy()})[0]
            return torch.from_numpy(result)
        previous, current = previous.contiguous(), current.contiguous()
        batch, _, height, width = current.shape
        flow = current.new_empty((batch, 2, height, width))
        binding = self.session.io_binding()
        for name, tensor in (("previous", previous), ("current", current)):
            binding.bind_input(name, "cuda", tensor.device.index, np.float32, tuple(tensor.shape), tensor.data_ptr())
        binding.bind_output("flow", "cuda", flow.device.index, np.float32, tuple(flow.shape), flow.data_ptr())
        self.session.run_with_iobinding(binding)
        return flow


class ExportableRAFT(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return self.model(previous, current, num_flow_updates=1)[-1]
