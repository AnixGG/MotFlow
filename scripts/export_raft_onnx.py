"""Export RAFT-large at 128 px and verify camera shifts on real frame pairs."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from raft.onnx_flow import ExportableRAFT, ONNXFlow
from raft.raft_wrapper import RAFTWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="SportsMOT clip's img1 directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    paths = sorted(path for path in args.images.glob("*.jpg") if not path.name.startswith("."))[:9]
    if len(paths) < 3:
        raise ValueError(f"Need at least three JPEG frames in {args.images}")
    frames = [cv2.imread(str(path)) for path in paths]
    if any(frame is None for frame in frames):
        raise ValueError("Could not decode the RAFT parity images")

    reference = RAFTWrapper(name="large", device="cpu")
    graph = ExportableRAFT(reference.model).eval()
    dummy = torch.zeros((1, 3, 128, 128))
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.out / "raft_large128.onnx"
    print("Exporting RAFT-large128 FP32, one flow update, ONNX opset 17", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            graph, (dummy, dummy), str(path), dynamo=False, opset_version=17,
            input_names=["previous", "current"], output_names=["flow"],
        )

    device = torch.device(args.device)
    reference.device = device
    graph.to(device)
    runtime = ONNXFlow(path, device, profile_prefix=args.out / "ort-profile")
    errors = []
    reduced = [cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR) for frame in frames]
    with torch.inference_mode():
        for i, (previous, current) in enumerate(zip(reduced, reduced[1:])):
            images = reference.prepare_pair(previous, current)
            height, width = frames[i + 1].shape[:2]
            scale = np.array([width / 128, height / 128])
            expected = graph(*images)[0].cpu().numpy()
            actual = runtime(*images)[0].cpu().numpy()
            if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                raise RuntimeError("Non-finite RAFT flow in parity check")
            error = np.abs((np.median(actual, axis=(1, 2)) - np.median(expected, axis=(1, 2))) * scale)
            errors.append({"frames": [paths[i].name, paths[i + 1].name], "abs_dx_dy_error_px": error.tolist()})

    profile = Path(runtime.session.end_profiling())
    events = json.loads(profile.read_text())
    operators = {name: set() for name in runtime.session.get_providers()}
    for event in events:
        metadata = event.get("args", {})
        if event.get("cat") == "Node" and "provider" in metadata and "op_name" in metadata:
            operators.setdefault(metadata["provider"], set()).add(metadata["op_name"])
    max_error = max(max(row["abs_dx_dy_error_px"]) for row in errors)
    report = {"max_shift_error_px": max_error, "pairs": errors,
              "operators_by_provider": {name: sorted(ops) for name, ops in operators.items()}}
    (args.out / "parity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if device.type == "cuda":
        required_cuda = {"Conv", "FusedConv", "MatMul", "Gemm", "GridSample"}
        if not operators.get("CUDAExecutionProvider") or operators.get("CPUExecutionProvider", set()) & required_cuda:
            raise RuntimeError("RAFT ONNX did not place its compute operators on CUDA. See parity.json.")
    if max_error > 0.25:
        raise RuntimeError(f"RAFT ONNX shift parity failed: {max_error:.4f} px exceeds 0.25 px")
    print(f"Parity passed over {len(errors)} pairs, max shift error {max_error:.4f} px", flush=True)


if __name__ == "__main__":
    main()
