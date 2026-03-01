from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from utils.config import BaselineConfig
from utils.io import write_mot_rows
from utils.timing import summarize_timing


def resolve_sequence_dir(data_root: Path, sequence: str) -> Path:
    seq_dir = data_root / sequence
    img_dir = seq_dir / "img1"
    gt_path = seq_dir / "gt" / "gt.txt"
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing frames directory: {img_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground truth file: {gt_path}")
    return seq_dir


def run_sequence(model: Any, seq_dir: Path, output_path: Path, config: BaselineConfig) -> dict[str, Any]:
    result_stream = model.track(
        source=str(seq_dir / "img1"),
        tracker=config.tracker,
        stream=True,
        persist=True,
        conf=config.conf,
        iou=config.iou,
        imgsz=config.imgsz,
        device=config.device,
        classes=config.classes,
        verbose=False,
        save=False,
    )

    mot_rows: list[list[float]] = []
    wall_latencies_ms: list[float] = []
    model_latencies_ms: list[float] = []

    frame_idx = 0
    stream_iter = iter(result_stream)
    while True:
        frame_start = time.perf_counter()
        try:
            result = next(stream_iter)
        except StopIteration:
            break
        frame_idx += 1

        boxes = result.boxes
        if boxes is not None and boxes.id is not None and len(boxes) > 0:
            xywh = boxes.xywh.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for det_idx, track_id in enumerate(ids):
                x_center, y_center, width, height = xywh[det_idx]
                mot_rows.append(
                    [
                        frame_idx,
                        int(track_id),
                        float(x_center - width / 2.0),
                        float(y_center - height / 2.0),
                        float(width),
                        float(height),
                        float(confs[det_idx]),
                        -1,
                        -1,
                        -1,
                    ]
                )

        model_ms = 0.0
        if getattr(result, "speed", None):
            model_ms = float(sum(result.speed.values()))
        model_latencies_ms.append(model_ms)
        wall_latencies_ms.append((time.perf_counter() - frame_start) * 1000.0)

    write_mot_rows(output_path, mot_rows)
    return {
        "rows": len(mot_rows),
        "wall_latencies_ms": wall_latencies_ms,
        "model_latencies_ms": model_latencies_ms,
        "timing": summarize_timing(seq_dir.name, wall_latencies_ms, model_latencies_ms),
    }
