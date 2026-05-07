from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from utils.config import BaselineConfig
from utils.io import write_mot_rows
from utils.timing import cuda_synchronize, summarize_timing


def _read_raft_gmc_timing_ms(model: Any) -> float | None:
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)
    if not trackers:
        return None

    gmc = getattr(trackers[0], "gmc", None)
    value = getattr(gmc, "last_timing_ms", None)
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_botsort_sequence_baseline(model: Any, seq_dir: Path, output_path: Path, config: BaselineConfig) -> dict[str, Any]:
    # # reset
    predictor = getattr(model, "predictor", None)
    if predictor and getattr(predictor, "trackers", None):
        for tracker in predictor.trackers:
            tracker.reset()
        predictor.vid_path = [None] * len(predictor.trackers)

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

    mot_rows = []
    wall_latencies_ms = []
    preprocess_latencies_ms = []
    inference_latencies_ms = []
    postprocess_latencies_ms = []
    model_latencies_ms = []
    residual_latencies_ms = []
    raft_gmc_latencies_ms = []

    frame_idx = 0
    stream_iter = iter(result_stream)
    while True:
        cuda_synchronize(config.device)
        frame_start = time.perf_counter()
        try:
            result = next(stream_iter)
        except StopIteration:
            break
        cuda_synchronize(config.device)
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
                        float(x_center - width / 2),
                        float(y_center - height / 2),
                        float(width),
                        float(height),
                        float(confs[det_idx]),
                        -1,
                        -1,
                        -1,
                    ]
                )

        preprocess_ms = 0
        inference_ms = 0
        postprocess_ms = 0
        if getattr(result, "speed", None):
            preprocess_ms = float(result.speed.get("preprocess", 0))
            inference_ms = float(result.speed.get("inference", 0))
            postprocess_ms = float(result.speed.get("postprocess", 0))

        model_ms = preprocess_ms + inference_ms + postprocess_ms
        wall_ms = (time.perf_counter() - frame_start) * 1000
        residual_ms = max(0.0, wall_ms - model_ms)
        raft_gmc_ms = _read_raft_gmc_timing_ms(model)

        wall_latencies_ms.append(wall_ms)
        preprocess_latencies_ms.append(preprocess_ms)
        inference_latencies_ms.append(inference_ms)
        postprocess_latencies_ms.append(postprocess_ms)
        model_latencies_ms.append(model_ms)
        residual_latencies_ms.append(residual_ms)
        if raft_gmc_ms is not None:
            raft_gmc_latencies_ms.append(raft_gmc_ms)

    write_mot_rows(output_path, mot_rows)
    return {
        "rows": len(mot_rows),
        "wall_latencies_ms": wall_latencies_ms,
        "preprocess_latencies_ms": preprocess_latencies_ms,
        "inference_latencies_ms": inference_latencies_ms,
        "postprocess_latencies_ms": postprocess_latencies_ms,
        "model_latencies_ms": model_latencies_ms,
        "residual_latencies_ms": residual_latencies_ms,
        "raft_gmc_latencies_ms": raft_gmc_latencies_ms,
        "timing": summarize_timing(
            seq_dir.name,
            wall_latencies_ms,
            preprocess_latencies_ms,
            inference_latencies_ms,
            postprocess_latencies_ms,
            model_latencies_ms,
            residual_latencies_ms,
            raft_gmc_latencies_ms,
        ),
    }
