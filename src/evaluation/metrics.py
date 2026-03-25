from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def ensure_numpy_compat() -> None:
    import numpy as np

    if not hasattr(np, "asfarray"):
        np.asfarray = lambda array, dtype=float: np.asarray(array, dtype=dtype)
    if not hasattr(np, "float"):
        np.float = float


def load_gt_file(gt_path: Path) -> Any:
    import pandas as pd

    gt = pd.read_csv(
        gt_path,
        header=None,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "class", "visibility"],
    )
    return gt[(gt["class"] == 1) & (gt["conf"] == 1)]


def load_pred_file(pred_path: Path) -> Any:
    import pandas as pd

    if pred_path.stat().st_size == 0:
        return pd.DataFrame(columns=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])
    return pd.read_csv(
        pred_path,
        header=None,
        names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"],
    )


def frame_boxes(df: Any) -> Any:
    import numpy as np

    if df.empty:
        return np.empty((0, 4), dtype=float)
    return df[["bb_left", "bb_top", "bb_width", "bb_height"]].to_numpy(dtype=float)


def frame_ids(df: Any) -> Any:
    import numpy as np

    if df.empty:
        return np.empty((0,), dtype=int)
    return df["id"].to_numpy(dtype=int)


def _iou_similarity_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), dtype=float)

    gt_x1 = gt_boxes[:, 0]
    gt_y1 = gt_boxes[:, 1]
    gt_x2 = gt_boxes[:, 0] + gt_boxes[:, 2]
    gt_y2 = gt_boxes[:, 1] + gt_boxes[:, 3]

    pr_x1 = pred_boxes[:, 0]
    pr_y1 = pred_boxes[:, 1]
    pr_x2 = pred_boxes[:, 0] + pred_boxes[:, 2]
    pr_y2 = pred_boxes[:, 1] + pred_boxes[:, 3]

    inter_x1 = np.maximum(gt_x1[:, None], pr_x1[None, :])
    inter_y1 = np.maximum(gt_y1[:, None], pr_y1[None, :])
    inter_x2 = np.minimum(gt_x2[:, None], pr_x2[None, :])
    inter_y2 = np.minimum(gt_y2[:, None], pr_y2[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    gt_area = np.maximum(0, (gt_x2 - gt_x1) * (gt_y2 - gt_y1))
    pr_area = np.maximum(0, (pr_x2 - pr_x1) * (pr_y2 - pr_y1))
    union = gt_area[:, None] + pr_area[None, :] - inter_area

    out = np.zeros_like(inter_area, dtype=float)
    valid = union > 0
    out[valid] = inter_area[valid] / union[valid]
    return out


def _compute_hota(gt_path: Path, pred_path: Path) -> float | None:
    ensure_numpy_compat()

    import trackeval
    
    metric = trackeval.metrics.HOTA()
    

    gt = load_gt_file(gt_path)
    pred = load_pred_file(pred_path)
    max_frame = int(max(gt["frame"].max() if not gt.empty else 0, pred["frame"].max() if not pred.empty else 0))

    gt_id_map = {}
    if not gt.empty:
        gt_id_map = {
            int(track_id): idx
            for idx, track_id in enumerate(sorted(int(track_id) for track_id in gt["id"].unique()))
        }
    
    pred_id_map = {}
    if not pred.empty:
        pred_id_map = {
            int(track_id): idx
            for idx, track_id in enumerate(sorted(int(track_id) for track_id in pred["id"].unique()))
        }

    gt_ids_per_frame: list[np.ndarray] = []
    pred_ids_per_frame: list[np.ndarray] = []
    similarities: list[np.ndarray] = []

    for frame_idx in range(1, max_frame + 1):
        gt_frame = gt[gt["frame"] == frame_idx]
        pred_frame = pred[pred["frame"] == frame_idx]

        gt_ids = np.asarray([gt_id_map[int(track_id)] for track_id in frame_ids(gt_frame)], dtype=int)
        pred_ids = np.asarray([pred_id_map[int(track_id)] for track_id in frame_ids(pred_frame)], dtype=int)
        sim = _iou_similarity_matrix(frame_boxes(gt_frame), frame_boxes(pred_frame))
        
        gt_ids_per_frame.append(gt_ids)
        pred_ids_per_frame.append(pred_ids)
        similarities.append(sim)

    data = {
        "gt_ids": gt_ids_per_frame,
        "tracker_ids": pred_ids_per_frame,
        "similarity_scores": similarities,
        "num_gt_ids": len(gt_id_map),
        "num_tracker_ids": len(pred_id_map),
        "num_gt_dets": int(len(gt)),
        "num_tracker_dets": int(len(pred)),
    }

    try:
        result = metric.eval_sequence(data)
    except Exception:
        return None

    raw_hota = result.get("HOTA")
    if raw_hota is None:
        return None
    
    if isinstance(raw_hota, (list, tuple, np.ndarray)):
        values = np.asarray(raw_hota, dtype=float)
        if values.size == 0:
            return None
        return float(np.nanmean(values))
    
    return float(raw_hota)


def compute_hota_scores(sequence_paths: dict[str, tuple[Path, Path]]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for name, (gt_path, pred_path) in sequence_paths.items():
        scores[name] = _compute_hota(gt_path, pred_path)
    return scores


def evaluate_sequence(gt_path: Path, pred_path: Path) -> Any:
    ensure_numpy_compat()
    import motmetrics as mm

    gt = load_gt_file(gt_path)
    pred = load_pred_file(pred_path)
    max_frame = int(max(gt["frame"].max() if not gt.empty else 0, pred["frame"].max() if not pred.empty else 0))
    acc = mm.MOTAccumulator(auto_id=True)

    for frame_idx in range(1, max_frame + 1):
        gt_frame = gt[gt["frame"] == frame_idx]
        pred_frame = pred[pred["frame"] == frame_idx]
        distances = mm.distances.iou_matrix(frame_boxes(gt_frame), frame_boxes(pred_frame), max_iou=0.5)
        acc.update(frame_ids(gt_frame).tolist(), frame_ids(pred_frame).tolist(), distances)
    return acc


def summarize_metrics(accumulators: dict[str, Any], hota_scores: dict[str, float | None] | None = None) -> Any:
    ensure_numpy_compat()
    import motmetrics as mm

    mh = mm.metrics.create()
    summary = mh.compute_many(
        [accumulators[name] for name in accumulators],
        names=list(accumulators.keys()),
        metrics=["mota", "idf1", "num_switches"],
        generate_overall=True,
    )
    if hota_scores is not None:
        summary["hota"] = np.nan
        for name, value in hota_scores.items():
            if value is None:
                continue
            if name in summary.index:
                summary.at[name, "hota"] = float(value)
        valid_values = [value for value in hota_scores.values() if value is not None]
        if valid_values and "OVERALL" in summary.index:
            summary.at["OVERALL", "hota"] = float(np.mean(valid_values))
    return summary
