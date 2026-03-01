from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_numpy_compat() -> None:
    import numpy as np

    if not hasattr(np, "asfarray"):
        np.asfarray = lambda array, dtype=float: np.asarray(array, dtype=dtype)


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


def summarize_metrics(accumulators: dict[str, Any]) -> Any:
    ensure_numpy_compat()
    import motmetrics as mm

    mh = mm.metrics.create()
    return mh.compute_many(
        [accumulators[name] for name in accumulators],
        names=list(accumulators.keys()),
        metrics=["mota", "idf1", "num_switches"],
        generate_overall=True,
    )
