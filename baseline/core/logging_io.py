from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from .config import BaselineConfig


class TeeStream:
    def __init__(self, original: Any, log_file: Any) -> None:
        self.original = original
        self.log_file = log_file

    def write(self, data: str) -> int:
        self.original.write(data)
        self.log_file.write(data)
        self.log_file.flush()
        return len(data)

    def flush(self) -> None:
        self.original.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.original, "isatty", lambda: False)())


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_mot_rows(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def dump_run_config(path: Path, config: BaselineConfig, sequences: list[str], tracker_params: dict[str, Any], data_root: Path) -> None:
    payload = {
        "data_root": str(data_root),
        "sequences": sequences,
        "detector": {
            "model": config.model,
            "conf": config.conf,
            "iou": config.iou,
            "classes": config.classes,
            "device": config.device,
        },
        "resize": {
            "imgsz": config.imgsz,
            "mode": config.resize,
        },
        "tracker": tracker_params,
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_metrics_csv(path: Path, summary: Any) -> None:
    rows: list[dict[str, Any]] = []
    for index_name, row in summary.iterrows():
        rows.append(
            {
                "sequence": index_name,
                "MOTA": round(float(row["mota"]), 6),
                "IDF1": round(float(row["idf1"]), 6),
                "IDS": int(row["num_switches"]),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "MOTA", "IDF1", "IDS"])
        writer.writeheader()
        writer.writerows(rows)


def write_timing_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sequence",
        "frames",
        "wall_latency_ms_mean",
        "wall_latency_ms_p50",
        "wall_latency_ms_p95",
        "wall_fps_mean",
        "model_latency_ms_mean",
        "model_latency_ms_p50",
        "model_latency_ms_p95",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
