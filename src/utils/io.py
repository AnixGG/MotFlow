from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_mot_rows(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

def write_metrics_csv(path: Path, summary: Any) -> None:
    rows= []
    for index_name, row in summary.iterrows():
        rows.append(
            {
                "sequence": index_name,
                "MOTA": round(float(row["mota"]), 6),
                "IDF1": round(float(row["idf1"]), 6),
                "HOTA": round(float(row["hota"]), 6),
                "IDS": int(row["num_switches"]),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "MOTA", "IDF1", "HOTA", "IDS"])
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
        "preprocess_latency_ms_mean",
        "preprocess_latency_ms_p50",
        "preprocess_latency_ms_p95",
        "inference_latency_ms_mean",
        "inference_latency_ms_p50",
        "inference_latency_ms_p95",
        "postprocess_latency_ms_mean",
        "postprocess_latency_ms_p50",
        "postprocess_latency_ms_p95",
        "model_latency_ms_mean",
        "model_latency_ms_p50",
        "model_latency_ms_p95",
        "residual_latency_ms_mean",
        "residual_latency_ms_p50",
        "residual_latency_ms_p95",
        "raft_gmc_latency_ms_mean",
        "raft_gmc_latency_ms_p50",
        "raft_gmc_latency_ms_p95",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
