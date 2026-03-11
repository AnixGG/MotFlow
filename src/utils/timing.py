from __future__ import annotations

import statistics
from typing import Any


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0
    
    if len(values) == 1:
        return float(values[0])
    
    sorted_values = sorted(values)

    position = (len(sorted_values) - 1) * q

    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower

    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize_timing(sequence: str, latencies_ms: list[float], model_ms: list[float]) -> dict[str, float | int | str]:
    mean_wall = 0
    mean_model = 0

    if latencies_ms:
        mean_wall = statistics.fmean(latencies_ms)
    if model_ms:
        mean_model = statistics.fmean(model_ms)

    return {
        "sequence": sequence,
        "frames": len(latencies_ms),
        "wall_latency_ms_mean": round(mean_wall, 4),
        "wall_latency_ms_p50": round(percentile(latencies_ms, 0.50), 4),
        "wall_latency_ms_p95": round(percentile(latencies_ms, 0.95), 4),
        "wall_fps_mean": round(1000.0 / mean_wall, 4) if mean_wall > 0 else 0.0,
        "model_latency_ms_mean": round(mean_model, 4),
        "model_latency_ms_p50": round(percentile(model_ms, 0.50), 4),
        "model_latency_ms_p95": round(percentile(model_ms, 0.95), 4),
    }


def build_overall_timing_row(
    all_wall_latencies: list[float], all_model_latencies: list[float], all_frames: int
) -> dict[str, float | int | str]:
    overall_wall_mean = 0
    if all_wall_latencies:
        overall_wall_mean = statistics.fmean(all_wall_latencies)
        
    return {
        "sequence": "OVERALL",
        "frames": all_frames,
        "wall_latency_ms_mean": round(overall_wall_mean, 4),
        "wall_latency_ms_p50": round(percentile(all_wall_latencies, 0.50), 4),
        "wall_latency_ms_p95": round(percentile(all_wall_latencies, 0.95), 4),
        "wall_fps_mean": round(1000.0 / overall_wall_mean, 4) if overall_wall_mean > 0 else 0.0,
        "model_latency_ms_mean": round(statistics.fmean(all_model_latencies), 4) if all_model_latencies else 0.0,
        "model_latency_ms_p50": round(percentile(all_model_latencies, 0.50), 4),
        "model_latency_ms_p95": round(percentile(all_model_latencies, 0.95), 4),
    }


def attach_raw_timing(row: dict[str, Any], wall_latencies: list[float], model_latencies: list[float]) -> dict[str, Any]:
    timing_entry = dict(row)
    timing_entry["_wall_latencies_ms"] = wall_latencies
    timing_entry["_model_latencies_ms"] = model_latencies
    return timing_entry
